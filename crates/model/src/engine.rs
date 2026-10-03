use crate::loader::ModelLoader;
use crate::tokenizer::Tokenizer;
use tensor::tensor::Tensor;
use tensor::nn::{Embedding, Linear, TransformerBlock};
use tensor::sampler::{GenerationConfig, Sampler};
use tensor::cache::KVCache;
use std::io::{self, Write};
use std::time::Instant;

pub struct LlmEngine {
    tokenizer: Tokenizer,
    embedding: Embedding,
    blocks: Vec<TransformerBlock>, 
    final_norm: Tensor,
    lm_head: Linear,
}

impl LlmEngine {
    pub fn load(model_path: &str, tokenizer_path: &str) -> Self {
        println!(" Loading model weights into RAM (this may take a few seconds)...");
        let weights = ModelLoader::load_safetensors(model_path).expect("Failed to load model");
        let tokenizer = Tokenizer::load(tokenizer_path).expect("Failed to load tokenizer");

        // Added a custom error message so we never get a blind "None" panic again!
        let get_linear = |name: &str| -> Linear {
            let raw = weights.get(name).unwrap_or_else(|| panic!(" Missing weight in safetensors: {}", name));
            // let weight = QuantizedTensor::quantize(raw).dequantize();
            let weight = raw.clone();
            let bias = Tensor::zeroes(vec![weight.shape.dims()[1]]);
            Linear::new(weight, bias)
        };

        let mut blocks = Vec::new();
        let mut layer_idx = 0;
        
        println!(" Building Neural Network Layers...");
        loop {
            let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
            if !weights.contains_key(&norm_name) {
                break;
            }

            blocks.push(TransformerBlock {
                norm1: weights.get(&norm_name).unwrap().clone(),
                norm2: weights.get(&format!("model.layers.{}.post_attention_layernorm.weight", layer_idx)).unwrap().clone(),
                w_q: get_linear(&format!("model.layers.{}.self_attn.q_proj.weight", layer_idx)),
                w_k: get_linear(&format!("model.layers.{}.self_attn.k_proj.weight", layer_idx)),
                w_v: get_linear(&format!("model.layers.{}.self_attn.v_proj.weight", layer_idx)),
                w_o: get_linear(&format!("model.layers.{}.self_attn.o_proj.weight", layer_idx)),
                ffn_gate: get_linear(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx)),
                ffn_up: get_linear(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx)),
                ffn_down: get_linear(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx)),
            });
            layer_idx += 1;
        }

        println!(" Built {} Transformer Layers.", blocks.len());

        // ==========================================
        // THE TIED EMBEDDINGS FIX
        // ==========================================
        let lm_head = if weights.contains_key("lm_head.weight") {
            get_linear("lm_head.weight")
        } else {
            println!(" Tied Embeddings detected! Re-using embed_tokens for lm_head.");
            let raw = weights.get("model.embed_tokens.weight").unwrap();
            let mut weight = raw.clone();
            
            if weight.shape.dims()[0] > weight.shape.dims()[1] {
                // Transpose [49152, 576] -> [576, 49152]
                let transposed = weight.transpose_2d().expect("Failed to transpose tied embeddings");
                // Flatten into physical contiguous row-major memory:
                let (rows, cols) = (transposed.shape.dims()[0], transposed.shape.dims()[1]);
                let mut contiguous_data = Vec::with_capacity(rows * cols);
                for r in 0..rows {
                    for c in 0..cols {
                        let idx = r * transposed.strides[0] + c * transposed.strides[1];
                        contiguous_data.push(transposed.data[idx]);
                    }
                }
                weight = Tensor::new(contiguous_data, vec![rows, cols]).unwrap();
            }
            
            let bias = Tensor::zeroes(vec![weight.shape.dims()[1]]);
            Linear::new(weight, bias)
        };

        Self {
            tokenizer,
            embedding: Embedding { weight: weights.get("model.embed_tokens.weight").unwrap().clone() },
            blocks,
            final_norm: weights.get("model.norm.weight").unwrap().clone(),
            lm_head,
        }
    }

    pub fn generate(&self, prompt: &str, config: GenerationConfig) {
        let mut tokens = self.tokenizer.encode(prompt);
        let num_layers = self.blocks.len();
        let mut kv_cache = KVCache::new(num_layers); // Initialize cache for ALL layers
        
        println!("--------------------------------------------------");
        print!("Generation: {}", prompt);
        io::stdout().flush().unwrap();

        let start_time = Instant::now();
        let mut decode_start = Instant::now();

        // 1. PREFILL PHASE
        let prompt_len = tokens.len();
        for pos in 0..prompt_len {
            let token_id = tokens[pos];
            let mut hidden_state = self.embedding.forward(token_id as usize);
            
            // Pass through EVERY layer
            for (i, block) in self.blocks.iter().enumerate() {
                hidden_state = block.forward(&hidden_state, pos, i, &mut kv_cache).unwrap();
            }
            
            if pos == prompt_len - 1 {
                hidden_state = hidden_state.rms_norm(&self.final_norm, 1e-5).unwrap();
                let logits = self.lm_head.forward(&hidden_state).unwrap();
                let next_token = Sampler::sample(&logits, &config,&tokens);
                let next_word = self.tokenizer.decode(next_token);
                print!("{}", next_word);
                io::stdout().flush().unwrap();
                
                tokens.push(next_token);
                decode_start = Instant::now();
            }
        }

        // 2. DECODE PHASE
        for _ in 0..config.max_tokens {
            let token_id = *tokens.last().unwrap();
            let pos = tokens.len() - 1; 
            
            let mut hidden_state = self.embedding.forward(token_id as usize);
            
            // Pass through EVERY layer
            for (i, block) in self.blocks.iter().enumerate() {
                hidden_state = block.forward(&hidden_state, pos, i, &mut kv_cache).unwrap();
            }
            
            hidden_state = hidden_state.rms_norm(&self.final_norm, 1e-5).unwrap();
            let logits = self.lm_head.forward(&hidden_state).unwrap();
            
            let next_token = Sampler::sample(&logits, &config,&tokens);
            let next_word = self.tokenizer.decode(next_token);
            
            print!("{}", next_word);
            io::stdout().flush().unwrap();
            tokens.push(next_token);
        }

        let prefill_time = start_time.elapsed() - decode_start.elapsed();
        let decode_time = decode_start.elapsed();
        let decode_tps = config.max_tokens as f64 / decode_time.as_secs_f64();

        let hidden_dim = self.final_norm.shape.dims()[0];
        let bytes_per_token = 2 * num_layers * hidden_dim * 4; 
        let kb_per_token = bytes_per_token as f64 / 1024.0;
        let total_kv_memory = kb_per_token * (tokens.len() as f64);

        println!("\n--------------------------------------------------");
        println!(" Performance Benchmarks:");
        println!("   - Prefill Latency: {:.2} ms", prefill_time.as_secs_f64() * 1000.0);
        println!("   - Decode Latency:  {:.2} ms", decode_time.as_secs_f64() * 1000.0);
        println!("   - Decode Speed:    {:.2} tokens/sec", decode_tps);
        println!("   - Memory/Token:    {:.2} KB/token (KV Cache)", kb_per_token);
        println!("   - Total KV Memory: {:.2} KB", total_kv_memory);
    }
}