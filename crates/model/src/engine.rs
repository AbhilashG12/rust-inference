use crate::loader::{LoadedWeight, ModelLoader};
use crate::tokenizer::Tokenizer;
use std::io::{self, Write};
use tensor::cache::KVCache;
use tensor::nn::{Embedding, Linear, TransformerBlock};
use tensor::sampler::{GenerationConfig, Sampler};
use tensor::tensor::Tensor;

pub struct GenerationOutput {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub ttft_ms: u64,
    pub total_latency_ms: u64,
    pub kv_memory_mb: f32,
}

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
        
        // Lock Rayon to Physical Cores only. This eliminates HyperThreading cache contention
        // and keeps threads "hot" for the M=1 decode phase!
        let physical_cores = num_cpus::get_physical();
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(physical_cores)
            .build_global();
        println!(" Optimized Thread Pool initialized with {} physical cores.", physical_cores);

        let mut weights = ModelLoader::load_safetensors(model_path).expect("Failed to load model");
        let tokenizer = Tokenizer::load(tokenizer_path).expect("Failed to load tokenizer");

        // Use local macros instead of closures to avoid borrow checker overlap on `weights`
        macro_rules! get_linear {
            ($name:expr) => {{
                let q8_weight = match weights
                    .remove($name)
                    .unwrap_or_else(|| panic!("Missing weight: {}", $name))
                {
                    LoadedWeight::Q8(q) => q,
                    _ => panic!("Expected Q8 Quantized Tensor for {}", $name),
                };

                let bias_name = $name.replace(".weight", ".bias");
                let bias = if let Some(b) = weights.remove(&bias_name) {
                    match b {
                        LoadedWeight::Fp32(t) => t,
                        _ => panic!("Expected FP32 Tensor for bias {}", bias_name),
                    }
                } else {
                    let out_features = q8_weight.shape.dims()[0];
                    Tensor::zeroes(vec![out_features])
                };

                Linear::new(q8_weight, bias)
            }};
        }

        macro_rules! get_fp32 {
            ($name:expr) => {
                match weights
                    .remove($name)
                    .unwrap_or_else(|| panic!("Missing weight: {}", $name))
                {
                    LoadedWeight::Fp32(t) => t,
                    _ => panic!("Expected FP32 Tensor for {}", $name),
                }
            };
        }

        let mut blocks = Vec::new();
        let mut layer_idx = 0;

        println!(" Building Neural Network Layers...");
        loop {
            let norm_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
            if !weights.contains_key(&norm_name) {
                break;
            }

            blocks.push(TransformerBlock {
                norm1: get_fp32!(&norm_name),
                norm2: get_fp32!(&format!(
                    "model.layers.{}.post_attention_layernorm.weight",
                    layer_idx
                )),
                w_q: get_linear!(&format!(
                    "model.layers.{}.self_attn.q_proj.weight",
                    layer_idx
                )),
                w_k: get_linear!(&format!(
                    "model.layers.{}.self_attn.k_proj.weight",
                    layer_idx
                )),
                w_v: get_linear!(&format!(
                    "model.layers.{}.self_attn.v_proj.weight",
                    layer_idx
                )),
                w_o: get_linear!(&format!(
                    "model.layers.{}.self_attn.o_proj.weight",
                    layer_idx
                )),
                ffn_gate: get_linear!(&format!("model.layers.{}.mlp.gate_proj.weight", layer_idx)),
                ffn_up: get_linear!(&format!("model.layers.{}.mlp.up_proj.weight", layer_idx)),
                ffn_down: get_linear!(&format!("model.layers.{}.mlp.down_proj.weight", layer_idx)),
            });
            layer_idx += 1;
        }

        println!(" Built {} Transformer Layers.", blocks.len());

        // Extract embed_tokens EXACTLY ONCE
        let embed_weight = get_fp32!("model.embed_tokens.weight");

        let lm_head = if weights.contains_key("lm_head.weight") {
            get_linear!("lm_head.weight")
        } else {
            println!(" Tied Embeddings detected! Re-using embed_tokens for lm_head.");
            let raw_embed = &embed_weight; 
            
            // SafeTensors stores embed_tokens as [vocab_size, hidden_dim].
            // This is EXACTLY the shape lm_head needs! No transpose required!
            let q8_lm_head = tensor::quant::QuantizedTensorQ8_0::quantize(raw_embed);
            
            let bias = Tensor::zeroes(vec![raw_embed.shape.dims()[0]]); // vocab_size
            Linear::new(q8_lm_head, bias)
        };

        Self {
            tokenizer,
            embedding: Embedding { weight: embed_weight }, // Safely consumes the original here
            blocks,
            final_norm: get_fp32!("model.norm.weight"),
            lm_head,
        }
    }

    pub fn generate_streaming<F>(&self, prompt: &str, config: GenerationConfig, mut on_token: F) -> GenerationOutput 
    where
        F: FnMut(&str)
    {
        let mut tokens = self.tokenizer.encode(prompt);
        let prompt_len = tokens.len();
        let num_layers = self.blocks.len();
        
        // Give the cache a 2048 token max capacity boundary!
        let mut kv_cache = KVCache::new(num_layers, 2048); 
        
        let start_time = std::time::Instant::now();
        let mut ttft_ms = 0;
        let mut generated_count = 0;
        let mut output_text = String::new();

        if prompt_len == 0 {
            return GenerationOutput {
                text: String::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
                ttft_ms: 0,
                total_latency_ms: 0,
                kv_memory_mb: 0.0,
            };
        }

        // 1. PREFILL PHASE
        for pos in 0..prompt_len {
            let token_id = tokens[pos];
            let mut hidden_state = self.embedding.forward(token_id as usize);
            
            for (i, block) in self.blocks.iter().enumerate() {
                hidden_state = block.forward(&hidden_state, pos, i, &mut kv_cache).unwrap();
            }
            
            // CRITICAL: Increment the cache clock AFTER all layers process the token
            kv_cache.current_seq_len += 1; 
            
            if pos == prompt_len - 1 {
                hidden_state = hidden_state.rms_norm(&self.final_norm, 1e-5).unwrap();
                let logits = self.lm_head.forward(&hidden_state).unwrap();
                let next_token = Sampler::sample(&logits, &config, &tokens);
                let next_word = self.tokenizer.decode(next_token);
                
                ttft_ms = start_time.elapsed().as_millis() as u64;
                if next_token != 0 && next_token != 2 && !next_word.contains("<|im_end|>") {
                    output_text.push_str(&next_word);
                    on_token(&next_word);
                    tokens.push(next_token);
                    generated_count += 1;
                }
            }
        }

        // 2. DECODE PHASE
        for _ in 0..config.max_tokens {
            let token_id = *tokens.last().unwrap();
            let pos = tokens.len() - 1;
            
            let mut hidden_state = self.embedding.forward(token_id as usize);
            
            for (i, block) in self.blocks.iter().enumerate() {
                hidden_state = block.forward(&hidden_state, pos, i, &mut kv_cache).unwrap();
            }
            
            // CRITICAL: Increment the cache clock AFTER all layers process the token
            kv_cache.current_seq_len += 1;
            
            hidden_state = hidden_state.rms_norm(&self.final_norm, 1e-5).unwrap();
            let logits = self.lm_head.forward(&hidden_state).unwrap();
            
            let next_token = Sampler::sample(&logits, &config, &tokens);
            let next_word = self.tokenizer.decode(next_token);
            
            // Core engine should ONLY stop on EOS tokens or max_tokens
            if next_token == 0 || next_token == 2 || next_word.contains("<|im_end|>") || next_word.contains("<|endoftext|>") {
                break;
            }
            
            output_text.push_str(&next_word);
            on_token(&next_word);
            tokens.push(next_token);
            generated_count += 1;
        }

        let hidden_dim = self.final_norm.shape.dims()[0];
        let bytes_per_token = 2 * num_layers * hidden_dim * 4;
        let total_kv_mb = (bytes_per_token * tokens.len()) as f32 / (1024.0 * 1024.0);

        GenerationOutput {
            text: output_text,
            prompt_tokens: prompt_len,
            completion_tokens: generated_count,
            ttft_ms,
            total_latency_ms: start_time.elapsed().as_millis() as u64,
            kv_memory_mb: total_kv_mb,
        }
    }

    pub fn generate_for_api(&self, prompt: &str, config: GenerationConfig) -> GenerationOutput {
        self.generate_streaming(prompt, config, |_| {})
    }

    pub fn generate(&self, prompt: &str, config: GenerationConfig) {
        let formatted_prompt = if prompt.contains("<|im_start|>") {
            prompt.to_string()
        } else {
            format!(
                "<|im_start|>system\nYou are a helpful, concise programming and AI assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                prompt
            )
        };

        println!("--------------------------------------------------");
        println!("Generation: {}", prompt);
        io::stdout().flush().unwrap();

        let output = self.generate_streaming(&formatted_prompt, config, |token| {
            print!("{}", token);
            io::stdout().flush().unwrap();
        });

        println!("\n--------------------------------------------------");
        println!(" Performance Benchmarks:");
        println!("   - Latency: {:.2} ms", output.total_latency_ms);
        println!("   - Memory:  {:.2} MB", output.kv_memory_mb);
    }
}
