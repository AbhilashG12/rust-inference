use clap::{Parser, Subcommand};
use model::loader::ModelLoader;
use model::tokenizer::Tokenizer;
use std::io::{self, Write};
use std::time::Instant;
use tensor::cache::KVCache;
use tensor::nn::{Embedding, Linear, TransformerBlock};
use tensor::quant::QuantizedTensor;
use tensor::sampler::{GenerationConfig, Sampler};
use tensor::tensor::Tensor;

#[derive(Parser)]
#[command(
    name = "rustinfer",
    version = "1.0",
    about = "Pure-Rust LLM Inference Engine"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Generate {
        #[arg(long, default_value = "tiny_llama.safetensors")]
        model: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        #[arg(long)]
        prompt: String,
    },
}

pub struct LlmEngine {
    tokenizer: Tokenizer,
    embedding: Embedding,
    block: TransformerBlock,
    final_norm: Tensor,
    lm_head: Linear,
}

impl LlmEngine {
    pub fn load(model_path: &str, tokenizer_path: &str) -> Self {
        let weights = ModelLoader::load_safetensors(model_path).expect("Failed to load model");
        let tokenizer = Tokenizer::load(tokenizer_path).expect("Failed to load tokenizer");
        let norm_weight = weights
            .get("model.layers.0.input_layernorm.weight")
            .unwrap();

        let get_linear = |name: &str| -> Linear {
            let raw = weights.get(name).unwrap();
            let weight = QuantizedTensor::quantize(raw).dequantize();
            let bias = Tensor::zeroes(vec![weight.shape.dims()[1]]);
            Linear::new(weight, bias)
        };

        Self {
            tokenizer,
            embedding: Embedding {
                weight: weights.get("model.embed_tokens.weight").unwrap().clone(),
            },
            block: TransformerBlock {
                norm1: norm_weight.clone(),
                norm2: weights
                    .get("model.layers.0.post_attention_layernorm.weight")
                    .unwrap()
                    .clone(),
                w_q: get_linear("model.layers.0.self_attn.q_proj.weight"),
                w_k: get_linear("model.layers.0.self_attn.k_proj.weight"),
                w_v: get_linear("model.layers.0.self_attn.v_proj.weight"),
                w_o: get_linear("model.layers.0.self_attn.o_proj.weight"),
                ffn_up: get_linear("model.layers.0.mlp.up_proj.weight"),
                ffn_down: get_linear("model.layers.0.mlp.down_proj.weight"),
            },
            final_norm: weights.get("model.norm.weight").unwrap().clone(),
            lm_head: get_linear("lm_head.weight"),
        }
    }

    pub fn generate(&self, prompt: &str, config: GenerationConfig) {
        let mut tokens = self.tokenizer.encode(prompt);
        let mut kv_cache = KVCache::new(1);

        println!("--------------------------------------------------");
        print!(" Generation: {}", prompt);
        io::stdout().flush().unwrap();

        let start_time = Instant::now();
        let mut decode_start = Instant::now();
        let prompt_len = tokens.len();
        for pos in 0..prompt_len {
            let token_id = tokens[pos];
            let mut hidden_state = self.embedding.forward(token_id as usize);

            hidden_state = self
                .block
                .forward(&hidden_state, pos, 0, &mut kv_cache)
                .unwrap();

            if pos == prompt_len - 1 {
                hidden_state = hidden_state.rms_norm(&self.final_norm, 1e-5).unwrap();
                let logits = self.lm_head.forward(&hidden_state).unwrap();
                let next_token = Sampler::sample(&logits, &config);
                let next_word = self.tokenizer.decode(next_token);
                print!("{}", next_word);
                io::stdout().flush().unwrap();
                tokens.push(next_token);
                decode_start = Instant::now();
            }
        }

        for _ in 0..config.max_tokens {
            let token_id = *tokens.last().unwrap();
            let pos = tokens.len() - 1;

            let mut hidden_state = self.embedding.forward(token_id as usize);
            hidden_state = self
                .block
                .forward(&hidden_state, pos, 0, &mut kv_cache)
                .unwrap();
            hidden_state = hidden_state.rms_norm(&self.final_norm, 1e-5).unwrap();
            let logits = self.lm_head.forward(&hidden_state).unwrap();

            let next_token = Sampler::sample(&logits, &config);
            let next_word = self.tokenizer.decode(next_token);

            print!("{}", next_word);
            io::stdout().flush().unwrap();
            tokens.push(next_token);
        }

        let prefill_time = start_time.elapsed() - decode_start.elapsed();
        let decode_time = decode_start.elapsed();
        let decode_tps = config.max_tokens as f64 / decode_time.as_secs_f64();

        let hidden_dim = self.final_norm.shape.dims()[0];
        let num_layers = 1; 
        let bytes_per_token = 2 * num_layers * hidden_dim * 4; 
        let kb_per_token = bytes_per_token as f64 / 1024.0;
        let total_kv_memory = kb_per_token * (tokens.len() as f64);

        println!("\n--------------------------------------------------");
        println!("Performance Benchmarks:");
        println!("   - Prefill Latency: {:.2} ms", prefill_time.as_secs_f64() * 1000.0);
        println!("   - Decode Latency:  {:.2} ms", decode_time.as_secs_f64() * 1000.0);
        println!("   - Decode Speed:    {:.2} tokens/sec", decode_tps);
        println!("   - Memory/Token:    {:.2} KB/token (KV Cache)", kb_per_token);
        println!("   - Total KV Memory: {:.2} KB", total_kv_memory);    }
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Generate {
            model,
            tokenizer,
            prompt,
        } => {
            let engine = LlmEngine::load(model, tokenizer);
            let config = GenerationConfig::default();
            engine.generate(prompt, config);
        }
    }
}
