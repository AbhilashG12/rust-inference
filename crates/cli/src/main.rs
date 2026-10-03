use clap::{Parser, Subcommand};
use model::LlmEngine;
use tensor::sampler::GenerationConfig;
use tensor::profiler::PROFILER;
use model::loader::ModelLoader;

#[derive(Parser)]
#[command(name = "rustinfer", version = "1.0", about = "Pure-Rust LLM Inference Engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Inspect {
        #[arg(long)]
        model: String,
    },
    Generate {
        #[arg(long, default_value = "tiny_llama.safetensors")]
        model: String,
        #[arg(long, default_value = "tokenizer.json")]
        tokenizer: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = false)]
        profile: bool, 
    },
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Inspect { model } => {
            println!("🔍 Inspecting Model: {}", model);
            match ModelLoader::load_safetensors(model) {
                Ok(weights) => {
                    println!(" Successfully parsed {} tensors.", weights.len());
                    let mut total_bytes = 0;
                    for (name, tensor) in &weights {
                        total_bytes += tensor.data.len() * 4; // f32 is 4 bytes
                        if name.contains("layer") && name.contains("0") {
                            println!("   - {}: {:?}", name, tensor.shape.dims());
                        }
                    }
                    println!("\n Total Unquantized Size: {:.2} MB", total_bytes as f64 / 1024.0 / 1024.0);
                }
                Err(e) => println!(" Error loading model: {}", e),
            }
        }
        Commands::Generate { model, tokenizer, prompt, profile } => {
            let engine = LlmEngine::load(model, tokenizer);
            let config = GenerationConfig::default();
            
            engine.generate(prompt, config);

            if *profile {
                if let Ok(prof) = PROFILER.lock() {
                    prof.print_summary();
                }
            }
        }
    }
}