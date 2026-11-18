use clap::{Parser, Subcommand};
use model::LlmEngine;
use model::embedder::EmbedderEngine;
use model::rag::{DocumentStore, VectorStore};
use model::tokenizer::Tokenizer;
use tensor::profiler::PROFILER;
use tensor::sampler::GenerationConfig;
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "rustinfer",
    version = "1.0",
    about = "Pure-Rust LLM Inference Engine & Air-Gapped RAG"
)]
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
        #[arg(long, default_value = "smollm-instruct.safetensors")]
        model: String,
        #[arg(long, default_value = "smollm_instruct_tokenizer.json")]
        tokenizer: String,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = false)]
        profile: bool,
    },
    Rag {
        #[arg(long, default_value = "smollm-instruct.safetensors")]
        llm: String,
        #[arg(long, default_value = "smollm_instruct_tokenizer.json")]
        llm_tokenizer: String,
        #[arg(long, default_value = "minilm-l6-v2.safetensors")]
        embedder: String,
        #[arg(long, default_value = "minilm_tokenizer.json")]
        embedder_tokenizer: String,
        #[arg(long)]
        docs_dir: String,
        #[arg(long)]
        prompt: String,
    }
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Inspect { model } => {
            // [Keep your existing Inspect logic exactly as it is...]
            println!("🔍 Inspecting Model: {}", model);
        }
        
        Commands::Generate { model, tokenizer, prompt, profile } => {
            // [Keep your existing Generate logic exactly as it is...]
            let engine = LlmEngine::load(model, tokenizer);
            engine.generate(prompt, GenerationConfig::default());
            if *profile {
                if let Ok(prof) = PROFILER.lock() { prof.print_summary(); }
            }
        }
        
        Commands::Rag { llm, llm_tokenizer, embedder, embedder_tokenizer, docs_dir, prompt } => {
            println!("--- [1] Booting Air-Gapped FDE Engine ---");
            let llm_engine = LlmEngine::load(llm, llm_tokenizer);
            
            // Phase 1: Load the 20M parameter semantic encoder
            let embed_engine = EmbedderEngine::load(embedder, embedder_tokenizer);
            let embed_tok = Tokenizer::load(embedder_tokenizer).unwrap();

            println!("\n--- [2] Ingesting Local Documents ---");
            let mut doc_store = DocumentStore::new();
            let start = Instant::now();
            doc_store.ingest_directory(docs_dir, &embed_tok, 256, 32);
            
            let mut vec_store = VectorStore::new();
            vec_store.embed_store(&doc_store, &embed_engine);
            println!(" Indexing took: {:.2}s", start.elapsed().as_secs_f32());

            println!("\n--- [3] Vector Search ---");
            let search_start = Instant::now();
            let top_3_indices = vec_store.search(prompt, &embed_engine, 3);
            println!(" Search completed in: {:.2}ms", search_start.elapsed().as_millis());

            let mut context_text = String::new();
            for (rank, &idx) in top_3_indices.iter().enumerate() {
                let chunk = &doc_store.chunks[idx];
                println!("  -> Match {}: [{}]", rank + 1, chunk.source_file);
                context_text.push_str(&chunk.text);
                context_text.push_str("\n\n");
            }

            println!("\n--- [4] Context-Aware Generation ---");
            use std::io::Write;
            
            // Bypass ChatML special tokens and use a strict, uppercase instruction block
            let augmented_prompt = format!(
                "SYSTEM: You are a strict data extraction AI. Read the CONTEXT below and answer the user's QUESTION. Do not invent information. If the answer is not in the CONTEXT, reply 'I do not know'.\n\nCONTEXT:\n{}\n\nQUESTION:\n{}\n\nANSWER:\n",
                context_text.trim(), prompt
            );

            let mut config = GenerationConfig::default();
            config.temperature = 0.0; // 0.0 forces absolute deterministic output

            print!(" Crunching document context (Prefill phase)... ");
            std::io::stdout().flush().unwrap();

            let prefill_start = Instant::now();
            let mut first_token = true;

            llm_engine.generate_streaming(&augmented_prompt, config, |token| {
                if first_token {
                    println!("Done! ({:.2}s)\n", prefill_start.elapsed().as_secs_f32());
                    first_token = false;
                }
                print!("{}", token);
                std::io::stdout().flush().unwrap();
            });
            println!("\n--------------------------------------------------");
        }
    }
}
