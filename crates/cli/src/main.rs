use clap::{Parser, Subcommand};
use model::LlmEngine;
use model::embedder::EmbedderEngine;
use model::rag::{DocumentStore, VectorStore};
use model::tokenizer::Tokenizer;
use tensor::profiler::PROFILER;
use tensor::sampler::GenerationConfig;
use std::time::Instant;
use std::path::Path;
use std::io::Write;
use std::io;

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
        #[arg(long, default_value = "rag_index.ridx")]
        index: String,
        #[arg(long, default_value_t = false)]
        reindex: bool,
        #[arg(long, default_value_t = 0.36)]
        threshold: f32,
        #[arg(long)]
        prompt: Option<String>,
    },
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
        
       Commands::Rag {
            llm,
            llm_tokenizer,
            embedder,
            embedder_tokenizer,
            docs_dir,
            index,
            reindex,
            threshold,
            prompt,
        } => {
            println!("--- [1] Booting Air-Gapped FDE Engine ---");
            let llm_engine = LlmEngine::load(llm, llm_tokenizer);
            let embed_engine = EmbedderEngine::load(embedder, embedder_tokenizer);

            println!("\n--- [2] Loading Knowledge Base ---");
            let start = Instant::now();
            let (doc_store, vec_store) = if !reindex && Path::new(index).exists() {
                println!(" Loading cached binary index from '{}'...", index);
                VectorStore::load_index(index).expect("Failed to load .ridx cache")
            } else {
                let embed_tok = Tokenizer::load(embedder_tokenizer).unwrap();
                let mut docs = DocumentStore::new();
                docs.ingest_directory(docs_dir, &embed_tok, 85,15);

                let mut vecs = VectorStore::new();
                vecs.embed_store(&docs, &embed_engine);
                vecs.save_index(&docs, index).expect("Failed to save .ridx cache");
                println!(" Saved binary index to '{}'.", index);
                (docs, vecs)
            };
            println!(
                " Knowledge Base ready ({} chunks) in {:.2}s.",
                doc_store.chunks.len(),
                start.elapsed().as_secs_f32()
            );

            if let Some(q) = prompt {
                run_rag_query(&llm_engine, &embed_engine, &doc_store, &vec_store, q, *threshold);
            } else {
                println!("\n--- Interactive Air-Gapped RAG REPL (type 'exit' to quit) ---");
                let stdin = io::stdin();
                loop {
                    print!("\nrag> ");
                    io::stdout().flush().unwrap();

                    let mut input = String::new();
                    if stdin.read_line(&mut input).is_err() {
                        break;
                    }
                    let q = input.trim();
                    if q.is_empty() {
                        continue;
                    }
                    if q.eq_ignore_ascii_case("exit") || q.eq_ignore_ascii_case("quit") {
                        break;
                    }
                    run_rag_query(&llm_engine, &embed_engine, &doc_store, &vec_store, q, *threshold);
                }
            }
        }
    }
}

fn run_rag_query(
    llm_engine: &LlmEngine,
    embed_engine: &EmbedderEngine,
    doc_store: &DocumentStore,
    vec_store: &VectorStore,
    query: &str,
    threshold: f32,
) {
    let search_start = Instant::now();
    let matches = vec_store.search(query, doc_store, embed_engine, 3, threshold);
    let search_ms = search_start.elapsed().as_millis();

    println!(
        "\n--- [Retrieval] {}ms ({} matches >= {:.2}) ---",
        search_ms,
        matches.len(),
        threshold
    );

    if matches.is_empty() {
        println!("Answer: I do not have data on this. (Short-circuited LLM)");
        return;
    }

    let mut context_text = String::new();
    for (rank, &(idx, score)) in matches.iter().enumerate() {
        let chunk = &doc_store.chunks[idx];
        println!(
            "  -> Match {}: [{}] (score: {:.3})",
            rank + 1,
            chunk.source_file,
            score
        );
        context_text.push_str(&chunk.text);
        context_text.push_str("\n\n");
    }

    let augmented_prompt = format!(
        "SYSTEM: You are a helpful data extraction assistant. Read the CONTEXT below and answer the user's QUESTION in one clear, complete English sentence. Do not output raw lists, pipes, or key-value pairs. If the answer is not in the CONTEXT, reply 'I do not know'.\n\nCONTEXT:\n{}\n\nQUESTION:\n{}\n\nANSWER:\n",
        context_text.trim(),
        query
    );

    let mut config = GenerationConfig::default();
    config.temperature = 0.0;
    config.max_tokens = 64;

    print!("\n--- [Generation] Prefilling KV Cache... ");
    io::stdout().flush().unwrap();

    let mut first_token = true;
    let mut generated = String::new();
    let _stopped = false;

    let output = llm_engine.generate_streaming(&augmented_prompt, config, |token| {
        if first_token {
            println!("Done! ---\n");
            first_token = false;
        }

        generated.push_str(token);
        let trimmed = generated.trim_start();

        if !trimmed.is_empty()
            && (trimmed.contains("\n\n")
                || trimmed.contains("\nQUESTION")
                || trimmed.contains("\nSystem:")
                || trimmed.contains("\nSYSTEM:")
                || trimmed.contains("\nCONTEXT:"))
        {
            return false;
        }

        print!("{}", token);
        io::stdout().flush().unwrap();
        true
    });

    let decode_ms = output.total_latency_ms.saturating_sub(output.ttft_ms);
    let tok_per_sec = if decode_ms > 0 {
        (output.completion_tokens as f64) / (decode_ms as f64 / 1000.0)
    } else {
        0.0
    };

    println!(
        "\n\n[Telemetry] Prompt: {} tok | TTFT: {:.2}s | Decode: {} tok ({:.1} tok/s) | KV: {:.2} MB",
        output.prompt_tokens,
        output.ttft_ms as f64 / 1000.0,
        output.completion_tokens,
        tok_per_sec,
        output.kv_memory_mb
    );
}
