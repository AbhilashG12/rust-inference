# Rust Inference Engine (v1.1.0)

A pure-Rust LLM inference runtime and air-gapped RAG engine built from first principles — no PyTorch, ONNX, llama.cpp, or external vector database dependencies anywhere in the execution core.

This project is a fully functional local AI appliance capable of loading pretrained HuggingFace models (`SmolLM2` / `LLaMA` decoders and `MiniLM` BERT encoders) via SafeTensors, managing memory layouts with custom strided tensors, performing hardware-accelerated `Q8_0` AVX2 SIMD inference, and running an embedded hybrid vector search + RAG REPL completely offline.

## What is this?

This is **not** a wrapper around PyTorch, ONNX, llama.cpp, or a third-party vector DB. It is a custom-built systems-level inference and retrieval runtime.

The project evolved from a basic Neural Network execution graph into a specialized Transformer runtime with an embedded RAG pipeline:

- **Decoder Runtime (SmolLM2 / LLaMA):** RMSNorm, Rotary Positional Embeddings (RoPE), Grouped Query Attention (GQA), SwiGLU feed-forward networks, and a flat transposed KV Cache.
- **Encoder Runtime (BERT / MiniLM):** LayerNorm, GELU, bi-directional self-attention, `token_type_embeddings`, `[PAD]` token filtering, Mean Pooling, and L2 Normalization.
- **Zero-Allocation Vector Store:** Flat contiguous `Vec<f32>` embedding matrix, dot-product cosine similarity, hybrid dense + lexical scoring, and zero-dependency `.ridx` binary index caching.

**On dependencies:** the tensor, attention, quantization, vector search, and SIMD execution core — the actual mathematical engine — is hand-built with no ML framework underneath it. External crates are used only for solved, non-core problems: `safetensors` (weight file parsing), `tokenizers` (HuggingFace BPE/WordPiece tokenization), `memmap2` (zero-copy file loading), `clap` (CLI), `rayon` (CPU parallelism), `core_affinity` (thread pinning), and `thiserror` (error handling).

## Architecture

```text
                    ┌─────────────────────┐      ┌──────────────────────┐
                    │ HuggingFace Models  │      │ Local Documents      │
                    │ (SmolLM2 + MiniLM)  │      │ (.txt, .md, .csv)    │
                    └──────────┬──────────┘      └──────────┬───────────┘
                               │                            │
                               ▼                            ▼
                    ┌─────────────────────┐      ┌──────────────────────┐
                    │ SafeTensors Loader  │      │ DocumentStore        │
                    │ (Dynamic Q8_0 Block)│      │ Chunking + CSV Prose │
                    └──────────┬──────────┘      └──────────┬───────────┘
                               │                            │
                               │                            ▼
                               │                 ┌──────────────────────┐
                               │                 │ EmbedderEngine       │
                               │                 │ (MiniLM-L6-v2 BERT)  │
                               │                 └──────────┬───────────┘
                               │                            │
                               │                            ▼
                               │                 ┌──────────────────────┐
                               │                 │ VectorStore (.ridx)  │
                               │                 │ Flat Matrix + Hybrid │
                               │                 └──────────┬───────────┘
                               │                            │ Top-K Context
                               ▼                            ▼
              ┌──────────────────────────────────────────────────────────┐
              │                Rust Inference Runtime                    │
              │                                                          │
              │  ChatML / RAG Prompt Injection & BPE                     │
              │        ↓                                                 │
              │  Tensor / Memory System (AVX2 SIMD + Q8_0)               │
              │        ↓                                                 │
              │  RMSNorm → RoPE → GQA (KV Cache) → SwiGLU                │
              │        ↓                                                 │
              │  Autoregressive Decode + Early-Abort Callback            │
              └────────────────────────────┬─────────────────────────────┘
                                           │
                                           ▼
                                    Grounded Output
```

## Air-Gapped RAG Pipeline (v1.1)

The engine includes a complete 4-phase Retrieval-Augmented Generation pipeline built directly into the binary:

1. **Phase 1 — Pure-Rust BERT Embedder (`all-MiniLM-L6-v2`):** Runs bi-directional attention and GELU feed-forward layers over `Q8_0` weights with FP32 bias broadcasting. Explicitly filters `[PAD]` tokens before Mean Pooling and L2 Normalization to prevent embedding collapse.
2. **Phase 2 — Native Document Ingestion (`DocumentStore`):** Recursively crawls `.txt`, `.md`, and `.csv` directories. Uses delimiter-aware raw-text sliding-window chunking to preserve newlines/casing within MiniLM's 128-token window, prepends `[File: <name>]` context headers, and linearizes `.csv` rows into natural-language prose records for row-level accuracy.
3. **Phase 3 — Zero-Allocation Vector Store (`VectorStore`):** Stores all *N* document embeddings in a single contiguous `Vec<f32>` (`[N, 384]`) to eliminate heap pointer chasing. Because vectors are L2-normalized, cosine similarity reduces to a pure dot product. Combines dense similarity with an exact-keyword lexical boost, `BinaryHeap` Top-K selection, score-gap filtering, `rayon` parallel indexing, and zero-dependency binary persistence (`rag_index.ridx`).
4. **Phase 4 — Context-Aware Generation & REPL:** Boots both models and the binary index once into RAM for interactive `rag>` queries. Includes zero-match short-circuiting (refusing out-of-domain queries in ~20ms without touching the LLM), callback-based early decode aborts (`on_token -> bool`), and live telemetry (prompt tokens, TTFT, decode tok/s, and KV cache footprint).

## Engineering Journey & Challenges

### Productionizing & Achieving Accuracy

Wiring up a real 360-million parameter model initially produced complete word salad. Debugging this required tensor-by-tensor parity checks against a PyTorch golden reference:

1. **SIMD Contiguity & Transposition:** The MatMul kernels use AVX2 hardware acceleration, which requires strict row-major contiguous memory. The loader physically allocates and flattens transposed matrices in RAM.
2. **Tied Embeddings:** Smaller models omit `lm_head.weight` and reuse `embed_tokens`. The engine dynamically detects this and quantizes the shared matrix for final logits projection.
3. **RoPE Rotation Order:** Adopted HuggingFace's split-half sinusoidal rotation across the head dimension rather than interleaved adjacent pairs.
4. **Multi-Head Attention & GQA:** Splits queries into distinct heads, routes them to grouped key/value heads, and scales dot products by `1/sqrt(d_k)` before Softmax.
5. **SwiGLU Missing Gate:** Multiplies the SiLU-activated `gate_proj` with `up_proj` before `down_proj`.
6. **Multi-Precision SafeTensors Ingestion (`BF16` / `F16` → `F32`):** Parses `BF16` and `F16` byte chunks directly into `F32` execution tensors using zero-dependency IEEE 754 bitwise shifts.
7. **Instruction Tuning, ChatML & Structured Stopping:** Preserves special control tokens (`<|im_start|>`, `<|im_end|>`), scopes Repetition Penalty to recent context, and supports immediate callback-driven early stopping.

### Systems Engineering & Performance Optimizations

1. **Block-Wise Integer Quantization (`Q8_0`):** Groups every 32 weights into a block with 1 `f32` scale factor and `i8` weights, slashing memory-bandwidth traffic by 4x during decoding.
2. **SIMD-Optimized KV Cache Layouts:** Pre-allocated flat ring-buffer with physically transposed Key cache `[num_heads, head_dim, seq_len]` for linear L1-friendly scans, plus `KVCache::rewind` for speculative decoding rollback.
3. **Asymmetric Thread-Pool Pinning (GEMV):** Uses `core_affinity` to permanently pin Rayon worker threads to physical CPU cores at boot, chunking matrices coarsely to eliminate OS scheduler overhead.

## Reproducible Benchmark

### Environment & Setup

| Setting    | This Engine                               | llama.cpp (baseline)                      |
| ---------- | ----------------------------------------- | ----------------------------------------- |
| Model      | SmolLM2-360M-Instruct                     | SmolLM2-360M-Instruct                     |
| Precision  | Q8_0 (8-bit block quantization)           | Q8_0 (GGUF)                               |
| Backend    | Rust CPU (AVX2 SIMD, Pinned Cores)        | GGML (CPU)                                |
| Target CPU | `target-cpu=native`                       | Native                                    |
| Prompt     | "Explain quantum entanglement in detail " | "Explain quantum entanglement in detail " |

### Decode Performance & Profiling

| Engine          | Decode Speed         | Compute Precision | Engine Architecture                  |
| --------------- | -------------------- | ----------------- | ------------------------------------ |
| **This engine** | **~15.5–24.9 tok/s** | Q8_0              | Safe Rust, SIMD Auto-vectorization   |
| llama.cpp       | 19.4 tok/s           | Q8_0              | C/Assembly, hand-tuned micro-kernels |

Operator profiling on the decode pass confirms ~89% of time is spent in `VecMatMul_Q8` and ~11% in `Attention_KVCached`.

### RAG Pipeline Benchmarks (180 Chunks / 160-Row CSV + Docs)

| Stage                                       | Metric                  |
| ------------------------------------------- | ----------------------- |
| Parallel Indexing (180 chunks, 12 cores)    | ~3.60 s                 |
| Binary Cache Load (`rag_index.ridx`)        | < 0.01 s                |
| Hybrid Vector Search (MiniLM + Flat Matrix) | 15 – 34 ms              |
| Out-of-Domain Short-Circuit                 | ~20 ms (0 LLM tokens)   |

## Download Models & Tokenizers

```bash
# 1. SmolLM2-360M-Instruct (Decoder LLM)
curl -L -o smollm-instruct.safetensors https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/model.safetensors
curl -L -o smollm_instruct_tokenizer.json https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer.json

# 2. all-MiniLM-L6-v2 (BERT Embedder for RAG)
curl -L -o minilm-l6-v2.safetensors https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors
curl -L -o minilm_tokenizer.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json
```

> **Windows (PowerShell):** use `curl.exe` instead of `curl`, since `curl` is an alias for `Invoke-WebRequest`.

## Usage

Ensure `target-cpu=native` is enabled in `.cargo/config.toml` to unlock hardware AVX2 acceleration.

```bash
# 1. Interactive Air-Gapped RAG REPL (builds & caches rag_index.ridx on first run)
cargo run --release -- rag --docs-dir "./docs" --reindex

# 2. Subsequent RAG runs (instant binary cache load)
cargo run --release -- rag --docs-dir "./docs"

# 3. One-Shot RAG Query
cargo run --release -- rag --docs-dir "./docs" --prompt "What port does quantum-vault run on?"

# 4. Standard Instruction / Code Generation (automatically wrapped in ChatML)
cargo run --release -- generate --model smollm-instruct.safetensors --tokenizer smollm_instruct_tokenizer.json --prompt "Write a for loop in java" --profile

# 5. Inspect SafeTensors architecture without executing
cargo run --release -- inspect --model smollm-instruct.safetensors
```

## Future Work (v1.2 and Beyond)

- **Batched Prefill (M > 1 GEMM):** Parallelize multi-token prompt prefill across physical cores so each Q8 weight row is loaded once per batch instead of once per token, slashing RAG TTFT from seconds to milliseconds.
- **Speculative Decoding CLI:** Wire the implemented `generate_speculative` and `KVCache::rewind` primitives to a draft/target CLI flag once M > 1 verification is batched.
- **Axum HTTP API Server:** Expose the local inference and RAG engine behind an OpenAI-compatible `/v1/chat/completions` endpoint.
- **Constrained Decoding:** Add formal JSON schema / grammar state machines for guaranteed structured outputs.
