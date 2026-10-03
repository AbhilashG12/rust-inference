Rust Inference Engine (v1.0.0)
A pure-Rust, zero-dependency Large Language Model (LLM) inference runtime built from first principles.

This project is a fully functional LLM execution engine capable of loading pretrained HuggingFace models (like SmolLM/LLaMA architectures) via SafeTensors, managing memory layouts with custom strided tensors, and performing autoregressive text generation using hardware-accelerated AVX2 SIMD operations.

What is this?
This is not a wrapper around PyTorch, ONNX, or llama.cpp. It is a custom-built systems-level inference runtime.

The project evolved from a basic Neural Network (NN) execution graph into a specialized Transformer runtime. While standard neural networks rely on sequential dense layers and basic activations (like ReLU), modern LLMs require a highly specific architecture to understand language context. To transition from a basic NN to a Transformer runtime, I implemented:

RMSNorm for stable gradient and signal flow.

Rotary Positional Embeddings (RoPE) so the model understands word order.

Grouped Query Attention (GQA) to map relationships between words efficiently.

SwiGLU feed-forward networks for complex reasoning.

Architecture
The engine loads standard model weights and executes the forward-pass computation sufficiently accurately to perform autoregressive generation.

Plaintext
                    ┌─────────────────────┐
                    │ HuggingFace Model   │
                    │ + Tokenizer         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SafeTensors Loader  │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │        Rust Inference Runtime  │
              │                                │
              │  Tensor / Memory System        │
              │       ↓                        │
              │  MatMul / SIMD / Tiling        │
              │       ↓                        │
              │  RMSNorm                       │
              │       ↓                        │
              │  RoPE                          │
              │       ↓                        │
              │  GQA (Attention)               │
              │       ↓                        │
              │  SwiGLU                        │
              │       ↓                        │
              │  KV Cache                      │
              │       ↓                        │
              │  Autoregressive Decode         │
              └──────────────┬─────────────────┘
                             │
                             ▼
                       Generated text


Engineering Journey & Challenges
Building an LLM runtime from scratch required solving several complex systems and mathematical challenges, particularly during the final phases of development.

The LLM Runtime & KV Cache
Transitioning from processing static tensors to generating continuous text required building a Tokenizer, Nucleus Sampling (Top-K, Top-P, Temperature), and an autoregressive decoding loop.

The primary challenge was computational amnesia. Generating one word at a time by recalculating the entire past sequence scales quadratically and destroys throughput. I implemented a KV Cache to store the Key and Value matrices of historical tokens, separating the fast, parallel "Prefill" phase from the iterative "Decode" phase.

Productionizing & Achieving Accuracy
Wiring up a real 135-million parameter model initially produced complete word salad. Debugging this required tensor-by-tensor parity checks against a PyTorch golden reference. Achieving true mathematical accuracy required resolving several critical architectural bottlenecks:

SIMD Contiguity & Transposition: Our MatMul kernels use AVX2 hardware acceleration, which requires strict row-major contiguous memory. Lazy transpositions resulted in scrambled memory reads. The loader was rewritten to physically allocate and flatten transposed matrices in RAM.

Tied Embeddings: Smaller models delete their final lm_head matrix to save disk space, expecting the runtime to reuse the embed_tokens matrix. The engine dynamically detects this, transposes the tensor, and allocates contiguous memory for the final logits projection.

RoPE Rotation Order: We originally applied rotary embeddings by interleaving adjacent pairs. True accuracy required adopting the standard HuggingFace approach: splitting the head dimension in half and applying the sinusoidal rotation across the two separate blocks.

Multi-Head Attention & GQA: Mashing all features into a single attention score washed out the signal. The engine now splits the query into distinct heads, routes them correctly to the grouped key heads, and scales the dot products by 1/sqrt(d_k) before the Softmax to prevent gradient saturation.

SwiGLU Missing Gate: Standard feed-forward networks use Up and Down projections. LLaMA architectures require a third gate_proj matrix. Accuracy was restored by multiplying the SiLU-activated gate with the up projection.

Probability Looping: Microscopic models frequently fall into repetitive loops (e.g., generating the same sentence endlessly). I implemented a Repetition Penalty in the sampler to penalize previously generated token logits, forcing the model to complete thoughts naturally.

Reproducible Benchmark
The following benchmark demonstrates the engine's performance running a fully local forward pass on CPU.

Environment & Setup:

Model: SmolLM

Parameters: 135M

Precision: FP32

Backend: Rust CPU (AVX2 SIMD)

KV Cache: Enabled

GQA: Enabled

Performance Metrics:

Prompt tokens: 4

Generated tokens: ~50

Prefill Latency: ~200 ms

Decode Speed: ~10.6 tokens/sec

KV Cache Memory/Token: 135.00 KB

Peak KV Cache Memory: ~7.2 MB

Usage
You can inspect models, run the profiler, or generate text directly from the CLI.

Bash
# Generate text and Profile operator execution time
cargo run --release -- generate --model smollm.safetensors --tokenizer smollm_tokenizer.json --prompt "Rust is a systems programming language that"

# Inspect SafeTensors architecture without executing
cargo run --release -- inspect --model smollm.safetensors


Future Work (v1.1 and Beyond)
Having established a mathematically sound baseline, future development will focus on scaling and optimization:

Quantization: Reintroducing robust INT8/INT4 kernels for larger models (like Llama-3-8B).

Sampling Improvements: Refining Top-K and Top-P penalty logic.

Serving Architecture: Paged KV Cache and continuous batching.

Hardware Backends: Exploring GPU execution (CUDA/Metal).