# Rust Inference Engine (v1.0.0)

A pure-Rust LLM inference runtime built from first principles — no PyTorch, ONNX, llama.cpp, or ML-framework dependencies anywhere in the execution core.

This project is a fully functional LLM execution engine capable of loading pretrained HuggingFace models (like SmolLM/LLaMA architectures) via SafeTensors, managing memory layouts with custom strided tensors, and performing autoregressive text generation using hardware-accelerated AVX2 SIMD operations.

## What is this?

This is **not** a wrapper around PyTorch, ONNX, or llama.cpp. It is a custom-built systems-level inference runtime.

The project evolved from a basic Neural Network (NN) execution graph into a specialized Transformer runtime. While standard neural networks rely on sequential dense layers and basic activations (like ReLU), modern LLMs require a highly specific architecture to understand language context. To transition from a basic NN to a Transformer runtime, I implemented:

- **RMSNorm** for stable gradient and signal flow.
- **Rotary Positional Embeddings (RoPE)** so the model understands word order.
- **Grouped Query Attention (GQA)** to map relationships between words efficiently.
- **SwiGLU** feed-forward networks for complex reasoning.

**On dependencies:** the tensor, attention, and SIMD execution core — the actual mathematical engine — is hand-built with no ML framework underneath it. External crates are used only for solved, non-core problems: `safetensors` (weight file parsing), `tokenizers` (HuggingFace's own BPE tokenizer), `memmap2` (zero-copy file loading), `clap` (CLI), `rayon` (CPU parallelism), and `thiserror` (error handling).

## Architecture

The engine loads standard model weights and executes the forward-pass computation sufficiently accurately to perform autoregressive generation.

```text
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
```

## Engineering Journey & Challenges

Building an LLM runtime from scratch required solving several complex systems and mathematical challenges, particularly during the final phases of development.

### The LLM Runtime & KV Cache

Transitioning from processing static tensors to generating continuous text required building a Tokenizer, Nucleus Sampling (Top-K, Top-P, Temperature), and an autoregressive decoding loop.

The primary challenge was **computational amnesia**. Generating one word at a time by recalculating the entire past sequence scales quadratically and destroys throughput. I implemented a **KV Cache** to store the Key and Value matrices of historical tokens, separating the fast, parallel "Prefill" phase from the iterative "Decode" phase.

### Productionizing & Achieving Accuracy

Wiring up a real 135-million parameter model initially produced complete word salad. Debugging this required tensor-by-tensor parity checks against a PyTorch golden reference. Achieving true mathematical accuracy required resolving several critical architectural bottlenecks:

1. **SIMD Contiguity & Transposition:** The MatMul kernels use AVX2 hardware acceleration, which requires strict row-major contiguous memory. Lazy transpositions resulted in scrambled memory reads. The loader was rewritten to physically allocate and flatten transposed matrices in RAM.

2. **Tied Embeddings:** Smaller models delete their final `lm_head` matrix to save disk space, expecting the runtime to reuse the `embed_tokens` matrix. The engine dynamically detects this, transposes the tensor, and allocates contiguous memory for the final logits projection.

3. **RoPE Rotation Order:** Rotary embeddings were originally applied by interleaving adjacent pairs. True accuracy required adopting the standard HuggingFace approach: splitting the head dimension in half and applying the sinusoidal rotation across the two separate blocks.

4. **Multi-Head Attention & GQA:** Mashing all features into a single attention score washed out the signal. The engine now splits the query into distinct heads, routes them correctly to the grouped key heads, and scales the dot products by `1/sqrt(d_k)` before the Softmax to prevent gradient saturation.

5. **SwiGLU Missing Gate:** Standard feed-forward networks use Up and Down projections. LLaMA architectures require a third `gate_proj` matrix. Accuracy was restored by multiplying the SiLU-activated gate with the up projection.

6. **Probability Looping:** Microscopic models frequently fall into repetitive loops (e.g., generating the same sentence endlessly). I implemented a **Repetition Penalty** in the sampler to penalize previously generated token logits, forcing the model to complete thoughts naturally.

## Reproducible Benchmark

The following benchmark measures a full local forward pass on CPU, compared directly against `llama.cpp` running the same model and prompt on the same machine.

### Environment & Setup

| Setting    | This Engine            | llama.cpp (baseline) |
|------------|-------------------------|------------------------|
| Model      | SmolLM-135M             | SmolLM-135M-Instruct  |
| Precision  | FP32                    | F16                    |
| Backend    | Rust CPU (AVX2 SIMD)    | GGML (CPU)             |
| KV Cache   | Enabled                 | Enabled                |
| GQA        | Enabled                 | Enabled                |
| Prompt     | "Rust is a systems programming language that" | same |

### This Engine — Performance Metrics

| Metric                 | Value                  |
|-------------------------|-------------------------|
| Prompt tokens           | 4                       |
| Generated tokens        | 50                      |
| Prefill latency         | 779.36 ms               |
| Decode latency          | 5081.29 ms              |
| Decode speed            | 9.84 tokens/sec         |
| KV cache memory/token   | 135.00 KB               |
| Total KV cache memory   | 7965.00 KB (~7.78 MB)   |

### Operator Profiling

| Operator   | Time (ms) | % of forward pass |
|------------|-----------|--------------------|
| MatMul     | 5367.48   | ~100%              |
| **Total**  | **5367.48** | **100%**         |

```bash
# Generate text and profile operator execution time
cargo run --release -- generate --model smollm.safetensors --tokenizer smollm_tokenizer.json --prompt "Rust is a systems programming language that" --profile

### Comparison vs. llama.cpp

| Engine              | Decode Speed | Precision |
|---------------------|--------------|-----------|
| **This engine**     | 9.84 tok/s   | FP32      |
| llama.cpp (run 1)   | 50.9 tok/s   | F16       |
| llama.cpp (run 2)   | 39.4 tok/s   | F16       |

This engine is roughly **4–5x slower** than llama.cpp on identical hardware and prompt. That gap is expected against a project with years of dedicated CPU-kernel engineering — the useful part is understanding *why*:

- **Decode is memory-bandwidth-bound, not compute-bound.** Each decode step is a matrix × *vector* operation (one new token against the weight matrices), not matrix × matrix — so the bottleneck is bytes streamed from RAM per token, not FLOPs/sec.
- **Precision compounds this directly.** This engine stores weights as FP32 (4 bytes/weight); llama.cpp's GGUF is F16 (2 bytes/weight) — roughly half the memory traffic per token before any kernel-quality difference even enters.
- llama.cpp's GGML kernels are specifically hand-tuned for this single-token (`M=1`) regime; this engine's AVX2 path is not yet specialized for it.

**Next optimization target, in order:**
1. Confirm the AVX2 path actually fires for the `M=1` decode shape rather than falling back to a scalar loop.
2. Move weight storage to F16 to roughly halve decode-time memory traffic.
3. Re-check whether `rayon` parallelism helps or hurts a single-row decode matmul — thread-spawn overhead can exceed the work itself at this size.

### Download the model and tokenizer

```bash
# Download the SmolLM-135M model weights (~540 MB)
curl -L -o smollm.safetensors https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/model.safetensors

# Download the corresponding tokenizer (~2 MB)
curl -L -o smollm_tokenizer.json https://huggingface.co/HuggingFaceTB/SmolLM-135M/resolve/main/tokenizer.json
```

> **Windows (PowerShell):** use `curl.exe` instead of `curl`, since `curl` is an alias for `Invoke-WebRequest`.

#### Optional: reproduce the llama.cpp comparison

```bash
# Download the F16 GGUF build of the same model
curl -L -o smollm-135m.gguf https://huggingface.co/second-state/SmolLM-135M-Instruct-GGUF/resolve/main/SmolLM-135M-Instruct-f16.gguf

# Run with the same prompt used above
llama-cli.exe -m smollm-135m.gguf -p "Rust is a systems programming language that" -n 50 -c 512
```

## Usage

You can inspect models, run the profiler, or generate text directly from the CLI.

```bash
# Generate text and profile operator execution time
cargo run --release -- generate --model smollm.safetensors --tokenizer smollm_tokenizer.json --prompt "Rust is a systems programming language that" --profile

# Inspect SafeTensors architecture without executing
cargo run --release -- inspect --model smollm.safetensors
```

> **Windows (PowerShell):** use `curl.exe` instead of `curl`, since `curl` is an alias for `Invoke-WebRequest`.

## Future Work (v1.1 and Beyond)

Having established a mathematically sound, benchmarked baseline, future development will focus on scaling and optimization:

- **Quantization (highest priority — see benchmark analysis above):** F16 weight storage first, then INT8/INT4 kernels for larger models (like Llama-3-8B).
- **Sampling Improvements:** Refining Top-K and Top-P penalty logic.
- **Serving Architecture:** Paged KV Cache and continuous batching.
- **Hardware Backends:** Exploring GPU execution (CUDA/Metal).
