# Rust Inference Engine (v1.0.0)

A pure-Rust LLM inference runtime built from first principles — no PyTorch, ONNX, llama.cpp, or ML-framework dependencies anywhere in the execution core.

This project is a fully functional LLM execution engine capable of loading pretrained HuggingFace models (like SmolLM2/LLaMA architectures) via SafeTensors, managing memory layouts with custom strided tensors, and performing instruction-tuned autoregressive text generation using hardware-accelerated AVX2 SIMD operations.

## What is this?

This is **not** a wrapper around PyTorch, ONNX, or llama.cpp. It is a custom-built systems-level inference runtime.

The project evolved from a basic Neural Network (NN) execution graph into a specialized Transformer runtime. While standard neural networks rely on sequential dense layers and basic activations (like ReLU), modern LLMs require a highly specific architecture to understand language context. To transition from a basic NN to a Transformer runtime, I implemented:

- **RMSNorm** for stable gradient and signal flow.
- **Rotary Positional Embeddings (RoPE)** so the model understands word order.
- **Grouped Query Attention (GQA)** to map relationships between words efficiently.
- **SwiGLU** feed-forward networks for complex reasoning.

**On dependencies:** the tensor, attention, and SIMD execution core — the actual mathematical engine — is hand-built with no ML framework underneath it. External crates are used only for solved, non-core problems: `safetensors` (weight file parsing), `tokenizers` (HuggingFace's own BPE tokenizer), `memmap2` (zero-copy file loading), `clap` (CLI), `rayon` (CPU parallelism), `core_affinity` (thread pinning), and `thiserror` (error handling).

## Architecture

The engine loads standard model weights and executes the forward-pass computation sufficiently accurately to perform instruction-following and structured JSON generation.

```text
                    ┌─────────────────────┐
                    │ HuggingFace Model   │
                    │ + Tokenizer         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ SafeTensors Loader  │
                    │ (Dynamic Q8 Block)  │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │        Rust Inference Runtime  │
              │                                │
              │  ChatML Formatting & BPE       │
              │        ↓                       │
              │  Tensor / Memory System        │
              │        ↓                       │
              │  MatMul / AVX2 SIMD / Tiling   │
              │        ↓                       │
              │  RMSNorm                       │
              │        ↓                       │
              │  RoPE                          │
              │        ↓                       │
              │  GQA (Attention)               │
              │        ↓                       │
              │  SwiGLU                        │
              │        ↓                       │
              │  KV Cache                      │
              │        ↓                       │
              │  Autoregressive Decode + EOS   │
              └──────────────┬─────────────────┘
                             │
                             ▼
                       Generated text
```

## Engineering Journey & Challenges

Building an LLM runtime from scratch required solving complex systems challenges to achieve strict mathematical parity with PyTorch. I resolved issues with Tied Embeddings, RoPE sinusoidal splitting, SwiGLU gating geometry, and ChatML structured stopping.

### Productionizing & Achieving Accuracy

Wiring up a real 360-million parameter model initially produced complete word salad. Debugging this required tensor-by-tensor parity checks against a PyTorch golden reference. Achieving true mathematical accuracy and reliable instruction adherence required resolving several critical architectural bottlenecks:

1. **SIMD Contiguity & Transposition:** The MatMul kernels use AVX2 hardware acceleration, which requires strict row-major contiguous memory. Lazy transpositions resulted in scrambled memory reads. The loader was rewritten to physically allocate and flatten transposed matrices in RAM.
2. **Tied Embeddings:** Smaller models delete their final `lm_head` matrix to save disk space, expecting the runtime to reuse the `embed_tokens` matrix. The engine dynamically detects this, transposes the tensor, and allocates contiguous memory for the final logits projection.
3. **RoPE Rotation Order:** Rotary embeddings were originally applied by interleaving adjacent pairs. True accuracy required adopting the standard HuggingFace approach: splitting the head dimension in half and applying the sinusoidal rotation across the two separate blocks.
4. **Multi-Head Attention & GQA:** Mashing all features into a single attention score washed out the signal. The engine now splits the query into distinct heads, routes them correctly to the grouped key heads, and scales the dot products by `1/sqrt(d_k)` before the Softmax to prevent gradient saturation.
5. **SwiGLU Missing Gate:** Standard feed-forward networks use Up and Down projections. LLaMA architectures require a third `gate_proj` matrix. Accuracy was restored by multiplying the SiLU-activated gate with the up projection.
6. **Multi-Precision SafeTensors Ingestion (`BF16` / `F16` → `F32`):** While base models are often stored in `F32`, official HuggingFace Instruct checkpoints (`SmolLM2-360M-Instruct`) serialize weights in 16-bit Brain Floating Point (`BF16`). The loader was upgraded to parse `BF16` and `F16` byte chunks directly into `F32` execution tensors using zero-dependency IEEE 754 bitwise shifts.
7. **Instruction Tuning, ChatML & Structured Stopping:** Moving from a base autocomplete model to `SmolLM2-360M-Instruct` required preserving special control tokens (`<|im_start|>`, `<|im_end|>`) during tokenization, automatically wrapping plain prompts in the ChatML conversation template, scoping the **Repetition Penalty** to recent context so prompt keywords aren't penalized, and implementing immediate End-of-Turn (`<|im_end|>`) and JSON delimiter (`}`) termination.

### Systems Engineering & Performance Optimizations

Once mathematically accurate, the engine ran at a sluggish ~3.8 tok/s. Profiling revealed that LLM decoding (the `M=1` phase) is heavily memory-bandwidth bound, not compute-bound. To bridge the 4x–5x performance gap between my naive FP32 engine and production runtimes like `llama.cpp`, I implemented three major optimizations:

1. **Block-Wise Integer Quantization (Q8_0):** At FP32, a 360M parameter model moves ~1.44 GB of data per token, completely saturating physical RAM bandwidth. The engine now groups every 32 weights into a block, stores 1 `f32` scale factor, and compresses the weights into `i8`. During decoding, the engine natively multiplies Q8 × Q8 using AVX2 integer dot-product instructions, slashing memory traffic by 4x.
2. **SIMD-Optimized KV Cache Memory Layouts:** Built a pre-allocated flat ring-buffer and physically transposed the Key cache to `[num_heads, head_dim, seq_len]`. When the Query vector scans across history, the sequence data for a specific attention head is perfectly contiguous in RAM, allowing the CPU to read linearly without L1 cache misses.
3. **Asymmetric Thread-Pool Pinning (GEMV):** Standard `rayon` iterators spawned a micro-task per row. Because Q8 dot-products execute in nanoseconds, OS context-switching overhead took longer than the math itself. Using `core_affinity`, Rayon worker threads are now permanently pinned to physical hardware cores at boot, and the matrix is chunked coarsely (one chunk per physical core) to bypass the OS scheduler entirely.

## Reproducible Benchmark

The following benchmark measures a full local forward pass on CPU, compared directly against `llama.cpp` running the exact same model, quantization, and prompt on the same machine.

### Environment & Setup

| Setting   | This Engine                          | llama.cpp (baseline)                     |
| --------- | ------------------------------------ | ---------------------------------------- |
| Model     | SmolLM2-360M-Instruct                | SmolLM2-360M-Instruct                    |
| Precision | Q8_0 (8-bit block quantization)      | Q8_0 (GGUF)                              |
| Backend   | Rust CPU (AVX2 SIMD, Pinned Cores)   | GGML (CPU)                               |
| Target CPU | `target-cpu=native`                 | Native                                   |
| Prompt    | "Explain quantum entanglement in detail " | "Explain quantum entanglement in detail " |

### This Engine — Performance Metrics

| Metric                | Value                |
| --------------------- | -------------------- |
| Prompt tokens         | ~30 (ChatML)         |
| Generated tokens      | ~160                 |
| Decode latency        | 10,658 ms            |
| **Decode speed**      | **~15.5 tokens/sec** |
| Total KV cache memory | ~54.38 MB            |

### Operator Profiling (Decode Phase)

| Operator             | Time (ms) | % of decode pass |
| -------------------- | --------- | ---------------- |
| `VecMatMul_Q8`       | 8293.82   | ~89%             |
| `Attention_KVCached` | 987.45    | ~11%             |

The profile confirms memory allocations and thread overhead have been successfully stripped away. The CPU spends ~90% of its time performing pure AVX2 matrix math.

### Comparison vs. llama.cpp

| Engine          | Decode Speed    | Compute Precision | Engine Architecture                  |
| --------------- | --------------- | ----------------- | ------------------------------------ |
| **This engine** | **~15.5 tok/s** | Q8_0              | Safe Rust, SIMD Auto-vectorization   |
| llama.cpp       | 19.4 tok/s      | Q8_0              | C/Assembly, hand-tuned micro-kernels |

This engine operates at **~80% of the speed** of `llama.cpp` on identical hardware. Bridging the gap from a naive FP32 script (~3.8 tok/s) to 15.5 tok/s validates the core architectural optimizations. The remaining 20% delta is the difference between compiler auto-vectorization and hand-written assembly micro-kernels mapped precisely to Intel cache lines.

### Download the model and tokenizer

```bash
# Download the SmolLM2-360M-Instruct model weights
curl -L -o smollm-instruct.safetensors https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/model.safetensors

# Download the corresponding Instruct tokenizer
curl -L -o smollm_instruct_tokenizer.json https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/resolve/main/tokenizer.json
```

> **Windows (PowerShell):** use `curl.exe` instead of `curl`, since `curl` is an alias for `Invoke-WebRequest`.

#### Optional: reproduce the llama.cpp comparison

```bash
# Download the Q8_0 GGUF build of the same Instruct model
curl -L -o smollm2-360m-instruct-q8_0.gguf https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q8_0.gguf

# Run with the same prompt used above
llama-cli.exe -m smollm2-360m-instruct-q8_0.gguf -p "Explain quantum entanglement in detail " -n 200 -c 512
```

## Usage

Ensure you run with `target-cpu=native` unlocked via a `.cargo/config.toml` file to enable hardware AVX2 acceleration.

```bash
# 1. Standard Instruction / Code Generation (automatically wrapped in ChatML)
cargo run --release -- generate --model smollm-instruct.safetensors --tokenizer smollm_instruct_tokenizer.json --prompt "Write a for loop in java" --profile

# 2. Structured JSON / Roleplay Generation (using explicit ChatML + JSON prefix seeding)
cargo run --release -- generate --model smollm-instruct.safetensors --tokenizer smollm_instruct_tokenizer.json --prompt '<|im_start|>system\nYour name is Dr. Aris. You are the chief physicist at the Mars Research Facility. Speak briefly in first person as Dr. Aris.<|im_end|>\n<|im_start|>user\nWho are you?<|im_end|>\n<|im_start|>assistant\n{\"dialogue\": \"I am Dr. Aris,'

# 3. Inspect SafeTensors architecture without executing
cargo run --release -- inspect --model smollm-instruct.safetensors
```

## Future Work (v1.1 and Beyond)

Having established a mathematically sound, benchmarked, and hardware-optimized baseline, future development will focus on control and serving:

- **Sampling & Grammar Constraints:** Refining Top-K/Top-P sampling and adding formal grammar/schema state machines for guaranteed structured JSON outputs.
- **Continuous Batching:** Upgrading the KV Cache manager to support continuous batching for multi-user asynchronous serving.
- **Game Engine Integration:** Wiring the inference loop into a Terminal UI async adapter for local NPC dialogue generation.
