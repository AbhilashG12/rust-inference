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
        
       // 1. ASYMMETRIC THREAD PINNING
        // Lock Rayon threads to Physical Cores. This eliminates OS context-switching
        // and keeps the L1 cache hot for the memory-bound Decode phase!
        if let Some(core_ids) = core_affinity::get_core_ids() {
            let physical_cores = core_ids.len();
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(physical_cores)
                .start_handler(move |thread_idx| {
                    // Pin this specific Rayon worker thread to a specific physical core
                    if let Some(core_id) = core_ids.get(thread_idx % core_ids.len()) {
                        core_affinity::set_for_current(*core_id);
                    }
                })
                .build_global();
            println!(" Optimized Thread Pool initialized and pinned to {} physical cores.", physical_cores);
        }
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
        F: FnMut(&str) -> bool
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
                    tokens.push(next_token);
                    generated_count += 1;
                    if !on_token(&next_word) {
                        // Early stop requested
                        let hidden_dim = self.final_norm.shape.dims()[0];
                        let bytes_per_token = 2 * num_layers * hidden_dim * 4;
                        return GenerationOutput {
                            text: output_text,
                            prompt_tokens: prompt_len,
                            completion_tokens: generated_count,
                            ttft_ms,
                            total_latency_ms: start_time.elapsed().as_millis() as u64,
                            kv_memory_mb: (bytes_per_token * tokens.len()) as f32 / (1024.0 * 1024.0),
                        };
                    }
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
            tokens.push(next_token);
            generated_count += 1;
            if !on_token(&next_word) {
                break;
            }
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

    /// Greedy argmax over a [1, vocab_size] logits tensor.
    fn greedy_token(logits: &Tensor) -> u32 {
        logits
            .data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0)
    }

    fn is_eos(&self, token_id: u32, word: &str) -> bool {
        token_id == 0
            || token_id == 2
            || word.contains("<|im_end|>")
            || word.contains("<|endoftext|>")
    }

    /// Runs a single token through all layers, advances `kv_cache`, and returns the greedy next token.
    fn forward_single_token(&self, token_id: u32, kv_cache: &mut KVCache) -> u32 {
        let pos = kv_cache.current_seq_len;
        let mut hidden_state = self.embedding.forward(token_id as usize);

        for (i, block) in self.blocks.iter().enumerate() {
            hidden_state = block.forward(&hidden_state, pos, i, kv_cache).unwrap();
        }
        kv_cache.current_seq_len += 1;

        hidden_state = hidden_state.rms_norm(&self.final_norm, 1e-5).unwrap();
        let logits = self.lm_head.forward(&hidden_state).unwrap();
        Self::greedy_token(&logits)
    }

    /// Prefills a slice of tokens into `kv_cache` without computing final logits.
    fn prefill_tokens(&self, tokens: &[u32], kv_cache: &mut KVCache) {
        for &token_id in tokens {
            let pos = kv_cache.current_seq_len;
            let mut hidden_state = self.embedding.forward(token_id as usize);
            for (i, block) in self.blocks.iter().enumerate() {
                hidden_state = block.forward(&hidden_state, pos, i, kv_cache).unwrap();
            }
            kv_cache.current_seq_len += 1;
        }
    }

    /// Verifies a batch of tokens sequentially (ready to swap for parallel GEMM prefill later).
    fn forward_batch(&self, tokens: &[u32], kv_cache: &mut KVCache) -> Vec<u32> {
        let mut predictions = Vec::with_capacity(tokens.len());
        for &token_id in tokens {
            predictions.push(self.forward_single_token(token_id, kv_cache));
        }
        predictions
    }

    pub fn generate_speculative(
        draft_engine: &LlmEngine,
        target_engine: &LlmEngine,
        prompt: &str,
        k_drafts: usize, // e.g., 4
    ) {
        let prompt_tokens = draft_engine.tokenizer.encode(prompt);
        if prompt_tokens.is_empty() {
            return;
        }

        let mut draft_kv = KVCache::new(draft_engine.blocks.len(), 2048);
        let mut target_kv = KVCache::new(target_engine.blocks.len(), 2048);

        // Prefill all prompt tokens except the last one; the last token kicks off the loop
        if prompt_tokens.len() > 1 {
            let prefix = &prompt_tokens[..prompt_tokens.len() - 1];
            target_engine.prefill_tokens(prefix, &mut target_kv);
            draft_engine.prefill_tokens(prefix, &mut draft_kv);
        }

        let mut last_token = *prompt_tokens.last().unwrap();
        let mut generated_count = 0;

        while generated_count < 200 {
            let base_seq_len = target_kv.current_seq_len;

            // --- PHASE 1: DRAFTING ---
            let mut draft_predictions = Vec::with_capacity(k_drafts);
            let mut draft_input = last_token;
            for _ in 0..k_drafts {
                let next_token = draft_engine.forward_single_token(draft_input, &mut draft_kv);
                draft_predictions.push(next_token);
                draft_input = next_token;
            }

            // --- PHASE 2: VERIFICATION ---
            // Target evaluates [last_token, d1, ..., dk] -> outputs k+1 predictions
            let mut verify_inputs = Vec::with_capacity(k_drafts + 1);
            verify_inputs.push(last_token);
            verify_inputs.extend_from_slice(&draft_predictions);

            let target_predictions = target_engine.forward_batch(&verify_inputs, &mut target_kv);

            // --- PHASE 3: ACCEPTANCE & ROLLBACK ---
            let mut accepted_count = 0;
            let mut hit_eos = false;

            for i in 0..k_drafts {
                if draft_predictions[i] == target_predictions[i] {
                    accepted_count += 1;
                    let word = target_engine.tokenizer.decode(draft_predictions[i]);
                    if target_engine.is_eos(draft_predictions[i], &word) {
                        hit_eos = true;
                        break;
                    }
                    print!("{}", word);
                } else {
                    break;
                }
            }

            if hit_eos {
                io::stdout().flush().unwrap();
                break;
            }

            let correction_token = target_predictions[accepted_count];
            let correction_word = target_engine.tokenizer.decode(correction_token);
            if target_engine.is_eos(correction_token, &correction_word) {
                io::stdout().flush().unwrap();
                break;
            }

            print!("{}", correction_word);
            io::stdout().flush().unwrap();

            // Sync both KV caches to [base_seq_len + last_token + accepted drafts]
            let valid_len = base_seq_len + 1 + accepted_count;
            target_kv.rewind(valid_len);

            if accepted_count == k_drafts {
                // Draft only ran up to d_{k-1}, so feed d_k to match target_kv length
                draft_engine.forward_single_token(draft_predictions[k_drafts - 1], &mut draft_kv);
            } else {
                draft_kv.rewind(valid_len);
            }

            last_token = correction_token;
            generated_count += accepted_count + 1;
        }
    }

    pub fn generate_for_api(&self, prompt: &str, config: GenerationConfig) -> GenerationOutput {
        self.generate_streaming(prompt, config, |_| true)
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
            true
        });
        println!("\n--------------------------------------------------");
        println!(" Performance Benchmarks:");
        println!("   - Latency: {:.2} ms", output.total_latency_ms);
        println!("   - Memory:  {:.2} MB", output.kv_memory_mb);
    }
}
