use crate::tensor::Tensor;
use std::collections::HashSet;

pub struct GenerationConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub max_tokens: usize,
    pub repetition_penalty: f32,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.9,
            max_tokens: 50,
            repetition_penalty: 1.2,
        }
    }
}

pub struct Sampler;

impl Sampler {
    pub fn sample(logits: &Tensor, config: &GenerationConfig, history: &[u32]) -> u32 {
        let vocab_size = *logits.shape.dims().last().unwrap();
        let mut data = logits.data.clone();

        // 0. REPETITION PENALTY
        if config.repetition_penalty > 1.0 {
            let mut seen = HashSet::new();
            for &token in history {
                seen.insert(token as usize);
            }
            
            for idx in seen {
                if idx < vocab_size {
                    let logit = data[idx];
                    // The standard HuggingFace penalty formula
                    if logit < 0.0 {
                        data[idx] = logit * config.repetition_penalty;
                    } else {
                        data[idx] = logit / config.repetition_penalty;
                    }
                }
            }
        }

        // 1. GREEDY
        if config.temperature == 0.0 {
            let mut max_idx = 0;
            let mut max_val = f32::NEG_INFINITY;
            for (i, &val) in data.iter().enumerate() {
                if val > max_val { max_val = val; max_idx = i; }
            }
            return max_idx as u32;
        }

        let mut temp_scaled: Vec<(usize, f32)> = data.iter()
            .enumerate()
            .map(|(i, &val)| (i, val / config.temperature))
            .collect();

        temp_scaled.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        if config.top_k > 0 && config.top_k < vocab_size {
            temp_scaled.truncate(config.top_k);
        }

        let max_logit = temp_scaled[0].1;
        let mut sum_exp = 0.0;
        let mut probs = Vec::with_capacity(temp_scaled.len());
        
        for &(idx, val) in &temp_scaled {
            let exp_val = (val - max_logit).exp();
            probs.push((idx, exp_val));
            sum_exp += exp_val;
        }

        let mut cumulative_prob = 0.0;
        let mut final_candidates = Vec::new();
        
        for (idx, exp_val) in probs {
            let prob = exp_val / sum_exp;
            cumulative_prob += prob;
            final_candidates.push((idx, prob));
            
            if cumulative_prob >= config.top_p {
                break;
            }
        }

        final_candidates[0].0 as u32
    }
}
