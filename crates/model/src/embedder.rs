use tensor::tensor::Tensor;
use tensor::nn::Linear;
use crate::loader::{LoadedWeight, ModelLoader};
use crate::tokenizer::Tokenizer;

pub struct BertBlock {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub norm1_weight: Tensor,
    pub norm1_bias: Tensor,
    
    pub intermediate_proj: Linear, // Expands hidden dim
    pub output_proj: Linear,       // Shrinks back
    pub norm2_weight: Tensor,
    pub norm2_bias: Tensor,
    
    pub num_heads: usize,
}

impl BertBlock {
    /// Helper to process a sequence through the Linear layer one token at a time,
    /// triggering the hyper-optimized M=1 AVX2 decode path without dimension panics.
    fn run_linear(&self, proj: &Linear, x: &Tensor) -> tensor::error::Result<Tensor> {
        let seq_len = x.shape.dims()[0];
        let hidden_dim = x.shape.dims()[1];
        let mut out = Vec::new();
        
        for i in 0..seq_len {
            let row = x.data[i * hidden_dim .. (i + 1) * hidden_dim].to_vec();
            let row_tensor = Tensor::new(row, vec![1, hidden_dim])?;
            let res = proj.forward(&row_tensor)?;
            out.extend_from_slice(&res.data);
        }
        
        let out_dim = out.len() / seq_len;
        Tensor::new(out, vec![seq_len, out_dim])
    }

    pub fn forward(&self, x: &Tensor) -> tensor::error::Result<Tensor> {
        // 1. Self Attention (Token-by-token linear projections)
        let q = self.run_linear(&self.q_proj, x)?;
        let k = self.run_linear(&self.k_proj, x)?;
        let v = self.run_linear(&self.v_proj, x)?;
        
        let attn_out = Tensor::attention_bidirectional(&q, &k, &v, self.num_heads)?;
        let proj_out = self.run_linear(&self.o_proj, &attn_out)?;
        
        // 2. Residual & LayerNorm 1
        let residual_1 = x.add(&proj_out)?;
        let norm_1 = residual_1.layer_norm(&self.norm1_weight, &self.norm1_bias, 1e-12)?;

        // 3. Feed Forward (GELU)
        let intermediate = self.run_linear(&self.intermediate_proj, &norm_1)?;
        let activated = intermediate.gelu()?;
        let ffn_out = self.run_linear(&self.output_proj, &activated)?;

        // 4. Residual & LayerNorm 2
        let residual_2 = norm_1.add(&ffn_out)?;
        residual_2.layer_norm(&self.norm2_weight, &self.norm2_bias, 1e-12)
    }
}

pub struct EmbedderEngine {
    tokenizer: Tokenizer,
    word_embeddings: Tensor,
    position_embeddings: Tensor,
    emb_norm_weight: Tensor,
    emb_norm_bias: Tensor,
    blocks: Vec<BertBlock>,
}

impl EmbedderEngine {
    pub fn load(model_path: &str, tokenizer_path: &str) -> Self {
        println!(" Loading Embedder weights...");
        let mut weights = ModelLoader::load_safetensors(model_path).expect("Failed to load Embedder");
        let tokenizer = Tokenizer::load(tokenizer_path).expect("Failed to load tokenizer");

        // Similar macro helpers as the LLM engine...
        macro_rules! get_fp32 {
            ($name:expr) => {
                match weights.remove($name).unwrap() {
                    LoadedWeight::Fp32(t) => t,
                    _ => panic!("Expected FP32 Tensor for {}", $name),
                }
            };
        }
        macro_rules! get_linear {
            ($name:expr) => {{
                let w = match weights.remove($name).unwrap() {
                    LoadedWeight::Q8(q) => q,
                    _ => panic!("Expected Q8 Tensor"),
                };
                let b = get_fp32!(&$name.replace(".weight", ".bias"));
                Linear::new(w, b)
            }};
        }

        let mut blocks = Vec::new();
        let mut layer_idx = 0;

        loop {
            let q_name = format!("encoder.layer.{}.attention.self.query.weight", layer_idx);
            if !weights.contains_key(&q_name) { break; }

            blocks.push(BertBlock {
                q_proj: get_linear!(&q_name),
                k_proj: get_linear!(&format!("encoder.layer.{}.attention.self.key.weight", layer_idx)),
                v_proj: get_linear!(&format!("encoder.layer.{}.attention.self.value.weight", layer_idx)),
                o_proj: get_linear!(&format!("encoder.layer.{}.attention.output.dense.weight", layer_idx)),
                norm1_weight: get_fp32!(&format!("encoder.layer.{}.attention.output.LayerNorm.weight", layer_idx)),
                norm1_bias: get_fp32!(&format!("encoder.layer.{}.attention.output.LayerNorm.bias", layer_idx)),
                intermediate_proj: get_linear!(&format!("encoder.layer.{}.intermediate.dense.weight", layer_idx)),
                output_proj: get_linear!(&format!("encoder.layer.{}.output.dense.weight", layer_idx)),
                norm2_weight: get_fp32!(&format!("encoder.layer.{}.output.LayerNorm.weight", layer_idx)),
                norm2_bias: get_fp32!(&format!("encoder.layer.{}.output.LayerNorm.bias", layer_idx)),
                num_heads: 12, // MiniLM uses 12 heads
            });
            layer_idx += 1;
        }

        Self {
            tokenizer,
            word_embeddings: get_fp32!("embeddings.word_embeddings.weight"),
            position_embeddings: get_fp32!("embeddings.position_embeddings.weight"),
            emb_norm_weight: get_fp32!("embeddings.LayerNorm.weight"),
            emb_norm_bias: get_fp32!("embeddings.LayerNorm.bias"),
            blocks,
        }
    }

    pub fn embed(&self, text: &str) -> Vec<f32> {
        // 1. Tokenize and add WordPiece [CLS] and [SEP] special tokens
        let formatted = format!("[CLS] {} [SEP]", text);
        let tokens = self.tokenizer.encode(&formatted);
        let seq_len = tokens.len();
        let hidden_dim = self.word_embeddings.shape.dims()[1];

        // 2. Absolute Embeddings = Word + Position + TokenType(Zeros)
        let mut x_data = vec![0.0; seq_len * hidden_dim];
        for (pos, &token_id) in tokens.iter().enumerate() {
            let word_offset = (token_id as usize) * hidden_dim;
            let pos_offset = pos * hidden_dim;
            let target_offset = pos * hidden_dim;
            
            for d in 0..hidden_dim {
                x_data[target_offset + d] = 
                    self.word_embeddings.data[word_offset + d] + 
                    self.position_embeddings.data[pos_offset + d]; // TokenType is 0
            }
        }
        
        let mut hidden_state = Tensor::new(x_data, vec![seq_len, hidden_dim]).unwrap();
        hidden_state = hidden_state.layer_norm(&self.emb_norm_weight, &self.emb_norm_bias, 1e-12).unwrap();

        // 3. Encoder Blocks
        for block in &self.blocks {
            hidden_state = block.forward(&hidden_state).unwrap();
        }

        // 4. Mean Pooling & L2 Normalization (The compression step)
        let mut pooled = vec![0.0; hidden_dim];
        for i in 0..seq_len {
            let offset = i * hidden_dim;
            for j in 0..hidden_dim {
                pooled[j] += hidden_state.data[offset + j];
            }
        }
        
        let mut mag = 0.0;
        for j in 0..hidden_dim {
            pooled[j] /= seq_len as f32;
            mag += pooled[j] * pooled[j];
        }
        
        let mag = mag.sqrt();
        for j in 0..hidden_dim {
            pooled[j] /= mag; // L2 Normalize for instant dot-product search later
        }

        pooled // Returns a pristine [384] vector!
    }
}
