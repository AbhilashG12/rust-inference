use crate::error::{Result, TensorError};
use crate::shape::Shape;
use crate::profiler::ProfileScope;
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Shape,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, dims: Vec<usize>) -> Result<Self> {
        let shape = Shape::new(dims);
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        let strides = Self::compute_strides(shape.dims());
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    pub fn zeroes(dims: Vec<usize>) -> Self {
        let shape = Shape::new(dims);
        let data = vec![0.0; shape.numel()];
        let strides = Self::compute_strides(shape.dims());
        Self {
            data,
            shape,
            strides,
        }
    }

    /// Standard LayerNorm (subtracts mean, divides by variance, adds bias).
    /// Used by BERT/MiniLM encoder architectures.
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Result<Self> {
        let seq_len = self.shape.dims()[0];
        let hidden_dim = self.shape.dims()[1];
        let mut out = vec![0.0; self.data.len()];

        for i in 0..seq_len {
            let offset = i * hidden_dim;
            let row = &self.data[offset..offset + hidden_dim];
            
            let mean = row.iter().sum::<f32>() / hidden_dim as f32;
            let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / hidden_dim as f32;
            let std = (var + eps).sqrt();

            for j in 0..hidden_dim {
                out[offset + j] = ((row[j] - mean) / std) * weight.data[j] + bias.data[j];
            }
        }
        Tensor::new(out, self.shape.dims().to_vec())
    }

    /// GELU Activation Function (Gaussian Error Linear Unit).
    pub fn gelu(&self) -> Result<Self> {
        let mut out = Vec::with_capacity(self.data.len());
        for &x in &self.data {
            // Blazing fast GELU approximation used in BERT
            let c = (2.0f32 / std::f32::consts::PI).sqrt();
            let val = 0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh());
            out.push(val);
        }
        Tensor::new(out, self.shape.dims().to_vec())
    }

    /// Bi-directional Attention. 
    /// No Causal Mask. No KV Cache. Every token looks at every other token.
    pub fn attention_bidirectional(q: &Tensor, k: &Tensor, v: &Tensor, num_heads: usize) -> Result<Self> {
        let seq_len = q.shape.dims()[0];
        let hidden_dim = q.shape.dims()[1];
        let head_dim = hidden_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let mut out = vec![0.0; seq_len * hidden_dim];

        // Process each attention head
        for h in 0..num_heads {
            let head_offset = h * head_dim;

            for i in 0..seq_len {
                let q_row = i * hidden_dim + head_offset;
                let mut scores = vec![0.0; seq_len];

                // 1. Q * K^T (Bi-directional, looks at all tokens 'j')
                for j in 0..seq_len {
                    let k_row = j * hidden_dim + head_offset;
                    let mut dot = 0.0;
                    for d in 0..head_dim {
                        dot += q.data[q_row + d] * k.data[k_row + d];
                    }
                    scores[j] = dot * scale;
                }

                // 2. Softmax
                let max_val = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0;
                for j in 0..seq_len {
                    scores[j] = (scores[j] - max_val).exp();
                    sum_exp += scores[j];
                }

                // 3. Scores * V
                for j in 0..seq_len {
                    let prob = scores[j] / sum_exp;
                    let v_row = j * hidden_dim + head_offset;
                    for d in 0..head_dim {
                        out[q_row + d] += prob * v.data[v_row + d];
                    }
                }
            }
        }
        Tensor::new(out, vec![seq_len, hidden_dim])
    }

    pub fn attention_kv_cached(
        q: &Tensor,
        layer_idx: usize,
        kv_cache: &crate::cache::KVCache,
    ) -> Result<Tensor> {
        let _prof = ProfileScope::new("Attention_KVCached");
        let features = *q.shape.dims().last().unwrap();
        let head_dim = kv_cache.head_dim;
        let num_q_heads = features / head_dim;
        let num_kv_heads = kv_cache.num_kv_heads;
        
        // This solves GQA (Grouped Query Attention) mathematically without memory copies!
        let q_per_kv = num_q_heads / num_kv_heads; 
        
        let scale = 1.0 / (head_dim as f32).sqrt();
        let seq_len = kv_cache.current_seq_len + 1; // Include the token we just appended

        let mut out_data = vec![0.0; features];

        for h in 0..num_q_heads {
            let kv_h = h / q_per_kv;
            let q_offset = h * head_dim;

            let k_head_offset = layer_idx * (num_kv_heads * head_dim * kv_cache.max_seq_len)
                              + kv_h * (head_dim * kv_cache.max_seq_len);

            let mut scores = vec![0.0; seq_len];

            // 1. Q * K^T (Inner loop over 't' is perfectly contiguous!)
            for d in 0..head_dim {
                let q_val = q.data[q_offset + d] * scale;
                let k_d_offset = k_head_offset + d * kv_cache.max_seq_len;
                for t in 0..seq_len {
                    scores[t] += q_val * kv_cache.k_cache[k_d_offset + t];
                }
            }

            // 2. Local Head Softmax
            let mut max_val = f32::NEG_INFINITY;
            for t in 0..seq_len {
                if scores[t] > max_val { max_val = scores[t]; }
            }
            let mut sum_exp = 0.0;
            for t in 0..seq_len {
                let exp_val = (scores[t] - max_val).exp();
                scores[t] = exp_val;
                sum_exp += exp_val;
            }
            for t in 0..seq_len {
                scores[t] /= sum_exp;
            }

            // 3. Scores * V (Inner loop over 'd' is perfectly contiguous!)
            let v_head_offset = layer_idx * (num_kv_heads * kv_cache.max_seq_len * head_dim)
                              + kv_h * (kv_cache.max_seq_len * head_dim);

            for t in 0..seq_len {
                let score = scores[t];
                let v_t_offset = v_head_offset + t * head_dim;
                for d in 0..head_dim {
                    out_data[q_offset + d] += score * kv_cache.v_cache[v_t_offset + d];
                }
            }
        }

        Ok(Tensor::new(out_data, vec![1, features]).unwrap())
    }

    pub fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];
        if dims.is_empty() {
            return strides;
        }
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == Self::compute_strides(self.shape.dims())
    }

    pub fn reshape(mut self, dims: Vec<usize>) -> Result<Self> {
        let new_shape = Shape::new(dims.clone());
        if self.shape.numel() != new_shape.numel() {
            return Err(TensorError::InvalidReshape {
                original: self.shape.dims().to_vec(),
                target: dims,
            });
        }
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }

        self.strides = Self::compute_strides(new_shape.dims());
        self.shape = new_shape;
        Ok(self)
    }

    pub fn transpose_2d(mut self) -> Result<Self> {
        if self.shape.rank() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![0, 0],
                got: self.shape.dims().to_vec(),
            });
        }
        self.shape.dims.swap(0, 1);
        self.strides.swap(0, 1);
        Ok(self)
    }
}
