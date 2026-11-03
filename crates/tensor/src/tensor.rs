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
