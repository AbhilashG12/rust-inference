use crate::error::{Result, TensorError};
use crate::kernels::MatMulDispatcher;
use crate::tensor::Tensor;
use crate::profiler::ProfileScope;

impl Tensor {
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        if !self.is_contiguous() || !other.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.dims().to_vec())
    }

    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        let _prof = ProfileScope::new("MatMul");
        if self.shape.rank() != 2 || other.shape.rank() != 2 {
            return Err(TensorError::InvalidMatmul {
                shape_a: self.shape.dims().to_vec(),
                shape_b: other.shape.dims().to_vec(),
            });
        }

        let m = self.shape.dims()[0];
        let k = self.shape.dims()[1];
        let other_k = other.shape.dims()[0];
        let n = other.shape.dims()[1];

        if k != other_k {
            return Err(TensorError::InvalidMatmul {
                shape_a: self.shape.dims().to_vec(),
                shape_b: other.shape.dims().to_vec(),
            });
        }

        let mut out = Tensor::zeroes(vec![m, n]);

        MatMulDispatcher::dispatch(self, other, &mut out, m, k, n);
        Ok(out)
    }

    pub fn add_broadcast_1d(&self, bias: &Tensor) -> Result<Tensor> {
        if self.shape.rank() != 2 || bias.shape.rank() != 1 {
            return Err(TensorError::InvalidBroadcast {
                shape_a: self.shape.dims().to_vec(),
                shape_b: bias.shape.dims().to_vec(),
            });
        }

        let batch_size = self.shape.dims()[0];
        let features = self.shape.dims()[1];

        if bias.shape.dims()[0] != features {
            return Err(TensorError::InvalidBroadcast {
                shape_a: self.shape.dims().to_vec(),
                shape_b: bias.shape.dims().to_vec(),
            });
        }

        let mut out = self.clone();
        for b in 0..batch_size {
            for f in 0..features {
                let out_idx = b * out.strides[0] + f * out.strides[1];
                let bias_idx = f * bias.strides[0];
                out.data[out_idx] += bias.data[bias_idx];
            }
        }

        Ok(out)
    }

    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();

        Tensor::new(data, self.shape.dims().to_vec()).unwrap()
    }

    pub fn softmax(&self) -> Result<Tensor> {
        if self.shape.rank() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![0, 0],
                got: self.shape.dims().to_vec(),
            });
        }

        let batch_size = self.shape.dims()[0];
        let classes = self.shape.dims()[1];
        let mut out = Tensor::zeroes(self.shape.dims().to_vec());

        for b in 0..batch_size {
            let mut max_val = f32::NEG_INFINITY;
            for c in 0..classes {
                let val = self.data[b * self.strides[0] + c * self.strides[1]];
                if val > max_val {
                    max_val = val;
                }
            }

            let mut sum_exp = 0.0;
            let mut exps = vec![0.0; classes];
            for c in 0..classes {
                let val = self.data[b * self.strides[0] + c * self.strides[1]];
                let exp_val = (val - max_val).exp();
                exps[c] = exp_val;
                sum_exp += exp_val;
            }

            for c in 0..classes {
                out.data[b * out.strides[0] + c * out.strides[1]] = exps[c] / sum_exp;
            }
        }

        Ok(out)
    }

    pub fn fused_matmul_add_relu(&self, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let m = self.shape.dims()[0];
        let k = self.shape.dims()[1];
        let n = weight.shape.dims()[1];

        let mut out = Tensor::zeroes(vec![m, n]);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    let a_idx = i * self.strides[0] + p * self.strides[1];
                    let b_idx = p * weight.strides[0] + j * weight.strides[1];
                    sum += self.data[a_idx] * weight.data[b_idx];
                }

                let bias_idx = j * bias.strides[0];
                sum += bias.data[bias_idx];

                if sum < 0.0 {
                    sum = 0.0;
                }
                let out_idx = i * out.strides[0] + j * out.strides[1];
                out.data[out_idx] = sum;
            }
        }
        Ok(out)
    }

    pub fn rms_norm(&self, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let features = self.shape.dims().last().unwrap();
        let batch = self.shape.numel() / features;

        let mut out = Tensor::zeroes(self.shape.dims().to_vec());

        for b in 0..batch {
            let offset = b * features;

            let mut sum_sq = 0.0;
            for f in 0..*features {
                let val = self.data[offset + f];
                sum_sq += val * val;
            }

            let mean_sq = sum_sq / (*features as f32);
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();

            for f in 0..*features {
                out.data[offset + f] = (self.data[offset + f] * inv_rms) * weight.data[f];
            }
        }
        Ok(out)
    }

    pub fn silu(&self) -> Tensor {
        let data = self
            .data
            .iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();
        Tensor::new(data, self.shape.dims().to_vec()).unwrap()
    }

    pub fn apply_rope(&self, pos: usize) -> Tensor {
        let mut out = self.clone();
        let features = *self.shape.dims().last().unwrap();
        let head_dim = 64;
        let half = head_dim / 2; // 32

        for h in 0..(features / head_dim) {
            let offset = h * head_dim;

            for i in 0..half {
                // Standard LLaMA theta calculation
                let freq = 1.0_f32 / 10000.0_f32.powf((2 * i) as f32 / head_dim as f32);
                let val = (pos as f32) * freq;
                let cos_val = val.cos();
                let sin_val = val.sin();

                let x0 = self.data[offset + i];
                let x1 = self.data[offset + i + half];

                // HuggingFace: [-x2, x1] rotated with cos/sin
                out.data[offset + i] = x0 * cos_val - x1 * sin_val;
                out.data[offset + i + half] = x1 * cos_val + x0 * sin_val;
            }
        }
        out
    }

    pub fn repeat_features(&self, target_dim: usize) -> Result<Tensor> {
        let current_dim = *self.shape.dims().last().unwrap();
        if current_dim == target_dim { return Ok(self.clone()); }
        
        let repeats = target_dim / current_dim;
        let head_dim = 64; // Block-repeat based on Head Dimension!
        let num_kv_heads = current_dim / head_dim;
        
        let mut data = Vec::with_capacity(target_dim);
        
        // Correct Block-Mapping for GQA (K1, K1, K1, K2, K2, K2)
        for h in 0..num_kv_heads {
            let start = h * head_dim;
            let end = start + head_dim;
            let head_data = &self.data[start..end];
            
            for _ in 0..repeats {
                data.extend_from_slice(head_data);
            }
        }
        
        Tensor::new(data, vec![1, target_dim])
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                expected: self.shape.dims().to_vec(),
                got: other.shape.dims().to_vec(),
            });
        }
        
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
            
        Tensor::new(data, self.shape.dims().to_vec())
    }

    pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let features = *q.shape.dims().last().unwrap();
        let seq_len = k.shape.dims()[0];
        
        let head_dim = 64; 
        let num_heads = features / head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt(); // Correct Multi-Head scaling!
        
        let mut out_data = Vec::with_capacity(features);

        // Process each Attention Head completely independently
        for h in 0..num_heads {
            let head_offset = h * head_dim;
            
            // 1. Extract Q for this specific head -> Shape: [1, 64]
            let mut q_head_data = Vec::with_capacity(head_dim);
            for i in 0..head_dim {
                q_head_data.push(q.data[head_offset + i]);
            }
            let q_head = Tensor::new(q_head_data, vec![1, head_dim]).unwrap();
            
            // 2. Extract K for this specific head across the whole sequence -> Shape: [Seq_Len, 64]
            let mut k_head_data = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                for i in 0..head_dim {
                    let idx = s * k.strides[0] + (head_offset + i) * k.strides[1];
                    k_head_data.push(k.data[idx]);
                }
            }
            let k_head = Tensor::new(k_head_data, vec![seq_len, head_dim]).unwrap();
            
            // 3. Extract V for this specific head across the whole sequence -> Shape: [Seq_Len, 64]
            let mut v_head_data = Vec::with_capacity(seq_len * head_dim);
            for s in 0..seq_len {
                for i in 0..head_dim {
                    let idx = s * v.strides[0] + (head_offset + i) * v.strides[1];
                    v_head_data.push(v.data[idx]);
                }
            }
            let v_head = Tensor::new(v_head_data, vec![seq_len, head_dim]).unwrap();

            // 4. Compute standard Attention for JUST this head!
            let k_t_lazy = k_head.transpose_2d().unwrap();
            let rows = k_t_lazy.shape.dims()[0];
            let cols = k_t_lazy.shape.dims()[1];
            let mut k_t_data = Vec::with_capacity(rows * cols);
            for r in 0..rows {
                for c in 0..cols {
                    let idx = r * k_t_lazy.strides[0] + c * k_t_lazy.strides[1];
                    k_t_data.push(k_t_lazy.data[idx]);
                }
            }
            let k_t = Tensor::new(k_t_data, vec![rows, cols]).unwrap();
            
            let mut scores = q_head.matmul(&k_t).unwrap();
            
            for val in scores.data.iter_mut() {
                *val *= scale;
            }
            
            let probs = scores.softmax().unwrap();
            let head_out = probs.matmul(&v_head).unwrap();
            
            // 5. Append this head's output to the final concatenated vector
            out_data.extend_from_slice(&head_out.data);
        }
        
        // Return the re-assembled 576-feature vector
        Ok(Tensor::new(out_data, vec![1, features]).unwrap())
    }


    pub fn concat_seq(a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let shape_a = a.shape.dims();
        let shape_b = b.shape.dims();

        if shape_a.len() != 2 || shape_b.len() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![0, 0],
                got: shape_a.to_vec(),
            });
        }

        if shape_a[1] != shape_b[1] {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape_a[0], shape_a[1]],
                got: shape_b.to_vec(),
            });
        }

        let features = shape_a[1];
        let new_seq_len = shape_a[0] + shape_b[0];

        let mut data = Vec::with_capacity(new_seq_len * features);

        data.extend_from_slice(&a.data);
        data.extend_from_slice(&b.data);
        Ok(Tensor::new(data, vec![new_seq_len, features]).unwrap())
    }
}
