use crate::tensor::Tensor;
use crate::error::{Result,TensorError};
use crate::kernels::MatMulDispatcher;

impl Tensor{

    pub fn add(&self, other:&Tensor) -> Result<Tensor> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch{
                expected : self.shape.dims().to_vec(),
                got : other.shape.dims().to_vec(),
            });
        }
        if !self.is_contiguous() || !other.is_contiguous(){
            return Err(TensorError::NotContiguous);
        }

        let data = self.data.iter().zip(other.data.iter()).map(|(a,b)| a+b).collect();
        Tensor::new(data, self.shape.dims().to_vec())
    }

    pub fn matmul(&self, other:&Tensor) -> Result<Tensor>{
        if self.shape.rank() != 2 || other.shape.rank() != 2 {
            return Err(TensorError::InvalidMatmul{
                shape_a : self.shape.dims().to_vec(),
                shape_b : other.shape.dims().to_vec(),
            });
        }

        let m = self.shape.dims()[0];
        let k = self.shape.dims()[1];
        let other_k = other.shape.dims()[0];
        let n = other.shape.dims()[1];

        if k != other_k {
            return Err(TensorError::InvalidMatmul{
                shape_a : self.shape.dims().to_vec(),
                shape_b : other.shape.dims().to_vec(),
            });
        }

        let mut out = Tensor::zeroes(vec![m,n]);

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
        let data: Vec<f32> = self.data.iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect();
            
        Tensor::new(data, self.shape.dims().to_vec()).unwrap()
    }

    pub fn softmax(&self) -> Result<Tensor> {
        if self.shape.rank() != 2 {
            return Err(TensorError::ShapeMismatch { 
                expected: vec![0, 0], got: self.shape.dims().to_vec() 
            });
        }

        let batch_size = self.shape.dims()[0];
        let classes = self.shape.dims()[1];
        let mut out = Tensor::zeroes(self.shape.dims().to_vec());

        for b in 0..batch_size {
            let mut max_val = f32::NEG_INFINITY;
            for c in 0..classes {
                let val = self.data[b * self.strides[0] + c * self.strides[1]];
                if val > max_val { max_val = val; }
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
        let data = self.data.iter().map(|&x| {
            x * (1.0 / (1.0 + (-x).exp()))
        }).collect();
        Tensor::new(data, self.shape.dims().to_vec()).unwrap()
    }
    
    pub fn apply_rope(&self, pos: usize) -> Tensor {
        let mut out = self.clone();
        let features = self.shape.dims().last().unwrap();
        
        for i in (0..*features).step_by(2) {
            let theta = 10000.0_f32.powf(-((i as f32) / (*features as f32)));
            let angle = (pos as f32) * theta;
            
            let cos_val = angle.cos();
            let sin_val = angle.sin();
            
            let x0 = self.data[i];
            let x1 = self.data[i + 1];
            
            out.data[i] = x0 * cos_val - x1 * sin_val;
            out.data[i + 1] = x0 * sin_val + x1 * cos_val;
        }
        out
    }

    pub fn attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let d_k = q.shape.dims().last().unwrap();
        let scale = 1.0 / (*d_k as f32).sqrt();

        let k_t = k.clone().transpose_2d()?;
        let mut scores = q.matmul(&k_t)?;
        
        for val in scores.data.iter_mut() {
            *val *= scale;
        }
        
        let probs = scores.softmax()?;
        
        let out = probs.matmul(v)?;
        
        Ok(out)
    }

    pub fn repeat_features(&self, target_dim: usize) -> Result<Tensor> {
        let current_dim = *self.shape.dims().last().unwrap();
        if current_dim == target_dim {
            return Ok(self.clone());
        }
        let repeats = target_dim / current_dim;
        let mut data = Vec::with_capacity(target_dim);
        
        for _ in 0..repeats {
            data.extend_from_slice(&self.data);
        }
        
        Tensor::new(data, vec![1, target_dim])
    }
}
