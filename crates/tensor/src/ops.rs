use crate::tensor::Tensor;
use crate::error::{Result,TensorError};

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

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    let a_idx = i * self.strides[0] + p * self.strides[1];
                    let b_idx = p * other.strides[0] + j * other.strides[1];
                    sum += self.data[a_idx] * other.data[b_idx];
                }
                let out_idx = i * out.strides[0] + j * out.strides[1];
                out.data[out_idx] = sum;
            }
        }

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


}
