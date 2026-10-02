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


}
