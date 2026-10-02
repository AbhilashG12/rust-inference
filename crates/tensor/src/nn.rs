use crate::tensor::Tensor;
use crate::error::Result;

pub struct Linear{
    pub weight : Tensor,
    pub bias : Tensor,
}

impl Linear {
    pub fn new(weight:Tensor, bias:Tensor) -> Self {
        Self {weight, bias}
    }

    pub fn forward(&self, x:&Tensor) -> Result<Tensor> {
        let out = x.matmul(&self.weight)?;
        let out_with_bias = out.add_broadcast_1d(&self.bias)?;

        Ok(out_with_bias)
    }

}
