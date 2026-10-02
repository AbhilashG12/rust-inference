use crate::tensor::Tensor;
use crate::error::Result;

pub struct Embedding {
    pub weight : Tensor,
}

pub struct Linear{
    pub weight : Tensor,
    pub bias : Tensor,
}

pub struct TransformerBlock {
    pub norm1: Tensor, 
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear, 
    
    pub norm2: Tensor, 
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl TransformerBlock {
    pub fn forward(&self, x: &Tensor, pos: usize) -> Result<Tensor> {
        let norm_x = x.rms_norm(&self.norm1, 1e-5)?;
        
        let mut q = self.w_q.forward(&norm_x)?;
        let mut k = self.w_k.forward(&norm_x)?;
        let v = self.w_v.forward(&norm_x)?;
        
        q = q.apply_rope(pos);
        k = k.apply_rope(pos);
        
        let attn_out = Tensor::attention(&q, &k, &v)?;
        
        let proj_out = self.w_o.forward(&attn_out)?;
        
        let mut residual_1 = x.add(&proj_out)?; 
        
        let norm_res_1 = residual_1.rms_norm(&self.norm2, 1e-5)?;
        
        let hidden = self.ffn_up.forward(&norm_res_1)?;
        let activated = hidden.silu();
        let ffn_out = self.ffn_down.forward(&activated)?;
        
        let final_out = residual_1.add(&ffn_out)?;
        
        Ok(final_out)
    }
}

impl Embedding {
    pub fn forward(&self, token_id :usize) -> Tensor {
        let hidden_dim = self.weight.shape.dims()[1];
        let offset = token_id * hidden_dim;
        let data = self.weight.data[offset..offset + hidden_dim].to_vec();
        Tensor::new(data, vec![1, hidden_dim]).unwrap()
    }


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
