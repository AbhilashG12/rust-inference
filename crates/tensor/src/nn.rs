use crate::cache::KVCache;
use crate::error::Result;
use crate::quant::QuantizedTensorQ8_0;
use crate::tensor::Tensor;

pub struct Embedding {
    pub weight: Tensor,
}

pub struct Linear {
    pub weight: QuantizedTensorQ8_0, // Now stores compressed 8-bit blocks!
    pub bias: Tensor,
}

pub struct TransformerBlock {
    pub norm1: Tensor,
    pub w_q: Linear,
    pub w_k: Linear,
    pub w_v: Linear,
    pub w_o: Linear,

    pub norm2: Tensor,
    pub ffn_gate: Linear,
    pub ffn_up: Linear,
    pub ffn_down: Linear,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &Tensor,
        pos: usize,
        layer_idx: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let norm_x = x.rms_norm(&self.norm1, 1e-5)?;

        let mut q = self.w_q.forward(&norm_x)?;
        let mut k = self.w_k.forward(&norm_x)?;
        let v = self.w_v.forward(&norm_x)?;

        // Notice: `repeat_features` is entirely deleted!
        q = q.apply_rope(pos);
        k = k.apply_rope(pos);

        // Map the new token straight into the continuous buffer
        kv_cache.append(layer_idx, &k, &v);

        // Extract attention scores reading natively from the flat array memory
        let attn_out = Tensor::attention_kv_cached(&q, layer_idx, kv_cache)?;

        let proj_out = self.w_o.forward(&attn_out)?;
        let residual_1 = x.add(&proj_out)?;

        let norm_res_1 = residual_1.rms_norm(&self.norm2, 1e-5)?;
        let gate = self.ffn_gate.forward(&norm_res_1)?;
        let activated_gate = gate.silu();
        let up = self.ffn_up.forward(&norm_res_1)?;
    
        let hidden = activated_gate.mul(&up)?;
        let ffn_out = self.ffn_down.forward(&hidden)?;
        let final_out = residual_1.add(&ffn_out)?;
        
        Ok(final_out)
    }
}

impl Embedding {
    pub fn forward(&self, token_id: usize) -> Tensor {
        let hidden_dim = self.weight.shape.dims()[1];
        let mut data = Vec::with_capacity(hidden_dim);
        for i in 0..hidden_dim {
            let idx = (token_id * self.weight.strides[0]) + (i * self.weight.strides[1]);
            data.push(self.weight.data[idx]);
        }

        Tensor::new(data, vec![1, hidden_dim]).unwrap()
    }
}

impl Linear {
    pub fn new(weight: QuantizedTensorQ8_0, bias: Tensor) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq_len = x.shape.dims()[0];

        let out = if seq_len == 1 {
            // M=1 Decode Phase: Route to lightning-fast Q8 SIMD kernel
            self.weight.vec_mat_mul(x)?
        } else {
            // Prefill Phase: Dequantize locally to run standard dense GEMM
            let fp32_weight = self.weight.dequantize();
            x.matmul(&fp32_weight)?
        };

        let out_with_bias = out.add_broadcast_1d(&self.bias)?;
        Ok(out_with_bias)
    }
}
