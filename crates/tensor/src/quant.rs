use crate::error::{Result, TensorError};
use crate::shape::Shape;
use crate::tensor::Tensor;
use rayon::prelude::*; // Ensure this matches your error path

/// Standard block size for Q8_0 quantization (same as GGML/llama.cpp)
pub const Q8_BLOCK_SIZE: usize = 32;

#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub struct BlockQ8_0 {
    pub scale: f32,
    pub qs: [i8; Q8_BLOCK_SIZE],
}

impl BlockQ8_0 {
    /// Computes the dot product of this Q8 block against a raw FP32 activation chunk.
    /// The compiler will auto-vectorize this tight loop into AVX2/FMA instructions.
    #[inline(always)]
    pub fn dot_fp32(&self, activation_chunk: &[f32]) -> f32 {
        let mut sum = 0.0;
        for i in 0..Q8_BLOCK_SIZE {
            // Integer to float conversion happens purely in registers, avoiding memory bandwidth bottlenecks.
            sum += (self.qs[i] as f32) * activation_chunk[i];
        }
        sum * self.scale
    }
}

#[derive(Debug, Clone)]
pub struct QuantizedTensorQ8_0 {
    pub blocks: Vec<BlockQ8_0>,
    pub shape: Shape,
}

impl QuantizedTensorQ8_0 {
    pub fn quantize(tensor: &Tensor) -> Self {
        let total_elements = tensor.data.len();

        assert!(
            total_elements % Q8_BLOCK_SIZE == 0,
            "Tensor size {} is not a multiple of Q8_BLOCK_SIZE ({})",
            total_elements,
            Q8_BLOCK_SIZE
        );

        let num_blocks = total_elements / Q8_BLOCK_SIZE;
        let mut blocks = Vec::with_capacity(num_blocks);

        for chunk in tensor.data.chunks_exact(Q8_BLOCK_SIZE) {
            let mut max_abs = 0.0_f32;
            for &val in chunk {
                let abs_val = val.abs();
                if abs_val > max_abs {
                    max_abs = abs_val;
                }
            }

            let scale = max_abs / 127.0;
            let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };

            let mut qs = [0i8; Q8_BLOCK_SIZE];
            for i in 0..Q8_BLOCK_SIZE {
                let q_val = (chunk[i] * inv_scale).round();
                qs[i] = q_val.clamp(-127.0, 127.0) as i8;
            }

            blocks.push(BlockQ8_0 { scale, qs });
        }

        Self {
            blocks,
            shape: tensor.shape.clone(),
        }
    }

    pub fn dequantize(&self) -> Tensor {
        let total_elements = self.blocks.len() * Q8_BLOCK_SIZE;
        let mut data = Vec::with_capacity(total_elements);

        for block in &self.blocks {
            for &q in &block.qs {
                data.push(q as f32 * block.scale);
            }
        }
        Tensor::new(data, self.shape.dims().to_vec()).unwrap()
    }

    /// Blazing fast, Multi-threaded Vector-Matrix multiplication (M=1) for Decode Phase.
    pub fn vec_mat_mul(&self, x: &Tensor) -> Result<Tensor> {
        let k = *x.shape.dims().last().unwrap();
        let n = self.shape.dims()[0];
        let weight_k = self.shape.dims()[1];

        if k != weight_k {
            return Err(TensorError::InvalidMatmul {
                shape_a: x.shape.dims().to_vec(),
                shape_b: self.shape.dims().to_vec(),
            });
        }

        let num_blocks_per_row = k / Q8_BLOCK_SIZE;
        let mut out_data = vec![0.0; n];

        // Rayon Data-Parallelism: Spits the output vector across CPU cores.
        // Each core independently calculates its assigned rows of the weight matrix.
        out_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, out_val)| {
                let mut row_sum = 0.0;
                let row_block_start = i * num_blocks_per_row;

                for b in 0..num_blocks_per_row {
                    let block = &self.blocks[row_block_start + b];
                    let act_start = b * Q8_BLOCK_SIZE;

                    // Extract the exact 32 floats from the hidden state
                    let act_chunk = &x.data[act_start..act_start + Q8_BLOCK_SIZE];

                    row_sum += block.dot_fp32(act_chunk);
                }
                *out_val = row_sum;
            });

        Ok(Tensor::new(out_data, vec![1, n]).unwrap())
    }
}
