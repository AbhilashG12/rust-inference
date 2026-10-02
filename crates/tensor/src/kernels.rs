use crate::tensor::Tensor;
use rayon::prelude::*;

const TILE_SIZE: usize = 32;

pub struct MatMulDispatcher;

impl MatMulDispatcher {
    
    pub fn dispatch(a: &Tensor, b: &Tensor, out: &mut Tensor, m: usize, k: usize, n: usize) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    Self::matmul_avx2_fma_tiled(a, b, out, m, k, n);
                }
                return;
            }
        }
        
        Self::matmul_scalar_tiled_parallel(a, b, out, m, k, n);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn matmul_avx2_fma_tiled(a: &Tensor, b: &Tensor, out: &mut Tensor, m: usize, k: usize, n: usize) {
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        out.data.par_chunks_exact_mut(n).enumerate().for_each(|(i, out_row)| {
            for j_step in (0..n).step_by(TILE_SIZE) {
                for p_step in (0..k).step_by(TILE_SIZE) {
                    
                    let j_end = (j_step + TILE_SIZE).min(n);
                    let p_end = (p_step + TILE_SIZE).min(k);

                    for p in p_step..p_end {
                        let a_val = a.data[i * a.strides[0] + p * a.strides[1]];
                        let a_vec = _mm256_set1_ps(a_val); 

                        let mut j = j_step;
                        while j + 8 <= j_end {
                            let out_ptr = out_row.as_mut_ptr().add(j);
                            let mut out_vec = _mm256_loadu_ps(out_ptr);

                            let b_ptr = b.data.as_ptr().add(p * b.strides[0] + j * b.strides[1]);
                            let b_vec = _mm256_loadu_ps(b_ptr);

                            out_vec = _mm256_fmadd_ps(a_vec, b_vec, out_vec);

                            _mm256_storeu_ps(out_ptr, out_vec);
                            j += 8;
                        }
                        
                        for r in j..j_end {
                            let b_val = b.data[p * b.strides[0] + r * b.strides[1]];
                            out_row[r] += a_val * b_val;
                        }
                    }
                }
            }
        });
    }

    fn matmul_scalar_tiled_parallel(a: &Tensor, b: &Tensor, out: &mut Tensor, m: usize, k: usize, n: usize) {
        out.data.par_chunks_exact_mut(n).enumerate().for_each(|(i, out_row)| {
            for j_step in (0..n).step_by(TILE_SIZE) {
                for p_step in (0..k).step_by(TILE_SIZE) {
                    let j_end = (j_step + TILE_SIZE).min(n);
                    let p_end = (p_step + TILE_SIZE).min(k);

                    for p in p_step..p_end {
                        let a_val = a.data[i * a.strides[0] + p * a.strides[1]];
                        for j in j_step..j_end {
                            let b_val = b.data[p * b.strides[0] + j * b.strides[1]];
                            out_row[j] += a_val * b_val;
                        }
                    }
                }
            }
        });
    }
}
