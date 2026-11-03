use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use tensor::tensor::Tensor;
use tensor::quant::QuantizedTensorQ8_0;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum LoaderError {
    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),
    #[error("SafeTensors Error: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("Tensor engine error")]
    EngineError,
}

pub enum LoadedWeight {
    Fp32(Tensor),
    Q8(QuantizedTensorQ8_0),
}

pub struct ModelLoader;

impl ModelLoader {
    pub fn load_safetensors(path: &str) -> Result<HashMap<String, LoadedWeight>, LoaderError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        let st = SafeTensors::deserialize(&mmap)?;
        let mut tensors = HashMap::new();

        for name in st.names() {
            let tensor_view = st.tensor(&name)?;
            let shape = tensor_view.shape().to_vec();
            let data: Vec<f32> = match tensor_view.dtype() {
                safetensors::Dtype::F32 => tensor_view
                    .data()
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect(),
                safetensors::Dtype::BF16 => tensor_view
                    .data()
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect(),
                safetensors::Dtype::F16 => tensor_view
                    .data()
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes(chunk.try_into().unwrap()) as u32;
                        let sign = (bits & 0x8000) << 16;
                        let exp = (bits & 0x7c00) >> 10;
                        let mant = bits & 0x03ff;
                        let f32_bits = if exp == 0 {
                            sign
                        } else if exp == 31 {
                            sign | 0x7f800000 | (mant << 13)
                        } else {
                            sign | ((exp + 112) << 23) | (mant << 13)
                        };
                        f32::from_bits(f32_bits)
                    })
                    .collect(),
                other => panic!("Unsupported dtype in safetensors: {:?}", other),
            };

            let raw_tensor = Tensor::new(data, shape).map_err(|_| LoaderError::EngineError)?;

            // Do NOT transpose here anymore! Keep native [out, in] layout for SIMD vector math!
            if raw_tensor.shape.rank() == 2 && !name.contains("embed") {
                let quantized_tensor = QuantizedTensorQ8_0::quantize(&raw_tensor);
                tensors.insert(name.to_string(), LoadedWeight::Q8(quantized_tensor));
            } else {
                tensors.insert(name.to_string(), LoadedWeight::Fp32(raw_tensor));
            }
        }

        Ok(tensors)
    }
}
