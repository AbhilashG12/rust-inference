use std::collections::HashMap;
use std::fs::File;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use tensor::tensor::Tensor;
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

pub struct ModelLoader;

impl ModelLoader {
    pub fn load_safetensors(path: &str) -> Result<HashMap<String, Tensor>, LoaderError> {
        let file = File::open(path)?;
        let mmap = unsafe { MmapOptions::new().map(&file)? };
        
        let st = SafeTensors::deserialize(&mmap)?;
        
        let mut tensors = HashMap::new();

        for name in st.names() {
            let tensor_view = st.tensor(&name)?;
            let shape = tensor_view.shape().to_vec();
            let data: Vec<f32> = tensor_view.data()
                .chunks_exact(4)
                .map(|chunk| {
                    let array: [u8; 4] = chunk.try_into().unwrap();
                    f32::from_le_bytes(array)
                })
                .collect();
            let my_tensor = Tensor::new(data, shape).map_err(|_| LoaderError::EngineError)?;
            tensors.insert(name.to_string(), my_tensor);
        }

        Ok(tensors)
    }
}
