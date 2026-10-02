use crate::tensor::Tensor;
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<i8>,   
    pub scale: f32,      
    pub shape: Shape,
}

impl QuantizedTensor {
    pub fn quantize(tensor: &Tensor) -> Self {
        let mut max_abs = 0.0_f32;
        for &val in &tensor.data {
            if val.abs() > max_abs {
                max_abs = val.abs();
            }
        }

        let scale = max_abs / 127.0;

        let data = tensor.data.iter().map(|&val| {
            let q_val = (val / scale).round();
            let clamped = q_val.clamp(-127.0, 127.0);
            clamped as i8
        }).collect();

        Self {
            data,
            scale,
            shape: tensor.shape.clone(),
        }
    }

    pub fn dequantize(&self) -> Tensor {
        let mut data = Vec::with_capacity(self.data.len());
        for &val in &self.data {
            data.push(val as f32 * self.scale);
        }
        Tensor::new(data, self.shape.dims().to_vec()).unwrap()
    }
}
