use crate::error::{Result, TensorError};
use crate::shape::Shape;

#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Shape,
    pub strides: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, dims: Vec<usize>) -> Result<Self> {
        let shape = Shape::new(dims);
        if data.len() != shape.numel() {
            return Err(TensorError::ShapeMismatch {
                expected: vec![shape.numel()],
                got: vec![data.len()],
            });
        }
        let strides = Self::compute_strides(shape.dims());
        Ok(Self {
            data,
            shape,
            strides,
        })
    }

    pub fn zeroes(dims: Vec<usize>) -> Self {
        let shape = Shape::new(dims);
        let data = vec![0.0; shape.numel()];
        let strides = Self::compute_strides(shape.dims());
        Self {
            data,
            shape,
            strides,
        }
    }

    pub fn compute_strides(dims: &[usize]) -> Vec<usize> {
        let mut strides = vec![1; dims.len()];
        if dims.is_empty() {
            return strides;
        }
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }
        strides
    }

    pub fn is_contiguous(&self) -> bool {
        self.strides == Self::compute_strides(self.shape.dims())
    }

    pub fn reshape(mut self, dims: Vec<usize>) -> Result<Self> {
        let new_shape = Shape::new(dims.clone());
        if self.shape.numel() != new_shape.numel() {
            return Err(TensorError::InvalidReshape {
                original: self.shape.dims().to_vec(),
                target: dims,
            });
        }
        if !self.is_contiguous() {
            return Err(TensorError::NotContiguous);
        }

        self.strides = Self::compute_strides(new_shape.dims());
        self.shape = new_shape;
        Ok(self)
    }

    pub fn transpose_2d(mut self) -> Result<Self> {
        if self.shape.rank() != 2 {
            return Err(TensorError::ShapeMismatch {
                expected: vec![0, 0],
                got: self.shape.dims().to_vec(),
            });
        }
        self.shape.dims.swap(0, 1);
        self.strides.swap(0, 1);
        Ok(self)
    }
}
