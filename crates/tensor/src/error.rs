use thiserror::Error;

#[derive(Error, Debug, PartialEq, Eq)]
pub enum TensorError {
    #[error("Shape mismatch : expected {expected:?}, got {got:?}")]
    ShapeMismatch {expected : Vec<usize>, got : Vec<usize>},

    #[error("Invalid Matrix Multiplicatiob : {shape_a:?} x {shape_b:?}")]
    InvalidMatmul {shape_a :Vec<usize>, shape_b : Vec<usize>},

    #[error("Cannot reshape {original:?} to {target:?}. Element count must match")]
    InvalidReshape {original : Vec<usize>, target : Vec<usize>},

    #[error("Operation requires contiguous memory layout")]
    NotContiguous,
}

pub type Result<T> = std::result::Result<T,TensorError>;
