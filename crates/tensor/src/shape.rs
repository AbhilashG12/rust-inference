#[derive(Debug, PartialEq, Clone, Eq)]
pub struct Shape {
    pub(crate) dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
}
