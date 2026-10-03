use tensor::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

#[derive(Debug,Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    MatMul,
    Add,
    Relu,
    Softmax,
    FusedMatMulAddRelu,
}

#[derive(Debug, Clone)]
pub struct Node {
    pub id : NodeId,
    pub op : Operator,
    pub inputs : Vec<TensorId>,
    pub outputs : Vec<TensorId>,
}

#[derive(Debug)]
pub struct Graph {
    pub nodes : Vec<Node>,
    pub tensors : Vec<Tensor>,
    pub network_inputs : Vec<TensorId>,
    pub network_outputs : Vec<TensorId>,
}
