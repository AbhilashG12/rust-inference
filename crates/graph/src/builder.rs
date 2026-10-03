use crate::ir::{Graph, Node, NodeId, Operator, TensorId};
use tensor::tensor::Tensor;
use tensor::error::{Result, TensorError};

pub struct GraphBuilder {
    graph: Graph,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: Graph {
                nodes: Vec::new(),
                tensors: Vec::new(),
                network_inputs: Vec::new(),
                network_outputs: Vec::new(),
            },
        }
    }

    pub fn add_tensor(&mut self, tensor: Tensor) -> TensorId {
        let id = TensorId(self.graph.tensors.len());
        self.graph.tensors.push(tensor);
        id
    }

    fn allocate_output(&mut self, shape: Vec<usize>) -> TensorId {
        let tensor = Tensor::zeroes(shape);
        self.add_tensor(tensor)
    }


    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> Result<TensorId> {
        let shape_a = self.graph.tensors[a.0].shape.dims();
        let shape_b = self.graph.tensors[b.0].shape.dims();

        if shape_a.len() != 2 || shape_b.len() != 2 || shape_a[1] != shape_b[0] {
            return Err(TensorError::InvalidMatmul {
                shape_a: shape_a.to_vec(),
                shape_b: shape_b.to_vec(),
            });
        }

        let out_shape = vec![shape_a[0], shape_b[1]];
        let out_id = self.allocate_output(out_shape);

        self.graph.nodes.push(Node {
            id: NodeId(self.graph.nodes.len()),
            op: Operator::MatMul,
            inputs: vec![a, b],
            outputs: vec![out_id],
        });

        Ok(out_id)
    }

    pub fn add_broadcast(&mut self, target: TensorId, bias: TensorId) -> Result<TensorId> {
        let shape_target = self.graph.tensors[target.0].shape.dims();
        let shape_bias = self.graph.tensors[bias.0].shape.dims();

        if shape_target.len() != 2 || shape_bias.len() != 1 || shape_target[1] != shape_bias[0] {
            return Err(TensorError::InvalidBroadcast {
                shape_a: shape_target.to_vec(),
                shape_b: shape_bias.to_vec(),
            });
        }

        let out_id = self.allocate_output(shape_target.to_vec());
        
        self.graph.nodes.push(Node {
            id: NodeId(self.graph.nodes.len()),
            op: Operator::Add,
            inputs: vec![target, bias],
            outputs: vec![out_id],
        });

        Ok(out_id)
    }

    pub fn relu(&mut self, target: TensorId) -> Result<TensorId> {
        let shape = self.graph.tensors[target.0].shape.dims().to_vec();
        let out_id = self.allocate_output(shape);

        self.graph.nodes.push(Node {
            id: NodeId(self.graph.nodes.len()),
            op: Operator::Relu,
            inputs: vec![target],
            outputs: vec![out_id],
        });

        Ok(out_id)
    }

    pub fn build(self) -> Graph {
        self.graph
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::Executor;

    fn assert_close(a: &[f32], b: &[f32]) {
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-4, "Mismatch: {} != {}", x, y);
        }
    }

    #[test]
    fn test_graph_compilation_and_execution() {
        let mut builder = GraphBuilder::new();

        let input = Tensor::new(vec![1.0, 2.0, -1.0], vec![1, 3]).unwrap();
        
        let l1_weight = Tensor::new(vec![
            0.1,  0.2, -0.1,  0.3,
           -0.2,  0.1,  0.5, -0.1,
            0.3, -0.4,  0.2,  0.1
        ], vec![3, 4]).unwrap();
        let l1_bias = Tensor::new(vec![0.1, -0.1, 0.2, 0.0], vec![4]).unwrap();
        
        let l2_weight = Tensor::new(vec![
            0.5, -0.2,
           -0.3,  0.1,
            0.2,  0.4,
           -0.1,  0.3
        ], vec![4, 2]).unwrap();
        let l2_bias = Tensor::new(vec![0.05, -0.05], vec![2]).unwrap();

        //  Add raw data to the Graph Arena
        let input_id = builder.add_tensor(input);
        let w1_id = builder.add_tensor(l1_weight);
        let b1_id = builder.add_tensor(l1_bias);
        let w2_id = builder.add_tensor(l2_weight);
        let b2_id = builder.add_tensor(l2_bias);

        //  Compile the Graph (Shape Inference happens here!)
        let x = builder.matmul(input_id, w1_id).expect("L1 MatMul failed");
        let x = builder.add_broadcast(x, b1_id).expect("L1 Bias failed");
        let x = builder.relu(x).expect("ReLU failed");
        
        let x = builder.matmul(x, w2_id).expect("L2 MatMul failed");
        let logits_id = builder.add_broadcast(x, b2_id).expect("L2 Bias failed");

        // Extract the finished blueprint
        let mut graph = builder.build();

        // Ensure Topological Order: 4 operations = 4 nodes.
        assert_eq!(graph.nodes.len(), 4);

        // Ensure Shape Inference worked: Logits should be [1, 2]
        let logits_shape = graph.tensors[logits_id.0].shape.dims();
        assert_eq!(logits_shape, &[1, 2]);

        //  Execute the Graph
        Executor::run(&mut graph).expect("Execution failed");

        // Verify against Golden Reference
        let logits_tensor = &graph.tensors[logits_id.0];
        let probabilities = logits_tensor.softmax().unwrap();

        // These are the exact numbers we mathematically proved in Phase 2
        assert_close(&probabilities.data, &[0.410959, 0.589040]);
    }

    #[test]
    fn test_shape_inference_catches_errors() {
        let mut builder = GraphBuilder::new();
        
        // Mismatched shapes: [1, 3] trying to MatMul with [4, 2]
        let input = Tensor::zeroes(vec![1, 3]);
        let weight = Tensor::zeroes(vec![4, 2]); // Should be [3, 2]
        
        let in_id = builder.add_tensor(input);
        let w_id = builder.add_tensor(weight);

        // The builder should reject this immediately, BEFORE execution
        let result = builder.matmul(in_id, w_id);
        assert!(result.is_err(), "Graph Builder failed to catch shape mismatch!");
    }
}
