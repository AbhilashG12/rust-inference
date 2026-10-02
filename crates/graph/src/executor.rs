use crate::ir::{Graph, Operator};
use tensor::error::Result;

pub struct Executor;

impl Executor {

    pub fn run(graph:&mut Graph) -> Result<()> {
        for node in &graph.nodes {
            match node.op {
                Operator::MatMul => {
                    let a = &graph.tensors[node.inputs[0].0];
                    let b = &graph.tensors[node.inputs[1].0];
                    let result = a.matmul(b)?;
                    graph.tensors[node.outputs[0].0] = result;
                }
                Operator::Add => {
                    let a = &graph.tensors[node.inputs[0].0];
                    let b = &graph.tensors[node.inputs[1].0];
                    let result = a.add_broadcast_1d(b)?;
                    graph.tensors[node.outputs[0].0] = result;
                }
                Operator::Relu => {
                    let a = &graph.tensors[node.inputs[0].0];
                    let result = a.relu();
                    graph.tensors[node.outputs[0].0] = result;
                }
                Operator::Softmax => {
                    let a = &graph.tensors[node.inputs[0].0];
                    let result = a.softmax()?;
                    graph.tensors[node.outputs[0].0] = result;
                }
            }
        }
        Ok(())
    }
}
