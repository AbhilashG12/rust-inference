use crate::ir::{Graph, Node, NodeId, Operator};

pub struct GraphOptimizer;

impl GraphOptimizer {
    pub fn optimize(mut graph: Graph) -> Graph {
        let mut optimized_nodes = Vec::new();
        let mut skip_next = 0;

        for i in 0..graph.nodes.len() {
            if skip_next > 0 {
                skip_next -= 1;
                continue;
            }

            if i + 2 < graph.nodes.len() {
                let n1 = &graph.nodes[i];
                let n2 = &graph.nodes[i + 1];
                let n3 = &graph.nodes[i + 2];

                if n1.op == Operator::MatMul && n2.op == Operator::Add && n3.op == Operator::Relu {
                    if n1.outputs[0] == n2.inputs[0] && n2.outputs[0] == n3.inputs[0] {
                        
                        println!("🔧 Optimizer: Fusing MatMul + Add + ReLU");
                        
                        let fused_node = Node {
                            id: NodeId(optimized_nodes.len()),
                            op: Operator::FusedMatMulAddRelu,
                            inputs: vec![n1.inputs[0], n1.inputs[1], n2.inputs[1]],
                            outputs: vec![n3.outputs[0]],
                        };
                        optimized_nodes.push(fused_node);
                        
                        skip_next = 2;
                        continue;
                    }
                }
            }

            let mut node_copy = graph.nodes[i].clone();
            node_copy.id = NodeId(optimized_nodes.len());
            optimized_nodes.push(node_copy);
        }

        graph.nodes = optimized_nodes;
        graph
    }
}
