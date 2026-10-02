use crate::ir::{Graph, Operator};
use crate::planner::MemoryPlan;
use tensor::error::Result;

pub struct ExecutionPlan {
    pub graph: Graph,
    pub mem_plan: MemoryPlan,
}

pub struct Runtime;

impl Runtime {
    pub fn execute(plan: &mut ExecutionPlan) -> Result<()> {
        let mut live_counts = plan.mem_plan.usage_counts.clone();

        for node in &plan.graph.nodes {
            match node.op {
                Operator::MatMul => {
                    let a = &plan.graph.tensors[node.inputs[0].0];
                    let b = &plan.graph.tensors[node.inputs[1].0];
                    let result = a.matmul(b)?;
                    plan.graph.tensors[node.outputs[0].0] = result;
                }
                Operator::FusedMatMulAddRelu => {
                    let input = &plan.graph.tensors[node.inputs[0].0];
                    let weight = &plan.graph.tensors[node.inputs[1].0];
                    let bias = &plan.graph.tensors[node.inputs[2].0];
                    
                    let result = input.fused_matmul_add_relu(weight, bias)?;
                    plan.graph.tensors[node.outputs[0].0] = result;
                }
                _ => unimplemented!(),
            }

            for input_id in &node.inputs {
                if let Some(count) = live_counts.get_mut(input_id) {
                    *count -= 1;
                    if *count == 0 {
                        // In a fully developed engine, we would return `input_id`'s 
                        // backing byte-buffer to an Arena for the next node to overwrite.
                        // For now, we logically mark it dead.
                        // println!("♻️ Tensor {:?} is dead. Buffer ready for reuse.", input_id);
                    }
                }
            }
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::GraphBuilder;
    use crate::optimizer::GraphOptimizer;
    use crate::planner::MemoryPlanner;
    use tensor::tensor::Tensor;

    #[test]
    fn test_phase4_inference_engine_pipeline() {
        let mut builder = GraphBuilder::new();
        let input_id = builder.add_tensor(Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap());
        let w_id = builder.add_tensor(Tensor::new(vec![0.5, -0.5, 0.1, 0.2], vec![2, 2]).unwrap());
        let b_id = builder.add_tensor(Tensor::new(vec![0.1, -0.1], vec![2]).unwrap());

        let x = builder.matmul(input_id, w_id).unwrap();
        let x = builder.add_broadcast(x, b_id).unwrap();
        let _out = builder.relu(x).unwrap();

        let naive_graph = builder.build();
        assert_eq!(naive_graph.nodes.len(), 3); // Naive graph has 3 steps

        let optimized_graph = GraphOptimizer::optimize(naive_graph);
        
        assert_eq!(optimized_graph.nodes.len(), 1); 
        assert_eq!(optimized_graph.nodes[0].op, Operator::FusedMatMulAddRelu);

        let mem_plan = MemoryPlanner::plan(&optimized_graph);

        let mut plan = ExecutionPlan {
            graph: optimized_graph,
            mem_plan,
        };

        Runtime::execute(&mut plan).unwrap();
        
        println!(" Phase 4 Pipeline Complete!");
    }
}
