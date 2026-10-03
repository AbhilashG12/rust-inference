use crate::ir::{Graph, TensorId};
use std::collections::{HashMap};

#[derive(Debug)]
pub struct MemoryPlan {
    pub usage_counts: HashMap<TensorId, usize>,
}

pub struct MemoryPlanner;

impl MemoryPlanner {
    pub fn plan(graph: &Graph) -> MemoryPlan {
        let mut usage_counts = HashMap::new();
        for node in &graph.nodes {
            for &input_id in &node.inputs {
                *usage_counts.entry(input_id).or_insert(0) += 1;
            }
        }

        MemoryPlan { usage_counts }
    }
}
