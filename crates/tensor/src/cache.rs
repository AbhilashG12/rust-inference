use crate::tensor::Tensor;

pub struct KVCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
    pub current_seq_len: usize,
}

impl KVCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: Vec::with_capacity(num_layers),
            values: Vec::with_capacity(num_layers),
            current_seq_len: 0,
        }
    }

    pub fn update(&mut self, layer_idx: usize, new_k: &Tensor, new_v: &Tensor) -> (Tensor, Tensor) {
        if self.keys.len() <= layer_idx {
            self.keys.push(new_k.clone());
            self.values.push(new_v.clone());
        } else {
            let old_k = &self.keys[layer_idx];
            let old_v = &self.values[layer_idx];

            self.keys[layer_idx] = Tensor::concat_seq(old_k, new_k).unwrap();
            self.values[layer_idx] = Tensor::concat_seq(old_v, new_v).unwrap();
        }

        (self.keys[layer_idx].clone(), self.values[layer_idx].clone())
    }
}
