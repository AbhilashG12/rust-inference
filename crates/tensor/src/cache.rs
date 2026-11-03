use crate::tensor::Tensor;

pub struct KVCache {
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub current_seq_len: usize,
    pub initialized: bool,
}

impl KVCache {
    pub fn new(num_layers: usize, max_seq_len: usize) -> Self {
        Self {
            k_cache: Vec::new(),
            v_cache: Vec::new(),
            max_seq_len,
            num_layers,
            num_kv_heads: 0,
            head_dim: 0,
            current_seq_len: 0,
            initialized: false,
        }
    }

    pub fn append(&mut self, layer_idx: usize, k: &Tensor, v: &Tensor) {
        // Lazy-allocation: We figure out head dimensions dynamically on token 0
        if !self.initialized {
            self.head_dim = 64; // SmolLM2 head dimension
            self.num_kv_heads = k.shape.dims()[1] / self.head_dim;
            
            let cache_size = self.num_layers * self.num_kv_heads * self.head_dim * self.max_seq_len;
            
            // Pre-allocate the ENTIRE context window memory pool immediately
            self.k_cache = vec![0.0; cache_size];
            self.v_cache = vec![0.0; cache_size];
            self.initialized = true;
        }

        let seq_pos = self.current_seq_len;
        if seq_pos >= self.max_seq_len {
            panic!("Context Window Limit Reached! Max is {}", self.max_seq_len);
        }

        // Write directly into the flat buffer layout
        for kv_h in 0..self.num_kv_heads {
            let head_offset = kv_h * self.head_dim;

            // KEY: [layer, kv_head, head_dim, max_seq_len] -> Contiguous across TIME
            let k_layer_offset = layer_idx * (self.num_kv_heads * self.head_dim * self.max_seq_len)
                               + kv_h * (self.head_dim * self.max_seq_len);
            
            for d in 0..self.head_dim {
                self.k_cache[k_layer_offset + d * self.max_seq_len + seq_pos] = k.data[head_offset + d];
            }

            // VALUE: [layer, kv_head, max_seq_len, head_dim] -> Contiguous across DIMENSION
            let v_layer_offset = layer_idx * (self.num_kv_heads * self.max_seq_len * self.head_dim)
                               + kv_h * (self.max_seq_len * self.head_dim);
            let v_t_offset = v_layer_offset + seq_pos * self.head_dim;
            
            for d in 0..self.head_dim {
                self.v_cache[v_t_offset + d] = v.data[head_offset + d];
            }
        }
    }
}
