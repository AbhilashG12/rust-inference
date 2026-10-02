use model::loader::ModelLoader;
use tensor::tensor::Tensor;
use tensor::nn::{Linear, TransformerBlock};
use tensor::quant::QuantizedTensor;

fn main() {
    println!(" Loading real Hugging Face LLaMA weights...");

    let weights = ModelLoader::load_safetensors("tiny_llama.safetensors")
        .expect("Failed to load tiny_llama.safetensors");

    let norm_weight = weights.get("model.layers.0.input_layernorm.weight").unwrap();
    let hidden_dim = norm_weight.shape.dims()[0];
    println!(" Detected Model Hidden Dimension: {}", hidden_dim);
    
    let get_linear = |name: &str| -> Linear {
        let raw = weights.get(name).expect(&format!("Missing {}", name));
        let q_tensor = QuantizedTensor::quantize(raw);
        let weight = q_tensor.dequantize();
        
        let out_dim = weight.shape.dims()[1];
        let bias = Tensor::zeroes(vec![out_dim]);
        
        Linear::new(weight, bias)
    };

    println!(" Building Transformer Block 0...");
    let block = TransformerBlock {
        norm1: norm_weight.clone(),
        norm2: weights.get("model.layers.0.post_attention_layernorm.weight").unwrap().clone(),
        
        w_q: get_linear("model.layers.0.self_attn.q_proj.weight"),
        w_k: get_linear("model.layers.0.self_attn.k_proj.weight"),
        w_v: get_linear("model.layers.0.self_attn.v_proj.weight"),
        w_o: get_linear("model.layers.0.self_attn.o_proj.weight"),
        
        ffn_up: get_linear("model.layers.0.mlp.up_proj.weight"),
        ffn_down: get_linear("model.layers.0.mlp.down_proj.weight"),
    };

    let dummy_input = Tensor::zeroes(vec![1, hidden_dim]);

    println!("Running Forward Pass...");
    let output = block.forward(&dummy_input, 0).expect("Forward pass crashed!");
    
    println!(" Forward Pass Complete!");
    println!(" Output Shape: {:?}", output.shape.dims());
}
