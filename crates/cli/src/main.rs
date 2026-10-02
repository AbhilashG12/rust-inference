use graph::builder::GraphBuilder;
use graph::executor::Executor;
use model::loader::ModelLoader;
use tensor::tensor::Tensor;


fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Starting Rust Inference Engine...");

    println!("Loading mnist.safetensors...");
    let weights = ModelLoader::load_safetensors("mnist.safetensors")
        .expect("Failed to load model weights. Ensure mnist.safetensors exists.");

    let mut builder = GraphBuilder::new();

    let w1_id = builder.add_tensor(weights.get("layer1.weight").unwrap().clone());
    let b1_id = builder.add_tensor(weights.get("layer1.bias").unwrap().clone());
    let w2_id = builder.add_tensor(weights.get("layer2.weight").unwrap().clone());
    let b2_id = builder.add_tensor(weights.get("layer2.bias").unwrap().clone());

    let dummy_image = Tensor::zeros(vec![1, 784]);
    let input_id = builder.add_tensor(dummy_image);

    println!("Compiling Graph...");
    
    let x = builder.matmul(input_id, w1_id)?;
    let x = builder.add_broadcast(x, b1_id)?;
    let x = builder.relu(x)?;

    let x = builder.matmul(x, w2_id)?;
    let logits = builder.add_broadcast(x, b2_id)?;

    let mut graph = builder.build();
    println!("Executing Graph...");
    Executor::run(&mut graph)?;
    let output_tensor = &graph.tensors[logits.0];
    let probabilities = output_tensor.softmax()?;
    let mut best_digit = 0;
    let mut highest_prob = f32::NEG_INFINITY;
    
    for (digit, &prob) in probabilities.data.iter().enumerate() {
        if prob > highest_prob {
            highest_prob = prob;
            best_digit = digit;
        }
    }

    println!("🎯 Prediction: Digit {}, Confidence: {:.2}%", best_digit, highest_prob * 100.0);
    Ok(())
}
