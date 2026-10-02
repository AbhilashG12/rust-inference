pub mod error;
pub mod shape;
pub mod tensor;
pub mod ops;
pub mod nn;


#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    fn assert_close(a:&[f32],b: &[f32]) {
        assert_eq!(a.len(), b.len());
        for (x,y) in a.iter().zip(b.iter()) {
            assert!((x-y).abs() < 1e-5, "Mismatch : {} != {}",x,y);
        }
    }

    #[test]
    fn test_tensor_creation_and_strides(){
        let t = Tensor::zeroes(vec![2,3,4]);
        assert_eq!(t.strides,vec![12,4,1]);
        assert!(t.is_contiguous());

    }

    #[test]
    fn test_zero_copy_transpose() {
        let data = vec![1., 2., 3., 4., 5., 6.];
        let t = Tensor::new(data, vec![2, 3]).unwrap();
        assert_eq!(t.strides, vec![3, 1]);

        let transposed = t.transpose_2d().unwrap();
        assert_eq!(transposed.shape.dims(), &[3, 2]);
        assert_eq!(transposed.strides, vec![1, 3]); 
        assert!(!transposed.is_contiguous());       
    }

    #[test]
    fn test_matmul() {
        let a = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7., 8., 9., 10., 11., 12.], vec![3, 2]).unwrap();

        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape.dims(), &[2, 2]);
        assert_close(&c.data, &[58., 64., 139., 154.]);
    }
}

mod phase2_tests {
    use super::*;
    use crate::tensor::Tensor;
    use crate::nn::Linear;

    fn assert_close(a:&[f32],b:&[f32]){
        for (x,y) in a.iter().zip(b.iter()){
            assert!((x-y).abs() < 1e-4, "Mismatch: {} != {}", x,y);
        }
    }

    #[test]
    fn test_tiny_neural_network() {
        // --- 1. Define the Weights ---
        // Layer 1: 3 in, 4 out
        let l1_weight = Tensor::new(vec![
            0.1,  0.2, -0.1,  0.3,
           -0.2,  0.1,  0.5, -0.1,
            0.3, -0.4,  0.2,  0.1
        ], vec![3, 4]).unwrap();
        let l1_bias = Tensor::new(vec![0.1, -0.1, 0.2, 0.0], vec![4]).unwrap();
        let layer1 = Linear::new(l1_weight, l1_bias);

        // Layer 2: 4 in, 2 out (classes)
        let l2_weight = Tensor::new(vec![
            0.5, -0.2,
           -0.3,  0.1,
            0.2,  0.4,
           -0.1,  0.3
        ], vec![4, 2]).unwrap();
        let l2_bias = Tensor::new(vec![0.05, -0.05], vec![2]).unwrap();
        let layer2 = Linear::new(l2_weight, l2_bias);

        // --- 2. Define the Input ---
        let input = Tensor::new(vec![1.0, 2.0, -1.0], vec![1, 3]).unwrap();

        // --- 3. Execute the Forward Pass ---
        // Step A: Linear 1
        let out1 = layer1.forward(&input).unwrap();
        
        // Step B: ReLU
        let out_relu = out1.relu();

        // Step C: Linear 2
        let out2 = layer2.forward(&out_relu).unwrap();

        // Step D: Softmax
        let probabilities = out2.softmax().unwrap();

        // --- 4. Golden Reference Verification ---
        // I calculated this reference output using PyTorch with the exact same weights.
        // Expected logits before softmax: [0.02, 0.38]
        // Expected softmax probs: [0.410959, 0.589040]
        assert_close(&probabilities.data, &[0.410959, 0.589040]);
    }
}   
