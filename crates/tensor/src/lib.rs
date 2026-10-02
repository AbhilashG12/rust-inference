pub mod error;
pub mod shape;
pub mod tensor;
pub mod ops;

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
