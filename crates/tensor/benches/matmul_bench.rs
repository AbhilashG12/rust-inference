use criterion::{Criterion, black_box, criterion_group, criterion_main};
use tensor::tensor::Tensor;

fn bench_matmul(c: &mut Criterion) {
    // We will multiply two 512x512 matrices.
    // Naive MatMul does 512^3 = 134 MILLION math operations.
    let m = 512;
    let k = 512;
    let n = 512;

    // Create random data
    let data_a: Vec<f32> = (0..m * k).map(|_| rand::random::<f32>()).collect();
    let data_b: Vec<f32> = (0..k * n).map(|_| rand::random::<f32>()).collect();

    let a = Tensor::new(data_a, vec![m, k]).unwrap();
    let b = Tensor::new(data_b, vec![k, n]).unwrap();

    let mut group = c.benchmark_group("MatMul_512x512");

    group.bench_function("Optimized_SIMD_Parallel", |b_env| {
        b_env.iter(|| {
            // We use black_box to stop the compiler from optimizing away the call
            let result = black_box(&a).matmul(black_box(&b)).unwrap();
            black_box(result);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_matmul);
criterion_main!(benches);
