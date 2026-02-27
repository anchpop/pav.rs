use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pav_regression::{IsotonicRegression, Point};
use rand::Rng;

fn generate_noisy_ascending_points(n: usize) -> Vec<Point<f64>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|i| {
            let x = i as f64;
            let y = x + rng.gen_range(-5.0..5.0);
            Point::new(x, y)
        })
        .collect()
}

fn bench_new_ascending(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_ascending");

    for size in [100, 1_000, 10_000, 100_000] {
        let points = generate_noisy_ascending_points(size);
        group.bench_function(format!("n={size}"), |b| {
            b.iter(|| IsotonicRegression::new_ascending(black_box(&points)))
        });
    }

    group.finish();
}

criterion_group!(benches, bench_new_ascending);
criterion_main!(benches);
