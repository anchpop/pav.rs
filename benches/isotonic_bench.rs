use criterion::{black_box, criterion_group, criterion_main, Criterion};
use pav_regression::{IsotonicRegression, Point, UnitWeight};
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

fn generate_noisy_ascending_points_unit(n: usize) -> Vec<Point<f64, UnitWeight>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|i| {
            let x = i as f64;
            let y = x + rng.gen_range(-5.0..5.0);
            Point::new_with_weight(x, y, UnitWeight)
        })
        .collect()
}

fn generate_noisy_ascending_points_f32_unit(n: usize) -> Vec<Point<f32, UnitWeight>> {
    let mut rng = rand::thread_rng();
    (0..n)
        .map(|i| {
            let x = i as f32;
            let y = x + rng.gen_range(-5.0_f32..5.0);
            Point::new_with_weight(x, y, UnitWeight)
        })
        .collect()
}

fn bench_new_ascending(c: &mut Criterion) {
    let mut group = c.benchmark_group("new_ascending");

    for size in [100, 1_000, 10_000, 100_000] {
        let points = generate_noisy_ascending_points(size);
        group.bench_function(format!("f64_weight/n={size}"), |b| {
            b.iter(|| IsotonicRegression::new_ascending(black_box(&points)))
        });

        let points_unit = generate_noisy_ascending_points_unit(size);
        group.bench_function(format!("f64_unit/n={size}"), |b| {
            b.iter(|| IsotonicRegression::new_ascending(black_box(&points_unit)))
        });

        let points_f32 = generate_noisy_ascending_points_f32_unit(size);
        group.bench_function(format!("f32_unit/n={size}"), |b| {
            b.iter(|| IsotonicRegression::new_ascending(black_box(&points_f32)))
        });
    }

    group.finish();
}

fn bench_sort_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("sort_only");

    for size in [100, 1_000, 10_000, 100_000] {
        let points = generate_noisy_ascending_points(size);
        group.bench_function(format!("f64_weight/n={size}"), |b| {
            b.iter(|| {
                let mut pts = black_box(&points).to_vec();
                pts.sort_unstable_by(|a, b| a.x().partial_cmp(b.x()).unwrap());
                pts
            })
        });

        let points_unit = generate_noisy_ascending_points_unit(size);
        group.bench_function(format!("f64_unit/n={size}"), |b| {
            b.iter(|| {
                let mut pts = black_box(&points_unit).to_vec();
                pts.sort_unstable_by(|a, b| a.x().partial_cmp(b.x()).unwrap());
                pts
            })
        });

        let points_f32 = generate_noisy_ascending_points_f32_unit(size);
        group.bench_function(format!("f32_unit/n={size}"), |b| {
            b.iter(|| {
                let mut pts = black_box(&points_f32).to_vec();
                pts.sort_unstable_by(|a, b| a.x().partial_cmp(b.x()).unwrap());
                pts
            })
        });
    }

    group.finish();
}

criterion_group!(benches, bench_new_ascending, bench_sort_only);
criterion_main!(benches);
