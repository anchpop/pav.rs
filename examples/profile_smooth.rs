use pav_regression::isotonic_regression::Direction;
use pav_regression::{IsotonicRegression, Point, SmoothRegression, UnitWeight};
use std::time::Instant;

fn generate_points(n: usize) -> Vec<Point<f64, UnitWeight>> {
    let mut x = 0.0_f64;
    let mut seed = 12345u64;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / u32::MAX as f64) * 10.0 - 5.0;
            let y = x + noise;
            let p = Point::new_with_weight(x, y, UnitWeight);
            x += 1.0;
            p
        })
        .collect()
}

fn main() {
    for &n in &[1_000, 10_000, 100_000] {
        println!("\n=== n = {} ===", n);
        let points = generate_points(n);

        // Path 1: from_regression (by value)
        let t0 = Instant::now();
        for _ in 0..10 {
            let reg = IsotonicRegression::new_ascending_sorted(&points).unwrap();
            let _ = SmoothRegression::from_regression(reg, 5.0);
        }
        let dt1 = t0.elapsed() / 10;
        println!("from_regression path: {:?}", dt1);

        // Path 2: new_sorted (direct)
        let t0 = Instant::now();
        for _ in 0..10 {
            let _ = SmoothRegression::new_sorted(&points, Direction::Ascending, 5.0);
        }
        let dt2 = t0.elapsed() / 10;
        println!("new_sorted path:      {:?}", dt2);

        // Breakdown: isotonic only
        let t0 = Instant::now();
        for _ in 0..10 {
            let _ = IsotonicRegression::new_ascending_sorted(&points).unwrap();
        }
        let dt_iso = t0.elapsed() / 10;
        println!("isotonic only:        {:?}", dt_iso);

        // Breakdown: get the isotonic points, then measure sub-steps of smooth build
        let reg = IsotonicRegression::new_ascending_sorted(&points).unwrap();
        let iso_points = reg.into_points();
        let n_pts = iso_points.len();

        // Step A: CumulativeIntegral::new (build xs, ys, cumulative_areas)
        let t0 = Instant::now();
        for _ in 0..10 {
            // Simulate building cumulative integral
            let mut xs = Vec::with_capacity(n_pts);
            let mut ys = Vec::with_capacity(n_pts);
            let mut cumulative_areas = Vec::with_capacity(n_pts);
            cumulative_areas.push(0.0_f64);
            for (i, p) in iso_points.iter().enumerate() {
                xs.push(*p.x());
                ys.push(*p.y());
                if i > 0 {
                    let dx = xs[i] - xs[i - 1];
                    let avg_y = (ys[i] + ys[i - 1]) / 2.0;
                    cumulative_areas.push(cumulative_areas[i - 1] + dx * avg_y);
                }
            }
            std::hint::black_box((&xs, &ys, &cumulative_areas));
        }
        let dt_cumul = t0.elapsed() / 10;
        println!("  cumulative build:   {:?}", dt_cumul);

        // Step B: Generate boundaries
        let t0 = Instant::now();
        let w = 5.0_f64;
        for _ in 0..10 {
            let mut boundaries = Vec::with_capacity(n_pts * 3);
            for p in &iso_points {
                let x = *p.x();
                boundaries.push(x - w);
                boundaries.push(x);
                boundaries.push(x + w);
            }
            boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap());
            boundaries.dedup();
            boundaries.retain(|&x| x >= *iso_points.first().unwrap().x() && x <= *iso_points.last().unwrap().x());
            std::hint::black_box(&boundaries);
        }
        let dt_boundaries = t0.elapsed() / 10;
        println!("  boundary gen+sort:  {:?}", dt_boundaries);

        // Step C: Compute coefficients (the integrate calls)
        // Rebuild cumulative
        let mut xs = Vec::with_capacity(n_pts);
        let mut ys = Vec::with_capacity(n_pts);
        let mut cumulative_areas = Vec::with_capacity(n_pts);
        cumulative_areas.push(0.0_f64);
        for (i, p) in iso_points.iter().enumerate() {
            xs.push(*p.x());
            ys.push(*p.y());
            if i > 0 {
                let dx = xs[i] - xs[i - 1];
                let avg_y = (ys[i] + ys[i - 1]) / 2.0;
                cumulative_areas.push(cumulative_areas[i - 1] + dx * avg_y);
            }
        }

        let mut boundaries = Vec::with_capacity(n_pts * 3);
        for p in &iso_points {
            let x = *p.x();
            boundaries.push(x - w);
            boundaries.push(x);
            boundaries.push(x + w);
        }
        boundaries.sort_by(|a, b| a.partial_cmp(b).unwrap());
        boundaries.dedup();
        let x_min = *iso_points.first().unwrap().x();
        let x_max = *iso_points.last().unwrap().x();
        boundaries.retain(|&x| x >= x_min && x <= x_max);

        let num_segments = boundaries.len().saturating_sub(1);
        println!("  num boundaries:     {}", boundaries.len());
        println!("  num segments:       {}", num_segments);

        // Measure just the coefficient computation loop
        let t0 = Instant::now();
        for _ in 0..10 {
            let mut coeffs = Vec::with_capacity(num_segments);
            let two_w = 2.0 * w;

            // Inline eval_cumulative for profiling
            let eval = |x: f64| -> f64 {
                if xs.is_empty() { return 0.0; }
                let y_min = ys[0];
                let y_max = *ys.last().unwrap();
                if x <= x_min { return y_min * (x - x_min); }
                if x >= x_max {
                    return *cumulative_areas.last().unwrap() + y_max * (x - x_max);
                }
                let i = match xs.binary_search_by(|&xi| xi.partial_cmp(&x).unwrap()) {
                    Ok(idx) => return cumulative_areas[idx],
                    Err(idx) => idx - 1,
                };
                let x0 = xs[i];
                let y0 = ys[i];
                let y1 = ys[i + 1];
                let x1 = xs[i + 1];
                let dx = x - x0;
                let slope = (y1 - y0) / (x1 - x0);
                let y_at_x = y0 + slope * dx;
                cumulative_areas[i] + dx * (y0 + y_at_x) / 2.0
            };

            let integrate = |a: f64, b: f64| -> f64 { eval(b) - eval(a) };

            let mut prev_y = integrate(boundaries[0] - w, boundaries[0] + w) / two_w;

            for i in 0..num_segments {
                let x0 = boundaries[i];
                let x1 = boundaries[i + 1];
                let xm = (x0 + x1) / 2.0;

                let y0 = prev_y;
                let y1 = integrate(x1 - w, x1 + w) / two_w;
                let ym = integrate(xm - w, xm + w) / two_w;

                prev_y = y1;

                // fit_quadratic inline
                let denom = (x0 - xm) * (x0 - x1) * (xm - x1);
                if denom.abs() < 1e-12 {
                    let b = (y1 - y0) / (x1 - x0);
                    let c = y0 - b * x0;
                    coeffs.push((0.0, b, c));
                } else {
                    let a = (y0 * (xm - x1) + ym * (x1 - x0) + y1 * (x0 - xm)) / denom;
                    let b = (y0 * (xm * xm - x1 * x1) + ym * (x1 * x1 - x0 * x0) + y1 * (x0 * x0 - xm * xm)) / (-denom);
                    let c = (y0 * (xm * xm * x1 - x1 * x1 * xm) + ym * (x1 * x1 * x0 - x0 * x0 * x1) + y1 * (x0 * x0 * xm - xm * xm * x0)) / denom;
                    coeffs.push((a, b, c));
                }
            }
            std::hint::black_box(&coeffs);
        }
        let dt_coeffs = t0.elapsed() / 10;
        println!("  coeff computation:  {:?}", dt_coeffs);

        // Count how many binary searches happen: 3 per segment (y0, ym, y1) minus caching
        // Each integrate call does 2 eval_cumulative calls, each with a binary search
        // So per segment: ~4 binary searches (y1 and ym each need 2, y0 cached)
        println!("  binary searches:    ~{} (4 per segment)", num_segments * 4);
        println!("  bs per point:       ~{:.1}", (num_segments * 4) as f64 / n_pts as f64);
    }
}
