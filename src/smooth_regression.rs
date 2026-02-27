use crate::coordinate::Coordinate;
use crate::isotonic_regression::{isotonic, isotonic_presorted, Direction, IsotonicRegression};
use crate::point::Point;
use crate::weight::Weight;
use serde::Serialize;

/// A smoothed regression using box filter on an isotonic regression.
///
/// The smoothed function is piecewise quadratic and preserves monotonicity.
/// Evaluation and inversion are O(log n) operations.
#[derive(Debug, Clone, Serialize)]
pub struct SmoothRegression<T: Coordinate> {
    /// Segment boundaries, sorted
    boundaries: Vec<T>,
    /// Quadratic coefficients (a, b, c) for each segment where y = ax² + bx + c
    coeffs: Vec<(T, T, T)>,
    /// Precomputed y-values at boundaries for inversion
    boundary_ys: Vec<T>,
    /// Half-width of the smoothing window
    window_half_width: T,
    /// Minimum x value in the domain
    x_min: T,
    /// Maximum x value in the domain
    x_max: T,
}

impl<T: Coordinate> SmoothRegression<T> {
    /// Create a smoothed regression directly from raw points.
    ///
    /// This runs isotonic regression (PAV) and then smooths in one step,
    /// avoiding any intermediate allocations.
    ///
    /// # Arguments
    /// * `points` - The raw data points
    /// * `direction` - Ascending or descending regression
    /// * `window_half_width` - Half-width of the box filter window (w)
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, SmoothRegression};
    /// use pav_regression::isotonic_regression::Direction;
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let smooth = SmoothRegression::new(&points, Direction::Ascending, 0.2);
    /// ```
    pub fn new<W: Weight>(
        points: &[Point<T, W>],
        direction: Direction,
        window_half_width: T,
    ) -> SmoothRegression<T> {
        let (iso_points, _) = isotonic(points, direction);
        Self::build(iso_points, window_half_width)
    }

    /// Create a smoothed regression directly from pre-sorted points.
    ///
    /// The caller must ensure the points are sorted in non-decreasing order by x.
    ///
    /// # Arguments
    /// * `points` - The raw data points, sorted by x
    /// * `direction` - Ascending or descending regression
    /// * `window_half_width` - Half-width of the box filter window (w)
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, SmoothRegression};
    /// use pav_regression::isotonic_regression::Direction;
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let smooth = SmoothRegression::new_sorted(&points, Direction::Ascending, 0.2);
    /// ```
    pub fn new_sorted<W: Weight>(
        points: &[Point<T, W>],
        direction: Direction,
        window_half_width: T,
    ) -> SmoothRegression<T> {
        let (iso_points, _) = isotonic_presorted(points.to_vec(), direction);
        Self::build(iso_points, window_half_width)
    }

    /// Create a smoothed regression from an isotonic regression (by value).
    ///
    /// Takes ownership of the regression to avoid cloning — the sorted points
    /// are used directly with zero copies.
    ///
    /// # Arguments
    /// * `regression` - The isotonic regression to smooth
    /// * `window_half_width` - Half-width of the box filter window (w)
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression, SmoothRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// let smooth = SmoothRegression::from_regression(regression, 0.2);
    /// ```
    pub fn from_regression<W: Weight>(
        regression: IsotonicRegression<T, W>,
        window_half_width: T,
    ) -> SmoothRegression<T> {
        let points = regression.into_points();
        Self::build(points, window_half_width)
    }

    /// Core builder: takes sorted isotonic points (any weight type) and builds
    /// the piecewise quadratic smooth regression.
    fn build<W: Weight>(points: Vec<Point<T, W>>, window_half_width: T) -> SmoothRegression<T> {
        if points.is_empty() {
            return SmoothRegression {
                boundaries: Vec::new(),
                coeffs: Vec::new(),
                boundary_ys: Vec::new(),
                window_half_width,
                x_min: T::zero(),
                x_max: T::zero(),
            };
        }

        // Get domain bounds
        let x_min = *points.first().unwrap().x();
        let x_max = *points.last().unwrap().x();
        let w = window_half_width;
        let two_w = T::from_float(2.0) * w;

        // Step 1: Build cumulative integral function (O(n))
        let cumulative = CumulativeIntegral::new(&points);

        // Step 2: Generate sorted boundaries via 3-way merge (O(n) vs O(n log n) sort)
        // The three sequences (x-w, x, x+w) are each already sorted.
        let boundaries = {
            let seq_minus: Vec<T> = points.iter().map(|p| *p.x() - w).collect();
            let seq_zero: Vec<T> = points.iter().map(|p| *p.x()).collect();
            let seq_plus: Vec<T> = points.iter().map(|p| *p.x() + w).collect();
            let ab = sorted_merge(&seq_minus, &seq_zero);
            let mut merged = sorted_merge(&ab, &seq_plus);
            merged.dedup();
            merged.retain(|&x| x >= x_min && x <= x_max);
            merged
        };

        let num_segments = boundaries.len().saturating_sub(1);

        // Step 3: Precompute smoothed y at all boundaries via batch linear scan (O(n))
        // Since boundaries are sorted, (b-w) and (b+w) are also sorted sequences,
        // so we replace O(log n) binary searches with O(1) amortized linear scans.
        let boundary_ys = {
            let bw_minus: Vec<T> = boundaries.iter().map(|&b| b - w).collect();
            let bw_plus: Vec<T> = boundaries.iter().map(|&b| b + w).collect();
            let eval_minus = cumulative.batch_eval_sorted(&bw_minus);
            let eval_plus = cumulative.batch_eval_sorted(&bw_plus);
            eval_plus
                .iter()
                .zip(eval_minus.iter())
                .map(|(&ep, &em)| (ep - em) / two_w)
                .collect::<Vec<T>>()
        };

        // Step 4: Compute midpoints and their smoothed y-values (O(n))
        let midpoints: Vec<T> = (0..num_segments)
            .map(|i| (boundaries[i] + boundaries[i + 1]) / T::from_float(2.0))
            .collect();
        let midpoint_ys = {
            let mw_minus: Vec<T> = midpoints.iter().map(|&m| m - w).collect();
            let mw_plus: Vec<T> = midpoints.iter().map(|&m| m + w).collect();
            let eval_minus = cumulative.batch_eval_sorted(&mw_minus);
            let eval_plus = cumulative.batch_eval_sorted(&mw_plus);
            eval_plus
                .iter()
                .zip(eval_minus.iter())
                .map(|(&ep, &em)| (ep - em) / two_w)
                .collect::<Vec<T>>()
        };

        // Step 5: Fit quadratics using precomputed values (O(n))
        let coeffs: Vec<(T, T, T)> = (0..num_segments)
            .map(|i| {
                fit_quadratic(
                    boundaries[i],
                    boundary_ys[i],
                    midpoints[i],
                    midpoint_ys[i],
                    boundaries[i + 1],
                    boundary_ys[i + 1],
                )
            })
            .collect();

        SmoothRegression {
            boundaries,
            coeffs,
            boundary_ys,
            window_half_width,
            x_min,
            x_max,
        }
    }

    /// Evaluate the smoothed function at a given x value.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression, SmoothRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// let smooth = SmoothRegression::from_regression(regression, 0.2);
    /// let y = smooth.interpolate(1.5);
    /// ```
    pub fn interpolate(&self, x: T) -> Option<T> {
        if self.coeffs.is_empty() {
            return None;
        }

        // Clamp x to the valid domain
        let x_clamped = x.max(self.x_min).min(self.x_max);

        // Binary search to find segment
        // Find the last boundary <= x
        let i = match self
            .boundaries
            .binary_search_by(|&b| b.partial_cmp(&x_clamped).unwrap())
        {
            Ok(idx) => idx.min(self.coeffs.len() - 1),
            Err(idx) => {
                if idx == 0 {
                    0
                } else {
                    (idx - 1).min(self.coeffs.len() - 1)
                }
            }
        };

        let (a, b, c) = self.coeffs[i];
        Some(a * x_clamped * x_clamped + b * x_clamped + c)
    }

    /// Invert the smoothed function: given y, find x.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression, SmoothRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 0.0),
    ///     Point::new(1.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// let smooth = SmoothRegression::from_regression(regression, 0.2);
    /// let x = smooth.invert(1.5);
    /// ```
    pub fn invert(&self, y: T) -> Option<T> {
        if self.coeffs.is_empty() {
            return None;
        }

        // Determine if y values are ascending or descending
        let ascending = if self.boundary_ys.len() >= 2 {
            self.boundary_ys[self.boundary_ys.len() - 1] >= self.boundary_ys[0]
        } else {
            true
        };

        // Binary search on y-values to find segment
        // For descending y values, we need to reverse the comparison
        let i = if ascending {
            self.boundary_ys
                .binary_search_by(|&by| by.partial_cmp(&y).unwrap())
                .unwrap_or_else(|i| i.saturating_sub(1))
                .min(self.coeffs.len() - 1)
        } else {
            // For descending, reverse the comparison
            self.boundary_ys
                .binary_search_by(|&by| y.partial_cmp(&by).unwrap())
                .unwrap_or_else(|i| i.saturating_sub(1))
                .min(self.coeffs.len() - 1)
        };

        let (a, b, c) = self.coeffs[i];

        let epsilon = T::from_float(1e-12);
        if a.abs() < epsilon {
            // Linear segment: y = bx + c
            Some((y - c) / b)
        } else {
            // Quadratic: ax² + bx + (c - y) = 0
            let c_shifted = c - y;
            let discriminant = b * b - T::from_float(4.0) * a * c_shifted;

            if discriminant < T::zero() {
                // No real solution, this shouldn't happen for monotonic function
                // Return the closest boundary point
                return Some(T::zero());
            }

            let sqrt_disc = discriminant.sqrt();
            let two_a = T::from_float(2.0) * a;

            // Compute both roots
            let root1 = (-b + sqrt_disc) / two_a;
            let root2 = (-b - sqrt_disc) / two_a;

            // Pick the root that's in bounds for this segment
            // This works for both ascending and descending regressions
            if i + 1 < self.boundaries.len() {
                let x_lo = self.boundaries[i];
                let x_hi = self.boundaries[i + 1];

                if root1 >= x_lo && root1 <= x_hi {
                    Some(root1)
                } else if root2 >= x_lo && root2 <= x_hi {
                    Some(root2)
                } else {
                    // Neither root is in bounds, return the closer one to segment center
                    let center = (x_lo + x_hi) / T::from_float(2.0);
                    let dist1 = (root1 - center).abs();
                    let dist2 = (root2 - center).abs();
                    if dist1 < dist2 {
                        Some(root1)
                    } else {
                        Some(root2)
                    }
                }
            } else {
                // Last segment, use the valid domain bounds
                let x_lo = self.boundaries[i];
                let x_hi = self.x_max;

                if root1 >= x_lo && root1 <= x_hi {
                    Some(root1)
                } else if root2 >= x_lo && root2 <= x_hi {
                    Some(root2)
                } else {
                    // Return the closer one
                    let center = (x_lo + x_hi) / T::from_float(2.0);
                    let dist1 = (root1 - center).abs();
                    let dist2 = (root2 - center).abs();
                    if dist1 < dist2 {
                        Some(root1)
                    } else {
                        Some(root2)
                    }
                }
            }
        }
    }
}

/// Helper struct for computing cumulative integrals of a piecewise linear function
struct CumulativeIntegral<T: Coordinate> {
    xs: Vec<T>,
    ys: Vec<T>,
    cumulative_areas: Vec<T>,
}

impl<T: Coordinate> CumulativeIntegral<T> {
    fn new<W: Weight>(points: &[Point<T, W>]) -> Self {
        let n = points.len();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut cumulative_areas = Vec::with_capacity(n);

        cumulative_areas.push(T::zero());

        for (i, point) in points.iter().enumerate() {
            xs.push(*point.x());
            ys.push(*point.y());

            if i > 0 {
                let dx = xs[i] - xs[i - 1];
                let avg_y = (ys[i] + ys[i - 1]) / T::from_float(2.0);
                let area = dx * avg_y;
                cumulative_areas.push(cumulative_areas[i - 1] + area);
            }
        }

        CumulativeIntegral {
            xs,
            ys,
            cumulative_areas,
        }
    }

    /// Evaluate the cumulative integral F(x) at multiple sorted x-values in O(n) time.
    ///
    /// Uses a linear scan instead of binary search — since query_xs is sorted,
    /// we advance a cursor through self.xs, giving O(1) amortized cost per query
    /// instead of O(log n).
    ///
    /// The function is extended as constant beyond its domain:
    /// - For t < x_min: f(t) = y_min
    /// - For t > x_max: f(t) = y_max
    fn batch_eval_sorted(&self, query_xs: &[T]) -> Vec<T> {
        let mut results = Vec::with_capacity(query_xs.len());

        if self.xs.is_empty() {
            results.resize(query_xs.len(), T::zero());
            return results;
        }

        let n = self.xs.len();
        let x_min = self.xs[0];
        let x_max = *self.xs.last().unwrap();
        let y_min = self.ys[0];
        let y_max = *self.ys.last().unwrap();
        let f_at_max = *self.cumulative_areas.last().unwrap();

        // Cursor into self.xs: invariant is xs[seg] <= x for interior queries
        let mut seg = 0usize;

        for &x in query_xs {
            if x <= x_min {
                results.push(y_min * (x - x_min));
                continue;
            }
            if x >= x_max {
                results.push(f_at_max + y_max * (x - x_max));
                continue;
            }

            // Advance seg so that xs[seg] <= x and (seg+1 == n or xs[seg+1] > x)
            while seg + 1 < n && self.xs[seg + 1] <= x {
                seg += 1;
            }

            if self.xs[seg] == x {
                results.push(self.cumulative_areas[seg]);
            } else {
                // Interpolate within segment [seg, seg+1]
                let x0 = self.xs[seg];
                let x1 = self.xs[seg + 1];
                let y0 = self.ys[seg];
                let y1 = self.ys[seg + 1];

                let dx = x - x0;
                let slope = (y1 - y0) / (x1 - x0);
                let y_at_x = y0 + slope * dx;
                let avg_y = (y0 + y_at_x) / T::from_float(2.0);
                results.push(self.cumulative_areas[seg] + dx * avg_y);
            }
        }

        results
    }
}

/// Merge two sorted slices into a single sorted Vec.
fn sorted_merge<T: Coordinate>(a: &[T], b: &[T]) -> Vec<T> {
    let mut result = Vec::with_capacity(a.len() + b.len());
    let (mut ia, mut ib) = (0, 0);
    while ia < a.len() && ib < b.len() {
        if a[ia].partial_cmp(&b[ib]).unwrap() != std::cmp::Ordering::Greater {
            result.push(a[ia]);
            ia += 1;
        } else {
            result.push(b[ib]);
            ib += 1;
        }
    }
    result.extend_from_slice(&a[ia..]);
    result.extend_from_slice(&b[ib..]);
    result
}

/// Fit a quadratic through three points
fn fit_quadratic<T: Coordinate>(x0: T, y0: T, x1: T, y1: T, x2: T, y2: T) -> (T, T, T) {
    // Solve the system:
    // y0 = a*x0^2 + b*x0 + c
    // y1 = a*x1^2 + b*x1 + c
    // y2 = a*x2^2 + b*x2 + c

    let x0_sq = x0 * x0;
    let x1_sq = x1 * x1;
    let x2_sq = x2 * x2;

    // Using Lagrange interpolation formulas for quadratic
    let denom = (x0 - x1) * (x0 - x2) * (x1 - x2);

    if denom.abs() < T::from_float(1e-12) {
        // Points are colinear, return linear fit
        let b = (y2 - y0) / (x2 - x0);
        let c = y0 - b * x0;
        return (T::zero(), b, c);
    }

    let a = (y0 * (x1 - x2) + y1 * (x2 - x0) + y2 * (x0 - x1)) / denom;
    let b = (y0 * (x1_sq - x2_sq) + y1 * (x2_sq - x0_sq) + y2 * (x0_sq - x1_sq)) / (-denom);
    let c = (y0 * (x1_sq * x2 - x2_sq * x1)
        + y1 * (x2_sq * x0 - x0_sq * x2)
        + y2 * (x0_sq * x1 - x1_sq * x0))
        / denom;

    (a, b, c)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::point::Point;

    #[test]
    fn test_smooth_regression_basic() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let smooth = SmoothRegression::from_regression(regression, 0.1);

        // Should be close to linear in the middle
        let y = smooth.interpolate(1.0).unwrap();
        assert!((y - 1.0).abs() < 0.2, "y={} should be close to 1.0", y);
    }

    #[test]
    fn test_smooth_regression_monotonic() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 5.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let smooth = SmoothRegression::from_regression(regression, 0.1);

        // Check monotonicity in the interior (away from boundaries)
        let mut prev_y = 0.0f64;
        for i in 1..29 {
            // Skip near boundaries
            let x = i as f64 * 0.1;
            if let Some(y) = smooth.interpolate(x) {
                assert!(
                    y >= prev_y - 0.01,
                    "Not monotonic at x={:?}: y={}, prev_y={}",
                    x,
                    y,
                    prev_y
                );
                prev_y = y;
            }
        }
    }

    #[test]
    fn test_smooth_invert() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
            Point::new(3.0, 3.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let smooth = SmoothRegression::from_regression(regression, 0.2);

        // Test round trip
        let test_x = 1.5;
        let y = smooth.interpolate(test_x).unwrap();
        let recovered_x = smooth.invert(y).unwrap();
        assert!((recovered_x - test_x).abs() < 0.1);
    }

    #[test]
    fn test_smooth_invert_descending() {
        // Test with descending regression to ensure quadratic root selection works
        let points = vec![
            Point::new(0.0, 5.0),
            Point::new(1.0, 4.0),
            Point::new(2.0, 3.0),
            Point::new(3.0, 2.0),
            Point::new(4.0, 1.0),
        ];
        let regression = IsotonicRegression::new_descending(&points).unwrap();
        let smooth = SmoothRegression::from_regression(regression, 0.3);

        // Test round trip for descending
        let test_x = 2.5;
        let y = smooth.interpolate(test_x).unwrap();
        let recovered_x = smooth.invert(y).unwrap();
        assert!(
            (recovered_x - test_x).abs() < 0.2,
            "Round trip failed: x={} -> y={} -> x={}",
            test_x,
            y,
            recovered_x
        );

        // Test specific inversions
        let y_mid = 3.0;
        let x_mid = smooth.invert(y_mid).unwrap();
        let verify_y = smooth.interpolate(x_mid).unwrap();
        assert!(
            (verify_y - y_mid).abs() < 0.1,
            "Inversion verification failed: y={} -> x={} -> y={}",
            y_mid,
            x_mid,
            verify_y
        );
    }

    #[test]
    fn test_monotonicity_near_boundaries() {
        // Boss Claude's counterexample: f(x) = 10x - 5 on [0, 1]
        // This goes from -5 to +5, testing monotonicity with negative values
        let points = vec![
            Point::new(0.0, -5.0),
            Point::new(0.5, 0.0),
            Point::new(1.0, 5.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let smooth = SmoothRegression::from_regression(regression, 0.3);

        // Check monotonicity near left boundary
        let mut prev_y = smooth.interpolate(0.0).unwrap();
        for i in 1..=10 {
            let x = i as f64 * 0.1;
            let y = smooth.interpolate(x).unwrap();
            assert!(
                y >= prev_y - 1e-10,
                "Not monotonic at x={}: y={} < prev_y={}",
                x,
                y,
                prev_y
            );
            prev_y = y;
        }

        // Check monotonicity across entire domain
        prev_y = smooth.interpolate(0.0).unwrap();
        for i in 1..=100 {
            let x = i as f64 * 0.01;
            if x <= 1.0 {
                let y = smooth.interpolate(x).unwrap();
                assert!(
                    y >= prev_y - 1e-10,
                    "Not monotonic at x={}: y={} < prev_y={}",
                    x,
                    y,
                    prev_y
                );
                prev_y = y;
            }
        }
    }

    #[test]
    fn test_smooth_new_direct() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        let smooth = SmoothRegression::new(&points, Direction::Ascending, 0.2);
        let y = smooth.interpolate(1.0).unwrap();
        assert!((y - 1.0).abs() < 0.2, "y={} should be close to 1.0", y);
    }

    #[test]
    fn test_smooth_new_sorted_direct() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(2.0, 2.0),
        ];
        let smooth = SmoothRegression::new_sorted(&points, Direction::Ascending, 0.2);
        let y = smooth.interpolate(1.0).unwrap();
        assert!((y - 1.0).abs() < 0.2, "y={} should be close to 1.0", y);
    }
}
