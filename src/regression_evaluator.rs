use crate::coordinate::Coordinate;
use crate::isotonic_regression::{Centroid, Direction, IsotonicRegression};
use crate::point::{interpolate_two_points, interpolate_x_from_y, Point};
use crate::weight::Weight;
use eytzinger::SliceExt;
use serde::Serialize;

/// An evaluator for isotonic regression queries. Constructed from an
/// `IsotonicRegression` by value, eytzingerizing the points for O(log n)
/// `interpolate` and `invert` queries.
#[derive(Debug, Clone, Serialize)]
pub struct RegressionEvaluator<T: Coordinate, W: Weight = f64> {
    direction: Direction,
    points: Vec<Point<T, W>>,
    centroid_point: Centroid<T>,
    intersect_origin: bool,
}

#[allow(dead_code)]
impl<T: Coordinate, W: Weight> RegressionEvaluator<T, W> {
    /// Create a new evaluator from an isotonic regression.
    ///
    /// The regression's sorted points are eytzingerized on construction for
    /// cache-friendly O(log n) queries.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression, RegressionEvaluator};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// let evaluator = RegressionEvaluator::new(regression);
    /// let y = evaluator.interpolate(1.5).unwrap();
    /// assert_eq!(y, 1.75);
    /// ```
    pub fn new(regression: IsotonicRegression<T, W>) -> Self {
        let mut points = regression.points;
        points.eytzingerize(&mut eytzinger::permutation::InplacePermutator);

        RegressionEvaluator {
            direction: regression.direction,
            points,
            centroid_point: regression.centroid_point,
            intersect_origin: regression.intersect_origin,
        }
    }

    /// Find the _y_ point at position `at_x` or None if the regression is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression, RegressionEvaluator};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// let evaluator = RegressionEvaluator::new(regression);
    /// let interpolated_y = evaluator.interpolate(1.5).unwrap();
    /// assert_eq!(interpolated_y, 1.75);
    /// ```
    #[must_use]
    pub fn interpolate(&self, at_x: T) -> Option<T> {
        if self.points.is_empty() {
            return None;
        }

        let interpolation = if self.points.len() == 1 {
            *self.points[0].y()
        } else {
            let (lte, gt) = self
                .points
                .eytzinger_interpolative_search_by(|p| p.x().partial_cmp(&at_x).unwrap());

            match (lte, gt) {
                // Found exact match or need to interpolate between two points
                (Some(lower), Some(upper)) => {
                    interpolate_two_points(&self.points[lower], &self.points[upper], at_x)
                }
                // Requested point meets or exceeds the upper bound
                (Some(upper), None) => {
                    // at_x is beyond the last point - interpolate with centroid
                    interpolate_two_points(
                        &self.get_centroid_point().unwrap(),
                        &self.points[upper],
                        at_x,
                    )
                }
                // Requested point is below the lower bound
                (None, Some(lower)) => {
                    // at_x is before the first point
                    if self.intersect_origin {
                        interpolate_two_points(
                            &Point::new_with_weight(T::zero(), T::zero(), W::unit()),
                            &self.points[lower],
                            at_x,
                        )
                    } else {
                        interpolate_two_points(
                            &self.points[lower],
                            &self.get_centroid_point().unwrap(),
                            at_x,
                        )
                    }
                }
                // Should never happen - only possible if the slice is empty, which we already checked
                (None, None) => {
                    debug_assert!(
                        false,
                        "Got None, None from eytzinger_interpolative_search_by on non-empty slice"
                    );
                    return None;
                }
            }
        };

        Some(interpolation)
    }

    /// Retrieve the points (in eytzinger order, not sorted by x).
    pub fn get_points(&self) -> &[Point<T, W>] {
        &self.points
    }

    /// Retrieve the mean point of the original point set.
    pub fn get_centroid_point(&self) -> Option<Point<T, W>> {
        if self.centroid_point.sum_weight == T::zero() {
            None
        } else {
            Some(Point::new_with_weight(
                self.centroid_point.sum_x / self.centroid_point.sum_weight,
                self.centroid_point.sum_y / self.centroid_point.sum_weight,
                W::unit(),
            ))
        }
    }

    /// Returns the number of points in the regression.
    pub fn len(&self) -> usize {
        self.centroid_point.sum_weight.to_float().round() as usize
    }

    /// Checks if the regression is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.centroid_point.sum_weight == T::zero()
    }

    /// Find the _x_ value that would produce the given `at_y` value, or None if the regression is empty.
    /// This is the inverse operation of `interpolate`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression, RegressionEvaluator};
    ///
    /// let points = vec![
    ///     Point::new(0.0_f64, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// let evaluator = RegressionEvaluator::new(regression);
    /// let inverted_x = evaluator.invert(1.75).unwrap();
    /// assert!((inverted_x - 1.5).abs() < 0.01);
    /// ```
    #[must_use]
    pub fn invert(&self, at_y: T) -> Option<T> {
        if self.points.is_empty() {
            return None;
        }

        if self.points.len() == 1 {
            return Some(*self.points[0].x());
        }

        // We need sorted points for binary search on y values
        let mut sorted_points = self.points.clone();
        sorted_points.sort_by(|a, b| a.x().partial_cmp(b.x()).unwrap());

        // Binary search to find the position where at_y would fit
        let pos = match self.direction {
            Direction::Ascending => {
                // For ascending, y values are non-decreasing
                sorted_points.binary_search_by(|p| p.y().partial_cmp(&at_y).unwrap())
            }
            Direction::Descending => {
                // For descending, y values are non-increasing, so reverse the comparison
                sorted_points.binary_search_by(|p| at_y.partial_cmp(p.y()).unwrap())
            }
        };

        match pos {
            Ok(exact_idx) => {
                // Found exact match - handle horizontal segments (pooled points)
                let y_value = sorted_points[exact_idx].y();

                let mut start_idx = exact_idx;
                while start_idx > 0 && sorted_points[start_idx - 1].y() == y_value {
                    start_idx -= 1;
                }

                let mut end_idx = exact_idx;
                while end_idx < sorted_points.len() - 1 && sorted_points[end_idx + 1].y() == y_value
                {
                    end_idx += 1;
                }

                if start_idx < end_idx {
                    let start_x = *sorted_points[start_idx].x();
                    let end_x = *sorted_points[end_idx].x();
                    Some((start_x + end_x) / T::from_float(2.0))
                } else {
                    Some(*sorted_points[exact_idx].x())
                }
            }
            Err(insert_idx) => {
                if insert_idx == 0 {
                    if self.intersect_origin {
                        let p1 = Point::new_with_weight(T::zero(), T::zero(), W::unit());
                        let p2 = &sorted_points[0];
                        Some(interpolate_x_from_y(&p1, p2, at_y))
                    } else {
                        let centroid = self.get_centroid_point()?;
                        let p2 = &sorted_points[0];
                        Some(interpolate_x_from_y(&centroid, p2, at_y))
                    }
                } else if insert_idx >= sorted_points.len() {
                    let p1 = &sorted_points[sorted_points.len() - 1];
                    let centroid = self.get_centroid_point()?;
                    Some(interpolate_x_from_y(p1, &centroid, at_y))
                } else {
                    let p1 = &sorted_points[insert_idx - 1];
                    let p2 = &sorted_points[insert_idx];
                    Some(interpolate_x_from_y(p1, p2, at_y))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::UnitWeight;

    #[test]
    fn test_interpolate_ascending() {
        let points = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.5),
            Point::new(3.0, 3.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let evaluator = RegressionEvaluator::new(regression);
        let y = evaluator.interpolate(1.5).unwrap();
        assert_eq!(y, 1.75);
    }

    #[test]
    fn test_empty() {
        let regression: IsotonicRegression<f64> = IsotonicRegression::new_ascending(&[]).unwrap();
        let evaluator = RegressionEvaluator::new(regression);
        assert!(evaluator.is_empty());
        assert!(evaluator.interpolate(1.0).is_none());
        assert!(evaluator.invert(1.0).is_none());
    }

    #[test]
    fn test_invert_ascending() {
        let points = vec![
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.5),
            Point::new(3.0, 3.0),
        ];
        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let evaluator = RegressionEvaluator::new(regression);

        let x_at_1 = evaluator.invert(1.0).unwrap();
        assert!((x_at_1 - 0.0).abs() < 0.01);

        let x_at_1_75 = evaluator.invert(1.75).unwrap();
        assert!((x_at_1_75 - 1.5).abs() < 0.01);

        // Round trip
        let test_x = 1.5;
        let y = evaluator.interpolate(test_x).unwrap();
        let inverted_x = evaluator.invert(y).unwrap();
        assert!((inverted_x - test_x).abs() < 0.01);
    }

    #[test]
    fn test_invert_descending() {
        let points = vec![
            Point::new(0.0, 3.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 2.5),
            Point::new(3.0, 1.0),
        ];
        let regression = IsotonicRegression::new_descending(&points).unwrap();
        let evaluator = RegressionEvaluator::new(regression);

        let x_at_3 = evaluator.invert(3.0).unwrap();
        assert!((x_at_3 - 0.0).abs() < 0.01);

        let x_at_1 = evaluator.invert(1.0).unwrap();
        assert!((x_at_1 - 3.0).abs() < 0.01);

        // Round trip
        let test_x = 1.5;
        let y = evaluator.interpolate(test_x).unwrap();
        let inverted_x = evaluator.invert(y).unwrap();
        assert!((inverted_x - test_x).abs() < 0.01);
    }

    #[test]
    fn test_invert_single_point() {
        let points = vec![Point::new(1.0, 2.0)];
        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let evaluator = RegressionEvaluator::new(regression);
        let x = evaluator.invert(2.0).unwrap();
        assert_eq!(x, 1.0);
    }

    #[test]
    fn test_unit_weight() {
        let points: Vec<Point<f64, UnitWeight>> = vec![
            Point::new_with_weight(0.0, 1.0, UnitWeight),
            Point::new_with_weight(1.0, 2.0, UnitWeight),
            Point::new_with_weight(2.0, 1.5, UnitWeight),
            Point::new_with_weight(3.0, 3.0, UnitWeight),
        ];

        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let evaluator = RegressionEvaluator::new(regression);

        let y = evaluator.interpolate(1.5).unwrap();
        assert_eq!(y, 1.75);

        let test_x = 1.5;
        let y = evaluator.interpolate(test_x).unwrap();
        let inverted_x = evaluator.invert(y).unwrap();
        assert!((inverted_x - test_x).abs() < 0.01);
    }
}
