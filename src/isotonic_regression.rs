use crate::coordinate::Coordinate;
use crate::point::{interpolate_two_points, interpolate_x_from_y, Point};
use eytzinger::SliceExt;
use serde::Serialize;
use std::fmt::{Display, Formatter};
use thiserror::Error;

/// Errors that can occur during isotonic regression
#[derive(Error, Debug)]
#[allow(dead_code)]
pub enum IsotonicRegressionError {
    /// Error when a negative point is encountered with intersect_origin set to true
    #[error("With intersect_origin = true, all points must be >= 0 on both x and y axes")]
    NegativePointWithIntersectOrigin,
}

/// A vector of points forming an isotonic regression, along with the
/// centroid point of the original set.
#[derive(Debug, Clone, Serialize)]
pub struct IsotonicRegression<T: Coordinate> {
    direction: Direction,
    points: Vec<Point<T>>,
    centroid_point: Centroid<T>,
    intersect_origin: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
struct Centroid<T: Coordinate> {
    sum_x: T,
    sum_y: T,
    sum_weight: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[allow(dead_code)]
/// Specifies the direction of the isotonic regression.
pub enum Direction {
    /// Indicates an ascending (non-decreasing) regression.
    Ascending,
    /// Indicates a descending (non-increasing) regression.
    Descending,
}

impl<T: Coordinate + Display> Display for IsotonicRegression<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "IsotonicRegression {{")?;
        writeln!(f, "\tdirection: {:?},", self.direction)?;
        writeln!(f, "\tpoints:")?;
        for point in &self.get_points_sorted() {
            writeln!(
                f,
                "\t\t{}\t{:.2}\t{:.2}",
                point.x(),
                point.y(),
                point.weight()
            )?;
        }
        writeln!(f, "\tcentroid_point:")?;
        writeln!(
            f,
            "\t\t{}\t{:.2}\t{:.2}",
            self.centroid_point.sum_x, self.centroid_point.sum_y, self.centroid_point.sum_weight
        )?;
        write!(f, "}}")
    }
}

#[allow(dead_code)]
impl<T: Coordinate> IsotonicRegression<T> {
    /// Find an ascending isotonic regression from a set of points.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// assert_eq!(regression.get_points().len(), 4); // All points preserved
    /// ```
    pub fn new_ascending(
        points: &[Point<T>],
    ) -> Result<IsotonicRegression<T>, IsotonicRegressionError> {
        IsotonicRegression::new(points, Direction::Ascending, false)
    }

    /// Find a descending isotonic regression from a set of points.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 3.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 2.5),
    ///     Point::new(3.0, 1.0),
    /// ];
    /// let regression = IsotonicRegression::new_descending(&points).unwrap();
    /// assert_eq!(regression.get_points().len(), 4); // All points preserved
    /// ```
    pub fn new_descending(
        points: &[Point<T>],
    ) -> Result<IsotonicRegression<T>, IsotonicRegressionError> {
        IsotonicRegression::new(points, Direction::Descending, false)
    }

    /// Find an isotonic regression in the specified direction.
    ///
    /// If `intersect_origin` is true, the regression will intersect the origin (0,0) and all points must be >= 0 on both axes.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    /// use pav_regression::isotonic_regression::Direction;
    ///
    /// let points = vec![
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new(&points, Direction::Ascending, false).unwrap();
    /// assert_eq!(regression.get_points().len(), 4); // All points preserved
    /// ```
    pub fn new(
        points: &[Point<T>],
        direction: Direction,
        intersect_origin: bool,
    ) -> Result<IsotonicRegression<T>, IsotonicRegressionError> {
        let (sum_x, sum_y, sum_weight) =
            points
                .iter()
                .try_fold((T::zero(), T::zero(), 0.0), |(sx, sy, sw), point| {
                    if intersect_origin
                        && (point.x().is_sign_negative() || point.y().is_sign_negative())
                    {
                        Err(IsotonicRegressionError::NegativePointWithIntersectOrigin)
                    } else {
                        Ok((
                            sx + *point.x() * T::from_float(point.weight()),
                            sy + *point.y() * T::from_float(point.weight()),
                            sw + point.weight(),
                        ))
                    }
                })?;

        let mut isotonic_points = isotonic(points, direction.clone());
        isotonic_points.eytzingerize(&mut eytzinger::permutation::InplacePermutator);

        Ok(IsotonicRegression {
            direction,
            points: isotonic_points,
            centroid_point: Centroid {
                sum_x,
                sum_y,
                sum_weight,
            },
            intersect_origin,
        })
    }

    /// Find the _y_ point at position `at_x` or None if the regression is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// let interpolated_y = regression.interpolate(1.5).unwrap();
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
                            &Point::new(T::zero(), T::zero()),
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

    /// Retrieve the points that make up the isotonic regression. The points are NOT sorted by x value - they are in eytzinger order.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// assert_eq!(regression.get_points().len(), 4); // All points preserved
    /// ```
    pub fn get_points(&self) -> &[Point<T>] {
        &self.points
    }

    /// Retrieve the points that make up the isotonic regression, sorted by x value.
    pub fn get_points_sorted(&self) -> Vec<Point<T>> {
        let mut points = self.points.clone();
        points.sort_by(|a, b| a.x().partial_cmp(b.x()).unwrap());
        points
    }

    /// Retrieve the mean point of the original point set.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// let centroid = regression.get_centroid_point().unwrap();
    /// assert_eq!(*centroid.x(), 1.5);
    /// assert_eq!(*centroid.y(), 1.875);
    /// ```
    pub fn get_centroid_point(&self) -> Option<Point<T>> {
        if self.centroid_point.sum_weight == 0.0 {
            None
        } else {
            Some(Point::new_with_weight(
                self.centroid_point.sum_x / T::from_float(self.centroid_point.sum_weight),
                self.centroid_point.sum_y / T::from_float(self.centroid_point.sum_weight),
                1.0,
            ))
        }
    }

    /// Add new points to the regression.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    ///
    /// let mut regression = IsotonicRegression::new_ascending(&[
    ///     Point::new(0.0, 1.0),
    ///     Point::new(2.0, 2.0),
    /// ]).unwrap();
    /// regression.add_points(&[Point::new(1.0, 1.5)]);
    /// assert_eq!(regression.get_points().len(), 3);
    /// ```
    pub fn add_points(&mut self, points: &[Point<T>]) {
        for point in points {
            assert!(
                !self.intersect_origin
                    || (!point.x().is_sign_negative() && !point.y().is_sign_negative()),
                "With intersect_origin = true, all points must be >= 0 on both x and y axes"
            );
            self.centroid_point.sum_x =
                self.centroid_point.sum_x + *point.x() * T::from_float(point.weight());
            self.centroid_point.sum_y =
                self.centroid_point.sum_y + *point.y() * T::from_float(point.weight());
            self.centroid_point.sum_weight = self.centroid_point.sum_weight + point.weight();
        }

        let mut new_points = self.points.clone();
        new_points.extend_from_slice(points);
        self.points = isotonic(&new_points, self.direction.clone());
        self.points
            .eytzingerize(&mut eytzinger::permutation::InplacePermutator);
    }

    /// Remove points from the regression.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    ///
    /// let mut regression = IsotonicRegression::new_ascending(&[
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 3.0),
    /// ]).unwrap();
    /// regression.remove_points(&[Point::new(1.0, 2.0)]);
    /// assert_eq!(regression.get_points().len(), 2);
    /// ```
    pub fn remove_points(&mut self, points: &[Point<T>]) {
        for point in points {
            assert!(
                !self.intersect_origin
                    || (!point.x().is_sign_negative() && !point.y().is_sign_negative()),
                "With intersect_origin = true, all points must be >= 0 on both x and y axes"
            );
            self.centroid_point.sum_x =
                self.centroid_point.sum_x - *point.x() * T::from_float(point.weight());
            self.centroid_point.sum_y =
                self.centroid_point.sum_y - *point.y() * T::from_float(point.weight());
            self.centroid_point.sum_weight = self.centroid_point.sum_weight - point.weight();
        }

        let mut new_points = self.points.clone();
        for point in points {
            if let Some(pos) = new_points.iter().position(|p| {
                p.x() == point.x() && p.y() == point.y() && p.weight() == point.weight()
            }) {
                new_points.remove(pos);
            }
        }
        self.points = isotonic(&new_points, self.direction.clone());
        self.points
            .eytzingerize(&mut eytzinger::permutation::InplacePermutator);
    }

    /// Returns the number of points in the regression.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// assert_eq!(regression.len(), 4);
    /// ```
    pub fn len(&self) -> usize {
        self.centroid_point.sum_weight.round() as usize
    }

    /// Checks if the regression is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::IsotonicRegression;
    ///
    /// let regression: IsotonicRegression<f64> = IsotonicRegression::new_ascending(&[]).unwrap();
    /// assert!(regression.is_empty());
    /// ```
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.centroid_point.sum_weight == 0.0
    }

    /// Find the _x_ value that would produce the given `at_y` value, or None if the regression is empty.
    /// This is the inverse operation of `interpolate`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::{Point, IsotonicRegression};
    ///
    /// let points = vec![
    ///     Point::new(0.0, 1.0),
    ///     Point::new(1.0, 2.0),
    ///     Point::new(2.0, 1.5),
    ///     Point::new(3.0, 3.0),
    /// ];
    /// let regression = IsotonicRegression::new_ascending(&points).unwrap();
    /// let inverted_x = regression.invert(1.75).unwrap();
    /// // 1.75 is the y-value at x=1.5, so we should get back approximately 1.5
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

        // We need to work with sorted points for binary search on y values
        let sorted_points = self.get_points_sorted();

        // Binary search to find the position where at_y would fit
        let pos = match self.direction {
            Direction::Ascending => {
                // For ascending, y values are non-decreasing
                sorted_points.binary_search_by(|p| {
                    p.y().partial_cmp(&at_y).unwrap()
                })
            }
            Direction::Descending => {
                // For descending, y values are non-increasing, so reverse the comparison
                sorted_points.binary_search_by(|p| {
                    at_y.partial_cmp(p.y()).unwrap()
                })
            }
        };

        match pos {
            Ok(exact_idx) => {
                // Found exact match - but we need to handle horizontal segments (pooled points)
                // Find the range of points with the same y value
                let y_value = sorted_points[exact_idx].y();

                // Find the first point with this y value
                let mut start_idx = exact_idx;
                while start_idx > 0 && sorted_points[start_idx - 1].y() == y_value {
                    start_idx -= 1;
                }

                // Find the last point with this y value
                let mut end_idx = exact_idx;
                while end_idx < sorted_points.len() - 1 && sorted_points[end_idx + 1].y() == y_value {
                    end_idx += 1;
                }

                // If there's a range of points with the same y, return the midpoint
                if start_idx < end_idx {
                    let start_x = *sorted_points[start_idx].x();
                    let end_x = *sorted_points[end_idx].x();
                    Some((start_x + end_x) / T::from_float(2.0))
                } else {
                    Some(*sorted_points[exact_idx].x())
                }
            }
            Err(insert_idx) => {
                // at_y falls between two points or outside the range
                if insert_idx == 0 {
                    // at_y is before the first point
                    if self.intersect_origin {
                        // Interpolate between origin and first point
                        let p1 = Point::new(T::zero(), T::zero());
                        let p2 = &sorted_points[0];
                        Some(interpolate_x_from_y(&p1, p2, at_y))
                    } else {
                        // Interpolate between centroid and first point
                        let centroid = self.get_centroid_point()?;
                        let p2 = &sorted_points[0];
                        Some(interpolate_x_from_y(&centroid, p2, at_y))
                    }
                } else if insert_idx >= sorted_points.len() {
                    // at_y is after the last point
                    let p1 = &sorted_points[sorted_points.len() - 1];
                    let centroid = self.get_centroid_point()?;
                    Some(interpolate_x_from_y(p1, &centroid, at_y))
                } else {
                    // at_y is between two points
                    let p1 = &sorted_points[insert_idx - 1];
                    let p2 = &sorted_points[insert_idx];
                    Some(interpolate_x_from_y(p1, p2, at_y))
                }
            }
        }
    }
}

#[allow(dead_code)]
fn isotonic<T: Coordinate>(points: &[Point<T>], direction: Direction) -> Vec<Point<T>> {
    if points.is_empty() {
        return Vec::new();
    }

    let mut sorted_points: Vec<Point<T>> = points.to_vec();
    
    // Sort the points by x
    sorted_points.sort_by(|a, b| {
        a.x()
            .partial_cmp(b.x())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Apply PAV algorithm - preserve all points, only adjust y-values
    let mut result = sorted_points.clone();
    let n = result.len();
    
    // Keep track of pools of points that should have the same y-value
    let mut pools: Vec<(usize, usize)> = (0..n).map(|i| (i, i + 1)).collect();
    
    loop {
        let mut merged = false;
        
        // Check each adjacent pair of pools
        for i in 0..pools.len() - 1 {
            let (start1, end1) = pools[i];
            let (start2, end2) = pools[i + 1];
            
            // Calculate weighted average y for each pool
            let (sum_y1, sum_weight1) = (start1..end1).fold((T::zero(), 0.0), |(sy, sw), j| {
                (sy + result[j].y * T::from_float(result[j].weight), sw + result[j].weight)
            });
            let avg_y1 = sum_y1 / T::from_float(sum_weight1);
            
            let (sum_y2, sum_weight2) = (start2..end2).fold((T::zero(), 0.0), |(sy, sw), j| {
                (sy + result[j].y * T::from_float(result[j].weight), sw + result[j].weight)
            });
            let avg_y2 = sum_y2 / T::from_float(sum_weight2);
            
            // Check if pools violate monotonicity
            let should_merge = match direction {
                Direction::Ascending => avg_y1 > avg_y2,
                Direction::Descending => avg_y1 < avg_y2,
            };
            
            if should_merge {
                // Merge the two pools
                pools[i] = (start1, end2);
                pools.remove(i + 1);
                
                // Calculate new weighted average for merged pool
                let total_weight = sum_weight1 + sum_weight2;
                let new_y = (sum_y1 + sum_y2) / T::from_float(total_weight);
                
                // Update all points in the merged pool with the new y-value
                for j in start1..end2 {
                    result[j].y = new_y;
                }
                
                merged = true;
                break;
            }
        }
        
        if !merged {
            break;
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ascending_regression() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.5),
            Point::new(3.0, 3.0),
        ];

        let regression = IsotonicRegression::new_ascending(points).unwrap();
        let sorted_points = regression.get_points_sorted();
        assert_eq!(sorted_points.len(), 4); // All points preserved
        assert_eq!(*sorted_points[0].x(), 0.0);
        assert_eq!(*sorted_points[0].y(), 1.0);
        assert_eq!(*sorted_points[1].x(), 1.0);
        assert_eq!(*sorted_points[1].y(), 1.75); // Pooled value
        assert_eq!(*sorted_points[2].x(), 2.0);
        assert_eq!(*sorted_points[2].y(), 1.75); // Pooled value
        assert_eq!(*sorted_points[3].x(), 3.0);
        assert_eq!(*sorted_points[3].y(), 3.0);
    }

    #[test]
    fn test_descending_regression() {
        let points = &[
            Point::new(0.0, 3.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 2.5),
            Point::new(3.0, 1.0),
        ];

        let regression = IsotonicRegression::new_descending(points).unwrap();
        let sorted_points = regression.get_points_sorted();
        assert_eq!(sorted_points.len(), 4); // All points preserved
        assert_eq!(*sorted_points[0].x(), 0.0);
        assert_eq!(*sorted_points[0].y(), 3.0);
        assert_eq!(*sorted_points[1].x(), 1.0);
        assert_eq!(*sorted_points[1].y(), 2.25); // Pooled value
        assert_eq!(*sorted_points[2].x(), 2.0);
        assert_eq!(*sorted_points[2].y(), 2.25); // Pooled value
        assert_eq!(*sorted_points[3].x(), 3.0);
        assert_eq!(*sorted_points[3].y(), 1.0);
    }

    #[test]
    fn test_add_points() {
        let mut regression =
            IsotonicRegression::new_ascending(&[Point::new(0.0, 1.0), Point::new(2.0, 2.0)])
                .unwrap();
        regression.add_points(&[Point::new(1.0, 1.5)]);
        assert_eq!(regression.get_points_sorted().len(), 3);
        assert_eq!(*regression.get_points_sorted()[1].x(), 1.0);
        assert_eq!(*regression.get_points_sorted()[1].y(), 1.5);
    }

    #[test]
    fn test_remove_points() {
        let mut regression = IsotonicRegression::new_ascending(&[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
        ])
        .unwrap();
        
        // Before removal, all 3 points should be present
        assert_eq!(regression.get_points_sorted().len(), 3);
        
        regression.remove_points(&[Point::new(1.0, 2.0)]);
        
        // After removal, 2 points should remain
        let sorted_points = regression.get_points_sorted();
        assert_eq!(sorted_points.len(), 2);
        assert_eq!(*sorted_points[0].x(), 0.0);
        assert_eq!(*sorted_points[1].x(), 2.0);
    }

    #[test]
    fn test_centroid_point() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
        ];
        let regression = IsotonicRegression::new_ascending(points).unwrap();
        let centroid = regression.get_centroid_point().unwrap();
        assert_eq!(*centroid.x(), 1.0);
        assert_eq!(*centroid.y(), 2.0);
    }

    #[test]
    fn test_empty_regression() {
        let regression: IsotonicRegression<f64> = IsotonicRegression::new_ascending(&[]).unwrap();
        assert!(regression.is_empty());
        assert_eq!(regression.len(), 0);
        assert!(regression.interpolate(1.0).is_none());
    }
    
    #[test]
    fn test_pav_preserves_all_points() {
        // Test case from the bug description
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 10.0),
            Point::new(2.0, 9.0),
            Point::new(3.0, 8.0),
            Point::new(4.0, 20.0),
        ];
        
        let regression = IsotonicRegression::new_ascending(points).unwrap();
        let sorted_points = regression.get_points_sorted();
        
        // All points must be preserved
        assert_eq!(sorted_points.len(), 5, "All 5 input points should be preserved");
        
        // X-coordinates must remain unchanged
        assert_eq!(*sorted_points[0].x(), 0.0);
        assert_eq!(*sorted_points[1].x(), 1.0);
        assert_eq!(*sorted_points[2].x(), 2.0);
        assert_eq!(*sorted_points[3].x(), 3.0);
        assert_eq!(*sorted_points[4].x(), 4.0);
        
        // Y-coordinates should be pooled for violating points
        assert_eq!(*sorted_points[0].y(), 1.0); // First point unchanged
        assert_eq!(*sorted_points[1].y(), 9.0); // Pooled average of 10, 9, 8
        assert_eq!(*sorted_points[2].y(), 9.0); // Same pooled value
        assert_eq!(*sorted_points[3].y(), 9.0); // Same pooled value
        assert_eq!(*sorted_points[4].y(), 20.0); // Last point unchanged
    }
    
    #[test]
    fn test_weighted_pav_preserves_all_points() {
        // Test with weighted points
        let points = &[
            Point::new_with_weight(0.0, 5.0, 1.0),
            Point::new_with_weight(1.0, 3.0, 2.0),
            Point::new_with_weight(2.0, 4.0, 1.0),
            Point::new_with_weight(3.0, 10.0, 1.0),
        ];

        let regression = IsotonicRegression::new_ascending(points).unwrap();
        let sorted_points = regression.get_points_sorted();

        // All points must be preserved
        assert_eq!(sorted_points.len(), 4);

        // X-coordinates unchanged
        assert_eq!(*sorted_points[0].x(), 0.0);
        assert_eq!(*sorted_points[1].x(), 1.0);
        assert_eq!(*sorted_points[2].x(), 2.0);
        assert_eq!(*sorted_points[3].x(), 3.0);

        // Y-values should be pooled where monotonicity is violated
        // Points 0 (y=5, w=1) and 1 (y=3, w=2) violate ascending order
        // They get pooled: (5*1 + 3*2)/(1+2) = 11/3 ≈ 3.666...
        // Point 2 (y=4) doesn't violate monotonicity with pooled value 11/3
        // Point 3 (y=10) doesn't violate monotonicity
        assert!((sorted_points[0].y() - 11.0 / 3.0).abs() < 0.0001);
        assert!((sorted_points[1].y() - 11.0 / 3.0).abs() < 0.0001);
        assert_eq!(*sorted_points[2].y(), 4.0);
        assert_eq!(*sorted_points[3].y(), 10.0);
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

        // Test exact point matches
        let x_at_1 = regression.invert(1.0).unwrap();
        assert!((x_at_1 - 0.0).abs() < 0.01);

        // Test interpolated values
        // At x=1.5, y=1.75 (from the example in the docstring)
        let x_at_1_75 = regression.invert(1.75).unwrap();
        assert!((x_at_1_75 - 1.5).abs() < 0.01);

        // Test that interpolate and invert are inverses
        let test_x = 1.5;
        let y = regression.interpolate(test_x).unwrap();
        let inverted_x = regression.invert(y).unwrap();
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

        // Test exact point matches
        let x_at_3 = regression.invert(3.0).unwrap();
        assert!((x_at_3 - 0.0).abs() < 0.01);

        let x_at_1 = regression.invert(1.0).unwrap();
        assert!((x_at_1 - 3.0).abs() < 0.01);

        // Test that interpolate and invert are inverses
        let test_x = 1.5;
        let y = regression.interpolate(test_x).unwrap();
        let inverted_x = regression.invert(y).unwrap();
        assert!((inverted_x - test_x).abs() < 0.01);
    }

    #[test]
    fn test_invert_empty() {
        let regression: IsotonicRegression<f64> = IsotonicRegression::new_ascending(&[]).unwrap();
        assert!(regression.invert(1.0).is_none());
    }

    #[test]
    fn test_invert_single_point() {
        let points = vec![Point::new(1.0, 2.0)];
        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let x = regression.invert(2.0).unwrap();
        assert_eq!(x, 1.0);
    }
}
