use crate::coordinate::Coordinate;
use crate::point::Point;
use crate::weight::Weight;
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
/// centroid point of the original set. Points are stored in sorted order by x.
#[derive(Debug, Clone, Serialize)]
pub struct IsotonicRegression<T: Coordinate, W: Weight = f64> {
    pub(crate) direction: Direction,
    pub(crate) points: Vec<Point<T, W>>,
    pub(crate) centroid_point: Centroid<T>,
    pub(crate) intersect_origin: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct Centroid<T: Coordinate> {
    pub(crate) sum_x: T,
    pub(crate) sum_y: T,
    pub(crate) sum_weight: T,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq)]
#[allow(dead_code)]
/// Specifies the direction of the isotonic regression.
pub enum Direction {
    /// Indicates an ascending (non-decreasing) regression.
    Ascending,
    /// Indicates a descending (non-increasing) regression.
    Descending,
}

impl<T: Coordinate + Display, W: Weight> Display for IsotonicRegression<T, W> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "IsotonicRegression {{")?;
        writeln!(f, "\tdirection: {:?},", self.direction)?;
        writeln!(f, "\tpoints:")?;
        for point in &self.points {
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
impl<T: Coordinate, W: Weight> IsotonicRegression<T, W> {
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
    /// assert_eq!(regression.get_points().len(), 4);
    /// ```
    pub fn new_ascending(
        points: &[Point<T, W>],
    ) -> Result<IsotonicRegression<T, W>, IsotonicRegressionError> {
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
    /// assert_eq!(regression.get_points().len(), 4);
    /// ```
    pub fn new_descending(
        points: &[Point<T, W>],
    ) -> Result<IsotonicRegression<T, W>, IsotonicRegressionError> {
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
    /// assert_eq!(regression.get_points().len(), 4);
    /// ```
    pub fn new(
        points: &[Point<T, W>],
        direction: Direction,
        intersect_origin: bool,
    ) -> Result<IsotonicRegression<T, W>, IsotonicRegressionError> {
        if intersect_origin {
            for point in points {
                if point.x().is_sign_negative() || point.y().is_sign_negative() {
                    return Err(IsotonicRegressionError::NegativePointWithIntersectOrigin);
                }
            }
        }

        let (isotonic_points, centroid) = isotonic(points, direction);

        Ok(IsotonicRegression {
            direction,
            points: isotonic_points,
            centroid_point: centroid,
            intersect_origin,
        })
    }

    /// Find an ascending isotonic regression from a set of points that are already sorted by x.
    ///
    /// This skips the sort step, which can be faster if the input is already sorted.
    /// The caller must ensure the points are sorted in non-decreasing order by x.
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
    /// let regression = IsotonicRegression::new_ascending_sorted(&points).unwrap();
    /// assert_eq!(regression.get_points().len(), 4);
    /// ```
    pub fn new_ascending_sorted(
        points: &[Point<T, W>],
    ) -> Result<IsotonicRegression<T, W>, IsotonicRegressionError> {
        IsotonicRegression::new_sorted(points, Direction::Ascending, false)
    }

    /// Find a descending isotonic regression from a set of points that are already sorted by x.
    ///
    /// This skips the sort step, which can be faster if the input is already sorted.
    /// The caller must ensure the points are sorted in non-decreasing order by x.
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
    /// let regression = IsotonicRegression::new_descending_sorted(&points).unwrap();
    /// assert_eq!(regression.get_points().len(), 4);
    /// ```
    pub fn new_descending_sorted(
        points: &[Point<T, W>],
    ) -> Result<IsotonicRegression<T, W>, IsotonicRegressionError> {
        IsotonicRegression::new_sorted(points, Direction::Descending, false)
    }

    /// Find an isotonic regression from a set of points that are already sorted by x.
    ///
    /// This skips the sort step, which can be faster if the input is already sorted.
    /// The caller must ensure the points are sorted in non-decreasing order by x.
    ///
    /// If `intersect_origin` is true, the regression will intersect the origin (0,0) and all points must be >= 0 on both axes.
    pub fn new_sorted(
        points: &[Point<T, W>],
        direction: Direction,
        intersect_origin: bool,
    ) -> Result<IsotonicRegression<T, W>, IsotonicRegressionError> {
        if intersect_origin {
            for point in points {
                if point.x().is_sign_negative() || point.y().is_sign_negative() {
                    return Err(IsotonicRegressionError::NegativePointWithIntersectOrigin);
                }
            }
        }

        let (isotonic_points, centroid) = isotonic_presorted(points.to_vec(), direction);

        Ok(IsotonicRegression {
            direction,
            points: isotonic_points,
            centroid_point: centroid,
            intersect_origin,
        })
    }

    /// Retrieve the points that make up the isotonic regression, sorted by x value.
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
    /// assert_eq!(regression.get_points().len(), 4);
    /// ```
    pub fn get_points(&self) -> &[Point<T, W>] {
        &self.points
    }

    /// Consume the regression and return the owned points, sorted by x value.
    pub fn into_points(self) -> Vec<Point<T, W>> {
        self.points
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
    pub fn add_points(&mut self, points: &[Point<T, W>]) {
        for point in points {
            assert!(
                !self.intersect_origin
                    || (!point.x().is_sign_negative() && !point.y().is_sign_negative()),
                "With intersect_origin = true, all points must be >= 0 on both x and y axes"
            );
            let w: T = point.weight.to_coord();
            self.centroid_point.sum_x += *point.x() * w;
            self.centroid_point.sum_y += *point.y() * w;
            self.centroid_point.sum_weight += w;
        }

        let mut new_points = self.points.clone();
        new_points.extend_from_slice(points);
        let (iso_points, _) = isotonic(&new_points, self.direction);
        self.points = iso_points;
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
    pub fn remove_points(&mut self, points: &[Point<T, W>]) {
        for point in points {
            assert!(
                !self.intersect_origin
                    || (!point.x().is_sign_negative() && !point.y().is_sign_negative()),
                "With intersect_origin = true, all points must be >= 0 on both x and y axes"
            );
            let w: T = point.weight.to_coord();
            self.centroid_point.sum_x = self.centroid_point.sum_x - *point.x() * w;
            self.centroid_point.sum_y = self.centroid_point.sum_y - *point.y() * w;
            self.centroid_point.sum_weight = self.centroid_point.sum_weight - w;
        }

        let mut new_points = self.points.clone();
        for point in points {
            if let Some(pos) = new_points.iter().position(|p| {
                p.x() == point.x() && p.y() == point.y() && p.weight() == point.weight()
            }) {
                new_points.remove(pos);
            }
        }
        let (iso_points, _) = isotonic(&new_points, self.direction);
        self.points = iso_points;
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
        self.centroid_point.sum_weight.to_float().round() as usize
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
        self.centroid_point.sum_weight == T::zero()
    }

    /// Returns the direction of the regression.
    pub fn direction(&self) -> Direction {
        self.direction
    }

    /// Returns whether the regression intersects the origin.
    pub fn intersect_origin(&self) -> bool {
        self.intersect_origin
    }
}

pub(crate) fn isotonic<T: Coordinate, W: Weight>(
    points: &[Point<T, W>],
    direction: Direction,
) -> (Vec<Point<T, W>>, Centroid<T>) {
    if points.is_empty() {
        return (
            Vec::new(),
            Centroid {
                sum_x: T::zero(),
                sum_y: T::zero(),
                sum_weight: T::zero(),
            },
        );
    }

    let mut result: Vec<Point<T, W>> = points.to_vec();

    // Sort the points by x
    result.sort_unstable_by(|a, b| {
        a.x()
            .partial_cmp(b.x())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    isotonic_presorted(result, direction)
}

pub(crate) fn isotonic_presorted<T: Coordinate, W: Weight>(
    mut result: Vec<Point<T, W>>,
    direction: Direction,
) -> (Vec<Point<T, W>>, Centroid<T>) {
    if result.is_empty() {
        return (
            result,
            Centroid {
                sum_x: T::zero(),
                sum_y: T::zero(),
                sum_weight: T::zero(),
            },
        );
    }

    let n = result.len();

    // Single-pass stack-based PAV algorithm with fused centroid accumulation.
    // Each pool is (start_index, end_index, sum_y_weighted, sum_weight).
    // We process points left-to-right, merging adjacent pools that violate monotonicity.
    let mut pools: Vec<(usize, usize, T, T)> = Vec::with_capacity(n);

    let mut sum_x = T::zero();
    let mut sum_y = T::zero();
    let mut sum_weight = T::zero();

    for (i, point) in result.iter().enumerate().take(n) {
        let w: T = point.weight.to_coord();
        let wy = *point.y() * w;

        // Accumulate centroid from original y values (before pooling)
        sum_x += *point.x() * w;
        sum_y += wy;
        sum_weight += w;

        pools.push((i, i + 1, wy, w));

        // Merge backwards while monotonicity is violated
        while pools.len() >= 2 {
            let len = pools.len();
            let (_, _, sum_y1, w1) = pools[len - 2];
            let (_, _, sum_y2, w2) = pools[len - 1];

            let avg1 = sum_y1 / w1;
            let avg2 = sum_y2 / w2;

            let violates = match direction {
                Direction::Ascending => avg1 > avg2,
                Direction::Descending => avg1 < avg2,
            };

            if violates {
                let (start, _, _, _) = pools[len - 2];
                let (_, end, _, _) = pools[len - 1];
                let merged = (start, end, sum_y1 + sum_y2, w1 + w2);
                pools[len - 2] = merged;
                pools.pop();
            } else {
                break;
            }
        }
    }

    // Apply pooled y-values back to all points
    for &(start, end, sum_y, sum_w) in &pools {
        let new_y = sum_y / sum_w;
        for point in result[start..end].iter_mut() {
            point.y = new_y;
        }
    }

    (
        result,
        Centroid {
            sum_x,
            sum_y,
            sum_weight,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight::UnitWeight;

    #[test]
    fn test_ascending_regression() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 1.5),
            Point::new(3.0, 3.0),
        ];

        let regression = IsotonicRegression::new_ascending(points).unwrap();
        let sorted_points = regression.get_points();
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
        let sorted_points = regression.get_points();
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
        assert_eq!(regression.get_points().len(), 3);
        assert_eq!(*regression.get_points()[1].x(), 1.0);
        assert_eq!(*regression.get_points()[1].y(), 1.5);
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
        assert_eq!(regression.get_points().len(), 3);

        regression.remove_points(&[Point::new(1.0, 2.0)]);

        // After removal, 2 points should remain
        let sorted_points = regression.get_points();
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
        let sorted_points = regression.get_points();

        // All points must be preserved
        assert_eq!(
            sorted_points.len(),
            5,
            "All 5 input points should be preserved"
        );

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
        let sorted_points = regression.get_points();

        // All points must be preserved
        assert_eq!(sorted_points.len(), 4);

        // X-coordinates unchanged
        assert_eq!(*sorted_points[0].x(), 0.0);
        assert_eq!(*sorted_points[1].x(), 1.0);
        assert_eq!(*sorted_points[2].x(), 2.0);
        assert_eq!(*sorted_points[3].x(), 3.0);

        // Y-values should be pooled where monotonicity is violated
        assert!((sorted_points[0].y() - 11.0 / 3.0).abs() < 0.0001);
        assert!((sorted_points[1].y() - 11.0 / 3.0).abs() < 0.0001);
        assert_eq!(*sorted_points[2].y(), 4.0);
        assert_eq!(*sorted_points[3].y(), 10.0);
    }

    #[test]
    fn test_into_points() {
        let points = &[
            Point::new(0.0, 1.0),
            Point::new(1.0, 2.0),
            Point::new(2.0, 3.0),
        ];
        let regression = IsotonicRegression::new_ascending(points).unwrap();
        let owned_points = regression.into_points();
        assert_eq!(owned_points.len(), 3);
        assert_eq!(*owned_points[0].x(), 0.0);
        assert_eq!(*owned_points[2].x(), 2.0);
    }

    #[test]
    fn test_unit_weight_size() {
        assert_eq!(
            std::mem::size_of::<Point<f64, UnitWeight>>(),
            16,
            "Point<f64, UnitWeight> should be 16 bytes (two f64s)"
        );
        assert_eq!(
            std::mem::size_of::<Point<f64>>(),
            24,
            "Point<f64> (with f64 weight) should be 24 bytes"
        );
    }

    #[test]
    fn test_unit_weight_ascending() {
        let points: Vec<Point<f64, UnitWeight>> = vec![
            Point::new_with_weight(0.0, 1.0, UnitWeight),
            Point::new_with_weight(1.0, 2.0, UnitWeight),
            Point::new_with_weight(2.0, 1.5, UnitWeight),
            Point::new_with_weight(3.0, 3.0, UnitWeight),
        ];

        let regression = IsotonicRegression::new_ascending(&points).unwrap();
        let sorted_points = regression.get_points();

        assert_eq!(sorted_points.len(), 4);
        assert_eq!(*sorted_points[0].y(), 1.0);
        assert_eq!(*sorted_points[1].y(), 1.75);
        assert_eq!(*sorted_points[2].y(), 1.75);
        assert_eq!(*sorted_points[3].y(), 3.0);
    }
}
