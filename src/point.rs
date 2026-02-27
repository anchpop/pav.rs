use crate::coordinate::Coordinate;
use crate::weight::Weight;
use serde::Serialize;

/// A point in 2D cartesian space
#[derive(Debug, PartialEq, Copy, Clone, Serialize)]
pub struct Point<T: Coordinate, W: Weight = f64> {
    pub(crate) x: T,
    pub(crate) y: T,
    pub(crate) weight: W,
}

impl<T: Coordinate, W: Weight> Default for Point<T, W> {
    fn default() -> Self {
        Point {
            x: T::zero(),
            y: T::zero(),
            weight: W::unit(),
        }
    }
}

/// Constructors that always produce `Point<T, f64>` (the default weight type).
///
/// These are separated so that `Point::new(0.0, 1.0)` compiles without
/// type annotations — Rust's default‑type‑parameter inference only kicks in
/// at type‑annotation sites, not in expression position.
impl<T: Coordinate> Point<T> {
    /// Create a new Point with unit weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::Point;
    ///
    /// let point = Point::new(1.0, 2.0);
    /// assert_eq!(*point.x(), 1.0);
    /// assert_eq!(*point.y(), 2.0);
    /// assert_eq!(point.weight(), 1.0);
    /// ```
    pub fn new(x: T, y: T) -> Point<T> {
        Point {
            x,
            y,
            weight: 1.0,
        }
    }
}

impl<T: Coordinate, W: Weight> Point<T, W> {
    /// Create a new Point with a specified weight.
    ///
    /// The weight type `W` is inferred from the `weight` argument, so this
    /// works for both `f64` and [`UnitWeight`](crate::UnitWeight):
    ///
    /// ```
    /// use pav_regression::{Point, UnitWeight};
    ///
    /// // f64 weight (inferred):
    /// let p1 = Point::new_with_weight(1.0, 2.0, 0.5);
    /// assert_eq!(p1.weight(), 0.5);
    ///
    /// // UnitWeight (inferred):
    /// let p2 = Point::new_with_weight(1.0, 2.0, UnitWeight);
    /// assert_eq!(p2.weight(), 1.0);
    /// ```
    pub fn new_with_weight(x: T, y: T, weight: W) -> Point<T, W> {
        Point { x, y, weight }
    }

    /// The x position of the point.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::Point;
    ///
    /// let point = Point::new(1.0, 2.0);
    /// assert_eq!(*point.x(), 1.0);
    /// ```
    pub fn x(&self) -> &T {
        &self.x
    }

    /// The y position of the point.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::Point;
    ///
    /// let point = Point::new(1.0, 2.0);
    /// assert_eq!(*point.y(), 2.0);
    /// ```
    pub fn y(&self) -> &T {
        &self.y
    }

    /// The weight of the point (initially 1.0).
    ///
    /// Always returns `f64` regardless of the underlying weight storage type.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::Point;
    ///
    /// let point = Point::new(1.0, 2.0);
    /// assert_eq!(point.weight(), 1.0);
    /// ```
    pub fn weight(&self) -> f64 {
        self.weight.to_f64()
    }

    /// Merges this point with another point, updating only the y-coordinate and weight.
    /// The x-coordinate is preserved.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::Point;
    ///
    /// let mut point1 = Point::new_with_weight(1.0, 2.0, 0.5);
    /// let point2 = Point::new_with_weight(3.0, 4.0, 1.5);
    /// point1.merge_with(&point2);
    /// assert_eq!(*point1.x(), 1.0); // x unchanged
    /// assert_eq!(*point1.y(), 3.5);
    /// assert_eq!(point1.weight(), 2.0);
    /// ```
    pub fn merge_with(&mut self, other: &Point<T, W>) {
        let self_w = self.weight.to_f64();
        let other_w = other.weight.to_f64();
        let total_weight = self_w + other_w;
        // Only update y-coordinate, preserve x
        self.y = (self.y * T::from_float(self_w) + other.y * T::from_float(other_w))
            / T::from_float(total_weight);
        self.weight = W::from_f64(total_weight);
    }
}

impl<T: Coordinate, W: Weight> From<(T, T)> for Point<T, W> {
    fn from(tuple: (T, T)) -> Self {
        Point::new_with_weight(tuple.0, tuple.1, W::unit())
    }
}

#[allow(dead_code)]
/// Interpolates the y value at a given x position between two points.
///
/// # Examples
///
/// ```
/// use pav_regression::Point;
/// use pav_regression::point::interpolate_two_points;
///
/// let point1 = Point::new(0.0, 0.0);
/// let point2 = Point::new(2.0, 2.0);
/// let interpolated_y = interpolate_two_points(&point1, &point2, 1.0);
/// assert_eq!(interpolated_y, 1.0);
/// ```
pub fn interpolate_two_points<T: Coordinate, W: Weight>(
    a: &Point<T, W>,
    b: &Point<T, W>,
    at_x: T,
) -> T {
    let prop = (at_x - a.x) / (b.x - a.x);
    a.y + (b.y - a.y) * prop
}

#[allow(dead_code)]
/// Interpolates the x value at a given y position between two points.
/// This is the inverse operation of `interpolate_two_points`.
///
/// # Examples
///
/// ```
/// use pav_regression::Point;
/// use pav_regression::point::interpolate_x_from_y;
///
/// let point1 = Point::new(0.0, 0.0);
/// let point2 = Point::new(2.0, 2.0);
/// let interpolated_x = interpolate_x_from_y(&point1, &point2, 1.0);
/// assert_eq!(interpolated_x, 1.0);
/// ```
pub fn interpolate_x_from_y<T: Coordinate, W: Weight>(
    a: &Point<T, W>,
    b: &Point<T, W>,
    at_y: T,
) -> T {
    let prop = (at_y - a.y) / (b.y - a.y);
    a.x + (b.x - a.x) * prop
}
