use crate::coordinate::Coordinate;
use serde::Serialize;

/// A point in 2D cartesian space
#[derive(Debug, PartialEq, Copy, Clone, Serialize)]
pub struct Point<T: Coordinate> {
    pub(crate) x: T,
    pub(crate) y: T,
    pub(crate) weight: f64,
}

impl<T: Coordinate> Default for Point<T> {
    fn default() -> Self {
        Point {
            x: T::zero(),
            y: T::zero(),
            weight: 1.0,
        }
    }
}

impl<T: Coordinate> Point<T> {
    /// Create a new Point.
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
        Point { x, y, weight: 1.0 }
    }

    /// Create a new Point with a specified weight.
    ///
    /// # Examples
    ///
    /// ```
    /// use pav_regression::Point;
    ///
    /// let point = Point::new_with_weight(1.0, 2.0, 0.5);
    /// assert_eq!(*point.x(), 1.0);
    /// assert_eq!(*point.y(), 2.0);
    /// assert_eq!(point.weight(), 0.5);
    /// ```
    pub fn new_with_weight(x: T, y: T, weight: f64) -> Point<T> {
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
    /// # Examples
    ///
    /// ```
    /// use pav_regression::Point;
    ///
    /// let point = Point::new(1.0, 2.0);
    /// assert_eq!(point.weight(), 1.0);
    /// ```
    pub fn weight(&self) -> f64 {
        self.weight
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
    pub fn merge_with(&mut self, other: &Point<T>) {
        let total_weight = self.weight + other.weight;
        // Only update y-coordinate, preserve x
        self.y = (self.y * T::from_float(self.weight) + other.y * T::from_float(other.weight))
            / T::from_float(total_weight);
        self.weight = total_weight;
    }
}

impl<T: Coordinate> From<(T, T)> for Point<T> {
    fn from(tuple: (T, T)) -> Self {
        Point::new(tuple.0, tuple.1)
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
pub fn interpolate_two_points<T: Coordinate>(a: &Point<T>, b: &Point<T>, at_x: T) -> T {
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
pub fn interpolate_x_from_y<T: Coordinate>(a: &Point<T>, b: &Point<T>, at_y: T) -> T {
    let prop = (at_y - a.y) / (b.y - a.y);
    a.x + (b.x - a.x) * prop
}
