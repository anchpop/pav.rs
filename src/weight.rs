use crate::coordinate::Coordinate;
use serde::Serialize;
use std::fmt::Debug;

/// Trait for weight types used in [`crate::Point`].
///
/// This trait abstracts over the weight storage in `Point`, allowing a zero-sized
/// [`UnitWeight`] type for cases where weights are not needed (saving 8 bytes per point).
pub trait Weight: Copy + Clone + Debug + PartialEq + Serialize {
    /// Returns the unit weight (equivalent to 1.0).
    fn unit() -> Self;
    /// Convert weight directly to a Coordinate type, bypassing f64.
    fn to_coord<T: Coordinate>(&self) -> T;
    /// Creates a weight from a Coordinate value.
    ///
    /// For [`UnitWeight`], this ignores the value and returns `UnitWeight`.
    fn from_coord<T: Coordinate>(value: T) -> Self;
}

impl Weight for f64 {
    fn unit() -> Self {
        1.0
    }
    fn to_coord<T: Coordinate>(&self) -> T {
        T::from_float(*self)
    }
    fn from_coord<T: Coordinate>(value: T) -> Self {
        value.to_float()
    }
}

/// A zero-sized weight type that always represents a weight of 1.0.
///
/// Using `Point<T, UnitWeight>` instead of `Point<T>` (which defaults to `Point<T, f64>`)
/// saves 8 bytes per point, which can improve cache performance for large datasets
/// where weights are not needed.
#[derive(Debug, Copy, Clone, PartialEq, Serialize)]
pub struct UnitWeight;

impl Weight for UnitWeight {
    fn unit() -> Self {
        UnitWeight
    }
    fn to_coord<T: Coordinate>(&self) -> T {
        T::one()
    }
    fn from_coord<T: Coordinate>(_value: T) -> Self {
        UnitWeight
    }
}
