use serde::Serialize;
use std::fmt::Debug;

/// Trait for weight types used in [`crate::Point`].
///
/// This trait abstracts over the weight storage in `Point`, allowing a zero-sized
/// [`UnitWeight`] type for cases where weights are not needed (saving 8 bytes per point).
pub trait Weight: Copy + Clone + Debug + PartialEq + Serialize {
    /// Returns the unit weight (equivalent to 1.0).
    fn unit() -> Self;
    /// Converts the weight to `f64` for arithmetic.
    fn to_f64(&self) -> f64;
    /// Creates a weight from an `f64` value.
    ///
    /// For [`UnitWeight`], this ignores the value and returns `UnitWeight`.
    fn from_f64(value: f64) -> Self;
}

impl Weight for f64 {
    fn unit() -> Self {
        1.0
    }
    fn to_f64(&self) -> f64 {
        *self
    }
    fn from_f64(value: f64) -> Self {
        value
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
    fn to_f64(&self) -> f64 {
        1.0
    }
    fn from_f64(_value: f64) -> Self {
        UnitWeight
    }
}
