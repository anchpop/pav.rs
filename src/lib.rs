#![warn(missing_docs)]

//! Pair adjacent violators algorithm for isotonic regression
//!
//! [Pair Adjacent Violators](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118763155.app3) algorithm for
//! [isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression)
//!
//! # What is "Isotonic Regression" and why should I care?
//!
//! Imagine you have two variables, _x_ and _y_, and you don't know the relationship between them, but you know that if _x_
//! increases then _y_ will increase, and if _x_ decreases then _y_ will decrease.  Alternatively it may be the opposite,
//! if _x_ increases then _y_ decreases, and if _x_ decreases then _y_ increases.
//!
//! Examples of such isotonic or monotonic relationships include:
//!
//! * _x_ is the pressure applied to the accelerator in a car, _y_ is the acceleration of the car (acceleration increases as more pressure is applied)
//! * _x_ is the rate at which a web server is receiving HTTP requests, _y_ is the CPU usage of the web server (server CPU usage will increase as the request rate increases)
//! * _x_ is the price of an item, and _y_ is the probability that someone will buy it (this would be a decreasing relationship, as _x_ increases _y_ decreases)
//!
//! These are all examples of an isotonic relationship between two variables, where the relationship is likely to be more complex than linear.
//!
//! So we know the relationship between _x_ and _y_ is isotonic, and let's also say that we've been able to collect data about actual _x_ and _y_ values that occur in practice.
//!
//! What we'd really like to be able to do is estimate, for any given _x_, what _y_ will be, or alternatively for any given _y_, what _x_ would be required.
//!
//! But of course real-world data is noisy, and is unlikely to be strictly isotonic, so we want something that allows us to feed in this raw noisy data, figure out the
//! actual relationship between _x_ and _y_, and then use this to allow us to predict _y_ given _x_ (using the `interpolate` method on `RegressionEvaluator`), or to predict what value of _x_ will
//! give us a particular value of _y_ (using the `invert` method on `RegressionEvaluator`).
//! This is the purpose of the pair-adjacent-violators algorithm.
//!
//! # ...and why should I care?
//!
//! Using the examples I provide above:
//!
//! * A self-driving car could use it to learn how much pressure to apply to the accelerator to give a desired amount of acceleration
//! * An autoscaling system could use it to help predict how many web servers they need to handle a given amount of web traffic
//! * A retailer could use it to choose a price for an item that maximizes their profit (aka "yield optimization")
//!
//! # Isotonic regression in online advertising
//!
//! If you have an hour to spare, and are interested in learning more about how online advertising works - you should check out [this lecture](https://vimeo.com/137999578)
//! that I gave in 2015 where I explain how we were able to use pair adjacent violators to solve some fun problems.
//!
//! # Examples
//!
//! ## Basic Usage
//!
//! ```
//! use pav_regression::{Point, IsotonicRegression, RegressionEvaluator};
//!
//! // Create some data points with noise
//! let points = vec![
//!     Point::new(0.0, 1.0),
//!     Point::new(1.0, 2.0),
//!     Point::new(2.0, 1.5),  // Violates monotonicity
//!     Point::new(3.0, 3.0),
//! ];
//!
//! // Build an ascending isotonic regression
//! let regression = IsotonicRegression::new_ascending(&points).unwrap();
//!
//! // Create an evaluator for O(log n) queries
//! let evaluator = RegressionEvaluator::new(regression);
//!
//! // Predict y for a given x (interpolate)
//! let y = evaluator.interpolate(1.5).unwrap();
//! println!("At x=1.5, predicted y={}", y);
//!
//! // Find x for a given y (invert)
//! let x = evaluator.invert(2.5).unwrap();
//! println!("To get y=2.5, need x={}", x);
//! ```
//!
//! ## Smooth Regression for Better Performance
//!
//! If you need to smooth the regression curve (e.g., to reduce sensitivity to noise),
//! use `SmoothRegression` which applies a box filter:
//!
//! ```
//! use pav_regression::{Point, IsotonicRegression, SmoothRegression};
//!
//! let points = vec![
//!     Point::new(100.0, 10.0),
//!     Point::new(200.0, 25.0),
//!     Point::new(300.0, 35.0),
//!     Point::new(400.0, 50.0),
//! ];
//!
//! // First build the isotonic regression
//! let regression = IsotonicRegression::new_ascending(&points).unwrap();
//!
//! // Then create a smoothed version with window half-width of 20
//! let smooth = SmoothRegression::from_regression(regression, 20.0);
//!
//! // Now interpolation is pre-smoothed - much faster than smoothing at query time!
//! let y = smooth.interpolate(250.0).unwrap();
//! let x = smooth.invert(30.0).unwrap();
//! ```
//!
//! This is particularly useful when you were doing manual smoothing like:
//!
//! ```
//! use pav_regression::{Point, IsotonicRegression, RegressionEvaluator, SmoothRegression};
//!
//! let points = vec![
//!     Point::new(100.0_f64, 10.0),
//!     Point::new(200.0, 25.0),
//!     Point::new(300.0, 35.0),
//! ];
//! let regression = IsotonicRegression::new_ascending(&points).unwrap();
//! let evaluator = RegressionEvaluator::new(regression.clone());
//!
//! // OLD WAY: Smooth at query time (3 lookups per query!)
//! let base = 150.0_f64;
//! let lower = evaluator.interpolate(base * 0.8).unwrap();
//! let middle = evaluator.interpolate(base).unwrap();
//! let upper = evaluator.interpolate(base * 1.2).unwrap();
//! let smoothed_old = (lower + middle + upper) / 3.0;
//!
//! // NEW WAY: Pre-smooth once, then fast lookups
//! let smooth = SmoothRegression::from_regression(regression, base * 0.2);
//! let smoothed_new = smooth.interpolate(base).unwrap();
//! // 3x faster and invertible!
//!
//! // Both approaches give similar results
//! assert!((smoothed_old - smoothed_new).abs() < 5.0);
//! ```

/// Module containing the `Coordinate` trait definition.
pub mod coordinate;

/// Module containing the `Weight` trait and `UnitWeight` type.
pub mod weight;

/// Module containing the `Point` struct definition.
pub mod point;

/// Module containing the `IsotonicRegression` struct and implementation.
pub mod isotonic_regression;

/// Module containing the `RegressionEvaluator` struct for O(log n) queries.
pub mod regression_evaluator;

/// Module containing the `SmoothRegression` struct and implementation.
pub mod smooth_regression;

pub use coordinate::Coordinate;
pub use isotonic_regression::IsotonicRegression;
pub use point::Point;
pub use regression_evaluator::RegressionEvaluator;
pub use smooth_regression::SmoothRegression;
pub use weight::{UnitWeight, Weight};
