/* holosphere/src/vector/polar.rs */
//!▫~•◦-------------------------------‣
//! # Circular Angular Metric & Polar Periodic Geometry
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides polar coordinate transformations ($r = \sqrt{x^2+y^2}$, $\theta = \text{atan2}(y, x)$),
//! wrap-around periodic angular distance metrics on the circle $S^1$, and circular statistics:
//!
//! $$d(\theta_1, \theta_2) = \min(|\theta_1 - \theta_2|,\, 2\pi - |\theta_1 - \theta_2|) \in [0, \pi]$$
//!
//! ## Invariants
//! 1. **Range Bounds**: $\forall \theta_1, \theta_2 \in [-\pi, \pi]: 0 \le d(\theta_1, \theta_2) \le \pi$.
//! 2. **Periodic Wrap-around**: $d(\pi - \epsilon, -\pi + \epsilon) \equiv 2\epsilon$ (no boundary cliff).
//! 3. **Metric Properties**: $d(a, b) = d(b, a)$, $d(a, a) = 0$, $d(a, b) + d(b, c) \ge d(a, c)$.
//! 4. **Round-trip Invariance**: $(r, \theta) \to (x, y) \to (r, \theta)$ up to floating-point precision.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use num_complex::Complex32;
use std::f32::consts::PI;

/// Reusable circular periodic metric and polar projection primitive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CircularAngularMetric;

impl CircularAngularMetric {
    pub const TWO_PI: f32 = 2.0 * PI;

    /// Converts a 2D Cartesian pair $(x, y)$ to Polar coordinates $(r, \theta)$ where $r \ge 0$ and $\theta \in [-\pi, \pi]$.
    #[inline(always)]
    pub fn cartesian_to_polar(x: f32, y: f32) -> (f32, f32) {
        let r = (x * x + y * y).sqrt();
        let theta = y.atan2(x);
        (r, theta)
    }

    /// Converts Polar coordinates $(r, \theta)$ to a 2D Cartesian pair $(x, y)$.
    #[inline(always)]
    pub fn polar_to_cartesian(r: f32, theta: f32) -> (f32, f32) {
        (r * theta.cos(), r * theta.sin())
    }

    /// Converts a `Complex32` number to Polar coordinates $(r, \theta)$.
    #[inline(always)]
    pub fn complex_to_polar(z: Complex32) -> (f32, f32) {
        (z.norm(), z.arg())
    }

    /// Converts Polar coordinates $(r, \theta)$ to a `Complex32` number.
    #[inline(always)]
    pub fn polar_to_complex(r: f32, theta: f32) -> Complex32 {
        Complex32::from_polar(r, theta)
    }

    /// Computes the exact geodesic angular distance between two angles on $S^1$ taking into account the $2\pi$ periodic wrap-around.
    /// Output is strictly in $[0, \pi]$.
    #[inline(always)]
    pub fn angular_distance(theta_a: f32, theta_b: f32) -> f32 {
        let diff = (theta_a - theta_b).abs() % Self::TWO_PI;
        if diff > PI { Self::TWO_PI - diff } else { diff }
    }

    /// Computes the mean pairwise circular distance across two slices of phase angles.
    pub fn mean_angular_distance(thetas_a: &[f32], thetas_b: &[f32]) -> f32 {
        let len = thetas_a.len().min(thetas_b.len());
        if len == 0 {
            return 0.0;
        }
        let sum: f32 = thetas_a
            .iter()
            .zip(thetas_b)
            .map(|(&a, &b)| Self::angular_distance(a, b))
            .sum();
        sum / len as f32
    }

    /// Computes circular / directional mean angle of a set of angles in radians.
    pub fn circular_mean(thetas: &[f32]) -> f32 {
        if thetas.is_empty() {
            return 0.0;
        }
        let mut sin_sum = 0.0f32;
        let mut cos_sum = 0.0f32;
        for &t in thetas {
            sin_sum += t.sin();
            cos_sum += t.cos();
        }
        sin_sum.atan2(cos_sum)
    }

    /// Computes circular variance $V = 1 - \bar{R} \in [0, 1]$, where $\bar{R}$ is the mean resultant vector length.
    /// $V = 0$ indicates perfect angular concentration; $V = 1$ indicates uniform angular dispersion.
    pub fn circular_variance(thetas: &[f32]) -> f32 {
        if thetas.is_empty() {
            return 0.0;
        }
        let n = thetas.len() as f32;
        let mut sin_sum = 0.0f32;
        let mut cos_sum = 0.0f32;
        for &t in thetas {
            sin_sum += t.sin();
            cos_sum += t.cos();
        }
        let r_bar = (sin_sum * sin_sum + cos_sum * cos_sum).sqrt() / n;
        (1.0 - r_bar).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_polar_cartesian_round_trip() {
        let test_points = [
            (1.0, 0.0),
            (0.0, 2.0),
            (-3.0, 4.0),
            (-5.0, -12.0),
            (7.0, -24.0),
        ];
        for (x, y) in test_points {
            let (r, theta) = CircularAngularMetric::cartesian_to_polar(x, y);
            assert!(r >= 0.0);
            assert!((-PI..=PI).contains(&theta));
            let (rx, ry) = CircularAngularMetric::polar_to_cartesian(r, theta);
            assert!((rx - x).abs() < 1e-5, "Expected x={x}, got {rx}");
            assert!((ry - y).abs() < 1e-5, "Expected y={y}, got {ry}");
        }
    }

    #[test]
    fn test_angular_distance_periodic_wrap_around() {
        // Points near +PI and -PI across the branch cut
        let theta_pos = PI - 0.05;
        let theta_neg = -PI + 0.05;
        let dist = CircularAngularMetric::angular_distance(theta_pos, theta_neg);
        assert!(
            (dist - 0.10).abs() < 1e-5,
            "Wrap-around distance should be 0.10, got {dist}"
        );

        // Identity
        assert_eq!(CircularAngularMetric::angular_distance(1.23, 1.23), 0.0);

        // Maximum distance is PI (antipodal)
        let dist_antipodal = CircularAngularMetric::angular_distance(0.0, PI);
        assert!((dist_antipodal - PI).abs() < 1e-5);

        // Symmetry
        let d1 = CircularAngularMetric::angular_distance(0.5, -1.2);
        let d2 = CircularAngularMetric::angular_distance(-1.2, 0.5);
        assert_eq!(d1, d2);
    }

    #[test]
    fn test_circular_mean_and_variance() {
        // Angles clustered around PI/2
        let clustered = [PI / 2.0 - 0.1, PI / 2.0, PI / 2.0 + 0.1];
        let mean = CircularAngularMetric::circular_mean(&clustered);
        assert!((mean - PI / 2.0).abs() < 1e-4);
        let var = CircularAngularMetric::circular_variance(&clustered);
        assert!(var < 0.01, "Variance should be near 0 for tight cluster");

        // Uniformly distributed 4 orthogonal angles (variance ~ 1.0)
        let orthogonal = [0.0, PI / 2.0, PI, -PI / 2.0];
        let var_ortho = CircularAngularMetric::circular_variance(&orthogonal);
        assert!(
            (var_ortho - 1.0).abs() < 1e-4,
            "Variance should be 1.0 for orthogonal set"
        );
    }
}
