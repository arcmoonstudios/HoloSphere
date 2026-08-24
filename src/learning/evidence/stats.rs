/* holosphere/src/learning/evidence/stats.rs */
//!▫~•◦-------------------------------‣
//! # Fixed-Point Utility & Normalization Rules
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides deterministic Q32.32 fixed-point arithmetic for metric utility calculation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::experience::id::MetricId;
use crate::experience::metric::MetricValue;
use serde::{Deserialize, Serialize};

/// Deterministic Q32.32 fixed-point utility value.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct FixedUtility {
    pub raw_q32: i64,
}

impl FixedUtility {
    pub const ZERO: Self = Self { raw_q32: 0 };
    pub const ONE: Self = Self {
        raw_q32: 1i64 << 32,
    };
    pub const NEG_ONE: Self = Self {
        raw_q32: -(1i64 << 32),
    };

    #[inline(always)]
    pub fn from_raw(raw: i64) -> Self {
        Self { raw_q32: raw }
    }

    #[inline(always)]
    pub fn from_i64(val: i64) -> Self {
        Self {
            raw_q32: val.saturating_mul(1i64 << 32),
        }
    }

    #[inline(always)]
    pub fn from_f64_checked(val: f64) -> Option<Self> {
        if val.is_nan() || val.is_infinite() {
            return None;
        }
        let scaled = val * ((1u64 << 32) as f64);
        if scaled < (i64::MIN as f64) || scaled > (i64::MAX as f64) {
            None
        } else {
            Some(Self {
                raw_q32: scaled as i64,
            })
        }
    }

    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        (self.raw_q32 as f64) / ((1u64 << 32) as f64)
    }

    #[inline(always)]
    pub fn saturating_add(self, other: Self) -> Self {
        Self {
            raw_q32: self.raw_q32.saturating_add(other.raw_q32),
        }
    }

    #[inline(always)]
    pub fn saturating_mul(self, other: Self) -> Self {
        let prod = (self.raw_q32 as i128).saturating_mul(other.raw_q32 as i128);
        let res = prod >> 32;
        let clamped = if res > (i64::MAX as i128) {
            i64::MAX
        } else if res < (i64::MIN as i128) {
            i64::MIN
        } else {
            res as i64
        };
        Self { raw_q32: clamped }
    }
}

/// Direction of improvement for a measurable metric.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricDirection {
    LowerIsBetter,
    HigherIsBetter,
}

/// Normalization rule translating raw baseline vs observed measurements into normalized utility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NormalizationRule {
    AbsoluteDelta,
    RelativeDelta,
    BoundedRelative { floor_q32: i64, ceiling_q32: i64 },
}

/// Rule governing how a specific metric observation contributes to policy utility.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MetricEvaluationRule {
    pub metric_id: MetricId,
    pub weight_q32: i64,
    pub direction: MetricDirection,
    pub normalization: NormalizationRule,
}

impl MetricEvaluationRule {
    /// Evaluates the normalized Q32.32 utility component for a single metric observation.
    pub fn evaluate(&self, baseline: &MetricValue, observed: &MetricValue) -> Option<FixedUtility> {
        let b = baseline.as_f64()?;
        let o = observed.as_f64()?;

        // Calculate delta based on improvement direction
        let raw_delta = match self.direction {
            MetricDirection::LowerIsBetter => b - o, // Lower observed is positive improvement
            MetricDirection::HigherIsBetter => o - b, // Higher observed is positive improvement
        };

        let normalized_ratio = match self.normalization {
            NormalizationRule::AbsoluteDelta => raw_delta,
            NormalizationRule::RelativeDelta => {
                if b.abs() < 1e-12 {
                    return None;
                }
                raw_delta / b.abs()
            }
            NormalizationRule::BoundedRelative {
                floor_q32,
                ceiling_q32,
            } => {
                if b.abs() < 1e-12 {
                    return None;
                }
                let ratio = raw_delta / b.abs();
                let f_floor = (floor_q32 as f64) / ((1u64 << 32) as f64);
                let f_ceil = (ceiling_q32 as f64) / ((1u64 << 32) as f64);
                ratio.clamp(f_floor, f_ceil)
            }
        };

        let norm_fixed = FixedUtility::from_f64_checked(normalized_ratio)?;
        let weight_fixed = FixedUtility::from_raw(self.weight_q32);

        Some(norm_fixed.saturating_mul(weight_fixed))
    }
}
