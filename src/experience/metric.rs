/* holosphere/src/experience/metric.rs */
//!▫~•◦-------------------------------‣
//! # Metric Schema & Typed Raw Observations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the immutable typed metric catalog and observation value representations.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::experience::id::{MetricId, SymbolId};

/// Data type classification for raw metric measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricValueKind {
    SignedInteger,
    UnsignedInteger,
    FixedQ32_32,
    Boolean,
    Symbol,
    Float64,
}

/// Typed measurement value preserving raw empirical precision without lossy interpretation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MetricValue {
    Signed(i64),
    Unsigned(u64),
    FixedQ32(i64),
    Boolean(bool),
    Symbol(SymbolId),
    Float(f64),
}

impl MetricValue {
    /// Extracts a numeric floating-point representation for derived delta calculations.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetricValue::Signed(v) => Some(*v as f64),
            MetricValue::Unsigned(v) => Some(*v as f64),
            MetricValue::FixedQ32(v) => Some((*v as f64) / (1i64 << 32) as f64),
            MetricValue::Float(v) => Some(*v),
            MetricValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            MetricValue::Symbol(_) => None,
        }
    }
}

/// Definition of a measurable empirical metric.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeMetricSchema {
    pub metric_id: MetricId,
    pub name: Arc<str>,
    pub unit: Arc<str>,
    pub value_kind: MetricValueKind,
    pub schema_version: u16,
}

impl OutcomeMetricSchema {
    /// Computes derived absolute delta: observed - baseline.
    pub fn compute_delta(&self, baseline: &MetricValue, observed: &MetricValue) -> Option<f64> {
        let b = baseline.as_f64()?;
        let o = observed.as_f64()?;
        Some(o - b)
    }

    /// Computes derived percentage delta: (observed - baseline) / baseline.
    pub fn compute_percentage_delta(
        &self,
        baseline: &MetricValue,
        observed: &MetricValue,
    ) -> Option<f64> {
        let b = baseline.as_f64()?;
        let o = observed.as_f64()?;
        if b.abs() < 1e-12 {
            None
        } else {
            Some((o - b) / b)
        }
    }
}
