/* holosphere/src/entity/outcome.rs */
//!▫~•◦-------------------------------‣
//! # Outcome Metric Schemas and Observations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides typed, integer/fixed-point durable outcome metrics for recording
//! experience, empirical attempts, and context-dependent utility.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// Evaluation direction for an outcome metric.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OutcomeMetricDirection {
    /// Higher values indicate improvement (e.g. throughput, accuracy, cache hit rate).
    HigherBetter = 0,
    /// Lower values indicate improvement (e.g. latency, error rate, memory footprint).
    LowerBetter = 1,
}

/// Catalog schema for an outcome metric.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutcomeMetricSchema {
    pub metric_id: u32,
    pub name: String,
    pub unit: String,
    pub direction: OutcomeMetricDirection,
}

/// Deterministic, integer fixed-point observation of a metric transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OutcomeObservation {
    pub metric_id: u32,
    /// Value before action in Q32.32 fixed-point (or raw integer for count/ns).
    pub before_q32: i64,
    /// Value after action in Q32.32 fixed-point (or raw integer for count/ns).
    pub after_q32: i64,
}

impl OutcomeObservation {
    #[inline(always)]
    pub fn delta_q32(&self) -> i64 {
        self.after_q32 - self.before_q32
    }

    #[inline(always)]
    pub fn delta_f64(&self) -> f64 {
        (self.delta_q32() as f64) / (4294967296.0) // 2^32
    }
}
