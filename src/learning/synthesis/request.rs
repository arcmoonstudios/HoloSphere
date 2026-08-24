/* holosphere/src/learning/synthesis/request.rs */
//!▫~•◦-------------------------------‣
//! # Structural Synthesis Requests & Objectives
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the entry point contract for synthesizing candidate resolution plans
//! grounded in a single pinned snapshot world.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::experience::id::{ContextId, MetricId, ProblemId};

/// Unique identifier for a synthesis policy configuration.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct SynthesisPolicyId(pub u64);

/// Target objective driving structural synthesis.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum SynthesisGoal {
    /// Mitigate or resolve an observed problem instance.
    MitigateProblem(ProblemId),
    /// Optimize an empirical performance metric to a target threshold.
    OptimizeMetric { metric: MetricId, target_value: f64 },
    /// Unconstrained geometric exploratory synthesis.
    Exploration,
}

/// Request for synthesizing resolution candidates from a pinned snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SynthesisRequest {
    pub problem: ProblemId,
    pub context: ContextId,
    pub snapshot_lsn: u64,
    pub goal: SynthesisGoal,
    pub policy: SynthesisPolicyId,
}
