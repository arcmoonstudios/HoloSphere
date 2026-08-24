/* holosphere/src/learning/adjudication/policy.rs */
//!▫~•◦-------------------------------‣
//! # Versioned Adjudication Policy & Decisions
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the versioned evaluation criteria and decision rules for relation promotion/falsification.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::experience::id::EvaluationPolicyId;
use crate::learning::evidence::stats::MetricEvaluationRule;

/// Deterministic adjudication decision classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdjudicationDecisionCode {
    InsufficientEvidence,
    SupportThresholdReached,
    ContradictionThresholdReached,
    ContextDependent,
    MixedEvidence,
}

/// Conditional or categorical disposition of a relation under evaluated evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdjudicationDisposition {
    Pending,
    Supported,
    Contradicted,
    ContextDependent,
}

/// Versioned adjudication policy defining thresholds for empirical promotion or falsification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdjudicationPolicy {
    pub id: EvaluationPolicyId,
    pub version: u32,
    pub min_observations: u32,
    pub min_support: u32,
    pub max_contradictions: u32,
    pub promote_utility_q32: i64,
    pub falsify_utility_q32: i64,
    pub rules: Vec<MetricEvaluationRule>,
}
