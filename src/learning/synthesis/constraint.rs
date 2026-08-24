/* holosphere/src/learning/synthesis/constraint.rs */
//!▫~•◦-------------------------------‣
//! # Synthesis Constraints & Incompatibility Checking
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Deterministic constraint evaluation checking for resource conflicts,
//! known negative precedents, parameter bounds, and ordering rules.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::entity::id::DurableEvidenceRef;

/// Standard constraint classification code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintCode {
    KnownContradiction,
    FailedCombinationPrecedent,
    ParameterOutOfRange,
    ResourceConflict,
    OrderingViolation,
    SchemaConstraint,
}

/// Outcome of evaluating an action constraint against empirical history.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ConstraintResult {
    /// Plan step or combination is fully admissible.
    Admissible,
    /// Plan step or combination is hard-rejected due to verified failure/conflict.
    Rejected(Arc<str>),
    /// Plan is admissible but carries an empirical penalty score (Q32).
    Penalized(i64),
}

/// Detailed evaluation record of a constraint check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConstraintCheck {
    pub code: ConstraintCode,
    pub result: ConstraintResult,
    pub evidence: Vec<DurableEvidenceRef>,
}

/// An explicit action constraint attached to a candidate plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionConstraint {
    pub constraint_id: u64,
    pub description: Arc<str>,
    pub check: ConstraintCheck,
}
