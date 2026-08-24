/* holosphere/src/learning/synthesis/precedent.rs */
//!▫~•◦-------------------------------‣
//! # Historical Empirical Precedents (Supporting & Contradicting)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Models past attempts, actions, and outcomes as first-class positive and negative
//! procedural evidence for structural synthesis.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::experience::action::ActionInvocation;
use crate::experience::id::{AttemptId, ContextId, OutcomeId, ProblemId};
use crate::learning::id::AdjudicationId;
use crate::learning::synthesis::alignment::{ContextApplicability, StructuralAnalogyArtifact};

/// Empirical disposition of a historical precedent relative to problem mitigation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecedentDisposition {
    /// Attempt succeeded and produced positive empirical utility.
    Supporting,
    /// Attempt failed, regressed performance, or caused outages (negative evidence).
    Contradicting,
    /// Mixed results across different evaluation metrics.
    Mixed,
    /// Empirical outcome recorded but not yet deterministically adjudicated.
    Unadjudicated,
}

/// A structured historical precedent retrieved from the pinned snapshot.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Precedent {
    pub problem: ProblemId,
    pub attempt: AttemptId,
    pub context: ContextId,
    pub actions: Vec<ActionInvocation>,
    pub outcome: OutcomeId,
    pub adjudication: Option<AdjudicationId>,
    pub analogy: Option<StructuralAnalogyArtifact>,
    pub context_applicability: ContextApplicability,
    pub evidence_disposition: PrecedentDisposition,
    pub measured_utility_q32: i64,
}
