/* holosphere/src/learning/synthesis/trace.rs */
//!▫~•◦-------------------------------‣
//! # Structural Synthesis Provenance Traces
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the complete, explainable audit trace detailing every input, precedent,
//! analogy, closure, and constraint check used to synthesize a plan.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::experience::id::{AttemptId, ContextId, ProblemId};
use crate::learning::inference::candidate::InferenceCandidateId;
use crate::learning::synthesis::constraint::ConstraintCheck;
use crate::learning::synthesis::request::SynthesisPolicyId;

/// Unique identifier for a $Cl(24)$ closure derivation artifact.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct ClosureArtifactId(pub u64);

/// Complete structural synthesis trace recording the exact provenance of a synthesized plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuralSynthesisTrace {
    pub snapshot_lsn: u64,
    pub target_problem: ProblemId,
    pub target_context: ContextId,
    pub precedent_attempts: Vec<AttemptId>,
    pub precedent_relations: Vec<u64>,
    pub analogy_artifacts: Vec<InferenceCandidateId>,
    pub closure_artifacts: Vec<ClosureArtifactId>,
    pub constraint_checks: Vec<ConstraintCheck>,
    pub synthesis_policy: SynthesisPolicyId,
    pub method_fingerprint: [u8; 32],
}
