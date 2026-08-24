/* holosphere/src/learning/synthesis/candidate.rs */
//!▫~•◦-------------------------------‣
//! # Resolution Candidates & Epistemic Boundaries
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Models provisional resolution proposals synthesized from empirical precedents
//! and structural inference.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::status::EpistemicStatus;
use crate::experience::id::{AttemptId, ContextId, ProblemId};
use crate::learning::synthesis::composition::ActionPlan;
use crate::learning::synthesis::trace::StructuralSynthesisTrace;

/// Unique identifier for a synthesized resolution candidate.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct ResolutionCandidateId(pub u64);

/// Lifecycle state of a synthesized candidate plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CandidateResolutionState {
    /// Newly synthesized proposal awaiting external authorization.
    Proposed,
    /// Authorized and queued by an external orchestrator/operator.
    AcceptedForExecution,
    /// An empirical attempt has begun executing this plan.
    ExecutionStarted,
    /// Execution finished and empirical outcome observations are committed.
    ExecutionCompleted,
    /// Execution was canceled or aborted before completion.
    ExecutionAborted,
}

/// Explainable decomposition of multi-criteria synthesis evaluation scores.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SynthesisScores {
    pub structural_alignment_q32: i64,
    pub context_applicability_q32: i64,
    pub supporting_precedent_count: u32,
    pub contradicting_precedent_count: u32,
    pub historical_utility_q32: i64,
    pub cl24_reference_novelty: f32,
    pub cl24_truncation_loss: f32,
    pub aggregate_ranking_score_q32: i64,
}

/// A complete, provisional resolution candidate proposing a structured action plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolutionCandidate {
    pub candidate_id: ResolutionCandidateId,
    pub problem: ProblemId,
    pub context: ContextId,
    pub action_plan: ActionPlan,
    pub supporting_precedents: Vec<AttemptId>,
    pub contradicting_precedents: Vec<AttemptId>,
    pub scores: SynthesisScores,
    pub structural_trace: StructuralSynthesisTrace,
    /// HARD INVARIANT: Always begins strictly at EpistemicStatus::Provisional.
    pub epistemic_status: EpistemicStatus,
    pub resolution_state: CandidateResolutionState,
}
