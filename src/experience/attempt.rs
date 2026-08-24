/* holosphere/src/experience/attempt.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Attempt Lifecycle & Records
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Models the execution lifecycle of attempts independently from outcome utility.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::entity::id::ProvenanceId;
use crate::experience::action::ActionInvocation;
use crate::experience::id::{AttemptId, ContextId, OutcomeId, ProblemId};

/// Operational execution state of an attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AttemptState {
    Planned,
    Running,
    Completed,
    Aborted,
    ExecutionError,
    TimedOut,
}

/// Empirical attempt record capturing actions tried under a specific context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AttemptRecord {
    pub attempt_id: AttemptId,
    pub problem_id: ProblemId,
    pub context_id: ContextId,
    pub state: AttemptState,
    pub action_invocations: Vec<ActionInvocation>,
    pub outcome_id: Option<OutcomeId>,
    pub started_lsn: u64,
    pub completed_lsn: Option<u64>,
    pub abort_reason: Option<Arc<str>>,
    pub provenance_id: ProvenanceId,
}
