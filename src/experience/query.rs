/* holosphere/src/experience/query.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Experience Queries
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides pattern queries over empirical problems, contexts, actions, and attempts.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::experience::attempt::{AttemptRecord, AttemptState};
use crate::experience::id::{ActionId, ContextId, ProblemId};
use crate::experience::read::ExperienceReadSnapshot;

/// Query predicate matching empirical attempts.
#[derive(Clone, Debug, Default)]
pub struct ExperienceQuery {
    pub problem_id: Option<ProblemId>,
    pub context_id: Option<ContextId>,
    pub action_id: Option<ActionId>,
    pub state: Option<AttemptState>,
    pub as_of_lsn: Option<u64>,
}

impl ExperienceQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_problem(mut self, id: ProblemId) -> Self {
        self.problem_id = Some(id);
        self
    }

    pub fn with_context(mut self, id: ContextId) -> Self {
        self.context_id = Some(id);
        self
    }

    pub fn with_action(mut self, id: ActionId) -> Self {
        self.action_id = Some(id);
        self
    }

    pub fn with_state(mut self, state: AttemptState) -> Self {
        self.state = Some(state);
        self
    }

    pub fn with_as_of(mut self, lsn: u64) -> Self {
        self.as_of_lsn = Some(lsn);
        self
    }

    /// Executes query against a pinned `ExperienceReadSnapshot`.
    pub fn execute(&self, snap: &ExperienceReadSnapshot) -> Vec<AttemptRecord> {
        let attempts_map = snap.segment.attempts.read();
        let mut results = Vec::new();

        for (&att_id, _) in attempts_map.iter() {
            if let Some(att) = snap.attempt(att_id) {
                if let Some(pid) = self.problem_id {
                    if att.problem_id != pid {
                        continue;
                    }
                }
                if let Some(cid) = self.context_id {
                    if att.context_id != cid {
                        continue;
                    }
                }
                if let Some(aid) = self.action_id {
                    let uses_action = att.action_invocations.iter().any(|a| a.action_id == aid);
                    if !uses_action {
                        continue;
                    }
                }
                if let Some(st) = self.state {
                    if att.state != st {
                        continue;
                    }
                }
                results.push(att);
            }
        }

        // Canonical deterministic ordering: AttemptId ASC
        results.sort_unstable_by_key(|a| a.attempt_id);
        results
    }
}
