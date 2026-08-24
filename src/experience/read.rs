/* holosphere/src/experience/read.rs */
//!▫~•◦-------------------------------‣
//! # Pinned Experience Snapshots & Unified Empirical Trace
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides point-in-time isolated queries and complete experience traces.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::entity::provenance::ProvenanceRecord;
use crate::entity::read::EntityReadSnapshot;
use crate::experience::action::{ActionDefinition, ActionInvocation};
use crate::experience::attempt::{AttemptRecord, AttemptState};
use crate::experience::context::ContextRecord;
use crate::experience::id::{ActionId, AttemptId, ContextId, MetricId, OutcomeId, ProblemId};
use crate::experience::metric::OutcomeMetricSchema;
use crate::experience::outcome::OutcomeRecord;
use crate::experience::problem::ProblemOccurrence;

/// Complete empirical trace containing all structured context, actions, outcomes, and provenance.
#[derive(Clone, Debug, PartialEq)]
pub struct ExperienceTrace {
    pub problem: ProblemOccurrence,
    pub context: ContextRecord,
    pub attempt: AttemptRecord,
    pub ordered_actions: Vec<ActionInvocation>,
    pub outcome: Option<OutcomeRecord>,
    pub provenance: Option<ProvenanceRecord>,
}

/// Generation-scoped container managing all empirical experience state.
pub struct ExperienceSegment {
    pub generation_id: u64,
    pub problems: RwLock<HashMap<ProblemId, ProblemOccurrence>>,
    pub contexts: RwLock<HashMap<ContextId, ContextRecord>>,
    pub actions: RwLock<HashMap<ActionId, ActionDefinition>>,
    pub attempts: RwLock<HashMap<AttemptId, AttemptRecord>>,
    pub outcomes: RwLock<HashMap<OutcomeId, OutcomeRecord>>,
    pub metrics: RwLock<HashMap<MetricId, OutcomeMetricSchema>>,
}

impl ExperienceSegment {
    pub fn new(generation_id: u64) -> Self {
        Self {
            generation_id,
            problems: RwLock::new(HashMap::new()),
            contexts: RwLock::new(HashMap::new()),
            actions: RwLock::new(HashMap::new()),
            attempts: RwLock::new(HashMap::new()),
            outcomes: RwLock::new(HashMap::new()),
            metrics: RwLock::new(HashMap::new()),
        }
    }

    pub fn read_snapshot(self: &Arc<Self>, lsn: u64) -> ExperienceReadSnapshot {
        ExperienceReadSnapshot {
            lsn,
            segment: Arc::clone(self),
        }
    }

    /// Performs physical compaction, cloning all durable empirical records.
    pub fn compact(&self, new_generation_id: u64) -> Arc<Self> {
        let compacted = Arc::new(Self::new(new_generation_id));
        *compacted.problems.write() = self.problems.read().clone();
        *compacted.contexts.write() = self.contexts.read().clone();
        *compacted.actions.write() = self.actions.read().clone();
        *compacted.attempts.write() = self.attempts.read().clone();
        *compacted.outcomes.write() = self.outcomes.read().clone();
        *compacted.metrics.write() = self.metrics.read().clone();
        compacted
    }
}

/// Point-in-time snapshot pinned at committed LSN `lsn`.
#[derive(Clone)]
pub struct ExperienceReadSnapshot {
    pub lsn: u64,
    pub segment: Arc<ExperienceSegment>,
}

impl ExperienceReadSnapshot {
    pub fn problem(&self, id: ProblemId) -> Option<ProblemOccurrence> {
        let problems = self.segment.problems.read();
        let p = problems.get(&id)?;
        if p.first_observed_lsn <= self.lsn {
            Some(p.clone())
        } else {
            None
        }
    }

    pub fn context(&self, id: ContextId) -> Option<ContextRecord> {
        let contexts = self.segment.contexts.read();
        contexts.get(&id).cloned()
    }

    pub fn attempt(&self, id: AttemptId) -> Option<AttemptRecord> {
        let attempts = self.segment.attempts.read();
        let att = attempts.get(&id)?;
        if att.started_lsn > self.lsn {
            return None;
        }

        let mut a = att.clone();
        // If query LSN is before completion, view as Running
        if let Some(comp_lsn) = a.completed_lsn {
            if self.lsn < comp_lsn {
                a.state = AttemptState::Running;
                a.outcome_id = None;
                a.completed_lsn = None;
            }
        }
        Some(a)
    }

    pub fn outcomes_for_attempt(&self, id: AttemptId) -> Option<OutcomeRecord> {
        let att = self.attempt(id)?;
        let outcome_id = att.outcome_id?;
        let outcomes = self.segment.outcomes.read();
        let out = outcomes.get(&outcome_id)?;
        if out.commit_lsn <= self.lsn {
            Some(out.clone())
        } else {
            None
        }
    }

    pub fn full_trace(
        &self,
        id: AttemptId,
        ent_snap: &EntityReadSnapshot,
    ) -> Option<ExperienceTrace> {
        let attempt = self.attempt(id)?;
        let problem = self.problem(attempt.problem_id)?;
        let context = self.context(attempt.context_id)?;
        let outcome = self.outcomes_for_attempt(id);

        let mut ordered_actions = attempt.action_invocations.clone();
        ordered_actions.sort_unstable_by_key(|a| a.ordinal);

        let provenance = ent_snap
            .segment
            .provenance
            .resolve_record_by_id(attempt.provenance_id);

        Some(ExperienceTrace {
            problem,
            context,
            attempt,
            ordered_actions,
            outcome,
            provenance,
        })
    }
}
