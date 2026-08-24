/* holosphere/src/experience/mutation.rs */
//!▫~•◦-------------------------------‣
//! # Replicated Experience State Machine Mutations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Durable mutations applied to the empirical experience state machine.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thiserror::Error;

use crate::entity::id::ProvenanceId;
use crate::experience::action::{ActionDefinition, ActionInvocation, DurableActionParameter};
use crate::experience::attempt::{AttemptRecord, AttemptState};
use crate::experience::context::{ContextRecord, DurableContextDimension};
use crate::experience::id::{ActionId, AttemptId, ContextId, OutcomeId, ProblemId};
use crate::experience::metric::OutcomeMetricSchema;
use crate::experience::outcome::{DurableOutcomeObservation, OutcomeRecord};
use crate::experience::problem::ProblemOccurrence;
use crate::experience::read::ExperienceSegment;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum ExperienceMutationError {
    #[error("Problem {0:?} already exists")]
    ProblemAlreadyExists(ProblemId),
    #[error("Problem {0:?} not found")]
    ProblemNotFound(ProblemId),
    #[error("Context {0:?} already exists")]
    ContextAlreadyExists(ContextId),
    #[error("Context {0:?} not found")]
    ContextNotFound(ContextId),
    #[error("Action {0:?} not found")]
    ActionNotFound(ActionId),
    #[error("Attempt {0:?} already exists")]
    AttemptAlreadyExists(AttemptId),
    #[error("Attempt {0:?} not found")]
    AttemptNotFound(AttemptId),
    #[error("Expected attempt state {expected:?} but found {actual:?}")]
    AttemptStateConflict {
        expected: AttemptState,
        actual: AttemptState,
    },
}

/// Durable commands replicated via the Raft log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ExperienceMutation {
    CreateProblem {
        problem_id: ProblemId,
        symptom: Arc<str>,
        component: Arc<str>,
        context_id: ContextId,
        provenance_id: ProvenanceId,
    },
    CreateContext {
        context_id: ContextId,
        schema_version: u16,
        dimensions: Vec<DurableContextDimension>,
        provenance_id: ProvenanceId,
    },
    RegisterAction {
        action_id: ActionId,
        name: Arc<str>,
        description: Arc<str>,
        provenance_id: ProvenanceId,
    },
    RegisterMetric {
        metric: OutcomeMetricSchema,
    },
    BeginAttempt {
        attempt_id: AttemptId,
        problem_id: ProblemId,
        context_id: ContextId,
        provenance_id: ProvenanceId,
    },
    RecordActionInvocation {
        attempt_id: AttemptId,
        invocation_id: u64,
        action_id: ActionId,
        ordinal: u32,
        parameters: Vec<DurableActionParameter>,
        started_lsn: u64,
        completed_lsn: u64,
        provenance_id: ProvenanceId,
    },
    CompleteAttempt {
        attempt_id: AttemptId,
        expected_state: AttemptState,
        outcome_id: OutcomeId,
        observations: Vec<DurableOutcomeObservation>,
        provenance_id: ProvenanceId,
    },
    AbortAttempt {
        attempt_id: AttemptId,
        expected_state: AttemptState,
        reason: Arc<str>,
        provenance_id: ProvenanceId,
    },
}

impl ExperienceMutation {
    pub fn apply(
        &self,
        seg: &ExperienceSegment,
        commit_lsn: u64,
    ) -> Result<(), ExperienceMutationError> {
        match self {
            ExperienceMutation::CreateProblem {
                problem_id,
                symptom,
                component,
                context_id,
                provenance_id,
            } => {
                let mut problems = seg.problems.write();
                if problems.contains_key(problem_id) {
                    return Err(ExperienceMutationError::ProblemAlreadyExists(*problem_id));
                }
                problems.insert(
                    *problem_id,
                    ProblemOccurrence {
                        problem_id: *problem_id,
                        symptom: Arc::clone(symptom),
                        component: Arc::clone(component),
                        first_observed_lsn: commit_lsn,
                        provenance_id: *provenance_id,
                        context_id: *context_id,
                    },
                );
                Ok(())
            }
            ExperienceMutation::CreateContext {
                context_id,
                schema_version,
                dimensions,
                provenance_id,
            } => {
                let mut contexts = seg.contexts.write();
                if contexts.contains_key(context_id) {
                    return Err(ExperienceMutationError::ContextAlreadyExists(*context_id));
                }
                let record = ContextRecord::new(
                    *context_id,
                    *schema_version,
                    dimensions.clone(),
                    *provenance_id,
                );
                contexts.insert(*context_id, record);
                Ok(())
            }
            ExperienceMutation::RegisterAction {
                action_id,
                name,
                description,
                provenance_id,
            } => {
                let mut actions = seg.actions.write();
                actions.insert(
                    *action_id,
                    ActionDefinition {
                        action_id: *action_id,
                        name: Arc::clone(name),
                        description: Arc::clone(description),
                        provenance_id: *provenance_id,
                    },
                );
                Ok(())
            }
            ExperienceMutation::RegisterMetric { metric } => {
                let mut metrics = seg.metrics.write();
                metrics.insert(metric.metric_id, metric.clone());
                Ok(())
            }
            ExperienceMutation::BeginAttempt {
                attempt_id,
                problem_id,
                context_id,
                provenance_id,
            } => {
                if !seg.problems.read().contains_key(problem_id) {
                    return Err(ExperienceMutationError::ProblemNotFound(*problem_id));
                }
                if !seg.contexts.read().contains_key(context_id) {
                    return Err(ExperienceMutationError::ContextNotFound(*context_id));
                }

                let mut attempts = seg.attempts.write();
                if attempts.contains_key(attempt_id) {
                    return Err(ExperienceMutationError::AttemptAlreadyExists(*attempt_id));
                }

                attempts.insert(
                    *attempt_id,
                    AttemptRecord {
                        attempt_id: *attempt_id,
                        problem_id: *problem_id,
                        context_id: *context_id,
                        state: AttemptState::Running,
                        action_invocations: Vec::new(),
                        outcome_id: None,
                        started_lsn: commit_lsn,
                        completed_lsn: None,
                        abort_reason: None,
                        provenance_id: *provenance_id,
                    },
                );
                Ok(())
            }
            ExperienceMutation::RecordActionInvocation {
                attempt_id,
                invocation_id,
                action_id,
                ordinal,
                parameters,
                started_lsn,
                completed_lsn,
                provenance_id,
            } => {
                if !seg.actions.read().contains_key(action_id) {
                    return Err(ExperienceMutationError::ActionNotFound(*action_id));
                }

                let mut attempts = seg.attempts.write();
                let att = attempts
                    .get_mut(attempt_id)
                    .ok_or(ExperienceMutationError::AttemptNotFound(*attempt_id))?;

                att.action_invocations.push(ActionInvocation {
                    invocation_id: *invocation_id,
                    attempt_id: *attempt_id,
                    action_id: *action_id,
                    ordinal: *ordinal,
                    parameters: parameters.clone(),
                    started_lsn: *started_lsn,
                    completed_lsn: *completed_lsn,
                    provenance_id: *provenance_id,
                });

                Ok(())
            }
            ExperienceMutation::CompleteAttempt {
                attempt_id,
                expected_state,
                outcome_id,
                observations,
                provenance_id,
            } => {
                let mut attempts = seg.attempts.write();
                let att = attempts
                    .get_mut(attempt_id)
                    .ok_or(ExperienceMutationError::AttemptNotFound(*attempt_id))?;

                if att.state != *expected_state {
                    return Err(ExperienceMutationError::AttemptStateConflict {
                        expected: *expected_state,
                        actual: att.state,
                    });
                }

                // 1. Create outcome record
                let mut outcomes = seg.outcomes.write();
                outcomes.insert(
                    *outcome_id,
                    OutcomeRecord {
                        outcome_id: *outcome_id,
                        attempt_id: *attempt_id,
                        observations: observations.clone(),
                        commit_lsn,
                        provenance_id: *provenance_id,
                    },
                );

                // 2. Publish complete attempt atomically
                att.state = AttemptState::Completed;
                att.outcome_id = Some(*outcome_id);
                att.completed_lsn = Some(commit_lsn);

                Ok(())
            }
            ExperienceMutation::AbortAttempt {
                attempt_id,
                expected_state,
                reason,
                ..
            } => {
                let mut attempts = seg.attempts.write();
                let att = attempts
                    .get_mut(attempt_id)
                    .ok_or(ExperienceMutationError::AttemptNotFound(*attempt_id))?;

                if att.state != *expected_state {
                    return Err(ExperienceMutationError::AttemptStateConflict {
                        expected: *expected_state,
                        actual: att.state,
                    });
                }

                att.state = AttemptState::Aborted;
                att.abort_reason = Some(Arc::clone(reason));
                att.completed_lsn = Some(commit_lsn);

                Ok(())
            }
        }
    }
}
