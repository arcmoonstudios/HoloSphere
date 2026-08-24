/* holosphere/src/learning/mutation.rs */
//!▫~•◦-------------------------------‣
//! # Replicated Learning State Machine Mutations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the durable commands applied deterministically to the learning subsystem.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::entity::status::EpistemicStatus;
use crate::experience::id::EvaluationPolicyId;
use crate::learning::adjudication::decision::AdjudicationRecord;
use crate::learning::adjudication::policy::AdjudicationPolicy;
use crate::learning::discovery::{DeclarativeOperator, DiscoveryStateMutation, OperatorLifecycle};
use crate::learning::evidence::accumulator::{EvidenceRecord, compute_evidence_digest};
use crate::learning::id::AdjudicationId;
use crate::learning::read::LearningSegment;
use crate::relation::id::RelationId;
use crate::relation::read::RelationSegment;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum LearningMutationError {
    #[error("Policy {0:?} not found")]
    PolicyNotFound(EvaluationPolicyId),
    #[error("Relation {0:?} not found")]
    RelationNotFound(RelationId),
    #[error("Adjudication {0:?} already exists")]
    AdjudicationAlreadyExists(AdjudicationId),
    #[error("Expected status {expected:?} but relation has {actual:?}")]
    StatusMismatch {
        expected: EpistemicStatus,
        actual: EpistemicStatus,
    },
    #[error("Evidence digest mismatch on replica")]
    DigestMismatch,
    #[error("Forbidden transition: {0}")]
    ForbiddenTransition(String),
    #[error("Discovered operator mutation rejected: {0}")]
    DiscoveryOperatorRejected(String),
}

/// Durable commands replicated via the consensus log.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningMutation {
    RegisterPolicy {
        policy: AdjudicationPolicy,
    },
    RecordEvidence {
        evidence: EvidenceRecord,
    },
    ApplyAdjudication {
        adjudication: AdjudicationRecord,
        expected_evidence_digest: [u8; 32],
    },
    UpsertDiscoveredOperator {
        operator: DeclarativeOperator,
        expected_previous: Option<OperatorLifecycle>,
    },
    ApplyDiscoveryState {
        mutation: DiscoveryStateMutation,
    },
}

impl LearningMutation {
    pub fn apply(
        &self,
        learn_seg: &LearningSegment,
        rel_seg: &RelationSegment,
        commit_lsn: u64,
    ) -> Result<(), LearningMutationError> {
        match self {
            LearningMutation::RegisterPolicy { policy } => {
                let mut policies = learn_seg.policies.write();
                policies.insert(policy.id, policy.clone());
                Ok(())
            }
            LearningMutation::RecordEvidence { evidence } => {
                learn_seg.accumulator.record(evidence.clone());
                Ok(())
            }
            LearningMutation::ApplyAdjudication {
                adjudication,
                expected_evidence_digest,
            } => {
                // 1. Verify relation exists and status matches
                let (rel_idx, mut header) = rel_seg
                    .arena
                    .get_by_id(adjudication.target_relation)
                    .ok_or(LearningMutationError::RelationNotFound(
                    adjudication.target_relation,
                ))?;

                if header.epistemic() != adjudication.previous_status {
                    return Err(LearningMutationError::StatusMismatch {
                        expected: adjudication.previous_status,
                        actual: header.epistemic(),
                    });
                }

                // Invariant: Inferred -> Observed or Provisional -> Observed is FORBIDDEN
                if adjudication.resulting_status == EpistemicStatus::Observed
                    || adjudication.resulting_status == EpistemicStatus::Asserted
                {
                    return Err(LearningMutationError::ForbiddenTransition(format!(
                        "Cannot adjudicate from {:?} to {:?}",
                        header.epistemic(),
                        adjudication.resulting_status
                    )));
                }

                // 2. Validate evidence digest
                let evidence_list = learn_seg.accumulator.get_evidence_for_relation(
                    adjudication.target_relation,
                    adjudication.evidence_snapshot_lsn,
                );
                let actual_digest = compute_evidence_digest(&evidence_list);
                if actual_digest != *expected_evidence_digest {
                    return Err(LearningMutationError::DigestMismatch);
                }

                // 3. Register adjudication record
                let mut adj_map = learn_seg.adjudications.write();
                if adj_map.contains_key(&adjudication.id) {
                    return Err(LearningMutationError::AdjudicationAlreadyExists(
                        adjudication.id,
                    ));
                }
                let mut adj_record = adjudication.clone();
                adj_record.committed_lsn = commit_lsn;
                adj_map.insert(adjudication.id, adj_record);
                learn_seg
                    .relation_adjudications
                    .write()
                    .entry(adjudication.target_relation)
                    .or_default()
                    .push(adjudication.id);

                // 4. Update relation header and version if status changed
                if adjudication.resulting_status != adjudication.previous_status {
                    let old_vrow_idx = header.version_row;
                    if old_vrow_idx != crate::entity::id::NULL_ROW_REF {
                        rel_seg.versions.close_version(old_vrow_idx, commit_lsn);
                    }

                    let new_vrow = crate::relation::version::RelationVersionRow {
                        relation_id: adjudication.target_relation,
                        version_id: (header.binding_len as u64) + 1,
                        valid_from_lsn: commit_lsn,
                        valid_until_lsn: u64::MAX,
                        prev_version_row: old_vrow_idx,
                        provenance_row: header.provenance_row,
                        epistemic_status: adjudication.resulting_status as u8,
                        lifecycle_status: header.lifecycle_status,
                        reserved: [0u8; 14],
                    };

                    let (_vid, vrow_idx) = rel_seg.versions.append(new_vrow);
                    header.version_row = vrow_idx;
                    header.set_epistemic(adjudication.resulting_status);
                    rel_seg.arena.update_header(rel_idx, header);
                }

                Ok(())
            }
            LearningMutation::UpsertDiscoveredOperator {
                operator,
                expected_previous,
            } => learn_seg
                .discovery
                .apply(operator.clone(), *expected_previous, commit_lsn)
                .map_err(|error| {
                    LearningMutationError::DiscoveryOperatorRejected(error.to_string())
                }),
            LearningMutation::ApplyDiscoveryState { mutation } => learn_seg
                .governed_discovery
                .apply(mutation.clone(), commit_lsn)
                .map_err(|error| {
                    LearningMutationError::DiscoveryOperatorRejected(error.to_string())
                }),
        }
    }
}
