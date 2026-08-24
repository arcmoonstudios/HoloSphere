/* holosphere/src/entity/mutation.rs */
//!▫~•◦-------------------------------‣
//! # Replicated Entity Mutations & State Machine Apply Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides deterministic state-machine mutations for the entity universe.
//!
//! ## Invariant Guarantees
//! - Replicated mutations name entities by `EntityId`, never `EntityIndex`.
//! - Version updates are applied atomically at committed LSN `L`.
//! - Expected-state guards reject race conditions and invalid epistemic transitions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::entity::epistemic::validate_epistemic_transition;
use crate::entity::header::EntityHeader;
use crate::entity::id::{DurableEvidenceRef, EntityId, NULL_ROW_REF, ProvenanceId, VersionId};
use crate::entity::provenance::ProvenanceRecord;
use crate::entity::segment::EntitySegment;
use crate::entity::status::{EpistemicStatus, LifecycleStatus};
use crate::entity::version::{VersionRelation, VersionRow};

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum MutationApplyError {
    #[error("Entity {0} already exists")]
    EntityAlreadyExists(EntityId),
    #[error("Entity {0} not found")]
    EntityNotFound(EntityId),
    #[error("Version head not found for entity {0}")]
    VersionHeadNotFound(EntityId),
    #[error("Expected epistemic state {expected:?}, but found {actual:?}")]
    EpistemicConflict {
        expected: EpistemicStatus,
        actual: EpistemicStatus,
    },
    #[error("Invalid epistemic transition: {0}")]
    InvalidEpistemicTransition(String),
}

/// Durable, deterministic mutation operations for the entity kernel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EntityMutation {
    /// Create a new entity with its initial version and provenance.
    Create {
        entity_id: EntityId,
        header: EntityHeader,
        initial_version_id: VersionId,
        provenance_id: ProvenanceId,
        provenance_record: Option<ProvenanceRecord>,
        epistemic_status: EpistemicStatus,
    },
    /// Advance an entity to a new version at committed LSN.
    CreateVersion {
        entity_id: EntityId,
        version_id: VersionId,
        provenance_id: ProvenanceId,
        provenance_record: Option<ProvenanceRecord>,
        epistemic_status: EpistemicStatus,
        lifecycle_status: LifecycleStatus,
        relation_kind: VersionRelation,
        property_row: u32,
        vector_row: u32,
    },
    /// Explicit epistemic state transition with expected-state conflict checking.
    TransitionEpistemic {
        entity_id: EntityId,
        version_id: VersionId,
        expected: EpistemicStatus,
        next: EpistemicStatus,
        evidence: Vec<DurableEvidenceRef>,
        provenance_id: ProvenanceId,
        provenance_record: Option<ProvenanceRecord>,
    },
    /// Tombstone an entity at committed LSN.
    Tombstone {
        entity_id: EntityId,
        version_id: VersionId,
        provenance_id: ProvenanceId,
        provenance_record: Option<ProvenanceRecord>,
    },
}

impl EntityMutation {
    /// Applies this mutation deterministically to an `EntitySegment` at committed `commit_lsn`.
    pub fn apply(self, segment: &EntitySegment, commit_lsn: u64) -> Result<(), MutationApplyError> {
        match self {
            EntityMutation::Create {
                entity_id,
                mut header,
                initial_version_id,
                provenance_id,
                provenance_record,
                epistemic_status,
            } => {
                if segment.arena.id_to_index(entity_id).is_some() {
                    return Err(MutationApplyError::EntityAlreadyExists(entity_id));
                }

                let prov_index = if let Some(rec) = provenance_record {
                    segment.provenance.bind(provenance_id, &rec)
                } else {
                    segment
                        .provenance
                        .id_to_index(provenance_id)
                        .unwrap_or(NULL_ROW_REF)
                };

                let vrow = VersionRow {
                    entity_id,
                    version_id: initial_version_id,
                    valid_from_lsn: commit_lsn,
                    valid_until_lsn: u64::MAX,
                    prev_version_row: NULL_ROW_REF,
                    provenance_row: prov_index,
                    vector_row: header.vector_row,
                    property_row: header.property_row,
                    epistemic_status: epistemic_status as u8,
                    lifecycle_status: LifecycleStatus::Active as u8,
                    relation_kind: VersionRelation::Supersedes as u8,
                    reserved: 0,
                    confidence_q16: 65536,
                };
                let vindex = segment.versions.bind(initial_version_id, vrow);

                header.version_row = vindex;
                header.provenance_row = prov_index;
                header.set_epistemic(epistemic_status);
                segment.arena.bind(entity_id, header);

                Ok(())
            }

            EntityMutation::CreateVersion {
                entity_id,
                version_id,
                provenance_id,
                provenance_record,
                epistemic_status,
                lifecycle_status,
                relation_kind,
                property_row,
                vector_row,
            } => {
                let (entity_index, header) = segment
                    .arena
                    .get_by_id(entity_id)
                    .ok_or(MutationApplyError::EntityNotFound(entity_id))?;

                let prev_vindex = header.version_row;
                if prev_vindex == NULL_ROW_REF {
                    return Err(MutationApplyError::VersionHeadNotFound(entity_id));
                }

                // 1. Close prior version at commit_lsn
                segment.versions.close_version(prev_vindex, commit_lsn);

                // 2. Append new version
                let prov_index = if let Some(rec) = provenance_record {
                    segment.provenance.bind(provenance_id, &rec)
                } else {
                    segment
                        .provenance
                        .id_to_index(provenance_id)
                        .unwrap_or(NULL_ROW_REF)
                };

                let new_vrow = VersionRow {
                    entity_id,
                    version_id,
                    valid_from_lsn: commit_lsn,
                    valid_until_lsn: u64::MAX,
                    prev_version_row: prev_vindex,
                    provenance_row: prov_index,
                    vector_row,
                    property_row,
                    epistemic_status: epistemic_status as u8,
                    lifecycle_status: lifecycle_status as u8,
                    relation_kind: relation_kind as u8,
                    reserved: 0,
                    confidence_q16: 65536,
                };
                let new_vindex = segment.versions.bind(version_id, new_vrow);

                // 3. Atomically update entity header
                segment.arena.publish_version_head(
                    entity_index,
                    new_vindex,
                    prov_index,
                    property_row,
                    vector_row,
                    epistemic_status,
                );

                Ok(())
            }

            EntityMutation::TransitionEpistemic {
                entity_id,
                version_id,
                expected,
                next,
                evidence: _evidence,
                provenance_id,
                provenance_record,
            } => {
                let (entity_index, header) = segment
                    .arena
                    .get_by_id(entity_id)
                    .ok_or(MutationApplyError::EntityNotFound(entity_id))?;

                let prev_vindex = header.version_row;
                if prev_vindex == NULL_ROW_REF {
                    return Err(MutationApplyError::VersionHeadNotFound(entity_id));
                }

                let current_vrow = segment
                    .versions
                    .get_row(prev_vindex)
                    .ok_or(MutationApplyError::VersionHeadNotFound(entity_id))?;

                if current_vrow.epistemic() != expected {
                    return Err(MutationApplyError::EpistemicConflict {
                        expected,
                        actual: current_vrow.epistemic(),
                    });
                }

                validate_epistemic_transition(expected, next)
                    .map_err(|e| MutationApplyError::InvalidEpistemicTransition(e.to_string()))?;

                // Close prior version
                segment.versions.close_version(prev_vindex, commit_lsn);

                // Append new transitioned version
                let prov_index = if let Some(rec) = provenance_record {
                    segment.provenance.bind(provenance_id, &rec)
                } else {
                    segment
                        .provenance
                        .id_to_index(provenance_id)
                        .unwrap_or(NULL_ROW_REF)
                };

                let new_vrow = VersionRow {
                    entity_id,
                    version_id,
                    valid_from_lsn: commit_lsn,
                    valid_until_lsn: u64::MAX,
                    prev_version_row: prev_vindex,
                    provenance_row: prov_index,
                    vector_row: current_vrow.vector_row,
                    property_row: current_vrow.property_row,
                    epistemic_status: next as u8,
                    lifecycle_status: current_vrow.lifecycle_status,
                    relation_kind: VersionRelation::Refines as u8,
                    reserved: 0,
                    confidence_q16: current_vrow.confidence_q16,
                };
                let new_vindex = segment.versions.bind(version_id, new_vrow);

                segment.arena.publish_version_head(
                    entity_index,
                    new_vindex,
                    prov_index,
                    current_vrow.property_row,
                    current_vrow.vector_row,
                    next,
                );

                Ok(())
            }

            EntityMutation::Tombstone {
                entity_id,
                version_id,
                provenance_id,
                provenance_record,
            } => {
                let (entity_index, header) = segment
                    .arena
                    .get_by_id(entity_id)
                    .ok_or(MutationApplyError::EntityNotFound(entity_id))?;

                let prev_vindex = header.version_row;
                if prev_vindex != NULL_ROW_REF {
                    segment.versions.close_version(prev_vindex, commit_lsn);
                }

                let prov_index = if let Some(rec) = provenance_record {
                    segment.provenance.bind(provenance_id, &rec)
                } else {
                    segment
                        .provenance
                        .id_to_index(provenance_id)
                        .unwrap_or(NULL_ROW_REF)
                };

                let new_vrow = VersionRow {
                    entity_id,
                    version_id,
                    valid_from_lsn: commit_lsn,
                    valid_until_lsn: u64::MAX,
                    prev_version_row: prev_vindex,
                    provenance_row: prov_index,
                    vector_row: NULL_ROW_REF,
                    property_row: NULL_ROW_REF,
                    epistemic_status: header.epistemic() as u8,
                    lifecycle_status: LifecycleStatus::Tombstoned as u8,
                    relation_kind: VersionRelation::Supersedes as u8,
                    reserved: 0,
                    confidence_q16: 0,
                };
                let new_vindex = segment.versions.bind(version_id, new_vrow);

                segment.arena.delete(entity_index);
                segment.arena.publish_version_head(
                    entity_index,
                    new_vindex,
                    prov_index,
                    NULL_ROW_REF,
                    NULL_ROW_REF,
                    header.epistemic(),
                );

                Ok(())
            }
        }
    }
}
