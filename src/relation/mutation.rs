/* holosphere/src/relation/mutation.rs */
//!▫~•◦-------------------------------‣
//! # Replicated Hypergraph Relation State Machine Mutations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Durable mutations applied to the relation state machine during Raft replication.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::entity::id::{DurableEvidenceRef, NULL_ROW_REF, ProvenanceId};
use crate::entity::provenance::ProvenanceRecord;
use crate::entity::segment::EntitySegment;
use crate::entity::status::{EpistemicStatus, LifecycleStatus};
use crate::relation::binding::SegmentRoleBinding;
use crate::relation::header::{RELATION_FLAG_LIVE, RelationHeader};
use crate::relation::id::{DurableRoleBinding, RelationId, RelationTypeId, RelationVersionId};
use crate::relation::read::RelationSegment;
use crate::relation::schema::{RelationType, RelationTypeState};
use crate::relation::version::RelationVersionRow;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum RelationMutationError {
    #[error("Relation type {type_id} not found in catalog")]
    TypeNotFound { type_id: RelationTypeId },
    #[error("Schema validation failed: {0}")]
    SchemaValidation(String),
    #[error("Relation {relation_id} already exists")]
    RelationAlreadyExists { relation_id: RelationId },
    #[error("Relation {relation_id} not found")]
    RelationNotFound { relation_id: RelationId },
    #[error("Entity {entity_id} in role binding not found in current entity arena")]
    EntityNotFound { entity_id: u64 },
    #[error("Epistemic state transition error: {0}")]
    EpistemicTransition(String),
    #[error("Expected relation type state {expected:?} but found {actual:?}")]
    TypeStateConflict {
        expected: RelationTypeState,
        actual: RelationTypeState,
    },
}

/// Durable command payload replicated via Raft log entries.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RelationMutation {
    ProposeType {
        type_id: RelationTypeId,
        schema: RelationType,
        provenance_id: ProvenanceId,
    },
    AdmitType {
        type_id: RelationTypeId,
        expected_state: RelationTypeState,
    },
    DeprecateType {
        type_id: RelationTypeId,
        expected_state: RelationTypeState,
    },
    CreateRelation {
        relation_id: RelationId,
        relation_type_id: RelationTypeId,
        bindings: Vec<DurableRoleBinding>,
        provenance_id: ProvenanceId,
        provenance_record: Option<ProvenanceRecord>,
        epistemic_status: EpistemicStatus,
    },
    CreateRelationVersion {
        relation_id: RelationId,
        version_id: RelationVersionId,
        provenance_id: ProvenanceId,
        provenance_record: Option<ProvenanceRecord>,
        epistemic_status: EpistemicStatus,
    },
    TransitionEpistemic {
        relation_id: RelationId,
        version_id: RelationVersionId,
        expected: EpistemicStatus,
        next: EpistemicStatus,
        evidence: Vec<DurableEvidenceRef>,
        provenance_id: ProvenanceId,
        provenance_record: Option<ProvenanceRecord>,
    },
    Tombstone {
        relation_id: RelationId,
    },
}

impl RelationMutation {
    pub fn apply(
        &self,
        rel_seg: &RelationSegment,
        ent_seg: &EntitySegment,
        commit_lsn: u64,
    ) -> Result<(), RelationMutationError> {
        match self {
            RelationMutation::ProposeType { schema, .. } => {
                let mut schema_clone = schema.clone();
                schema_clone.state = RelationTypeState::Proposed;
                rel_seg.register_type(schema_clone);
                Ok(())
            }
            RelationMutation::AdmitType {
                type_id,
                expected_state,
            } => {
                let mut types = rel_seg.types.write();
                if let Some(t) = types.iter_mut().find(|t| t.id == *type_id) {
                    if t.state != *expected_state {
                        return Err(RelationMutationError::TypeStateConflict {
                            expected: *expected_state,
                            actual: t.state,
                        });
                    }
                    t.state = RelationTypeState::Admitted;
                    Ok(())
                } else {
                    Err(RelationMutationError::TypeNotFound { type_id: *type_id })
                }
            }
            RelationMutation::DeprecateType {
                type_id,
                expected_state,
            } => {
                let mut types = rel_seg.types.write();
                if let Some(t) = types.iter_mut().find(|t| t.id == *type_id) {
                    if t.state != *expected_state {
                        return Err(RelationMutationError::TypeStateConflict {
                            expected: *expected_state,
                            actual: t.state,
                        });
                    }
                    t.state = RelationTypeState::Deprecated;
                    Ok(())
                } else {
                    Err(RelationMutationError::TypeNotFound { type_id: *type_id })
                }
            }
            RelationMutation::CreateRelation {
                relation_id,
                relation_type_id,
                bindings,
                provenance_id,
                provenance_record,
                epistemic_status,
            } => {
                if rel_seg.arena.get_by_id(*relation_id).is_some() {
                    return Err(RelationMutationError::RelationAlreadyExists {
                        relation_id: *relation_id,
                    });
                }

                let schema = rel_seg.get_type_schema(*relation_type_id).ok_or(
                    RelationMutationError::TypeNotFound {
                        type_id: *relation_type_id,
                    },
                )?;

                schema
                    .validate_bindings(bindings)
                    .map_err(|e| RelationMutationError::SchemaValidation(e.to_string()))?;

                // Map durable EntityId -> localized EntityIndex
                let mut segment_bindings = Vec::with_capacity(bindings.len());
                for b in bindings {
                    if let Some((ent_idx, _)) = ent_seg.arena.get_by_id(b.entity_id) {
                        segment_bindings.push(SegmentRoleBinding {
                            entity: ent_idx,
                            role_id: b.role_id,
                            flags: 0,
                        });
                    } else {
                        return Err(RelationMutationError::EntityNotFound {
                            entity_id: b.entity_id,
                        });
                    }
                }

                // Register provenance if provided
                let prov_row = if let Some(rec) = provenance_record {
                    let (_, row) = ent_seg.provenance.append(rec);
                    row
                } else {
                    ent_seg
                        .provenance
                        .id_to_index(*provenance_id)
                        .unwrap_or(NULL_ROW_REF)
                };

                let vrow = RelationVersionRow {
                    relation_id: *relation_id,
                    version_id: 1,
                    valid_from_lsn: commit_lsn,
                    valid_until_lsn: u64::MAX,
                    prev_version_row: NULL_ROW_REF,
                    provenance_row: prov_row,
                    epistemic_status: *epistemic_status as u8,
                    lifecycle_status: LifecycleStatus::Active as u8,
                    reserved: [0u8; 14],
                };

                let (_vid, vrow_idx) = rel_seg.versions.append(vrow);

                let header = RelationHeader {
                    relation_type_id: *relation_type_id,
                    binding_start: 0,
                    version_row: vrow_idx,
                    provenance_row: prov_row,
                    binding_len: segment_bindings.len() as u16,
                    schema_version: schema.schema_version,
                    epistemic_status: *epistemic_status as u8,
                    lifecycle_status: LifecycleStatus::Active as u8,
                    flags: RELATION_FLAG_LIVE,
                    reserved: [0u8; 8],
                };

                let rel_idx = rel_seg.arena.bind(*relation_id, header, &segment_bindings);

                // Populate incidence index
                for b in &segment_bindings {
                    rel_seg
                        .incidence
                        .insert(*relation_type_id, b.role_id, b.entity, rel_idx);
                }

                Ok(())
            }
            RelationMutation::CreateRelationVersion {
                relation_id,
                version_id,
                provenance_id,
                provenance_record,
                epistemic_status,
            } => {
                let (rel_idx, mut header) = rel_seg.arena.get_by_id(*relation_id).ok_or(
                    RelationMutationError::RelationNotFound {
                        relation_id: *relation_id,
                    },
                )?;

                let old_vrow_idx = header.version_row;
                if old_vrow_idx != NULL_ROW_REF {
                    rel_seg.versions.close_version(old_vrow_idx, commit_lsn);
                }

                let prov_row = if let Some(rec) = provenance_record {
                    let (_, row) = ent_seg.provenance.append(rec);
                    row
                } else {
                    ent_seg
                        .provenance
                        .id_to_index(*provenance_id)
                        .unwrap_or(NULL_ROW_REF)
                };

                let new_vrow = RelationVersionRow {
                    relation_id: *relation_id,
                    version_id: *version_id,
                    valid_from_lsn: commit_lsn,
                    valid_until_lsn: u64::MAX,
                    prev_version_row: old_vrow_idx,
                    provenance_row: prov_row,
                    epistemic_status: *epistemic_status as u8,
                    lifecycle_status: LifecycleStatus::Active as u8,
                    reserved: [0u8; 14],
                };

                let vrow_idx = rel_seg.versions.bind(*version_id, new_vrow);
                header.version_row = vrow_idx;
                header.provenance_row = prov_row;
                header.set_epistemic(*epistemic_status);
                rel_seg.arena.update_header(rel_idx, header);

                Ok(())
            }
            RelationMutation::TransitionEpistemic {
                relation_id,
                version_id,
                expected,
                next,
                evidence,
                provenance_id,
                provenance_record,
            } => {
                let (rel_idx, mut header) = rel_seg.arena.get_by_id(*relation_id).ok_or(
                    RelationMutationError::RelationNotFound {
                        relation_id: *relation_id,
                    },
                )?;

                if header.epistemic() != *expected {
                    return Err(RelationMutationError::EpistemicTransition(format!(
                        "Expected {:?} but relation has {:?}",
                        expected,
                        header.epistemic()
                    )));
                }

                crate::entity::epistemic::validate_epistemic_transition(*expected, *next)
                    .map_err(|e| RelationMutationError::EpistemicTransition(e.to_string()))?;

                let old_vrow_idx = header.version_row;
                if old_vrow_idx != NULL_ROW_REF {
                    rel_seg.versions.close_version(old_vrow_idx, commit_lsn);
                }

                let prov_row = if let Some(rec) = provenance_record {
                    let mut rec_clone = rec.clone();
                    rec_clone.evidence.extend_from_slice(evidence);
                    let (_, row) = ent_seg.provenance.append(&rec_clone);
                    row
                } else {
                    ent_seg
                        .provenance
                        .id_to_index(*provenance_id)
                        .unwrap_or(NULL_ROW_REF)
                };

                let new_vrow = RelationVersionRow {
                    relation_id: *relation_id,
                    version_id: *version_id,
                    valid_from_lsn: commit_lsn,
                    valid_until_lsn: u64::MAX,
                    prev_version_row: old_vrow_idx,
                    provenance_row: prov_row,
                    epistemic_status: *next as u8,
                    lifecycle_status: LifecycleStatus::Active as u8,
                    reserved: [0u8; 14],
                };

                let vrow_idx = rel_seg.versions.bind(*version_id, new_vrow);
                header.version_row = vrow_idx;
                header.provenance_row = prov_row;
                header.set_epistemic(*next);
                rel_seg.arena.update_header(rel_idx, header);

                Ok(())
            }
            RelationMutation::Tombstone { relation_id } => {
                if !rel_seg.arena.delete(*relation_id) {
                    return Err(RelationMutationError::RelationNotFound {
                        relation_id: *relation_id,
                    });
                }
                Ok(())
            }
        }
    }
}
