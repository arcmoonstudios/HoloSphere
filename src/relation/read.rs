/* holosphere/src/relation/read.rs */
//!▫~•◦-------------------------------‣
//! # Pinned Relation Read Snapshots & Point-In-Time Resolution
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides point-in-time isolation and historical resolution for relations
//! obeying the committed LSN invariant.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::sync::Arc;

use crate::entity::id::NULL_ROW_REF;
use crate::entity::provenance::ProvenanceRecord;
use crate::entity::read::EntityReadSnapshot;
use crate::entity::status::{EpistemicStatus, LifecycleStatus};
use crate::relation::arena::RelationArena;
use crate::relation::id::{DurableRoleBinding, RelationId, RelationTypeId, RelationVersionId};
use crate::relation::incidence::IncidenceIndex;
use crate::relation::projection::BinaryProjectionCache;
use crate::relation::schema::RelationType;
use crate::relation::version::RelationVersionTable;

/// High-level resolved view of a relation at a specific point in time.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedRelationVersion {
    pub relation_id: RelationId,
    pub version_id: RelationVersionId,
    pub type_id: RelationTypeId,
    pub schema_version: u16,
    pub valid_from_lsn: u64,
    pub valid_until_lsn: Option<u64>,
    pub epistemic_status: EpistemicStatus,
    pub lifecycle_status: LifecycleStatus,
    pub bindings: Vec<DurableRoleBinding>,
    pub provenance: Option<ProvenanceRecord>,
}

/// Generation-scoped container holding all physical relation structures.
pub struct RelationSegment {
    pub generation_id: u64,
    pub arena: Arc<RelationArena>,
    pub versions: Arc<RelationVersionTable>,
    pub incidence: Arc<IncidenceIndex>,
    pub projection_cache: Arc<BinaryProjectionCache>,
    pub types: RwLock<Vec<RelationType>>,
}

impl RelationSegment {
    pub fn new(generation_id: u64, start_relation_id: RelationId) -> Self {
        Self {
            generation_id,
            arena: Arc::new(RelationArena::new(start_relation_id)),
            versions: Arc::new(RelationVersionTable::new(1)),
            incidence: Arc::new(IncidenceIndex::new()),
            projection_cache: Arc::new(BinaryProjectionCache::new()),
            types: RwLock::new(Vec::new()),
        }
    }

    pub fn read_snapshot(self: &Arc<Self>, lsn: u64) -> RelationReadSnapshot {
        RelationReadSnapshot {
            lsn,
            segment: Arc::clone(self),
        }
    }

    pub fn register_type(&self, rtype: RelationType) -> RelationTypeId {
        let mut types = self.types.write();
        let id = rtype.id;
        types.push(rtype);
        id
    }

    /// Checks whether a governed evolved type can be projected into this
    /// canonical catalog without colliding with an unrelated local schema.
    pub fn prevalidate_evolved_type(&self, rtype: &RelationType) -> Result<(), String> {
        if let Some(existing) = self
            .types
            .read()
            .iter()
            .find(|existing| existing.id == rtype.id)
        {
            if existing.structural_fingerprint != rtype.structural_fingerprint {
                return Err(format!(
                    "evolved relation type {} collides with a different structural schema",
                    rtype.id
                ));
            }
        }
        Ok(())
    }

    /// Infallible after `prevalidate_evolved_type`; updates only the lifecycle
    /// and evidence metadata of the same structural definition.
    pub fn synchronize_evolved_type(&self, rtype: RelationType) {
        let mut types = self.types.write();
        if let Some(existing) = types.iter_mut().find(|existing| existing.id == rtype.id) {
            debug_assert_eq!(
                existing.structural_fingerprint,
                rtype.structural_fingerprint
            );
            existing.state = rtype.state;
            existing.provenance_id = rtype.provenance_id;
            existing.schema_version = rtype.schema_version;
        } else {
            types.push(rtype);
        }
    }

    pub fn get_type_schema(&self, type_id: RelationTypeId) -> Option<RelationType> {
        let types = self.types.read();
        types.iter().find(|t| t.id == type_id).cloned()
    }

    /// Performs physical compaction, re-indexing physical role bindings against the compacted EntityArena.
    pub fn compact(
        &self,
        new_generation_id: u64,
        old_entity_segment: &crate::entity::segment::EntitySegment,
        compacted_entity_segment: &crate::entity::segment::EntitySegment,
    ) -> Arc<Self> {
        let compacted = Arc::new(Self::new(new_generation_id, 1));
        *compacted.types.write() = self.types.read().clone();

        // 1. Copy over version table
        let (v_rows, v_ids) = self.versions.snapshot_data();
        for (i, &vrow) in v_rows.iter().enumerate() {
            if i < v_ids.len() {
                compacted.versions.bind(v_ids[i], vrow);
            }
        }

        // 2. Compact live relations: remap EntityIndex in SegmentRoleBindings
        for rel_id in self.arena.live_ids() {
            if let Some((_old_rel_idx, header)) = self.arena.get_by_id(rel_id) {
                let old_bindings = self.arena.get_bindings(&header);
                let mut new_bindings = Vec::with_capacity(old_bindings.len());

                for b in old_bindings {
                    if let Some(ent_id) = old_entity_segment.arena.index_to_id(b.entity) {
                        if let Some((new_ent_idx, _)) =
                            compacted_entity_segment.arena.get_by_id(ent_id)
                        {
                            new_bindings.push(crate::relation::binding::SegmentRoleBinding {
                                entity: new_ent_idx,
                                role_id: b.role_id,
                                flags: b.flags,
                            });
                        }
                    }
                }

                let new_rel_idx = compacted.arena.bind(rel_id, header, &new_bindings);

                // Populate incidence postings
                for b in &new_bindings {
                    compacted.incidence.insert(
                        header.relation_type_id,
                        b.role_id,
                        b.entity,
                        new_rel_idx,
                    );
                }
            }
        }

        compacted
    }
}

/// Immutable point-in-time relation snapshot pinned at committed LSN `lsn`.
#[derive(Clone)]
pub struct RelationReadSnapshot {
    pub lsn: u64,
    pub segment: Arc<RelationSegment>,
}

impl RelationReadSnapshot {
    pub fn resolve_relation_at(
        &self,
        relation_id: RelationId,
        query_lsn: u64,
        ent_snap: &EntityReadSnapshot,
    ) -> Option<ResolvedRelationVersion> {
        let (_rel_idx, header) = self.segment.arena.get_by_id(relation_id)?;
        if header.version_row == NULL_ROW_REF {
            return None;
        }

        let mut curr_row_idx = header.version_row;
        let mut target_vrow = None;

        while curr_row_idx != NULL_ROW_REF {
            if let Some(vrow) = self.segment.versions.get_row(curr_row_idx) {
                if vrow.visible_at(query_lsn) {
                    target_vrow = Some(vrow);
                    break;
                }
                curr_row_idx = vrow.prev_version_row;
            } else {
                break;
            }
        }

        let vrow = target_vrow?;
        let physical_bindings = self.segment.arena.get_bindings(&header);

        // Convert physical SegmentRoleBinding (EntityIndex) -> DurableRoleBinding (EntityId)
        let mut durable_bindings = Vec::with_capacity(physical_bindings.len());
        for pb in physical_bindings {
            if let Some(entity_id) = ent_snap.segment.arena.index_to_id(pb.entity) {
                durable_bindings.push(DurableRoleBinding {
                    entity_id,
                    role_id: pb.role_id,
                });
            }
        }

        let provenance = if vrow.provenance_row != NULL_ROW_REF {
            ent_snap
                .segment
                .provenance
                .resolve_record(vrow.provenance_row)
        } else {
            None
        };

        Some(ResolvedRelationVersion {
            relation_id,
            version_id: vrow.version_id,
            type_id: header.relation_type_id,
            schema_version: header.schema_version,
            valid_from_lsn: vrow.valid_from_lsn,
            valid_until_lsn: if vrow.valid_until_lsn == u64::MAX {
                None
            } else {
                Some(vrow.valid_until_lsn)
            },
            epistemic_status: vrow.epistemic(),
            lifecycle_status: vrow.lifecycle(),
            bindings: durable_bindings,
            provenance,
        })
    }

    pub fn current(
        &self,
        relation_id: RelationId,
        ent_snap: &EntityReadSnapshot,
    ) -> Option<ResolvedRelationVersion> {
        self.resolve_relation_at(relation_id, self.lsn, ent_snap)
    }

    pub fn as_of(
        &self,
        relation_id: RelationId,
        lsn: u64,
        ent_snap: &EntityReadSnapshot,
    ) -> Option<ResolvedRelationVersion> {
        self.resolve_relation_at(relation_id, lsn, ent_snap)
    }
}
