/* holosphere/src/entity/read.rs */
//!▫~•◦-------------------------------‣
//! # Unified Read Snapshot & Temporal Resolution Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the primary query interface for resolving entities, histories,
//! provenance, and epistemic states as of a pinned committed LSN.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::entity::id::{EntityId, NULL_ROW_REF, VersionId};
use crate::entity::provenance::ProvenanceRecord;
use crate::entity::segment::EntitySegment;
use crate::entity::status::{EpistemicStatus, LifecycleStatus};
use crate::entity::version::VersionRelation;
use std::sync::Arc;

/// High-level resolved view of an entity version snapshot.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedEntityVersion {
    pub entity_id: EntityId,
    pub version_id: VersionId,
    pub valid_from_lsn: u64,
    pub valid_until_lsn: Option<u64>,
    pub epistemic_status: EpistemicStatus,
    pub lifecycle_status: LifecycleStatus,
    pub relation_kind: VersionRelation,
    pub property_row: u32,
    pub vector_row: u32,
    pub provenance: Option<ProvenanceRecord>,
}

/// Immutable, pinned read snapshot representing an exact point-in-time view
/// of the HoloSphere entity universe.
pub struct EntityReadSnapshot {
    pub lsn: u64,
    pub segment: Arc<EntitySegment>,
}

impl EntityReadSnapshot {
    /// Creates a new pinned read snapshot at committed LSN `lsn`.
    pub fn new(lsn: u64, segment: Arc<EntitySegment>) -> Self {
        Self { lsn, segment }
    }

    /// Primary internal resolution primitive: resolves an entity's version at LSN `query_lsn`.
    pub fn resolve_version_at(
        &self,
        entity_id: EntityId,
        query_lsn: u64,
    ) -> Option<ResolvedEntityVersion> {
        let (_entity_index, header) = self.segment.arena.get_by_id(entity_id)?;
        if header.version_row == NULL_ROW_REF {
            return None;
        }

        let version_row_idx = self
            .segment
            .versions
            .find_as_of(header.version_row, query_lsn)?;
        let row = self.segment.versions.get_row(version_row_idx)?;

        let provenance = if row.provenance_row != NULL_ROW_REF {
            self.segment.provenance.resolve_record(row.provenance_row)
        } else {
            None
        };

        Some(ResolvedEntityVersion {
            entity_id,
            version_id: row.version_id,
            valid_from_lsn: row.valid_from_lsn,
            valid_until_lsn: if row.valid_until_lsn == u64::MAX {
                None
            } else {
                Some(row.valid_until_lsn)
            },
            epistemic_status: row.epistemic(),
            lifecycle_status: row.lifecycle(),
            relation_kind: match row.relation_kind {
                0 => VersionRelation::Supersedes,
                1 => VersionRelation::Refines,
                2 => VersionRelation::Reinforces,
                _ => VersionRelation::Supersedes,
            },
            property_row: row.property_row,
            vector_row: row.vector_row,
            provenance,
        })
    }

    /// Resolves the current active state of an entity as of this snapshot's pinned LSN.
    pub fn current(&self, entity_id: EntityId) -> Option<ResolvedEntityVersion> {
        self.resolve_version_at(entity_id, self.lsn)
    }

    /// Resolves the historical state of an entity AS OF a specific past LSN.
    pub fn as_of(&self, entity_id: EntityId, as_of_lsn: u64) -> Option<ResolvedEntityVersion> {
        let clamped_lsn = as_of_lsn.min(self.lsn);
        self.resolve_version_at(entity_id, clamped_lsn)
    }

    /// Retrieves the complete chronological history of versions for an entity (from root to current).
    pub fn history(&self, entity_id: EntityId) -> Vec<ResolvedEntityVersion> {
        let (_, header) = match self.segment.arena.get_by_id(entity_id) {
            Some(h) => h,
            None => return Vec::new(),
        };

        if header.version_row == NULL_ROW_REF {
            return Vec::new();
        }

        let raw_history = self.segment.versions.history_of(header.version_row);
        let mut results = Vec::with_capacity(raw_history.len());

        // Reverse so that index 0 is root (oldest) and last is head (newest).
        for row in raw_history.into_iter().rev() {
            let provenance = if row.provenance_row != NULL_ROW_REF {
                self.segment.provenance.resolve_record(row.provenance_row)
            } else {
                None
            };

            results.push(ResolvedEntityVersion {
                entity_id,
                version_id: row.version_id,
                valid_from_lsn: row.valid_from_lsn,
                valid_until_lsn: if row.valid_until_lsn == u64::MAX {
                    None
                } else {
                    Some(row.valid_until_lsn)
                },
                epistemic_status: row.epistemic(),
                lifecycle_status: row.lifecycle(),
                relation_kind: match row.relation_kind {
                    0 => VersionRelation::Supersedes,
                    1 => VersionRelation::Refines,
                    2 => VersionRelation::Reinforces,
                    _ => VersionRelation::Supersedes,
                },
                property_row: row.property_row,
                vector_row: row.vector_row,
                provenance,
            });
        }

        results
    }

    /// Returns the active provenance for an entity at a given LSN.
    pub fn provenance(&self, entity_id: EntityId, lsn: u64) -> Option<ProvenanceRecord> {
        let version = self.as_of(entity_id, lsn)?;
        version.provenance
    }

    /// Returns the epistemic status of an entity at a given LSN.
    pub fn epistemic_state(&self, entity_id: EntityId, lsn: u64) -> Option<EpistemicStatus> {
        let version = self.as_of(entity_id, lsn)?;
        Some(version.epistemic_status)
    }
}
