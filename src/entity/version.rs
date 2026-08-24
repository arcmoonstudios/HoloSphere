/* holosphere/src/entity/version.rs */
//!▫~•◦-------------------------------‣
//! # Version History & Canonical Temporal Lineage
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Models temporal entity lineage and version-to-version evolution.
//!
//! ## Invariant Guarantees
//! - Half-open interval `[valid_from, valid_until)`: a version is visible at LSN `s`
//!   iff `valid_from <= s < valid_until`.
//! - Version updates are atomic: previous head is closed at LSN `L`, new version
//!   is opened at LSN `L`, and EntityHeader version pointer is updated in one publication step.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entity::id::{EntityId, NULL_ROW_REF, ProvenanceId, VersionId, VersionIndex};
use crate::entity::status::{EpistemicStatus, LifecycleStatus};

/// Version-to-version relation ontology.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VersionRelation {
    /// Newer version directly supersedes the previous version in the entity's lineage.
    Supersedes = 0,
    /// Version refines or adds detail to the target version without replacing it completely.
    Refines = 1,
    /// Version reinforces the validity/evidence of the prior version.
    Reinforces = 2,
}

/// Exactly-56-byte, deterministic, padding-free version row.
///
/// Layout (56 bytes, 8-byte aligned):
/// ```text
/// offset 0  — entity_id         : u64 (8 bytes) ← canonical durable entity ID
/// offset 8  — version_id        : u64 (8 bytes) ← unique durable version ID
/// offset 16 — valid_from_lsn    : u64 (8 bytes) ← commit LSN from which this version is active (inclusive)
/// offset 24 — valid_until_lsn   : u64 (8 bytes) ← commit LSN when superseded (exclusive; u64::MAX if current)
/// offset 32 — prev_version_row  : u32 (4 bytes) ← row index to predecessor version (NULL_ROW_REF if root)
/// offset 36 — provenance_row    : u32 (4 bytes) ← row index in ProvenanceArena
/// offset 40 — vector_row        : u32 (4 bytes) ← vector embedding row for this version
/// offset 44 — property_row      : u32 (4 bytes) ← property snapshot row for this version
/// offset 48 — epistemic_status  : u8  (1 byte)  ← EpistemicStatus at this version
/// offset 49 — lifecycle_status  : u8  (1 byte)  ← LifecycleStatus
/// offset 50 — relation_kind     : u8  (1 byte)  ← VersionRelation enum
/// offset 51 — reserved          : u8  (1 byte)
/// offset 52 — confidence_q16    : u32 (4 bytes) ← confidence in Q16 fixed-point
/// total      56 bytes, zero padding
/// ```
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct VersionRow {
    pub entity_id: u64,
    pub version_id: u64,
    pub valid_from_lsn: u64,
    pub valid_until_lsn: u64,

    pub prev_version_row: u32,
    pub provenance_row: u32,
    pub vector_row: u32,
    pub property_row: u32,

    pub epistemic_status: u8,
    pub lifecycle_status: u8,
    pub relation_kind: u8,
    pub reserved: u8,
    pub confidence_q16: u32,
}

const _: () = assert!(std::mem::size_of::<VersionRow>() == 56);
const _: () = assert!(std::mem::align_of::<VersionRow>() == 8);

impl Default for VersionRow {
    fn default() -> Self {
        Self {
            entity_id: 0,
            version_id: 0,
            valid_from_lsn: 0,
            valid_until_lsn: u64::MAX,
            prev_version_row: NULL_ROW_REF,
            provenance_row: NULL_ROW_REF,
            vector_row: NULL_ROW_REF,
            property_row: NULL_ROW_REF,
            epistemic_status: EpistemicStatus::Observed as u8,
            lifecycle_status: LifecycleStatus::Active as u8,
            relation_kind: VersionRelation::Supersedes as u8,
            reserved: 0,
            confidence_q16: 65536,
        }
    }
}

impl VersionRow {
    /// Canonical temporal predicate: version is visible at committed LSN `s`
    /// iff `valid_from <= s` and `s < valid_until` (half-open interval `[from, until)`).
    #[inline(always)]
    pub fn visible_at(&self, lsn: u64) -> bool {
        self.valid_from_lsn <= lsn
            && (self.valid_until_lsn == u64::MAX || lsn < self.valid_until_lsn)
    }

    #[inline(always)]
    pub fn epistemic(&self) -> EpistemicStatus {
        match self.epistemic_status {
            0 => EpistemicStatus::Observed,
            1 => EpistemicStatus::Asserted,
            2 => EpistemicStatus::Inferred,
            3 => EpistemicStatus::Provisional,
            4 => EpistemicStatus::Contradicted,
            _ => EpistemicStatus::Observed,
        }
    }

    #[inline(always)]
    pub fn lifecycle(&self) -> LifecycleStatus {
        match self.lifecycle_status {
            0 => LifecycleStatus::Active,
            1 => LifecycleStatus::Superseded,
            2 => LifecycleStatus::Tombstoned,
            _ => LifecycleStatus::Active,
        }
    }
}

/// Durable representation of an entity version used across replicated Raft logs and snapshots.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurableEntityVersion {
    pub version_id: VersionId,
    pub entity_id: EntityId,
    pub valid_from_lsn: u64,
    pub valid_until_lsn: Option<u64>,
    pub epistemic_status: EpistemicStatus,
    pub lifecycle_status: LifecycleStatus,
    pub provenance_id: ProvenanceId,
    pub relation_kind: VersionRelation,
    pub property_row: u32,
    pub vector_row: u32,
}

/// Append-only, thread-safe version history table.
pub struct VersionTable {
    rows: RwLock<Vec<VersionRow>>,
    id_to_index: RwLock<HashMap<VersionId, VersionIndex>>,
    index_to_id: RwLock<Vec<VersionId>>,
    next_id: AtomicU64,
}

impl Default for VersionTable {
    fn default() -> Self {
        Self::new(1)
    }
}

impl VersionTable {
    pub fn new(start_id: u64) -> Self {
        Self {
            rows: RwLock::new(Vec::new()),
            id_to_index: RwLock::new(HashMap::new()),
            index_to_id: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(start_id),
        }
    }

    /// Appends a new version row, generating a new `VersionId`.
    pub fn append(&self, mut row: VersionRow) -> (VersionId, VersionIndex) {
        let version_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        row.version_id = version_id;
        let version_index = self.bind(version_id, row);
        (version_id, version_index)
    }

    /// Binds an existing durable `VersionId` to a new generation row.
    pub fn bind(&self, version_id: VersionId, row: VersionRow) -> VersionIndex {
        let mut rows = self.rows.write();
        let mut id_map = self.id_to_index.write();
        let mut idx_vec = self.index_to_id.write();

        let index = rows.len() as VersionIndex;
        rows.push(row);
        idx_vec.push(version_id);
        id_map.insert(version_id, index);

        index
    }

    /// Resolves `VersionId` to generation `VersionIndex`.
    #[inline]
    pub fn id_to_index(&self, id: VersionId) -> Option<VersionIndex> {
        self.id_to_index.read().get(&id).copied()
    }

    /// Resolves generation `VersionIndex` to durable `VersionId`.
    #[inline]
    pub fn index_to_id(&self, index: VersionIndex) -> Option<VersionId> {
        self.index_to_id.read().get(index as usize).copied()
    }

    /// Retrieves a version row by index.
    pub fn get_row(&self, row_index: VersionIndex) -> Option<VersionRow> {
        let rows = self.rows.read();
        rows.get(row_index as usize).copied()
    }

    /// Closes a prior version row at committed LSN `until_lsn` (atomic transition step).
    pub fn close_version(&self, row_index: VersionIndex, until_lsn: u64) -> bool {
        let mut rows = self.rows.write();
        if let Some(row) = rows.get_mut(row_index as usize) {
            row.valid_until_lsn = until_lsn;
            row.lifecycle_status = LifecycleStatus::Superseded as u8;
            true
        } else {
            false
        }
    }

    /// Resolves the valid version row index for an entity as of a given LSN.
    /// Traverses the backward ancestor chain starting from `head_row`.
    pub fn find_as_of(&self, head_row: u32, as_of_lsn: u64) -> Option<u32> {
        let mut curr = head_row;
        let rows = self.rows.read();

        while curr != NULL_ROW_REF {
            let row = rows.get(curr as usize)?;
            if row.visible_at(as_of_lsn) {
                return Some(curr);
            }
            curr = row.prev_version_row;
        }

        None
    }

    /// Retrieves the entire ordered lineage of version rows (from head back to root).
    pub fn history_of(&self, head_row: u32) -> Vec<VersionRow> {
        let mut result = Vec::new();
        let mut curr = head_row;
        let rows = self.rows.read();

        while curr != NULL_ROW_REF {
            if let Some(&row) = rows.get(curr as usize) {
                result.push(row);
                curr = row.prev_version_row;
            } else {
                break;
            }
        }

        result
    }

    /// Returns total row count.
    pub fn len(&self) -> usize {
        self.rows.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Snapshot rows and ID mappings for serialization.
    pub fn snapshot_data(&self) -> (Vec<VersionRow>, Vec<VersionId>) {
        (self.rows.read().clone(), self.index_to_id.read().clone())
    }
}
