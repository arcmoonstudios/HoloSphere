/* holosphere/src/relation/version.rs */
//!▫~•◦-------------------------------‣
//! # Temporal Relation Versioning & Resolution
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the physical 56-byte Pod `RelationVersionRow` and `RelationVersionTable`
//! implementing exact point-in-time temporal visibility [valid_from, valid_until).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entity::id::NULL_ROW_REF;
use crate::entity::status::{EpistemicStatus, LifecycleStatus};
use crate::relation::id::{RelationId, RelationVersionId};

/// Fixed 56-byte Pod relation version record.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable, Serialize, Deserialize)]
pub struct RelationVersionRow {
    pub relation_id: RelationId,
    pub version_id: RelationVersionId,
    pub valid_from_lsn: u64,
    pub valid_until_lsn: u64,
    pub prev_version_row: u32,
    pub provenance_row: u32,
    pub epistemic_status: u8,
    pub lifecycle_status: u8,
    pub reserved: [u8; 14],
}

const _: () = assert!(std::mem::size_of::<RelationVersionRow>() == 56);
const _: () = assert!(std::mem::align_of::<RelationVersionRow>() == 8);

impl Default for RelationVersionRow {
    fn default() -> Self {
        Self {
            relation_id: 0,
            version_id: 0,
            valid_from_lsn: 0,
            valid_until_lsn: u64::MAX,
            prev_version_row: NULL_ROW_REF,
            provenance_row: NULL_ROW_REF,
            epistemic_status: EpistemicStatus::Observed as u8,
            lifecycle_status: LifecycleStatus::Active as u8,
            reserved: [0u8; 14],
        }
    }
}

impl RelationVersionRow {
    #[inline(always)]
    pub fn visible_at(&self, lsn: u64) -> bool {
        self.valid_from_lsn <= lsn
            && (self.valid_until_lsn == u64::MAX || lsn < self.valid_until_lsn)
    }

    #[inline(always)]
    pub fn is_active_head(&self) -> bool {
        self.valid_until_lsn == u64::MAX
    }

    #[inline(always)]
    pub fn epistemic(&self) -> EpistemicStatus {
        EpistemicStatus::from_u8(self.epistemic_status)
    }

    #[inline(always)]
    pub fn lifecycle(&self) -> LifecycleStatus {
        LifecycleStatus::from_u8(self.lifecycle_status)
    }
}

/// Thread-safe table managing relation historical versions.
pub struct RelationVersionTable {
    next_version_id: AtomicU64,
    rows: RwLock<Vec<RelationVersionRow>>,
    version_id_to_row: RwLock<HashMap<RelationVersionId, u32>>,
    index_to_id: RwLock<Vec<RelationVersionId>>,
}

impl RelationVersionTable {
    pub fn new(start_version_id: RelationVersionId) -> Self {
        Self {
            next_version_id: AtomicU64::new(start_version_id),
            rows: RwLock::new(Vec::new()),
            version_id_to_row: RwLock::new(HashMap::new()),
            index_to_id: RwLock::new(Vec::new()),
        }
    }

    pub fn append(&self, mut row: RelationVersionRow) -> (RelationVersionId, u32) {
        let vid = self.next_version_id.fetch_add(1, Ordering::Relaxed);
        row.version_id = vid;

        let mut rows = self.rows.write();
        let mut map = self.version_id_to_row.write();
        let mut idx_map = self.index_to_id.write();

        let row_idx = rows.len() as u32;
        rows.push(row);
        map.insert(vid, row_idx);
        idx_map.push(vid);

        (vid, row_idx)
    }

    pub fn bind(&self, vid: RelationVersionId, row: RelationVersionRow) -> u32 {
        let mut rows = self.rows.write();
        let mut map = self.version_id_to_row.write();
        let mut idx_map = self.index_to_id.write();

        let row_idx = rows.len() as u32;
        rows.push(row);
        map.insert(vid, row_idx);
        idx_map.push(vid);

        let mut curr = self.next_version_id.load(Ordering::Relaxed);
        while vid >= curr {
            if self
                .next_version_id
                .compare_exchange_weak(curr, vid + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            curr = self.next_version_id.load(Ordering::Relaxed);
        }

        row_idx
    }

    pub fn close_version(&self, row_idx: u32, close_lsn: u64) {
        let mut rows = self.rows.write();
        if let Some(row) = rows.get_mut(row_idx as usize) {
            row.valid_until_lsn = close_lsn;
        }
    }

    pub fn get_row(&self, row_idx: u32) -> Option<RelationVersionRow> {
        let rows = self.rows.read();
        rows.get(row_idx as usize).copied()
    }

    pub fn snapshot_data(&self) -> (Vec<RelationVersionRow>, Vec<RelationVersionId>) {
        (self.rows.read().clone(), self.index_to_id.read().clone())
    }
}
