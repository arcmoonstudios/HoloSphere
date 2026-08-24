/* holosphere/src/relation/header.rs */
//!▫~•◦-------------------------------‣
//! # 32-Byte Hypergraph Relation Header
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Compact, cache-aligned Pod struct representing a localized relation instance
//! within a relation segment.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::entity::status::{EpistemicStatus, LifecycleStatus};
use crate::relation::id::RelationTypeId;

pub const RELATION_FLAG_LIVE: u16 = 1 << 0;
pub const RELATION_FLAG_HAS_VERSION_HISTORY: u16 = 1 << 1;
pub const RELATION_FLAG_HAS_PROVENANCE: u16 = 1 << 2;

/// Fixed 32-byte cache-aligned relation header.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable, Serialize, Deserialize)]
pub struct RelationHeader {
    /// Schema type ID of this hypergraph relation.
    pub relation_type_id: RelationTypeId,
    /// Physical start index into the segment role binding table.
    pub binding_start: u32,
    /// Generation-local version table row offset for temporal resolution.
    pub version_row: u32,
    /// Generation-local provenance table row offset.
    pub provenance_row: u32,
    /// Number of role bindings belonging to this relation instance.
    pub binding_len: u16,
    /// Version of the schema at time of creation.
    pub schema_version: u16,
    /// Materialized current epistemic status summary (Observed, Inferred, Provisional, Contradicted).
    pub epistemic_status: u8,
    /// Materialized current lifecycle status summary (Active, Superseded, Tombstoned).
    pub lifecycle_status: u8,
    /// Bitflags for live/tombstone/provenance presence.
    pub flags: u16,
    /// Reserved zero padding to ensure strict 32-byte alignment.
    pub reserved: [u8; 8],
}

const _: () = assert!(std::mem::size_of::<RelationHeader>() == 32);
const _: () = assert!(std::mem::align_of::<RelationHeader>() == 32);

impl Default for RelationHeader {
    fn default() -> Self {
        Self {
            relation_type_id: 0,
            binding_start: 0,
            version_row: crate::entity::id::NULL_ROW_REF,
            provenance_row: crate::entity::id::NULL_ROW_REF,
            binding_len: 0,
            schema_version: 1,
            epistemic_status: EpistemicStatus::Observed as u8,
            lifecycle_status: LifecycleStatus::Active as u8,
            flags: RELATION_FLAG_LIVE,
            reserved: [0u8; 8],
        }
    }
}

impl RelationHeader {
    #[inline(always)]
    pub fn is_live(&self) -> bool {
        (self.flags & RELATION_FLAG_LIVE) != 0
    }

    #[inline(always)]
    pub fn epistemic(&self) -> EpistemicStatus {
        EpistemicStatus::from_u8(self.epistemic_status)
    }

    #[inline(always)]
    pub fn set_epistemic(&mut self, status: EpistemicStatus) {
        self.epistemic_status = status as u8;
    }

    #[inline(always)]
    pub fn lifecycle(&self) -> LifecycleStatus {
        LifecycleStatus::from_u8(self.lifecycle_status)
    }

    #[inline(always)]
    pub fn set_lifecycle(&mut self, status: LifecycleStatus) {
        self.lifecycle_status = status as u8;
    }
}
