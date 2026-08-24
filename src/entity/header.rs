/* holosphere/src/entity/header.rs */
//!▫~•◦-------------------------------‣
//! # Cache-Aligned 32-Byte Entity Header
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Compact, zero-padding, cache-aligned entity header record.
//! Represents the hot core of an entity in memory and memory-mapped segments.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::entity::id::NULL_ROW_REF;
use crate::entity::status::EpistemicStatus;
use bytemuck::{Pod, Zeroable};

use serde::{Deserialize, Serialize};

/// Bitflags for `EntityHeader.flags`.
pub const ENTITY_FLAG_LIVE: u8 = 1 << 0;
pub const ENTITY_FLAG_HAS_VECTOR: u8 = 1 << 1;
pub const ENTITY_FLAG_HAS_PROPERTIES: u8 = 1 << 2;
pub const ENTITY_FLAG_HAS_PROVENANCE: u8 = 1 << 3;
pub const ENTITY_FLAG_HAS_VERSION_HISTORY: u8 = 1 << 4;
pub const ENTITY_FLAG_HAS_INFERENCE_SIDECAR: u8 = 1 << 5;

/// Exactly-32-byte, padding-free, cache-line-friendly entity header.
///
/// Every field is explicitly aligned so `Pod + Zeroable` are verified at compile time.
///
/// Layout (32 bytes):
/// ```text
/// offset 0  — label_fast_mask    : u64 (8 bytes) ← fast labels 0..63
/// offset 8  — version_row        : u32 (4 bytes) ← row index in VersionTable
/// offset 12 — provenance_row     : u32 (4 bytes) ← row index in ProvenanceArena
/// offset 16 — vector_row         : u32 (4 bytes) ← row index in VectorArena
/// offset 20 — property_row       : u32 (4 bytes) ← row index in PropertyColumns
/// offset 24 — label_overflow_row : u32 (4 bytes) ← row index in overflow label bitset
/// offset 28 — epistemic_status   : u8  (1 byte)  ← materialized current EpistemicStatus
/// offset 29 — flags              : u8  (1 byte)  ← liveness & capability bitflags
/// offset 30 — vector_layout_id   : u16 (2 bytes) ← VectorLayout schema handle
/// total      32 bytes, zero padding
/// ```
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Pod, Zeroable)]
pub struct EntityHeader {
    /// Fast bitmask for labels 0..63. Bit k set <-> LabelId == k is present.
    pub label_fast_mask: u64,
    /// Head reference into the version history table.
    pub version_row: u32,
    /// Row index into the immutable provenance arena.
    pub provenance_row: u32,
    /// Row index into the contiguous vector arena.
    pub vector_row: u32,
    /// Row index into the columnar property store.
    pub property_row: u32,
    /// Row index into the overflow label store (NULL_ROW_REF if none).
    pub label_overflow_row: u32,
    /// Materialized current epistemic status summary.
    pub epistemic_status: u8,
    /// Operational flags (live, has_vector, sidecar attachments, etc.).
    pub flags: u8,
    /// Vector schema descriptor handle (dimensions, normalization, quantization).
    pub vector_layout_id: u16,
}

// Compile-time size & alignment guards.
const _: () = assert!(std::mem::size_of::<EntityHeader>() == 32);
const _: () = assert!(std::mem::align_of::<EntityHeader>() == 32);

impl Default for EntityHeader {
    fn default() -> Self {
        Self {
            label_fast_mask: 0,
            version_row: NULL_ROW_REF,
            provenance_row: NULL_ROW_REF,
            vector_row: NULL_ROW_REF,
            property_row: NULL_ROW_REF,
            label_overflow_row: NULL_ROW_REF,
            epistemic_status: EpistemicStatus::Observed as u8,
            flags: ENTITY_FLAG_LIVE,
            vector_layout_id: 0,
        }
    }
}

impl EntityHeader {
    #[inline(always)]
    pub fn is_live(&self) -> bool {
        (self.flags & ENTITY_FLAG_LIVE) != 0
    }

    #[inline(always)]
    pub fn set_live(&mut self, live: bool) {
        if live {
            self.flags |= ENTITY_FLAG_LIVE;
        } else {
            self.flags &= !ENTITY_FLAG_LIVE;
        }
    }

    #[inline(always)]
    pub fn has_vector(&self) -> bool {
        (self.flags & ENTITY_FLAG_HAS_VECTOR) != 0 && self.vector_row != NULL_ROW_REF
    }

    #[inline(always)]
    pub fn has_provenance(&self) -> bool {
        (self.flags & ENTITY_FLAG_HAS_PROVENANCE) != 0 && self.provenance_row != NULL_ROW_REF
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
    pub fn set_epistemic(&mut self, status: EpistemicStatus) {
        self.epistemic_status = status as u8;
    }
}
