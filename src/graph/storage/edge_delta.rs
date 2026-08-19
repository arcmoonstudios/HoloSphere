/* hnsqr/src/graph/storage/edge_delta.rs */
//!▫~•◦-------------------------------‣
//! # Edge Delta — Mutable Generation Append-Only Relationship Arena
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Stores relationships as **fixed-stride 32-byte records** in a contiguous
//! arena.  Traversal follows the intrinsic linked list:
//!
//! ```text
//! node.out_ref → EdgeRecord { next_src: … } → EdgeRecord { next_src: NULL_EDGE }
//! ```
//!
//! This is the write-optimised form.  At segment seal time the entire delta is
//! compacted into `CsrAdjacency` + `CscAdjacency` for read-optimised traversal.
//!
//! ## Record layout (32 bytes, no padding)
//! ```text
//! offset 0  — rel_type   : u16  (2 bytes)
//! offset 2  — _pad       : u16  (2 bytes)  explicit padding field
//! offset 4  — src_node   : u32  (4 bytes)
//! offset 8  — dst_node   : u32  (4 bytes)
//! offset 12 — next_src   : u32  (4 bytes)  linked-list forward (out-chain)
//! offset 16 — next_dst   : u32  (4 bytes)  linked-list forward (in-chain)
//! offset 20 — prop_offset: u32  (4 bytes)  into GraphPropertyStore
//! offset 24 — weight     : f32  (4 bytes)
//! offset 28 — _pad2      : u32  (4 bytes)  explicit padding to hit 32 bytes
//! total       32 bytes
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;

use crate::graph::catalog::RelTypeId;
use crate::NodeIndex;

/// Sentinel for empty linked-list pointers.
pub const NULL_EDGE: u32 = u32::MAX;

/// Stable identity for a relationship record inside an `EdgeDelta`.
pub type RelationshipId = u32;

/// Exactly-32-byte relationship record.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct EdgeRecord {
    pub rel_type: u16,
    /// Explicit padding — ensures no compiler-inserted padding bytes.
    _pad: u16,
    pub src_node: u32,
    pub dst_node: u32,
    /// Next relationship in the source node's out-chain. `NULL_EDGE` = end.
    pub next_src: u32,
    /// Next relationship in the destination node's in-chain. `NULL_EDGE` = end.
    pub next_dst: u32,
    /// Offset into the columnar property store (0 = no properties).
    pub prop_offset: u32,
    /// Relationship weight (1.0 = unit weight for unweighted graphs).
    pub weight: f32,
    _pad2: u32,
}

// Compile-time size guard.
const _: () = assert!(std::mem::size_of::<EdgeRecord>() == 32);

impl EdgeRecord {
    pub fn new(
        rel_type: RelTypeId,
        src_node: NodeIndex,
        dst_node: NodeIndex,
        weight: f32,
        prop_offset: u32,
    ) -> Self {
        Self {
            rel_type,
            _pad: 0,
            src_node,
            dst_node,
            next_src: NULL_EDGE,
            next_dst: NULL_EDGE,
            prop_offset,
            weight,
            _pad2: 0,
        }
    }
}

/// Summary statistics for an `EdgeDelta`.
#[derive(Clone, Debug, Default)]
pub struct EdgeDeltaStats {
    pub total_records: usize,
    pub live_records: usize,
    pub tombstone_records: usize,
}

/// Append-only mutable relationship arena for one graph generation.
///
/// All writes must arrive via the authoritative Raft mutation path; direct
/// mutation methods on this struct are `pub(crate)` only.
pub struct EdgeDelta {
    records: RwLock<Vec<EdgeRecord>>,
    live: RwLock<Vec<bool>>,
    next_id: AtomicU32,
}

impl Default for EdgeDelta {
    fn default() -> Self {
        Self::new()
    }
}

impl EdgeDelta {
    pub fn new() -> Self {
        Self {
            records: RwLock::new(Vec::new()),
            live: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }

    /// Appends a new edge record; returns its stable [`RelationshipId`].
    pub fn append(&self, record: EdgeRecord) -> RelationshipId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.records.write().push(record);
        self.live.write().push(true);
        id
    }

    /// Tombstones an edge by ID (soft delete; record slot is preserved).
    pub fn delete(&self, id: RelationshipId) -> bool {
        let idx = id as usize;
        let mut live = self.live.write();
        if idx < live.len() && live[idx] {
            live[idx] = false;
            true
        } else {
            false
        }
    }

    /// Updates an existing record in-place (e.g. to patch `next_src` / `next_dst`).
    pub fn update(&self, id: RelationshipId, record: EdgeRecord) -> bool {
        let idx = id as usize;
        let live = self.live.read();
        if idx < live.len() {
            drop(live);
            self.records.write()[idx] = record;
            true
        } else {
            false
        }
    }

    /// Returns a copy of the record, or `None` if out of range.
    #[inline]
    pub fn get(&self, id: RelationshipId) -> Option<EdgeRecord> {
        let idx = id as usize;
        let live = self.live.read();
        if idx < live.len() {
            Some(self.records.read()[idx])
        } else {
            None
        }
    }

    /// Returns `true` if the record is live (not tombstoned).
    pub fn is_live(&self, id: RelationshipId) -> bool {
        let idx = id as usize;
        let live = self.live.read();
        idx < live.len() && live[idx]
    }

    /// Number of live edges in this delta.
    pub fn len(&self) -> usize {
        self.live.read().iter().filter(|&&b| b).count()
    }

    /// Iterates the outgoing edge chain for `node` starting at `head`.
    ///
    /// Skips tombstoned records so callers always see consistent live edges.
    pub fn iter_out_chain(
        &self,
        head: u32,
        mut visitor: impl FnMut(RelationshipId, &EdgeRecord),
    ) {
        let records = self.records.read();
        let live = self.live.read();
        let mut cur = head;
        while cur != NULL_EDGE {
            let idx = cur as usize;
            if idx >= records.len() {
                break;
            }
            let rec = &records[idx];
            let is_live = *live.get(idx).unwrap_or(&false);
            let next = rec.next_src;
            if is_live {
                visitor(cur, rec);
            }
            cur = next;
        }
    }

    /// Iterates the incoming edge chain for `node` starting at `head`.
    pub fn iter_in_chain(
        &self,
        head: u32,
        mut visitor: impl FnMut(RelationshipId, &EdgeRecord),
    ) {
        let records = self.records.read();
        let live = self.live.read();
        let mut cur = head;
        while cur != NULL_EDGE {
            let idx = cur as usize;
            if idx >= records.len() {
                break;
            }
            let rec = &records[idx];
            let is_live = *live.get(idx).unwrap_or(&false);
            let next = rec.next_dst;
            if is_live {
                visitor(cur, rec);
            }
            cur = next;
        }
    }

    pub fn stats(&self) -> EdgeDeltaStats {
        let live = self.live.read();
        let total = live.len();
        let live_count = live.iter().filter(|&&b| b).count();
        EdgeDeltaStats {
            total_records: total,
            live_records: live_count,
            tombstone_records: total - live_count,
        }
    }

    pub fn edge_count(&self) -> usize {
        self.live.read().iter().filter(|&&b| b).count()
    }

    /// Returns a snapshot of all live edge records for CSR/CSC compaction.
    pub fn live_edges(&self) -> Vec<(RelationshipId, EdgeRecord)> {
        let records = self.records.read();
        let live = self.live.read();
        live.iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some((i as RelationshipId, records[i])) } else { None })
            .collect()
    }
}
