/* hnsqr/src/graph/storage/node_arena.rs */
//!▫~•◦-------------------------------‣
//! # Node Arena — Cache-Aligned Fixed-Stride Node Records
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Each `GraphNodeRecord` is exactly 32 bytes and cache-line–friendly.
//! Every field is named explicitly so `bytemuck::Pod` can be verified at
//! compile time with no implicit padding.
//!
//! Layout (32 bytes):
//! ```text
//! offset 0  — label_fast_mask    : u64  (8 bytes)  ← bits 0-63 = fast labels
//! offset 8  — out_ref            : u32  (4 bytes)  ← delta/CSR head reference
//! offset 12 — in_ref             : u32  (4 bytes)  ← delta/CSC head reference
//! offset 16 — out_degree         : u32  (4 bytes)  ← u32 to avoid hub overflow
//! offset 20 — in_degree          : u32  (4 bytes)
//! offset 24 — vector_slot        : u32  (4 bytes)  ← HNSQR NodeIndex binding
//! offset 28 — label_overflow_ref : u32  (4 bytes)  ← 0 = no overflow labels
//! total      32 bytes, no padding
//! ```
//!
//! ## Label storage contract
//! - Labels 0–63 → set the corresponding bit in `label_fast_mask`.
//! - Labels ≥ 64 → store index into the per-node overflow bitmap via
//!   `label_overflow_ref`.  `NULL_OVERFLOW_REF` (u32::MAX) means no overflow.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::atomic::{AtomicU32, Ordering};

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;

use crate::NodeIndex;

/// Sentinel meaning "no overflow labels".
pub const NULL_OVERFLOW_REF: u32 = u32::MAX;

/// Exactly-32-byte, padding-free, cache-line–aligned node record.
///
/// `Pod + Zeroable` are safe: every bit pattern is a valid value (u64, u32 × 5).
/// The `repr(C, align(32))` guarantees layout stability and alignment.
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GraphNodeRecord {
    /// Bitmask for labels 0–63. Bit `k` set ↔ label with `LabelId == k` is present.
    pub label_fast_mask: u64,
    /// Head reference into the mutable `EdgeDelta` (out-direction) or the CSR row.
    /// Meaning depends on which adjacency form is active (see `GraphGeneration`).
    pub out_ref: u32,
    /// Head reference into the mutable `EdgeDelta` (in-direction) or the CSC row.
    pub in_ref: u32,
    /// Out-degree. u32 to handle hub nodes without silent wrapping.
    pub out_degree: u32,
    /// In-degree.
    pub in_degree: u32,
    /// Direct binding to HNSQR's vector/metadata arena (`NodeIndex`).
    /// `u32::MAX` means "graph node without a vector embedding".
    pub vector_slot: u32,
    /// Reference into the per-graph overflow-label bitmap store.
    /// `NULL_OVERFLOW_REF` means no overflow labels are set.
    pub label_overflow_ref: u32,
}

// Compile-time size guard.
const _: () = assert!(std::mem::size_of::<GraphNodeRecord>() == 32);

/// Contiguous arena of [`GraphNodeRecord`]s backed by a `Vec`.
///
/// Allocation is append-only during a mutable generation; deletions set a
/// tombstone bit in `live_nodes` rather than physically removing the record
/// so that existing indices remain stable.
pub struct NodeArena {
    records: RwLock<Vec<GraphNodeRecord>>,
    /// Roaring-style dense bitset tracking live nodes.  Using a plain `Vec<bool>`
    /// here is intentional: it is simpler and fast enough; a Roaring bitmap is
    /// not worth the complexity until node counts exceed a few hundred million.
    live: RwLock<Vec<bool>>,
    /// Monotonically increasing allocation counter.
    next_id: AtomicU32,
}

impl Default for NodeArena {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeArena {
    pub fn new() -> Self {
        Self {
            records: RwLock::new(Vec::new()),
            live: RwLock::new(Vec::new()),
            next_id: AtomicU32::new(0),
        }
    }

    /// Allocates a new node record and returns its `NodeIndex`.
    pub fn alloc(&self, record: GraphNodeRecord) -> NodeIndex {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut r = self.records.write();
        let mut l = self.live.write();
        r.push(record);
        l.push(true);
        id
    }

    /// Returns a copy of the record for `node`, or `None` if out of range or deleted.
    #[inline]
    pub fn get(&self, node: NodeIndex) -> Option<GraphNodeRecord> {
        let idx = node as usize;
        let live = self.live.read();
        if idx < live.len() && live[idx] {
            Some(self.records.read()[idx])
        } else {
            None
        }
    }

    /// Updates the record for an existing node in-place.
    pub fn update(&self, node: NodeIndex, record: GraphNodeRecord) -> bool {
        let idx = node as usize;
        let live = self.live.read();
        if idx < live.len() && live[idx] {
            drop(live);
            self.records.write()[idx] = record;
            true
        } else {
            false
        }
    }

    /// Marks a node as deleted (tombstone).  The slot is not reclaimed.
    pub fn delete(&self, node: NodeIndex) -> bool {
        let idx = node as usize;
        let mut live = self.live.write();
        if idx < live.len() && live[idx] {
            live[idx] = false;
            true
        } else {
            false
        }
    }

    pub fn is_live(&self, node: NodeIndex) -> bool {
        let idx = node as usize;
        let live = self.live.read();
        idx < live.len() && live[idx]
    }

    pub fn node_count(&self) -> usize {
        self.live.read().iter().filter(|&&b| b).count()
    }

    pub fn capacity(&self) -> usize {
        self.records.read().len()
    }

    /// Iterate over all live node IDs.
    pub fn live_nodes(&self) -> Vec<NodeIndex> {
        self.live
            .read()
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i as NodeIndex) } else { None })
            .collect()
    }
}
