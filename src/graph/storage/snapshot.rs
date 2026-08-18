/* hnsqr/src/graph/storage/snapshot.rs */
//!▫~•◦-------------------------------‣
//! # Graph Snapshot — Catalog + Generation Pinned Together
//!▫~•◦-------------------------------------------------------------------‣
//!
//! A `GraphSnapshot` is the unit of consistency for graph reads.  It bundles:
//! - The `LabelCatalog` snapshot (so label IDs are stable).
//! - The `RelTypeCatalog` snapshot.
//! - The `generation` counter at the time of snapshot creation.
//!
//! The combined `UnifiedReadSnapshot` (in `graph/mod.rs`) attaches this
//! alongside the Raft read index, vector generation, and metadata generation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::catalog::labels::LabelCatalogSnapshot;
use crate::graph::catalog::relationships::RelTypeCatalogSnapshot;

/// Lightweight snapshot descriptor for graph topology at a point in time.
///
/// Actual node/edge data is held by the `GraphReadGeneration` pin.
/// This struct carries the catalog snapshots that make label and rel-type IDs
/// stable for the lifetime of the query.
#[derive(Clone, Debug)]
pub struct GraphSnapshot {
    /// Monotonic generation counter.  Matches `GraphGeneration::generation`.
    pub generation: u64,
    /// Raft log index at which this snapshot was obtained.
    pub raft_read_index: u64,
    /// Frozen label catalog for this snapshot.
    pub label_catalog: LabelCatalogSnapshot,
    /// Frozen relationship-type catalog.
    pub rel_type_catalog: RelTypeCatalogSnapshot,
    /// Live node count at snapshot time.
    pub node_count: usize,
    /// Live edge count at snapshot time.
    pub edge_count: usize,
}
