/* holosphere/src/graph/storage/snapshot.rs */
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

use crate::NodeIndex;
use crate::graph::catalog::labels::LabelCatalogSnapshot;
use crate::graph::catalog::relationships::RelTypeCatalogSnapshot;
use crate::graph::storage::edge_delta::EdgeRecord;
use crate::graph::storage::node_arena::GraphNodeRecord;
use crate::graph::storage::properties::GraphPropertyStore;
use std::collections::HashMap;
use std::sync::Arc;

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

/// Fully materialized point-in-time immutable snapshot of graph topology, catalogs, and properties at LSN k.
/// Completely decoupled from subsequent mutations to live graph generation.
#[derive(Clone)]
pub struct ImmutableGraphSnapshot {
    pub generation: u64,
    pub lsn: u64,
    pub label_catalog: LabelCatalogSnapshot,
    pub rel_type_catalog: RelTypeCatalogSnapshot,
    pub nodes: Vec<GraphNodeRecord>,
    pub live_nodes: Vec<bool>,
    pub edge_records: Vec<EdgeRecord>,
    pub live_edges: Vec<bool>,
    pub properties: Arc<GraphPropertyStore>,
    pub node_id_map: HashMap<String, NodeIndex>,
    pub rel_id_map: HashMap<u64, u32>,
}

impl ImmutableGraphSnapshot {
    pub fn node_count(&self) -> usize {
        self.live_nodes.iter().filter(|&&l| l).count()
    }

    pub fn edge_count(&self) -> usize {
        self.live_edges.iter().filter(|&&l| l).count()
    }

    pub fn get_node_index(&self, external_id: &str) -> Option<NodeIndex> {
        self.node_id_map.get(external_id).copied()
    }

    pub fn get_node(&self, idx: NodeIndex) -> Option<&GraphNodeRecord> {
        let i = idx as usize;
        if i < self.nodes.len() && self.live_nodes.get(i).copied().unwrap_or(false) {
            Some(&self.nodes[i])
        } else {
            None
        }
    }
}
