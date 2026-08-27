/* holosphere/src/codegraph/ingest.rs */
//!▫~•◦-------------------------------‣
//! # Atomic CodeGraphDelta Ingestion & Hypergraph Store Bridge
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Applies atomic CodeGraphDelta transactions into persistent memory, updating
//! node arenas, bidirectional adjacency blocks, and inverted symbol indices in one publication step.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::schema::{
    CodeEdge, CodeEdgeId, CodeGraphDelta, CodeNode, CodeNodeId, CodeNodeKind, CodeRelation,
};
use crate::HNSQRResult;

/// Authoritative in-memory / persistent CodeGraph index store.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CodeGraphStoreState {
    pub workspace_id: String,
    pub commit_lsn: u64,
    pub nodes: BTreeMap<CodeNodeId, CodeNode>,
    pub edges: BTreeMap<CodeEdgeId, CodeEdge>,
    pub outgoing_edges: BTreeMap<CodeNodeId, Vec<CodeEdgeId>>,
    pub incoming_edges: BTreeMap<CodeNodeId, Vec<CodeEdgeId>>,
    pub symbols_by_name: BTreeMap<String, Vec<CodeNodeId>>,
    pub symbols_by_qualified: BTreeMap<String, CodeNodeId>,
    pub nodes_by_file: BTreeMap<PathBuf, Vec<CodeNodeId>>,
    pub nodes_by_kind: BTreeMap<CodeNodeKind, Vec<CodeNodeId>>,
}

pub struct CodeGraphStore {
    inner: Arc<RwLock<CodeGraphStoreState>>,
    next_lsn: AtomicU64,
}

impl Default for CodeGraphStore {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(CodeGraphStoreState::default())),
            next_lsn: AtomicU64::new(1),
        }
    }
}

impl CodeGraphStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Atomically commits a CodeGraphDelta into the store.
    pub fn commit_delta(&self, delta: CodeGraphDelta) -> u64 {
        let mut state = self.inner.write();
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);

        state.workspace_id = delta.workspace_id;
        state.commit_lsn = lsn;

        // 1. Delete tombstoned edges
        for edge_id in &delta.delete_edges {
            if let Some(edge) = state.edges.remove(edge_id) {
                if let Some(out_list) = state.outgoing_edges.get_mut(&edge.source) {
                    out_list.retain(|id| id != edge_id);
                }
                if let Some(in_list) = state.incoming_edges.get_mut(&edge.target) {
                    in_list.retain(|id| id != edge_id);
                }
            }
        }

        // 2. Delete tombstoned nodes
        for node_id in &delta.delete_nodes {
            if let Some(node) = state.nodes.remove(node_id) {
                if let Some(ids) = state.symbols_by_name.get_mut(&node.name) {
                    ids.retain(|id| id != node_id);
                }
                state.symbols_by_qualified.remove(&node.qualified_name);
                if let Some(file_nodes) = state.nodes_by_file.get_mut(&node.source_file) {
                    file_nodes.retain(|id| id != node_id);
                }
                if let Some(kind_nodes) = state.nodes_by_kind.get_mut(&node.kind) {
                    kind_nodes.retain(|id| id != node_id);
                }
                state.outgoing_edges.remove(node_id);
                state.incoming_edges.remove(node_id);
            }
        }

        // 3. Insert / update nodes
        for node in delta.insert_nodes {
            let node_id = node.id.clone();
            state
                .symbols_by_name
                .entry(node.name.clone())
                .or_default()
                .push(node_id.clone());
            state
                .symbols_by_qualified
                .insert(node.qualified_name.clone(), node_id.clone());
            state
                .nodes_by_file
                .entry(node.source_file.clone())
                .or_default()
                .push(node_id.clone());
            state
                .nodes_by_kind
                .entry(node.kind)
                .or_default()
                .push(node_id.clone());

            state.nodes.insert(node_id, node);
        }

        // 4. Insert / update edges
        for edge in delta.insert_edges {
            let edge_id = edge.id.clone();
            state
                .outgoing_edges
                .entry(edge.source.clone())
                .or_default()
                .push(edge_id.clone());
            state
                .incoming_edges
                .entry(edge.target.clone())
                .or_default()
                .push(edge_id.clone());

            state.edges.insert(edge_id, edge);
        }

        lsn
    }

    #[must_use]
    pub fn snapshot(&self) -> CodeGraphStoreState {
        self.inner.read().clone()
    }

    #[must_use]
    pub fn get_node(&self, id: &CodeNodeId) -> Option<CodeNode> {
        self.inner.read().nodes.get(id).cloned()
    }

    #[must_use]
    pub fn lookup_exact(&self, qualified_name: &str) -> Option<CodeNode> {
        let state = self.inner.read();
        let id = state.symbols_by_qualified.get(qualified_name)?;
        state.nodes.get(id).cloned()
    }

    #[must_use]
    pub fn lookup_by_name(&self, name: &str) -> Vec<CodeNode> {
        let state = self.inner.read();
        state
            .symbols_by_name
            .get(name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| state.nodes.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[must_use]
    pub fn get_outgoing_edges(&self, node_id: &CodeNodeId) -> Vec<CodeEdge> {
        let state = self.inner.read();
        state
            .outgoing_edges
            .get(node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|id| state.edges.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[must_use]
    pub fn get_incoming_edges(&self, node_id: &CodeNodeId) -> Vec<CodeEdge> {
        let state = self.inner.read();
        state
            .incoming_edges
            .get(node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|id| state.edges.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }
}
