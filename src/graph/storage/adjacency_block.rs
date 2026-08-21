/* hnsqr/src/graph/storage/adjacency_block.rs */
//!▫~•◦-------------------------------‣
//! # Adjacency Block — Unified Traversal Interface
//!▫~•◦-------------------------------------------------------------------‣
//!
//! `AdjacencyBlock` is the single traversal interface shared by query
//! executors and GDS algorithms.  It routes calls to the active form
//! (mutable delta or immutable CSR/CSC) without exposing the distinction
//! to callers.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;

use crate::NodeIndex;
use crate::graph::catalog::RelTypeId;
use crate::graph::storage::csr::{CscAdjacency, CsrAdjacency};
use crate::graph::storage::edge_delta::EdgeDelta;
use crate::graph::storage::node_arena::NodeArena;

/// A borrowed slice of neighbour IDs alongside optional weights.
pub struct NeighborSlice<'a> {
    pub nodes: &'a [NodeIndex],
    pub weights: &'a [f32],
}

/// Unified read-path adjacency over either mutable delta or immutable CSR/CSC.
pub enum AdjacencyBlock {
    /// Active mutable generation — traversal follows linked-list chains.
    Delta {
        nodes: Arc<NodeArena>,
        edges: Arc<EdgeDelta>,
    },
    /// Sealed immutable generation — traversal uses CSR/CSC row slices.
    Immutable {
        outgoing: Arc<CsrAdjacency>,
        incoming: Arc<CscAdjacency>,
    },
}

impl AdjacencyBlock {
    /// Calls `visitor` for each live outgoing neighbour of `node`, optionally
    /// filtered by `rel_type`.
    pub fn expand_out(
        &self,
        node: NodeIndex,
        rel_type_filter: Option<RelTypeId>,
        mut visitor: impl FnMut(NodeIndex, f32),
    ) {
        match self {
            AdjacencyBlock::Delta { nodes, edges } => {
                if let Some(rec) = nodes.get(node) {
                    edges.iter_out_chain(rec.out_ref, |_id, edge| {
                        if rel_type_filter.is_none_or(|t| t == edge.rel_type) {
                            visitor(edge.dst_node, edge.weight);
                        }
                    });
                }
            }
            AdjacencyBlock::Immutable { outgoing, .. } => {
                let neighbors = outgoing.out_neighbors(node);
                let weights = outgoing.out_weights(node);
                // CSR has no per-edge rel_type filter inline yet; scan is still O(deg).
                let row_start = outgoing
                    .row_offsets
                    .get(node as usize)
                    .copied()
                    .unwrap_or(0) as usize;
                for (i, &dst) in neighbors.iter().enumerate() {
                    let rt = outgoing.rel_types.get(row_start + i).copied().unwrap_or(0);
                    if rel_type_filter.is_none_or(|t| t == rt) {
                        visitor(dst, weights[i]);
                    }
                }
            }
        }
    }

    /// Calls `visitor` for each live incoming neighbour of `node`.
    pub fn expand_in(
        &self,
        node: NodeIndex,
        rel_type_filter: Option<RelTypeId>,
        mut visitor: impl FnMut(NodeIndex, f32),
    ) {
        match self {
            AdjacencyBlock::Delta { nodes, edges } => {
                if let Some(rec) = nodes.get(node) {
                    edges.iter_in_chain(rec.in_ref, |_id, edge| {
                        if rel_type_filter.is_none_or(|t| t == edge.rel_type) {
                            visitor(edge.src_node, edge.weight);
                        }
                    });
                }
            }
            AdjacencyBlock::Immutable { incoming, .. } => {
                let sources = incoming.in_neighbors(node);
                let weights = incoming.in_weights(node);
                let col_start = incoming
                    .col_offsets
                    .get(node as usize)
                    .copied()
                    .unwrap_or(0) as usize;
                for (i, &src) in sources.iter().enumerate() {
                    let rt = incoming.rel_types.get(col_start + i).copied().unwrap_or(0);
                    if rel_type_filter.is_none_or(|t| t == rt) {
                        visitor(src, weights[i]);
                    }
                }
            }
        }
    }

    /// Out-degree of `node`.
    pub fn out_degree(&self, node: NodeIndex) -> usize {
        match self {
            AdjacencyBlock::Delta { nodes, .. } => {
                nodes.get(node).map(|r| r.out_degree as usize).unwrap_or(0)
            }
            AdjacencyBlock::Immutable { outgoing, .. } => outgoing.out_degree(node),
        }
    }

    /// In-degree of `node`.
    pub fn in_degree(&self, node: NodeIndex) -> usize {
        match self {
            AdjacencyBlock::Delta { nodes, .. } => {
                nodes.get(node).map(|r| r.in_degree as usize).unwrap_or(0)
            }
            AdjacencyBlock::Immutable { incoming, .. } => incoming.in_degree(node),
        }
    }
}
