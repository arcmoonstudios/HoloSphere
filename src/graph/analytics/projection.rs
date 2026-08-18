/* hnsqr/src/graph/analytics/projection.rs */
//!▫~•◦-------------------------------‣
//! # Graph Projection — Algorithm Input Abstraction
//!▫~•◦-------------------------------------------------------------------‣

use std::sync::Arc;

use crate::graph::storage::csr::{CscAdjacency, CsrAdjacency};
use crate::NodeIndex;

/// Uniform interface consumed by all GDS algorithms.
///
/// A projection may represent a subgraph, a filtered view, or the whole
/// generation.  Algorithms receive `&dyn GraphProjection` so they remain
/// independent of the underlying storage form.
pub trait GraphProjection: Send + Sync {
    fn node_count(&self) -> usize;
    fn out_neighbors(&self, node: NodeIndex) -> &[NodeIndex];
    fn in_neighbors(&self, node: NodeIndex) -> &[NodeIndex];
    fn out_weights(&self, node: NodeIndex) -> &[f32];
    fn in_weights(&self, node: NodeIndex) -> &[f32];
    fn out_degree(&self, node: NodeIndex) -> usize {
        self.out_neighbors(node).len()
    }
    fn in_degree(&self, node: NodeIndex) -> usize {
        self.in_neighbors(node).len()
    }
    fn edge_count(&self) -> usize;
}

/// Full-graph projection backed by sealed `CsrAdjacency` + `CscAdjacency`.
pub struct CsrProjection {
    pub outgoing: Arc<CsrAdjacency>,
    pub incoming: Arc<CscAdjacency>,
}

impl CsrProjection {
    pub fn new(outgoing: Arc<CsrAdjacency>, incoming: Arc<CscAdjacency>) -> Self {
        Self { outgoing, incoming }
    }
}

impl GraphProjection for CsrProjection {
    fn node_count(&self) -> usize {
        self.outgoing.node_count
    }

    fn out_neighbors(&self, node: NodeIndex) -> &[NodeIndex] {
        self.outgoing.out_neighbors(node)
    }

    fn in_neighbors(&self, node: NodeIndex) -> &[NodeIndex] {
        self.incoming.in_neighbors(node)
    }

    fn out_weights(&self, node: NodeIndex) -> &[f32] {
        self.outgoing.out_weights(node)
    }

    fn in_weights(&self, node: NodeIndex) -> &[f32] {
        self.incoming.in_weights(node)
    }

    fn edge_count(&self) -> usize {
        self.outgoing.edge_count()
    }
}
