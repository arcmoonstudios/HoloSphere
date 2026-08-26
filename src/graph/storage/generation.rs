/* holosphere/src/graph/storage/generation.rs */
//!▫~•◦-------------------------------‣
//! # Graph Generation — Dual-Form Lifecycle Management
//!▫~•◦-------------------------------------------------------------------‣
//!
//! A `GraphGeneration` holds the complete snapshot of graph topology at one
//! point in time.  It transitions through two phases:
//!
//! ```text
//! MUTABLE  (NodeArena + EdgeDelta)
//!     │  seal()
//!     ▼
//! IMMUTABLE  (NodeArena + CsrAdjacency + CscAdjacency)
//! ```
//!
//! Query readers hold a [`GraphReadGeneration`] RAII pin that keeps the
//! `Arc`-counted generation alive across compactions without global locks.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;

use parking_lot::RwLock;

use crate::NodeIndex;
use crate::graph::storage::adjacency_block::AdjacencyBlock;
use crate::graph::storage::csr::{CscAdjacency, CsrAdjacency};
use crate::graph::storage::edge_delta::EdgeDelta;
use crate::graph::storage::node_arena::NodeArena;
use crate::graph::storage::properties::GraphPropertyStore;

/// The physical adjacency form for a sealed generation.
pub enum SealedAdjacency {
    Csr {
        outgoing: Arc<CsrAdjacency>,
        incoming: Arc<CscAdjacency>,
    },
}

/// A single graph generation.
pub struct GraphGeneration {
    /// Monotonically increasing generation counter.
    pub generation: u64,
    /// Node records (shared between mutable and immutable phases).
    pub nodes: Arc<NodeArena>,
    /// Mutable edge delta — `Some` while the generation is open for writes.
    pub edge_delta: Option<Arc<EdgeDelta>>,
    /// Immutable CSR/CSC — `Some` after sealing.
    pub sealed: Option<SealedAdjacency>,
    /// Columnar property store.
    pub properties: Arc<GraphPropertyStore>,
    /// Node count at seal time (or current live count if still mutable).
    pub node_count_at_seal: usize,
}

impl GraphGeneration {
    /// Creates a new open (mutable) generation.
    pub fn new_mutable(generation: u64) -> Self {
        Self {
            generation,
            nodes: Arc::new(NodeArena::new()),
            edge_delta: Some(Arc::new(EdgeDelta::new())),
            sealed: None,
            properties: Arc::new(GraphPropertyStore::default()),
            node_count_at_seal: 0,
        }
    }

    /// Returns `true` if the generation is still open for writes.
    pub fn is_mutable(&self) -> bool {
        self.edge_delta.is_some()
    }

    /// Returns the live node count.
    pub fn node_count(&self) -> usize {
        self.nodes.node_count()
    }

    /// Returns the live edge count.
    pub fn edge_count(&self) -> usize {
        match &self.edge_delta {
            Some(delta) => delta.edge_count(),
            None => match &self.sealed {
                Some(SealedAdjacency::Csr { outgoing, .. }) => outgoing.edge_count(),
                None => 0,
            },
        }
    }

    /// Seals this generation: materialises CSR and CSC from the delta, then
    /// drops the delta.  Returns `Err` if already sealed.
    pub fn seal(&mut self) -> Result<(), &'static str> {
        if self.sealed.is_some() {
            return Err("Generation is already sealed");
        }
        let delta = self.edge_delta.take().ok_or("No edge delta to seal")?;
        let node_count = self.nodes.capacity(); // include tombstoned slots for stable indices
        self.node_count_at_seal = node_count;
        let outgoing = Arc::new(CsrAdjacency::build(&self.nodes, &delta, node_count));
        let incoming = Arc::new(CscAdjacency::build(&self.nodes, &delta, node_count));
        self.sealed = Some(SealedAdjacency::Csr { outgoing, incoming });
        Ok(())
    }

    /// Returns the adjacency block appropriate for the current phase.
    pub fn adjacency(&self) -> AdjacencyBlock {
        match &self.sealed {
            Some(SealedAdjacency::Csr { outgoing, incoming }) => AdjacencyBlock::Immutable {
                outgoing: outgoing.clone(),
                incoming: incoming.clone(),
            },
            None => {
                let delta = self
                    .edge_delta
                    .as_ref()
                    .expect("no edge delta on mutable generation");
                AdjacencyBlock::Delta {
                    nodes: self.nodes.clone(),
                    edges: delta.clone(),
                }
            }
        }
    }

    /// Returns a reference to the outgoing CSR, or `None` if not yet sealed.
    pub fn csr(&self) -> Option<&Arc<CsrAdjacency>> {
        match &self.sealed {
            Some(SealedAdjacency::Csr { outgoing, .. }) => Some(outgoing),
            None => None,
        }
    }

    /// Returns a reference to the incoming CSC, or `None` if not yet sealed.
    pub fn csc(&self) -> Option<&Arc<CscAdjacency>> {
        match &self.sealed {
            Some(SealedAdjacency::Csr { incoming, .. }) => Some(incoming),
            None => None,
        }
    }
}

/// RAII read pin holding a generation alive during a query.
///
/// Callers obtain this from `GraphStore::pin_read_generation()`.  Dropping
/// the pin releases the reference.  No locks are held while the pin is live.
pub struct GraphReadGeneration {
    pub generation: Arc<RwLock<GraphGeneration>>,
    pub generation_id: u64,
}

impl GraphReadGeneration {
    pub fn new(generation: Arc<RwLock<GraphGeneration>>, id: u64) -> Self {
        Self {
            generation,
            generation_id: id,
        }
    }

    /// Convenience: expand outgoing neighbours of `node` via the pinned generation.
    pub fn expand_out(
        &self,
        node: NodeIndex,
        rel_type_filter: Option<crate::graph::catalog::RelTypeId>,
        visitor: impl FnMut(NodeIndex, f32),
    ) {
        self.generation
            .read()
            .adjacency()
            .expand_out(node, rel_type_filter, visitor);
    }

    /// Convenience: expand incoming neighbours of `node` via the pinned generation.
    pub fn expand_in(
        &self,
        node: NodeIndex,
        rel_type_filter: Option<crate::graph::catalog::RelTypeId>,
        visitor: impl FnMut(NodeIndex, f32),
    ) {
        self.generation
            .read()
            .adjacency()
            .expand_in(node, rel_type_filter, visitor);
    }
}
