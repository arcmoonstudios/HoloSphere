/* hnsqr/src/graph/storage/mod.rs */
//!▫~•◦-------------------------------‣
//! # Graph Storage — Dual-Form Adjacency & Generation Management
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Implements **Generation-Pinned Dual-Form Adjacency**:
//!
//! - **Mutable generation** (`EdgeDelta`): append-friendly doubly-linked record
//!   arena for low-latency writes; all writes go through Raft before touching
//!   this layer.
//! - **Immutable generation** (`CsrAdjacency` + `CscAdjacency`): CSR (outgoing)
//!   and CSC (incoming) materialised at segment seal time; guarantees neighbour
//!   contiguity for cache-efficient traversal and GDS projections.
//!
//! Both forms coexist inside a [`GraphGeneration`].  Query reads hold a
//! [`GraphReadGeneration`] pin that retains the snapshot without global locks.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod adjacency_block;
pub mod csr;
pub mod edge_delta;
pub mod generation;
pub mod node_arena;
pub mod properties;
pub mod snapshot;

pub use adjacency_block::{AdjacencyBlock, NeighborSlice};
pub use csr::{CscAdjacency, CsrAdjacency};
pub use edge_delta::{EdgeDelta, EdgeDeltaStats};
pub use generation::{GraphGeneration, GraphReadGeneration};
pub use node_arena::{GraphNodeRecord, NodeArena, NULL_OVERFLOW_REF};
pub use properties::{GraphPropertyStore, GraphPropertyValue};
pub use snapshot::{GraphSnapshot, ImmutableGraphSnapshot};
