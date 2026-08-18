/* hnsqr/src/graph/mod.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Graph Engine — Native Vector-Graph Unified Substrate
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Adds a first-class graph topology layer to HNSQR, sharing NodeIndex
//! identities, Raft consensus, metadata bitmaps, and generation pinning
//! with the existing vector and metadata engines.
//!
//! ## Architecture
//! ```text
//! DataMutation::Graph(GraphMutation)
//!     │
//!     └── Raft quorum commit
//!             │
//!             └── ShardStateMachine::apply
//!                     │
//!                     └── GraphMutationApplier::apply
//!                             │
//!                     ┌───────┴────────┐
//!                     │                │
//!               NodeArena         EdgeDelta
//!                (mutable)        (mutable)
//!                     │
//!                  seal()
//!                     │
//!             ┌───────┴────────┐
//!             │                │
//!      CsrAdjacency     CscAdjacency
//!       (outgoing)       (incoming)
//!             │
//!      GraphProjection
//!             │
//!       GDS Algorithms
//! ```
//!
//! ## Module layout (src/lib.rs invariant: only one root-level file)
//! All graph code is under `src/graph/`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod analytics;
pub mod catalog;
pub mod mutation;
pub mod query;
pub mod stats;
pub mod storage;

// Flat re-exports for convenient `use hnsqr::graph::*` access.
pub use analytics::{
    BfsResult, CsrProjection, DegreeCentrality, ConnectedComponents, GraphProjection,
    KCoreDecomposition, LouvainEngine, LouvainResult, PageRankEngine, PathfindingEngine,
    ShortestPath, TriangleCount,
};
pub use catalog::{
    LabelCatalog, LabelId, LabelResolution, PropertyKey, PropertyKeyCatalog,
    RelTypeCatalog, RelTypeId, RelTypeResolution,
};
pub use mutation::{
    GraphMutation, GraphMutationApplier, GraphProperties, RelationshipId,
};
pub use query::{
    BindingColumn, Direction, ExecutionContext, ExplainOutput, GraphPattern,
    LogicalPlan, Morsel, PhysicalPlan, QueryAst, QueryResult, ReturnClause,
    SemanticAnalyzer, SemanticError, SymbolId, SymbolTable, WhereClause,
};
pub use stats::{DegreeStats, GraphCardinalityStats};
pub use storage::{
    AdjacencyBlock, CscAdjacency, CsrAdjacency, EdgeDelta, EdgeDeltaStats,
    GraphGeneration, GraphNodeRecord, GraphPropertyStore, GraphPropertyValue,
    GraphReadGeneration, GraphSnapshot, NeighborSlice, NodeArena, NULL_OVERFLOW_REF,
};
