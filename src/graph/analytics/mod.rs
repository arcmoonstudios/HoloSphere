/* hnsqr/src/graph/analytics/mod.rs */
//!▫~•◦-------------------------------‣
//! # Graph Analytics — GDS Projection-Based Algorithms
//!▫~•◦-------------------------------------------------------------------‣
//!
//! All algorithms operate over the `GraphProjection` trait rather than
//! directly over `GraphGeneration`.  A projection can represent:
//!   - the whole graph
//!   - a filtered subgraph (specific labels, rel types, property predicates)
//!   - a single tenant's namespace
//!   - a Cypher-derived subgraph
//!
//! This separation means algorithms never need rewriting when the projection
//! source changes.
//!
//! ## Performance claims
//!
//! All throughput figures in this module are **[BENCH REQUIRED]** per the
//! Pinnacle-State-Module rules.  Hypotheses that require physical measurement:
//!   - CSR PageRank vs linked-edge traversal
//!   - rayon parallelism scaling vs single-threaded for various |V|, |E|
//!   - Louvain convergence speed vs competing community-detection libraries
//!   - BFS / Dijkstra cache-miss rate on CSR vs delta adjacency
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod bfs;
pub mod centrality;
pub mod components;
pub mod k_core;
pub mod louvain;
pub mod pagerank;
pub mod pathfinding;
pub mod projection;
pub mod triangles;

pub use bfs::BfsResult;
pub use centrality::DegreeCentrality;
pub use components::ConnectedComponents;
pub use k_core::KCoreDecomposition;
pub use louvain::{LouvainResult, LouvainEngine};
pub use pagerank::PageRankEngine;
pub use pathfinding::{ShortestPath, PathfindingEngine};
pub use projection::{CsrProjection, GraphProjection};
pub use triangles::TriangleCount;
