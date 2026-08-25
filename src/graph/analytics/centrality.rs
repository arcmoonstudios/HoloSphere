/* holosphere/src/graph/analytics/centrality.rs */
//!▫~•◦-------------------------------‣
//! # Graph Degree & Centrality Analytics
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Computes normalized in-degree and out-degree centrality metrics over
//! CSR/CSC graph projections for topological influence ranking.
//!
//! ## Key Capabilities
//! - **Normalized Degree Centrality:** (|V|)$ computation of directional node influence.
//! - **Zero-Copy Adjacency Traversal:** Direct evaluation over projected graph topology.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::NodeIndex;
use crate::graph::analytics::projection::GraphProjection;

/// Degree centrality scores per node.
#[derive(Debug, Default)]
pub struct DegreeCentrality {
    /// Normalised out-degree centrality (0.0–1.0).
    pub out_centrality: Vec<f64>,
    /// Normalised in-degree centrality (0.0–1.0).
    pub in_centrality: Vec<f64>,
}

impl DegreeCentrality {
    pub fn compute(projection: &dyn GraphProjection) -> Self {
        let n = projection.node_count();
        if n <= 1 {
            return Self::default();
        }
        let norm = (n - 1) as f64;
        let out_centrality = (0..n as NodeIndex)
            .map(|v| projection.out_degree(v) as f64 / norm)
            .collect();
        let in_centrality = (0..n as NodeIndex)
            .map(|v| projection.in_degree(v) as f64 / norm)
            .collect();
        Self {
            out_centrality,
            in_centrality,
        }
    }
}
