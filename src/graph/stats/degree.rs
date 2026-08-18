/* hnsqr/src/graph/stats/degree.rs */
//!▫~•◦-------------------------------‣
//! # Degree Statistics — Out/In-Degree Histograms
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::storage::generation::GraphGeneration;
use crate::NodeIndex;

/// Per-generation degree summary.
#[derive(Clone, Debug, Default)]
pub struct DegreeStats {
    pub min_out_degree: usize,
    pub max_out_degree: usize,
    pub avg_out_degree: f64,
    pub min_in_degree: usize,
    pub max_in_degree: usize,
    pub avg_in_degree: f64,
    /// Top-10 hub nodes by out-degree.
    pub top_out_hubs: Vec<(NodeIndex, usize)>,
}

impl DegreeStats {
    /// Computes degree stats by scanning the generation.  O(V).
    pub fn compute(generation: &GraphGeneration) -> Self {
        let live = generation.nodes.live_nodes();
        let adj = generation.adjacency();

        if live.is_empty() {
            return Self::default();
        }

        let mut min_out = usize::MAX;
        let mut max_out = 0usize;
        let mut sum_out = 0usize;
        let mut min_in = usize::MAX;
        let mut max_in = 0usize;
        let mut sum_in = 0usize;
        let mut out_degs: Vec<(NodeIndex, usize)> = Vec::with_capacity(live.len());

        for &node in &live {
            let od = adj.out_degree(node);
            let id_ = adj.in_degree(node);
            min_out = min_out.min(od);
            max_out = max_out.max(od);
            sum_out += od;
            min_in = min_in.min(id_);
            max_in = max_in.max(id_);
            sum_in += id_;
            out_degs.push((node, od));
        }

        out_degs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
        let top_out_hubs = out_degs.into_iter().take(10).collect();

        let n = live.len() as f64;
        Self {
            min_out_degree: if min_out == usize::MAX { 0 } else { min_out },
            max_out_degree: max_out,
            avg_out_degree: sum_out as f64 / n,
            min_in_degree: if min_in == usize::MAX { 0 } else { min_in },
            max_in_degree: max_in,
            avg_in_degree: sum_in as f64 / n,
            top_out_hubs,
        }
    }
}
