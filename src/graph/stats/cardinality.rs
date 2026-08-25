/* holosphere/src/graph/stats/cardinality.rs */
//!▫~•◦-------------------------------‣
//! # Graph Cardinality Statistics
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Scans the live generation and produces label / relationship-type
//! cardinalities used by the query planner for selectivity estimation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;

use crate::graph::catalog::labels::LabelId;
use crate::graph::catalog::relationships::RelTypeId;
use crate::graph::storage::generation::GraphGeneration;

/// Per-generation cardinality summary.
#[derive(Clone, Debug, Default)]
pub struct GraphCardinalityStats {
    /// Total live node count.
    pub nodes: usize,
    /// Total live edge count.
    pub edges: usize,
    /// Live count per label ID.
    pub label_cardinality: HashMap<LabelId, usize>,
    /// Live count per relationship type ID.
    pub relationship_cardinality: HashMap<RelTypeId, usize>,
    /// Average out-degree across all live nodes.
    pub average_out_degree: f64,
    /// Maximum out-degree observed.
    pub max_out_degree: usize,
}

impl GraphCardinalityStats {
    /// Computes statistics by scanning the generation.
    ///
    /// **O(V + E)** — call once at seal time or on demand; cache the result.
    pub fn compute(generation: &GraphGeneration) -> Self {
        let live_node_ids = generation.nodes.live_nodes();
        let adjacency = generation.adjacency();

        let mut label_cardinality: HashMap<LabelId, usize> = HashMap::new();
        let mut max_out_degree = 0usize;
        let mut total_out_degree = 0usize;

        for &node in &live_node_ids {
            if let Some(record) = generation.nodes.get(node) {
                // Fast label bits 0–63.
                for bit in 0u32..64 {
                    if record.label_fast_mask & (1u64 << bit) != 0 {
                        *label_cardinality.entry(bit).or_insert(0) += 1;
                    }
                }
                let deg = adjacency.out_degree(node);
                max_out_degree = max_out_degree.max(deg);
                total_out_degree += deg;
            }
        }

        let nodes = live_node_ids.len();
        let average_out_degree = if nodes == 0 {
            0.0
        } else {
            total_out_degree as f64 / nodes as f64
        };

        // Relationship-type cardinality from live edges.
        let mut relationship_cardinality: HashMap<RelTypeId, usize> = HashMap::new();
        if let Some(delta) = &generation.edge_delta {
            for (_, rec) in delta.live_edges() {
                *relationship_cardinality.entry(rec.rel_type).or_insert(0) += 1;
            }
        } else if let Some(csr) = generation.csr() {
            for &rt in &csr.rel_types {
                *relationship_cardinality.entry(rt).or_insert(0) += 1;
            }
        }

        Self {
            nodes,
            edges: generation.edge_count(),
            label_cardinality,
            relationship_cardinality,
            average_out_degree,
            max_out_degree,
        }
    }

    /// Returns the fraction of nodes carrying a given label (0.0–1.0).
    pub fn label_selectivity(&self, label: LabelId) -> f64 {
        if self.nodes == 0 {
            return 0.0;
        }
        self.label_cardinality.get(&label).copied().unwrap_or(0) as f64 / self.nodes as f64
    }

    /// Returns the fraction of edges with a given relationship type (0.0–1.0).
    pub fn rel_type_selectivity(&self, rel_type: RelTypeId) -> f64 {
        if self.edges == 0 {
            return 0.0;
        }
        self.relationship_cardinality
            .get(&rel_type)
            .copied()
            .unwrap_or(0) as f64
            / self.edges as f64
    }
}
