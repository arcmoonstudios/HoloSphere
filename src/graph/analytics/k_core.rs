/* holosphere/src/graph/analytics/k_core.rs */
//!▫~•◦-------------------------------‣
//! # K-Core Decomposition — Iterative Degree Peeling
//!▫~•◦-------------------------------------------------------------------‣

use crate::NodeIndex;
use crate::graph::analytics::projection::GraphProjection;

/// K-core decomposition result.
#[derive(Debug, Default)]
pub struct KCoreDecomposition {
    /// Core number per node.  `coreness[v]` is the maximum `k` for which
    /// node `v` belongs to the k-core.
    pub coreness: Vec<u32>,
    /// Maximum coreness observed in the graph.
    pub max_coreness: u32,
}

impl KCoreDecomposition {
    /// Computes k-core decomposition by iterative degree peeling.
    /// Treats the graph as undirected (uses out_degree + in_degree).
    pub fn compute(projection: &dyn GraphProjection) -> Self {
        let n = projection.node_count();
        if n == 0 {
            return Self::default();
        }

        // Combined degree: out + in (undirected treatment).
        let mut deg: Vec<u32> = (0..n as NodeIndex)
            .map(|v| (projection.out_degree(v) + projection.in_degree(v)) as u32)
            .collect();

        let mut coreness = vec![0u32; n];
        let mut removed = vec![false; n];
        let mut remaining = n;
        let mut k = 0u32;

        while remaining > 0 {
            // Find the minimum degree among non-removed nodes.
            let min_deg = (0..n)
                .filter(|&v| !removed[v])
                .map(|v| deg[v])
                .min()
                .unwrap_or(0);

            k = k.max(min_deg);

            // Peel all nodes with degree <= k (iteratively, since removing
            // a node may reduce neighbours' degrees below k).
            let mut changed = true;
            while changed {
                changed = false;
                for v in 0..n {
                    if !removed[v] && deg[v] <= k {
                        coreness[v] = k;
                        removed[v] = true;
                        remaining -= 1;
                        changed = true;
                        // Reduce degree of neighbours.
                        for &nb in projection.out_neighbors(v as NodeIndex) {
                            if !removed[nb as usize] {
                                deg[nb as usize] = deg[nb as usize].saturating_sub(1);
                            }
                        }
                        for &nb in projection.in_neighbors(v as NodeIndex) {
                            if !removed[nb as usize] {
                                deg[nb as usize] = deg[nb as usize].saturating_sub(1);
                            }
                        }
                    }
                }
            }

            k += 1;
        }

        let max_coreness = coreness.iter().copied().max().unwrap_or(0);
        Self {
            coreness,
            max_coreness,
        }
    }
}
