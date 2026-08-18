/* hnsqr/src/graph/analytics/triangles.rs */
//!▫~•◦-------------------------------‣
//! # Triangle Count — Sorted-Merge Node-Iterator
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Counts undirected triangles using the standard sorted-adjacency merge
//! approach.  O(m^{3/2}) time.  [BENCH REQUIRED] for large graphs.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::analytics::projection::GraphProjection;
use crate::NodeIndex;

/// Triangle count result.
#[derive(Debug, Default)]
pub struct TriangleCount {
    /// Total number of triangles in the undirected projection.
    pub triangles: u64,
    /// Clustering coefficient per node (ratio of actual triangles to possible).
    pub local_clustering: Vec<f64>,
}

impl TriangleCount {
    /// Counts triangles.  Treats the graph as undirected by using both
    /// outgoing and incoming edges combined into a sorted neighbour set.
    pub fn compute(projection: &dyn GraphProjection) -> Self {
        let n = projection.node_count();

        // Build sorted undirected neighbour sets for each node.
        let adj: Vec<Vec<NodeIndex>> = (0..n as NodeIndex)
            .map(|v| {
                let mut nb: Vec<NodeIndex> = projection
                    .out_neighbors(v)
                    .iter()
                    .chain(projection.in_neighbors(v).iter())
                    .copied()
                    .filter(|&u| u != v)
                    .collect();
                nb.sort_unstable();
                nb.dedup();
                nb
            })
            .collect();

        let mut triangles = 0u64;
        let mut local_t = vec![0u64; n]; // triangle contributions per node

        for u in 0..n as NodeIndex {
            for &v in &adj[u as usize] {
                if v <= u {
                    continue; // Only process each pair (u,v) with u < v once.
                }
                // Count common neighbours using sorted merge.
                let nu = &adj[u as usize];
                let nv = &adj[v as usize];
                let common = Self::sorted_intersection_count(nu, nv);
                triangles += common as u64;
                local_t[u as usize] += common as u64;
                local_t[v as usize] += common as u64;
            }
        }

        // Local clustering coefficient: 2 * t_v / (k_v * (k_v - 1))
        let local_clustering: Vec<f64> = (0..n)
            .map(|v| {
                let kv = adj[v].len() as f64;
                if kv < 2.0 {
                    0.0
                } else {
                    2.0 * local_t[v] as f64 / (kv * (kv - 1.0))
                }
            })
            .collect();

        Self { triangles, local_clustering }
    }

    fn sorted_intersection_count(a: &[NodeIndex], b: &[NodeIndex]) -> usize {
        let mut count = 0;
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            match a[i].cmp(&b[j]) {
                std::cmp::Ordering::Equal => {
                    count += 1;
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
            }
        }
        count
    }
}
