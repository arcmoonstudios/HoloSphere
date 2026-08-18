/* hnsqr/src/graph/analytics/bfs.rs */
//!▫~•◦-------------------------------‣
//! # Breadth-First Search — Level-Synchronous BFS
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Produces hop-distances from a source node to all reachable nodes.
//! Used as a building block for bidirectional BFS shortest-path in
//! unweighted graphs and for diameter estimation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::VecDeque;

use crate::graph::analytics::projection::GraphProjection;
use crate::NodeIndex;

/// BFS result from a single source.
#[derive(Debug, Default)]
pub struct BfsResult {
    /// Hop-distance from source to each node. `u32::MAX` = unreachable.
    pub distances: Vec<u32>,
    /// BFS discovery order.
    pub order: Vec<NodeIndex>,
}

impl BfsResult {
    /// Runs BFS from `source` on the projection's outgoing edges.
    pub fn from_source(projection: &dyn GraphProjection, source: NodeIndex) -> Self {
        let n = projection.node_count();
        let mut distances = vec![u32::MAX; n];
        let mut order = Vec::new();
        let mut queue = VecDeque::new();

        if (source as usize) < n {
            distances[source as usize] = 0;
            queue.push_back(source);
        }

        while let Some(u) = queue.pop_front() {
            order.push(u);
            let d = distances[u as usize];
            for &v in projection.out_neighbors(u) {
                if distances[v as usize] == u32::MAX {
                    distances[v as usize] = d + 1;
                    queue.push_back(v);
                }
            }
        }

        Self { distances, order }
    }
}
