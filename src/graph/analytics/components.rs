/* hnsqr/src/graph/analytics/components.rs */
//!▫~•◦-------------------------------‣
//! # Connected Components — Parallel Union-Find
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Computes weakly-connected components using a path-compressed Union-Find
//! over the undirected projection (both out and in edges).
//!
//! [BENCH REQUIRED] — parallelism scaling for large sparse graphs.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::atomic::{AtomicU32, Ordering};

use crate::NodeIndex;
use crate::graph::analytics::projection::GraphProjection;

/// Result of connected-component computation.
#[derive(Debug, Default)]
pub struct ConnectedComponents {
    /// Component ID per node (compact, 0-based after relabelling).
    pub component: Vec<u32>,
    /// Number of distinct components.
    pub num_components: usize,
    /// Size of each component, indexed by component ID.
    pub component_sizes: Vec<usize>,
}

struct UnionFind {
    parent: Vec<AtomicU32>,
    rank: Vec<AtomicU32>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n as u32).map(AtomicU32::new).collect(),
            rank: (0..n).map(|_| AtomicU32::new(0)).collect(),
        }
    }

    fn find(&self, mut x: u32) -> u32 {
        loop {
            let px = self.parent[x as usize].load(Ordering::Relaxed);
            if px == x {
                return x;
            }
            // Path compression (best-effort, not guaranteed under concurrency).
            let gpx = self.parent[px as usize].load(Ordering::Relaxed);
            let _ = self.parent[x as usize].compare_exchange(
                px,
                gpx,
                Ordering::Relaxed,
                Ordering::Relaxed,
            );
            x = px;
        }
    }

    fn union(&self, a: u32, b: u32) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        // Union by rank.
        if self.rank[ra as usize].load(Ordering::Relaxed)
            < self.rank[rb as usize].load(Ordering::Relaxed)
        {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize].store(ra, Ordering::Relaxed);
        if self.rank[ra as usize].load(Ordering::Relaxed)
            == self.rank[rb as usize].load(Ordering::Relaxed)
        {
            self.rank[ra as usize].fetch_add(1, Ordering::Relaxed);
        }
    }
}

impl ConnectedComponents {
    /// Computes weakly-connected components.
    pub fn compute(projection: &dyn GraphProjection) -> Self {
        let n = projection.node_count();
        if n == 0 {
            return Self::default();
        }

        let uf = UnionFind::new(n);

        // Process all edges (treat as undirected).
        for u in 0..n as NodeIndex {
            for &v in projection.out_neighbors(u) {
                uf.union(u, v);
            }
        }

        // Collect component assignments.
        let raw: Vec<u32> = (0..n as u32).map(|v| uf.find(v)).collect();

        // Relabel to compact 0-based component IDs.
        let mut label_map = std::collections::HashMap::new();
        let mut next_label = 0u32;
        let component: Vec<u32> = raw
            .iter()
            .map(|&r| {
                *label_map.entry(r).or_insert_with(|| {
                    let l = next_label;
                    next_label += 1;
                    l
                })
            })
            .collect();

        let num_components = next_label as usize;
        let mut component_sizes = vec![0usize; num_components];
        for &c in &component {
            component_sizes[c as usize] += 1;
        }

        Self {
            component,
            num_components,
            component_sizes,
        }
    }
}
