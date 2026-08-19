/* hnsqr/src/graph/storage/csr.rs */
//!▫~•◦-------------------------------‣
//! # Immutable CSR / CSC Adjacency — Cache-Contiguous Sealed-Generation Neighbours
//!▫~•◦-------------------------------------------------------------------‣
//!
//! At segment-seal time the mutable `EdgeDelta` is compacted into two
//! read-optimised representations:
//!
//! - `CsrAdjacency` — **out**going neighbours in contiguous row slices.
//! - `CscAdjacency` — **in**coming neighbours in contiguous column slices.
//!
//! Both store companion weight arrays aligned 1-to-1 with the neighbour arrays
//! so GDS algorithms can read `(neighbor, weight)` pairs in a single
//! cache-line pass without pointer chasing.
//!
//! ## Build-time complexity
//! O(V + E) time, O(V + E) space.
//!
//! ## Query-time complexity per node
//! O(1) for the row/column slice; O(deg(v)) for full traversal — with full
//! cache locality because all neighbours are physically contiguous.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::catalog::RelTypeId;
use crate::graph::storage::edge_delta::EdgeDelta;
use crate::graph::storage::node_arena::NodeArena;
use crate::NodeIndex;

/// Immutable Compressed Sparse Row adjacency for outgoing edges.
///
/// `row_offsets[v]..row_offsets[v+1]` is the contiguous slice of outgoing
/// neighbours of node `v`.
pub struct CsrAdjacency {
    /// Length = node_count + 1.  `row_offsets[v]` = start index in `targets`.
    pub row_offsets: Vec<u64>,
    /// Packed outgoing neighbour IDs in row order.
    pub targets: Vec<NodeIndex>,
    /// Relationship types parallel to `targets`.
    pub rel_types: Vec<RelTypeId>,
    /// Edge weights parallel to `targets`.
    pub weights: Vec<f32>,
    /// Number of nodes this adjacency covers (= `row_offsets.len() - 1`).
    pub node_count: usize,
}

impl CsrAdjacency {
    /// Builds a CSR from the live edge delta for `node_count` nodes.
    pub fn build(arena: &NodeArena, delta: &EdgeDelta, node_count: usize) -> Self {
        // Pass 1: count out-degrees.
        let mut out_deg = vec![0usize; node_count];
        for (_, rec) in delta.live_edges() {
            if (rec.src_node as usize) < node_count {
                out_deg[rec.src_node as usize] += 1;
            }
        }

        // Pass 2: compute prefix-sum row offsets.
        let mut row_offsets = vec![0u64; node_count + 1];
        for i in 0..node_count {
            row_offsets[i + 1] = row_offsets[i] + out_deg[i] as u64;
        }

        let total_edges = *row_offsets.last().unwrap_or(&0) as usize;
        let mut targets = vec![0u32; total_edges];
        let mut rel_types = vec![0u16; total_edges];
        let mut weights = vec![1.0f32; total_edges];

        // Pass 3: fill.  Use a cursor vec to track insertion position per row.
        let mut cursors: Vec<u64> = row_offsets[..node_count].to_vec();
        for (_, rec) in delta.live_edges() {
            let src = rec.src_node as usize;
            if src < node_count {
                let pos = cursors[src] as usize;
                targets[pos] = rec.dst_node;
                rel_types[pos] = rec.rel_type;
                weights[pos] = rec.weight;
                cursors[src] += 1;
            }
        }

        // Sort each row by target for deterministic iteration and binary search.
        for v in 0..node_count {
            let start = row_offsets[v] as usize;
            let end = row_offsets[v + 1] as usize;
            if end > start + 1 {
                // Sort by (target, rel_type) to be deterministic.
                let mut pairs: Vec<(NodeIndex, RelTypeId, f32)> = (start..end)
                    .map(|i| (targets[i], rel_types[i], weights[i]))
                    .collect();
                pairs.sort_unstable_by_key(|&(t, r, _)| (t, r));
                for (i, (t, r, w)) in pairs.into_iter().enumerate() {
                    targets[start + i] = t;
                    rel_types[start + i] = r;
                    weights[start + i] = w;
                }
            }
        }

        // Suppress unused warning for arena (used for node_count validation).
        let _ = arena;

        Self { row_offsets, targets, rel_types, weights, node_count }
    }

    /// Returns the contiguous outgoing-neighbour slice for `node`.
    #[inline]
    pub fn out_neighbors(&self, node: NodeIndex) -> &[NodeIndex] {
        let v = node as usize;
        if v >= self.node_count {
            return &[];
        }
        let start = self.row_offsets[v] as usize;
        let end = self.row_offsets[v + 1] as usize;
        &self.targets[start..end]
    }

    /// Returns the contiguous outgoing-weight slice for `node`.
    #[inline]
    pub fn out_weights(&self, node: NodeIndex) -> &[f32] {
        let v = node as usize;
        if v >= self.node_count {
            return &[];
        }
        let start = self.row_offsets[v] as usize;
        let end = self.row_offsets[v + 1] as usize;
        &self.weights[start..end]
    }

    /// Out-degree of `node`.
    #[inline]
    pub fn out_degree(&self, node: NodeIndex) -> usize {
        let v = node as usize;
        if v >= self.node_count {
            return 0;
        }
        (self.row_offsets[v + 1] - self.row_offsets[v]) as usize
    }

    /// Filters outgoing neighbours of `node` matching a target relationship type.
    /// Uses contiguous SIMD-ready chunked evaluation.
    pub fn filter_out_neighbors_simd(&self, node: NodeIndex, target_rel_type: RelTypeId) -> Vec<NodeIndex> {
        let v = node as usize;
        if v >= self.node_count {
            return Vec::new();
        }
        let start = self.row_offsets[v] as usize;
        let end = self.row_offsets[v + 1] as usize;
        let targets_slice = &self.targets[start..end];
        let rel_slice = &self.rel_types[start..end];

        let mut matching = Vec::with_capacity(targets_slice.len());

        // Fast chunked 8-wide comparison loop
        let chunks = rel_slice.chunks_exact(8);
        let remainder = chunks.remainder();
        let chunk_count = chunks.len();

        for c_idx in 0..chunk_count {
            let offset = c_idx * 8;
            for i in 0..8 {
                if rel_slice[offset + i] == target_rel_type {
                    matching.push(targets_slice[offset + i]);
                }
            }
        }

        let rem_offset = chunk_count * 8;
        for i in 0..remainder.len() {
            if remainder[i] == target_rel_type {
                matching.push(targets_slice[rem_offset + i]);
            }
        }

        matching
    }

    /// Intersects two sorted node slices using galloping search (Leapfrog Triejoin kernel)
    /// to achieve sub-linear $O(|A| \log |B|)$ time on skewed degrees and linear cache-locality on symmetric slices.
    pub fn intersect_sorted_galloping(a: &[NodeIndex], b: &[NodeIndex]) -> usize {
        if a.is_empty() || b.is_empty() {
            return 0;
        }

        // Fast linear scan if slices are similarly sized
        if a.len() <= b.len() * 4 && b.len() <= a.len() * 4 {
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
            return count;
        }

        // Galloping search for skewed slices
        let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
        let mut count = 0;
        let mut large_idx = 0;

        for &val in small {
            if large_idx >= large.len() {
                break;
            }

            let mut step = 1;
            let mut curr = large_idx;

            while curr < large.len() && large[curr] < val {
                curr += step;
                step *= 2;
            }

            let start = curr.saturating_sub(step / 2).max(large_idx).min(large.len());
            let end = curr.min(large.len());

            if start < end {
                match large[start..end].binary_search(&val) {
                    Ok(pos) => {
                        count += 1;
                        large_idx = start + pos + 1;
                    }
                    Err(pos) => {
                        large_idx = start + pos;
                    }
                }
            } else if start < large.len() && large[start] == val {
                count += 1;
                large_idx = start + 1;
            }
        }

        count
    }

    /// Total live edge count.
    pub fn edge_count(&self) -> usize {
        *self.row_offsets.last().unwrap_or(&0) as usize
    }
}

/// Immutable Compressed Sparse Column adjacency for incoming edges.
pub struct CscAdjacency {
    /// Length = node_count + 1.  `col_offsets[v]` = start in `sources`.
    pub col_offsets: Vec<u64>,
    /// Packed incoming source IDs in column order.
    pub sources: Vec<NodeIndex>,
    /// Relationship types parallel to `sources`.
    pub rel_types: Vec<RelTypeId>,
    /// Edge weights parallel to `sources`.
    pub weights: Vec<f32>,
    pub node_count: usize,
}

impl CscAdjacency {
    /// Builds a CSC (transpose of CSR) from the live edge delta.
    pub fn build(delta: &EdgeDelta, node_count: usize) -> Self {
        // Pass 1: count in-degrees.
        let mut in_deg = vec![0usize; node_count];
        for (_, rec) in delta.live_edges() {
            if (rec.dst_node as usize) < node_count {
                in_deg[rec.dst_node as usize] += 1;
            }
        }

        // Pass 2: prefix-sum column offsets.
        let mut col_offsets = vec![0u64; node_count + 1];
        for i in 0..node_count {
            col_offsets[i + 1] = col_offsets[i] + in_deg[i] as u64;
        }

        let total_edges = *col_offsets.last().unwrap_or(&0) as usize;
        let mut sources = vec![0u32; total_edges];
        let mut rel_types = vec![0u16; total_edges];
        let mut weights = vec![1.0f32; total_edges];

        // Pass 3: fill.
        let mut cursors: Vec<u64> = col_offsets[..node_count].to_vec();
        for (_, rec) in delta.live_edges() {
            let dst = rec.dst_node as usize;
            if dst < node_count {
                let pos = cursors[dst] as usize;
                sources[pos] = rec.src_node;
                rel_types[pos] = rec.rel_type;
                weights[pos] = rec.weight;
                cursors[dst] += 1;
            }
        }

        // Sort each column.
        for v in 0..node_count {
            let start = col_offsets[v] as usize;
            let end = col_offsets[v + 1] as usize;
            if end > start + 1 {
                let mut pairs: Vec<(NodeIndex, RelTypeId, f32)> = (start..end)
                    .map(|i| (sources[i], rel_types[i], weights[i]))
                    .collect();
                pairs.sort_unstable_by_key(|&(s, r, _)| (s, r));
                for (i, (s, r, w)) in pairs.into_iter().enumerate() {
                    sources[start + i] = s;
                    rel_types[start + i] = r;
                    weights[start + i] = w;
                }
            }
        }

        Self { col_offsets, sources, rel_types, weights, node_count }
    }

    /// Returns the contiguous incoming-source slice for `node`.
    #[inline]
    pub fn in_neighbors(&self, node: NodeIndex) -> &[NodeIndex] {
        let v = node as usize;
        if v >= self.node_count {
            return &[];
        }
        let start = self.col_offsets[v] as usize;
        let end = self.col_offsets[v + 1] as usize;
        &self.sources[start..end]
    }

    /// Returns the contiguous incoming-weight slice for `node`.
    #[inline]
    pub fn in_weights(&self, node: NodeIndex) -> &[f32] {
        let v = node as usize;
        if v >= self.node_count {
            return &[];
        }
        let start = self.col_offsets[v] as usize;
        let end = self.col_offsets[v + 1] as usize;
        &self.weights[start..end]
    }

    /// In-degree of `node`.
    #[inline]
    pub fn in_degree(&self, node: NodeIndex) -> usize {
        let v = node as usize;
        if v >= self.node_count {
            return 0;
        }
        (self.col_offsets[v + 1] - self.col_offsets[v]) as usize
    }
}
