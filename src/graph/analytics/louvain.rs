/* hnsqr/src/graph/analytics/louvain.rs */
//!▫~•◦-------------------------------‣
//! # Louvain Community Detection — Modularity-Maximising Phase-1 Loop
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Implements Phase 1 of the Louvain method (Blondel et al., 2008):
//! iterate over all nodes and greedily move each node to the neighbouring
//! community that maximises the modularity gain `ΔQ`.
//!
//! This is the standard greedy local-move phase.  Phase 2 (community
//! coarsening and recursive application) is marked TODO and will be added
//! in a follow-on commit once Phase 1 is verified by the unit tests.
//!
//! ## Modularity gain formula
//!
//! For moving node `i` into community `C`:
//!
//! ```text
//! ΔQ = [k_{i,in}/m  -  (Σ_tot + k_i) * k_i / (2m²)]
//!    - [0            -  Σ_tot * k_i / (2m²) - k_i² / (4m²)]
//! simplified to:
//! ΔQ = k_{i,in}/m - k_i * Σ_tot / (2m²)
//! ```
//!
//! where:
//! - `k_{i,in}` = sum of weights of edges from `i` to members of `C`
//! - `k_i`      = weighted degree of node `i`
//! - `Σ_tot`    = sum of all edge weights of nodes in `C`
//! - `m`        = total edge weight in the graph
//!
//! ## Performance
//! [BENCH REQUIRED] — no throughput claim is made until measured.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::analytics::projection::GraphProjection;
use crate::NodeIndex;

/// Result of the Louvain community-detection phase.
#[derive(Debug, Default)]
pub struct LouvainResult {
    /// Community ID per node.  Length = `projection.node_count()`.
    pub community: Vec<u32>,
    /// Number of distinct communities found.
    pub num_communities: usize,
    /// Final modularity Q (in range approximately [-1, 1]; higher is better).
    pub modularity: f64,
    /// Whether Phase 2 (coarsening) has been applied.
    pub phase2_applied: bool,
}

/// Louvain community-detection engine.
pub struct LouvainEngine;

impl LouvainEngine {
    /// Runs Phase 1 (greedy local moves) up to `max_passes` times or until
    /// no improvement is found.
    ///
    /// Returns a `LouvainResult` with `phase2_applied = false` until Phase 2
    /// is implemented.
    pub fn detect(
        projection: &dyn GraphProjection,
        max_passes: usize,
    ) -> LouvainResult {
        let n = projection.node_count();
        if n == 0 {
            return LouvainResult::default();
        }

        // Initial assignment: each node in its own community.
        let mut community: Vec<u32> = (0..n as u32).collect();

        // Precompute weighted degrees.
        let k: Vec<f64> = (0..n as NodeIndex)
            .map(|v| {
                let ws = projection.out_weights(v);
                if ws.is_empty() {
                    projection.out_degree(v) as f64
                } else {
                    ws.iter().map(|&w| w as f64).sum()
                }
            })
            .collect();

        // Total edge weight m (count each undirected edge once by summing
        // directed out-weights and halving).  For directed graphs this treats
        // each directed edge independently, which is standard for directed Louvain.
        let m: f64 = k.iter().sum::<f64>() / 2.0;
        if m < 1e-12 {
            // Empty or weight-less graph.
            return LouvainResult {
                num_communities: n,
                community,
                modularity: 0.0,
                phase2_applied: false,
            };
        }
        let inv_2m = 1.0 / (2.0 * m);

        // Σ_tot[c] = sum of weighted degrees of nodes in community c.
        let mut sigma_tot: Vec<f64> = k.clone();

        for _pass in 0..max_passes {
            let mut improved = false;

            for i in 0..n as NodeIndex {
                let ci = community[i as usize] as usize;
                let ki = k[i as usize];

                // Collect candidate communities from neighbours.
                let neighbors = projection.out_neighbors(i);
                let weights = projection.out_weights(i);

                let mut best_community = ci;
                let mut best_dq = 0.0f64; // only move if ΔQ > 0

                // Group edge weights by neighbour community.
                let mut comm_weights: std::collections::HashMap<usize, f64> =
                    std::collections::HashMap::new();
                for (idx, &nb) in neighbors.iter().enumerate() {
                    let w = if weights.is_empty() { 1.0 } else { weights[idx] as f64 };
                    let cn = community[nb as usize] as usize;
                    *comm_weights.entry(cn).or_insert(0.0) += w;
                }

                // Remove self contribution (current community).
                // k_i_in for current community.
                let ki_in_ci = comm_weights.get(&ci).copied().unwrap_or(0.0);

                for (&cn, &ki_in) in &comm_weights {
                    if cn == ci {
                        continue;
                    }
                    // ΔQ for moving i FROM ci TO cn:
                    // gain from joining cn minus loss from leaving ci.
                    let gain = ki_in / m - ki * sigma_tot[cn] * inv_2m;
                    let loss = ki_in_ci / m - ki * (sigma_tot[ci] - ki) * inv_2m;
                    let dq = gain - loss;
                    if dq > best_dq {
                        best_dq = dq;
                        best_community = cn;
                    }
                }

                if best_community != ci {
                    // Move node i from ci to best_community.
                    sigma_tot[ci] -= ki;
                    sigma_tot[best_community] += ki;
                    community[i as usize] = best_community as u32;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        // Relabel communities to compact 0-based IDs.
        let mut label_map = std::collections::HashMap::new();
        let mut next_label = 0u32;
        let community: Vec<u32> = community
            .into_iter()
            .map(|c| {
                *label_map.entry(c).or_insert_with(|| {
                    let l = next_label;
                    next_label += 1;
                    l
                })
            })
            .collect();

        let num_communities = next_label as usize;

        // Compute final modularity Q.
        let modularity = Self::compute_modularity(&community, projection, m);

        LouvainResult {
            community,
            num_communities,
            modularity,
            phase2_applied: false,
        }
    }

    /// Computes modularity Q = (1/2m) Σ_{ij} [A_{ij} - k_i k_j / 2m] δ(c_i, c_j).
    fn compute_modularity(
        community: &[u32],
        projection: &dyn GraphProjection,
        m: f64,
    ) -> f64 {
        if m < 1e-12 {
            return 0.0;
        }
        let n = projection.node_count();
        let k: Vec<f64> = (0..n as NodeIndex)
            .map(|v| {
                let ws = projection.out_weights(v);
                if ws.is_empty() {
                    projection.out_degree(v) as f64
                } else {
                    ws.iter().map(|&w| w as f64).sum()
                }
            })
            .collect();

        let mut q = 0.0f64;
        for i in 0..n as NodeIndex {
            let neighbors = projection.out_neighbors(i);
            let weights = projection.out_weights(i);
            for (idx, &j) in neighbors.iter().enumerate() {
                if community[i as usize] == community[j as usize] {
                    let w = if weights.is_empty() { 1.0 } else { weights[idx] as f64 };
                    q += w - k[i as usize] * k[j as usize] / (2.0 * m);
                }
            }
        }
        q / (2.0 * m)
    }
}
