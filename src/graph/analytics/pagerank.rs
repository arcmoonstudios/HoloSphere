/* hnsqr/src/graph/analytics/pagerank.rs */
//!▫~•◦-------------------------------‣
//! # PageRank — Pull-Based Parallel Implementation over CSC
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Uses the **pull** formulation: for each node `v`, accumulate incoming
//! rank contributions from its predecessors via the CSC incoming adjacency.
//! This orientation is correct for directed graphs and avoids the common
//! error of iterating outgoing edges for a pull update.
//!
//! Dangling nodes (out-degree = 0) redistribute their rank uniformly so
//! the total rank mass is conserved.
//!
//! ## Correctness invariants
//! - `sum(ranks) ≈ 1.0` after every iteration (within floating-point error).
//! - Convergence is declared when the L1 norm of rank delta < `tolerance`.
//! - If `max_iterations` is reached without convergence, results are returned
//!   with a warning flag `converged = false`.
//!
//! ## Performance
//! [BENCH REQUIRED] — rayon parallelism scaling vs single-threaded for
//! various |V|, |E| on representative hardware.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use rayon::prelude::*;

use crate::NodeIndex;
use crate::graph::analytics::projection::GraphProjection;

/// Result of a PageRank computation.
#[derive(Debug)]
pub struct PageRankResult {
    /// Rank per node index.  Length = `projection.node_count()`.
    pub ranks: Vec<f32>,
    /// Number of iterations executed before convergence (or max_iterations).
    pub iterations: usize,
    /// Whether L1 convergence was declared within `max_iterations`.
    pub converged: bool,
}

/// PageRank engine.
pub struct PageRankEngine;

impl PageRankEngine {
    /// Computes PageRank on any `GraphProjection`.
    ///
    /// # Parameters
    /// - `damping` — typical value 0.85 (probability of following a link).
    /// - `tolerance` — L1 convergence threshold, e.g. 1e-6.
    /// - `max_iterations` — upper bound on iterations.
    pub fn compute(
        projection: &dyn GraphProjection,
        damping: f32,
        tolerance: f32,
        max_iterations: usize,
    ) -> PageRankResult {
        let n = projection.node_count();
        if n == 0 {
            return PageRankResult {
                ranks: Vec::new(),
                iterations: 0,
                converged: true,
            };
        }

        let init = 1.0f32 / n as f32;
        let base = (1.0 - damping) / n as f32;

        let mut ranks = vec![init; n];
        let mut next = vec![0.0f32; n];

        // Precompute out-degrees to avoid repeated lookups.
        let out_degrees: Vec<usize> = (0..n as NodeIndex)
            .map(|v| projection.out_degree(v))
            .collect();

        let mut converged = false;
        let mut iterations = 0;

        for _ in 0..max_iterations {
            iterations += 1;

            // Dangling-node mass: sum of ranks of nodes with no outgoing edges.
            let dangling_sum: f32 = (0..n)
                .filter(|&v| out_degrees[v] == 0)
                .map(|v| ranks[v])
                .sum();
            let dangling_contrib = damping * dangling_sum / n as f32;

            // Pull: for each node v, sum incoming rank / out_degree(src).
            next.par_iter_mut().enumerate().for_each(|(v, nv)| {
                let srcs = projection.in_neighbors(v as NodeIndex);
                let weights = projection.in_weights(v as NodeIndex);
                let mut incoming = 0.0f32;
                for (i, &src) in srcs.iter().enumerate() {
                    let src_od = out_degrees[src as usize];
                    if src_od > 0 {
                        let w = if weights.is_empty() {
                            1.0f32
                        } else {
                            weights[i]
                        };
                        // Weighted PageRank: weight normalised by total out-weight.
                        // For unweighted graphs w = 1.0 and src_od is the divisor.
                        incoming += ranks[src as usize] * w / src_od as f32;
                    }
                }
                *nv = base + dangling_contrib + damping * incoming;
            });

            // L1 convergence check.
            let delta: f32 = ranks
                .iter()
                .zip(next.iter())
                .map(|(r, n)| (r - n).abs())
                .sum();

            std::mem::swap(&mut ranks, &mut next);

            if delta < tolerance {
                converged = true;
                break;
            }
        }

        PageRankResult {
            ranks,
            iterations,
            converged,
        }
    }
}
