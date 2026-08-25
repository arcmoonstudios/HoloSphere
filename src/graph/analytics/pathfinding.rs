/* holosphere/src/graph/analytics/pathfinding.rs */
//!▫~•◦-------------------------------‣
//! # Pathfinding — Bidirectional BFS (unweighted) & Bidirectional Dijkstra
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Two separate algorithms are provided and kept distinct:
//!
//! - `ShortestPath::unweighted` — bidirectional BFS for unit-weight graphs.
//!   Correct termination: both frontiers meet and the shared node's combined
//!   distance is verified to be optimal.
//!
//! - `ShortestPath::weighted` — bidirectional Dijkstra for non-negative edge
//!   weights.  Correct termination: the stopping criterion uses the
//!   `dist_fwd[u] + dist_rev[u] >= μ` condition, not merely the first meeting.
//!
//! The original proposal used a single-function mix of both and had a
//! premature termination bug.  Those bugs are fixed here.
//!
//! [BENCH REQUIRED] — neither function claims to be faster than the
//! standard Dijkstra without measurement.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::NodeIndex;
use crate::graph::analytics::projection::GraphProjection;

/// Shortest-path result.
#[derive(Debug)]
pub struct ShortestPath {
    /// Total cost from source to target.  `None` if unreachable.
    pub cost: Option<f32>,
    /// Hop count (for unweighted) or `None` when using weighted variant.
    pub hops: Option<u32>,
}

// ── Min-heap entry ────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq)]
struct State {
    cost: f32,
    node: NodeIndex,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower cost has higher priority.
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| self.node.cmp(&other.node))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Pathfinding engine.
pub struct PathfindingEngine;

impl PathfindingEngine {
    /// Bidirectional BFS for **unweighted** graphs (unit edge cost).
    ///
    /// Correct termination: once a node is settled in both frontier sets,
    /// verify the minimum over all meeting nodes.
    pub fn unweighted(
        projection: &dyn GraphProjection,
        source: NodeIndex,
        target: NodeIndex,
    ) -> ShortestPath {
        if source == target {
            return ShortestPath {
                cost: Some(0.0),
                hops: Some(0),
            };
        }
        let n = projection.node_count();
        if (source as usize) >= n || (target as usize) >= n {
            return ShortestPath {
                cost: None,
                hops: None,
            };
        }

        let mut dist_fwd = vec![u32::MAX; n];
        let mut dist_rev = vec![u32::MAX; n];
        let mut frontier_fwd = std::collections::VecDeque::new();
        let mut frontier_rev = std::collections::VecDeque::new();

        dist_fwd[source as usize] = 0;
        dist_rev[target as usize] = 0;
        frontier_fwd.push_back(source);
        frontier_rev.push_back(target);

        let mut best = u32::MAX;

        while !frontier_fwd.is_empty() || !frontier_rev.is_empty() {
            // Expand the smaller frontier for balance.
            if !frontier_fwd.is_empty()
                && (frontier_rev.is_empty() || frontier_fwd.len() <= frontier_rev.len())
            {
                if let Some(u) = frontier_fwd.pop_front() {
                    let d = dist_fwd[u as usize];
                    if d >= best {
                        continue;
                    }
                    for &v in projection.out_neighbors(u) {
                        if dist_fwd[v as usize] == u32::MAX {
                            dist_fwd[v as usize] = d + 1;
                            frontier_fwd.push_back(v);
                        }
                        // Meeting point check.
                        if dist_rev[v as usize] != u32::MAX {
                            let candidate = d + 1 + dist_rev[v as usize];
                            if candidate < best {
                                best = candidate;
                            }
                        }
                    }
                }
            } else if let Some(u) = frontier_rev.pop_front() {
                let d = dist_rev[u as usize];
                if d >= best {
                    continue;
                }
                for &v in projection.in_neighbors(u) {
                    if dist_rev[v as usize] == u32::MAX {
                        dist_rev[v as usize] = d + 1;
                        frontier_rev.push_back(v);
                    }
                    if dist_fwd[v as usize] != u32::MAX {
                        let candidate = dist_fwd[v as usize] + d + 1;
                        if candidate < best {
                            best = candidate;
                        }
                    }
                }
            }

            // Stopping criterion: minimum frontier distances exceed best.
            let min_fwd = frontier_fwd
                .front()
                .map(|&u| dist_fwd[u as usize])
                .unwrap_or(u32::MAX);
            let min_rev = frontier_rev
                .front()
                .map(|&u| dist_rev[u as usize])
                .unwrap_or(u32::MAX);
            if min_fwd.saturating_add(min_rev) >= best {
                break;
            }
        }

        if best == u32::MAX {
            ShortestPath {
                cost: None,
                hops: None,
            }
        } else {
            ShortestPath {
                cost: Some(best as f32),
                hops: Some(best),
            }
        }
    }

    /// Bidirectional Dijkstra for **non-negative weighted** graphs.
    ///
    /// Uses the standard stopping criterion:
    ///   `dist_fwd[u] + dist_rev[u] >= μ`  where `μ` is the best path so far
    /// (not merely the first time a node appears in both frontiers).
    pub fn weighted(
        projection: &dyn GraphProjection,
        source: NodeIndex,
        target: NodeIndex,
    ) -> ShortestPath {
        if source == target {
            return ShortestPath {
                cost: Some(0.0),
                hops: None,
            };
        }
        let n = projection.node_count();
        if (source as usize) >= n || (target as usize) >= n {
            return ShortestPath {
                cost: None,
                hops: None,
            };
        }

        let mut dist_fwd = vec![f32::INFINITY; n];
        let mut dist_rev = vec![f32::INFINITY; n];
        let mut settled_fwd = vec![false; n];
        let mut settled_rev = vec![false; n];
        let mut pq_fwd: BinaryHeap<State> = BinaryHeap::new();
        let mut pq_rev: BinaryHeap<State> = BinaryHeap::new();

        dist_fwd[source as usize] = 0.0;
        dist_rev[target as usize] = 0.0;
        pq_fwd.push(State {
            cost: 0.0,
            node: source,
        });
        pq_rev.push(State {
            cost: 0.0,
            node: target,
        });

        let mut mu = f32::INFINITY; // best complete-path cost found so far

        loop {
            let top_fwd = pq_fwd.peek().map(|s| s.cost).unwrap_or(f32::INFINITY);
            let top_rev = pq_rev.peek().map(|s| s.cost).unwrap_or(f32::INFINITY);

            // Stopping criterion: frontiers cannot improve mu.
            if top_fwd + top_rev >= mu {
                break;
            }

            // Expand the cheaper frontier.
            if top_fwd <= top_rev {
                if let Some(State { cost, node: u }) = pq_fwd.pop() {
                    if cost > dist_fwd[u as usize] {
                        continue; // Stale entry.
                    }
                    if settled_fwd[u as usize] {
                        continue;
                    }
                    settled_fwd[u as usize] = true;

                    let neighbors = projection.out_neighbors(u);
                    let weights = projection.out_weights(u);
                    for (i, &v) in neighbors.iter().enumerate() {
                        let w = if weights.is_empty() { 1.0 } else { weights[i] };
                        debug_assert!(
                            w >= 0.0,
                            "Negative edge weight violates Dijkstra precondition"
                        );
                        let nd = cost + w;
                        if nd < dist_fwd[v as usize] {
                            dist_fwd[v as usize] = nd;
                            pq_fwd.push(State { cost: nd, node: v });
                        }
                        // Meeting check.
                        if dist_rev[v as usize].is_finite() {
                            let candidate = nd + dist_rev[v as usize];
                            if candidate < mu {
                                mu = candidate;
                            }
                        }
                    }
                }
            } else {
                if let Some(State { cost, node: u }) = pq_rev.pop() {
                    if cost > dist_rev[u as usize] {
                        continue;
                    }
                    if settled_rev[u as usize] {
                        continue;
                    }
                    settled_rev[u as usize] = true;

                    let neighbors = projection.in_neighbors(u);
                    let weights = projection.in_weights(u);
                    for (i, &v) in neighbors.iter().enumerate() {
                        let w = if weights.is_empty() { 1.0 } else { weights[i] };
                        let nd = cost + w;
                        if nd < dist_rev[v as usize] {
                            dist_rev[v as usize] = nd;
                            pq_rev.push(State { cost: nd, node: v });
                        }
                        if dist_fwd[v as usize].is_finite() {
                            let candidate = dist_fwd[v as usize] + nd;
                            if candidate < mu {
                                mu = candidate;
                            }
                        }
                    }
                }
            }

            if pq_fwd.is_empty() && pq_rev.is_empty() {
                break;
            }
        }

        if mu.is_finite() {
            ShortestPath {
                cost: Some(mu),
                hops: None,
            }
        } else {
            ShortestPath {
                cost: None,
                hops: None,
            }
        }
    }
}
