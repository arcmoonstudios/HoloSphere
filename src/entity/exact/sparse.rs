/* holosphere/src/entity/exact/sparse.rs */
//!▫~•◦-------------------------------‣
//! # Sparse Gather Scan Operator
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Physical exact retrieval operator optimized for sparse eligible sets.
//! Gathers and scores candidate vectors with physical page/cacheline locality sorting.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::entity::exact::metric::{DistanceFunction, ScoredEntity, resolve_metric};
use crate::entity::id::{EntityIndex, NULL_ROW_REF};
use crate::entity::read::EntityReadSnapshot;

/// Executes an exact Top-K scan over sparse eligible entity indices.
pub fn sparse_gather_scan(
    snapshot: &EntityReadSnapshot,
    query: &[f32],
    eligible_indices: &[EntityIndex],
    k: usize,
    metric_fn: DistanceFunction,
) -> (Vec<ScoredEntity>, usize) {
    if k == 0 || eligible_indices.is_empty() {
        return (Vec::new(), 0);
    }

    let metric = resolve_metric(metric_fn);
    let arena = &snapshot.segment.arena;
    let vector_arena = &snapshot.segment.vector_arena;

    // 1. Locality optimization: resolve candidates and sort by physical vector_row
    let mut candidates: Vec<(EntityIndex, u64, u32)> = Vec::with_capacity(eligible_indices.len());
    for &idx in eligible_indices {
        if let Some(header) = arena.get(idx) {
            if header.vector_row != NULL_ROW_REF {
                if let Some(entity_id) = arena.index_to_id(idx) {
                    candidates.push((idx, entity_id, header.vector_row));
                }
            }
        }
    }

    // Sort by vector_row to maximize memory streaming and TLB hit rates
    candidates.sort_unstable_by_key(|&(_, _, vrow)| vrow);

    // 2. Score candidates and accumulate in bounded min-heap
    // Using Reverse<ScoredEntity> so the lowest score sits at the root of the heap.
    let mut heap: BinaryHeap<Reverse<ScoredEntity>> = BinaryHeap::with_capacity(k + 1);
    let mut scored_count = 0;

    for (idx, entity_id, vrow) in candidates {
        if let Some(score) = vector_arena.with_row(vrow, |v| metric.score_simd(query, v)) {
            scored_count += 1;
            let scored = ScoredEntity {
                entity_id,
                entity_index: idx,
                score,
            };

            if heap.len() < k {
                heap.push(Reverse(scored));
            } else if let Some(min_top) = heap.peek() {
                if scored > min_top.0 {
                    heap.pop();
                    heap.push(Reverse(scored));
                }
            }
        }
    }

    // 3. Extract and sort results in canonical descending order
    let mut results: Vec<ScoredEntity> = heap.into_iter().map(|Reverse(e)| e).collect();
    results.sort_unstable_by(|a, b| b.cmp(a)); // Score DESC, EntityId ASC

    (results, scored_count)
}
