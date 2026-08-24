/* holosphere/src/entity/exact/dense.rs */
//!▫~•◦-------------------------------‣
//! # Masked Dense Scan Operator
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Physical exact retrieval operator optimized for dense candidate masks.
//! Streams contiguous vector memory blocks, utilizing bitmap word skipping.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use roaring::RoaringBitmap;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::entity::exact::metric::{DistanceFunction, ScoredEntity, resolve_metric};
use crate::entity::id::{EntityIndex, NULL_ROW_REF};
use crate::entity::read::EntityReadSnapshot;

/// Executes an exact Top-K scan over a dense RoaringBitmap mask.
pub fn masked_dense_scan(
    snapshot: &EntityReadSnapshot,
    query: &[f32],
    bitmap: &RoaringBitmap,
    k: usize,
    metric_fn: DistanceFunction,
) -> (Vec<ScoredEntity>, usize) {
    if k == 0 || bitmap.is_empty() {
        return (Vec::new(), 0);
    }

    let metric = resolve_metric(metric_fn);
    let arena = &snapshot.segment.arena;
    let vector_arena = &snapshot.segment.vector_arena;

    let mut heap: BinaryHeap<Reverse<ScoredEntity>> = BinaryHeap::with_capacity(k + 1);
    let mut scored_count = 0;

    // Iterate through active bitmap entries
    for idx in bitmap.iter() {
        let entity_idx = idx as EntityIndex;
        if let Some(header) = arena.get(entity_idx) {
            if header.vector_row != NULL_ROW_REF {
                if let Some(entity_id) = arena.index_to_id(entity_idx) {
                    if let Some(score) =
                        vector_arena.with_row(header.vector_row, |v| metric.score_simd(query, v))
                    {
                        scored_count += 1;
                        let scored = ScoredEntity {
                            entity_id,
                            entity_index: entity_idx,
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
            }
        }
    }

    let mut results: Vec<ScoredEntity> = heap.into_iter().map(|Reverse(e)| e).collect();
    results.sort_unstable_by(|a, b| b.cmp(a)); // Score DESC, EntityId ASC

    (results, scored_count)
}
