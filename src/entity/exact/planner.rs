/* holosphere/src/entity/exact/planner.rs */
//!▫~•◦-------------------------------‣
//! # Exact Retrieval Planner & Cost Model Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates candidate eligibility sets E and dispatches to the optimal
//! physical operator (MaskedDenseScan vs SparseGatherScan) without sacrificing
//! bit-exact Top-K precision.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::entity::eligibility::EligibilitySet;
use crate::entity::exact::dense::masked_dense_scan;
use crate::entity::exact::metric::{DistanceFunction, ScoredEntity, resolve_metric};
use crate::entity::exact::sparse::sparse_gather_scan;
use crate::entity::id::{NULL_ROW_REF, VectorLayoutId};
use crate::entity::read::EntityReadSnapshot;

/// Physical execution operator selected for exact scan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExactScanOperator {
    /// Gather-based vectorized scan over sorted physical rows (optimal for sparse sets).
    SparseGather,
    /// Streaming memory scan with word-skipping over dense masks (optimal for high density).
    MaskedDense,
    /// Reference scalar implementation (ground truth).
    ScalarReference,
}

/// Execution proof for exact retrieval.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExactEligibilityProof {
    pub snapshot_lsn: u64,
    pub entity_generation: u64,
    pub eligible_count: usize,
    pub scored_count: usize,
    pub operator: ExactScanOperator,
    pub metric: DistanceFunction,
    pub globally_exact_over_eligibility: bool,
}

/// Cost model calibrating operator selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExactEligibilityCostModel {
    /// Crossover selectivity threshold in parts-per-million (50,000 = 5.0%).
    pub sparse_dense_crossover_ppm: u32,
}

impl Default for ExactEligibilityCostModel {
    fn default() -> Self {
        Self {
            sparse_dense_crossover_ppm: 50_000, // 5% default [BENCH REQUIRED]
        }
    }
}

/// Exact scan physical plan descriptor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExactScanPlan {
    pub operator: ExactScanOperator,
    pub vector_layout: VectorLayoutId,
    pub eligible_count: usize,
    pub total_count: usize,
}

/// Pinned context binding a point-in-time snapshot and candidate eligibility set.
pub struct ExactRetrievalContext {
    pub snapshot: EntityReadSnapshot,
    pub eligibility: EligibilitySet,
}

impl ExactRetrievalContext {
    pub fn new(snapshot: EntityReadSnapshot, eligibility: EligibilitySet) -> Result<Self, String> {
        if let Err(e) =
            eligibility.validate_against_snapshot(snapshot.lsn, snapshot.segment.generation_id)
        {
            return Err(e.to_string());
        }
        Ok(Self {
            snapshot,
            eligibility,
        })
    }
}

/// Ground-truth scalar reference scorer.
///
/// Scores every eligible entity using the scalar metric implementation.
pub fn exact_top_k_scalar(
    snapshot: &EntityReadSnapshot,
    query: &[f32],
    eligibility: &EligibilitySet,
    k: usize,
    metric_fn: DistanceFunction,
) -> (Vec<ScoredEntity>, usize) {
    if k == 0 || eligibility.eligible_count == 0 {
        return (Vec::new(), 0);
    }

    let metric = resolve_metric(metric_fn);
    let arena = &snapshot.segment.arena;
    let vector_arena = &snapshot.segment.vector_arena;

    let mut heap: BinaryHeap<Reverse<ScoredEntity>> = BinaryHeap::with_capacity(k + 1);
    let mut scored_count = 0;

    let sparse_indices = eligibility.to_sparse_indices();
    for idx in sparse_indices {
        if let Some(header) = arena.get(idx) {
            if header.vector_row != NULL_ROW_REF {
                if let Some(entity_id) = arena.index_to_id(idx) {
                    if let Some(score) =
                        vector_arena.with_row(header.vector_row, |v| metric.score_scalar(query, v))
                    {
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
            }
        }
    }

    let mut results: Vec<ScoredEntity> = heap.into_iter().map(|Reverse(e)| e).collect();
    results.sort_unstable_by(|a, b| b.cmp(a)); // Score DESC, EntityId ASC

    (results, scored_count)
}

/// High-level entry point executing exact Top-K retrieval over an eligibility set.
pub fn exact_top_k(
    snapshot: &EntityReadSnapshot,
    query: &[f32],
    eligibility: &EligibilitySet,
    k: usize,
    metric_fn: DistanceFunction,
    cost_model: Option<&ExactEligibilityCostModel>,
) -> (Vec<ScoredEntity>, ExactEligibilityProof) {
    if let Err(e) =
        eligibility.validate_against_snapshot(snapshot.lsn, snapshot.segment.generation_id)
    {
        panic!("Eligibility validation failed: {e}");
    }

    let default_model = ExactEligibilityCostModel::default();
    let model = cost_model.unwrap_or(&default_model);

    // Operator selection based on cost model crossover
    let selectivity_ppm = (eligibility.selectivity() * 1_000_000.0).round() as u32;
    let operator = if selectivity_ppm <= model.sparse_dense_crossover_ppm {
        ExactScanOperator::SparseGather
    } else {
        ExactScanOperator::MaskedDense
    };

    let (results, scored_count) = match operator {
        ExactScanOperator::SparseGather => {
            let sparse = eligibility.to_sparse_indices();
            sparse_gather_scan(snapshot, query, &sparse, k, metric_fn)
        }
        ExactScanOperator::MaskedDense => {
            let bitmap = eligibility.to_dense_bitmap();
            masked_dense_scan(snapshot, query, &bitmap, k, metric_fn)
        }
        ExactScanOperator::ScalarReference => {
            exact_top_k_scalar(snapshot, query, eligibility, k, metric_fn)
        }
    };

    let proof = ExactEligibilityProof {
        snapshot_lsn: snapshot.lsn,
        entity_generation: snapshot.segment.generation_id,
        eligible_count: eligibility.eligible_count,
        scored_count,
        operator,
        metric: metric_fn,
        globally_exact_over_eligibility: true,
    };

    (results, proof)
}
