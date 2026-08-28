/* holosphere/src/hybrid.rs */
//!▫~•◦-------------------------------‣
//! # Hybrid Fusion Engine (Reciprocal Rank Fusion & Score Normalization)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides rank-based fusion (RRF) and distribution-aware weighted linear fusion across
//! dense vectors, sparse lexical terms, and multi-vector late interaction.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::SimilarityScore;

pub const RRF_DEFAULT_K: f32 = 60.0;

/// Modality ranking list input for hybrid fusion.
#[derive(Clone, Debug, PartialEq)]
pub struct ModalityRankings {
    pub name: String,
    pub weight: f32,
    pub results: Vec<(Arc<str>, SimilarityScore)>,
}

/// Hybrid Fusion algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum HybridFusionMethod {
    /// Reciprocal Rank Fusion: $\text{RRF}(d) = \sum_m \frac{w_m}{k + \text{rank}_m(d)}$
    Rrf { k: f32 },
    /// Weighted linear score combination after min-max normalization.
    WeightedLinear,
}

impl Default for HybridFusionMethod {
    fn default() -> Self {
        Self::Rrf { k: RRF_DEFAULT_K }
    }
}

pub struct HybridFusionEngine;

impl HybridFusionEngine {
    /// Combines multiple search results using Reciprocal Rank Fusion (RRF).
    ///
    /// DERIVED: Eliminates Arc atomic refcount updates on hit paths and uses O(M + k log k) top-k selection.
    pub fn fuse_rrf(
        modalities: &[ModalityRankings],
        rrf_k: f32,
        top_k: usize,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if modalities.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let mut doc_scores: HashMap<Arc<str>, f32> = HashMap::with_capacity(top_k * 4);

        for modality in modalities {
            let weight = modality.weight.max(0.0);
            for (rank, (doc_id, _)) in modality.results.iter().enumerate() {
                let rrf_score = weight / (rrf_k + (rank as f32) + 1.0);
                if let Some(score) = doc_scores.get_mut(doc_id) {
                    *score += rrf_score;
                } else {
                    doc_scores.insert(doc_id.clone(), rrf_score);
                }
            }
        }

        let mut fused: Vec<(Arc<str>, SimilarityScore)> = doc_scores.into_iter().collect();
        if fused.len() > top_k {
            fused.select_nth_unstable_by(top_k, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            fused.truncate(top_k);
        }
        fused.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        fused
    }

    /// Combines multiple search results using weighted linear normalized score fusion.
    ///
    /// DERIVED: Single-pass min/max normalization, precomputed reciprocal range, and O(M + k log k) top-k selection.
    pub fn fuse_weighted(
        modalities: &[ModalityRankings],
        top_k: usize,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if modalities.is_empty() || top_k == 0 {
            return Vec::new();
        }

        let mut doc_scores: HashMap<Arc<str>, f32> = HashMap::with_capacity(top_k * 4);

        for modality in modalities {
            let weight = modality.weight.max(0.0);
            if modality.results.is_empty() {
                continue;
            }

            let mut min_score = f32::INFINITY;
            let mut max_score = f32::NEG_INFINITY;
            for (_, s) in &modality.results {
                if *s < min_score {
                    min_score = *s;
                }
                if *s > max_score {
                    max_score = *s;
                }
            }
            let range = (max_score - min_score).max(1e-6);
            let inv_range = 1.0 / range;

            for (doc_id, raw_score) in &modality.results {
                let normalized = (raw_score - min_score) * inv_range;
                let inc = weight * normalized;
                if let Some(score) = doc_scores.get_mut(doc_id) {
                    *score += inc;
                } else {
                    doc_scores.insert(doc_id.clone(), inc);
                }
            }
        }

        let mut fused: Vec<(Arc<str>, SimilarityScore)> = doc_scores.into_iter().collect();
        if fused.len() > top_k {
            fused.select_nth_unstable_by(top_k, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            fused.truncate(top_k);
        }
        fused.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        fused
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_rrf_fusion() {
        let dense_rankings = ModalityRankings {
            name: "dense".to_string(),
            weight: 1.0,
            results: vec![
                (Arc::from("doc_a"), 0.95),
                (Arc::from("doc_b"), 0.85),
                (Arc::from("doc_c"), 0.75),
            ],
        };

        let sparse_rankings = ModalityRankings {
            name: "sparse".to_string(),
            weight: 1.0,
            results: vec![
                (Arc::from("doc_b"), 12.0),
                (Arc::from("doc_d"), 8.0),
                (Arc::from("doc_a"), 4.0),
            ],
        };

        let fused = HybridFusionEngine::fuse_rrf(&[dense_rankings, sparse_rankings], 60.0, 3);
        assert_eq!(fused.len(), 3);
        // doc_b has rank 1 (0-based) in dense and rank 0 in sparse: RRF = 1/62 + 1/61 = highest!
        assert_eq!(fused[0].0.as_ref(), "doc_b");
    }
}
