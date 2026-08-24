/* holosphere/src/entity/exact/metric.rs */
//!▫~•◦-------------------------------‣
//! # Exact Vector Metric Dispatch & SIMD Kernels
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the unified metric dispatch contract ensuring:
//! Score_scalar == Score_dense == Score_gather
//! with deterministic tie-breaking (Score DESC, EntityId ASC).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::cmp::Ordering;

use crate::entity::id::{EntityId, EntityIndex};

/// Scored candidate item returned by exact retrieval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScoredEntity {
    pub entity_id: EntityId,
    pub entity_index: EntityIndex,
    pub score: f32,
}

impl Eq for ScoredEntity {}

impl Ord for ScoredEntity {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher score is better.
        match self.score.partial_cmp(&other.score) {
            Some(Ordering::Equal) | None => {
                // Deterministic tie-breaking: lower EntityId comes first.
                other.entity_id.cmp(&self.entity_id)
            }
            Some(ord) => ord,
        }
    }
}

impl PartialOrd for ScoredEntity {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub use crate::DistanceFunction;

/// Unified exact metric evaluation trait.
pub trait ExactVectorMetric: Send + Sync {
    /// Reference scalar implementation (ground truth).
    fn score_scalar(&self, query: &[f32], vector: &[f32]) -> f32;
    /// SIMD-accelerated implementation.
    fn score_simd(&self, query: &[f32], vector: &[f32]) -> f32 {
        self.score_scalar(query, vector)
    }
}

pub struct CosineMetric;
pub struct InnerProductMetric;
pub struct EuclideanMetric;
pub struct ProjectiveOverlapMetric;

impl ExactVectorMetric for InnerProductMetric {
    #[inline]
    fn score_scalar(&self, query: &[f32], vector: &[f32]) -> f32 {
        let len = query.len().min(vector.len());
        let mut sum = 0.0f32;
        let chunks_q = query[..len].chunks_exact(4);
        let chunks_v = vector[..len].chunks_exact(4);
        let rem_q = chunks_q.remainder();
        let rem_v = chunks_v.remainder();

        for (q, v) in chunks_q.zip(chunks_v) {
            sum += q[0] * v[0] + q[1] * v[1] + q[2] * v[2] + q[3] * v[3];
        }
        for (q, v) in rem_q.iter().zip(rem_v.iter()) {
            sum += q * v;
        }
        sum
    }

    #[inline]
    fn score_simd(&self, query: &[f32], vector: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            unsafe { dot_product_f32_avx2(query, vector) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            self.score_scalar(query, vector)
        }
    }
}

impl ExactVectorMetric for CosineMetric {
    #[inline]
    fn score_scalar(&self, query: &[f32], vector: &[f32]) -> f32 {
        let len = query.len().min(vector.len());
        let mut dot = 0.0f32;
        let mut norm_q = 0.0f32;
        let mut norm_v = 0.0f32;

        for i in 0..len {
            dot += query[i] * vector[i];
            norm_q += query[i] * query[i];
            norm_v += vector[i] * vector[i];
        }

        let denom = (norm_q * norm_v).sqrt();
        if denom > 1e-12 {
            (dot / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }

    #[inline]
    fn score_simd(&self, query: &[f32], vector: &[f32]) -> f32 {
        // Query norm can be precomputed or computed here
        let dot = InnerProductMetric.score_simd(query, vector);
        let norm_q = InnerProductMetric.score_simd(query, query);
        let norm_v = InnerProductMetric.score_simd(vector, vector);
        let denom = (norm_q * norm_v).sqrt();
        if denom > 1e-12 {
            (dot / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    }
}

impl ExactVectorMetric for EuclideanMetric {
    #[inline]
    fn score_scalar(&self, query: &[f32], vector: &[f32]) -> f32 {
        let len = query.len().min(vector.len());
        let mut sum = 0.0f32;
        for i in 0..len {
            let diff = query[i] - vector[i];
            sum += diff * diff;
        }
        -sum.sqrt() // Negative distance for Top-K maximization
    }

    #[inline]
    fn score_simd(&self, query: &[f32], vector: &[f32]) -> f32 {
        self.score_scalar(query, vector)
    }
}

impl ExactVectorMetric for ProjectiveOverlapMetric {
    #[inline]
    fn score_scalar(&self, query: &[f32], vector: &[f32]) -> f32 {
        let q_pairs = query.chunks_exact(2);
        let v_pairs = vector.chunks_exact(2);

        let mut acc_re = 0.0f32;
        let mut acc_im = 0.0f32;
        let mut norm_q = 0.0f32;
        let mut norm_v = 0.0f32;

        for (q, v) in q_pairs.zip(v_pairs) {
            let (q_re, q_im) = (q[0], q[1]);
            let (v_re, v_im) = (v[0], v[1]);

            acc_re += q_re * v_re + q_im * v_im;
            acc_im += q_re * v_im - q_im * v_re;
            norm_q += q_re * q_re + q_im * q_im;
            norm_v += v_re * v_re + v_im * v_im;
        }

        let num = acc_re * acc_re + acc_im * acc_im;
        let denom = (norm_q * norm_v).max(1e-12);
        (num / denom).clamp(0.0, 1.0)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let len = a.len().min(b.len());
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let chunks16 = len / 16;
    let mut offset = 0;

    for _ in 0..chunks16 {
        let va0 = _mm256_loadu_ps(a_ptr.add(offset));
        let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
        let va1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(offset + 8));

        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        acc1 = _mm256_fmadd_ps(va1, vb1, acc1);

        offset += 16;
    }

    let chunks8 = (len - offset) / 8;
    for _ in 0..chunks8 {
        let va0 = _mm256_loadu_ps(a_ptr.add(offset));
        let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
        acc0 = _mm256_fmadd_ps(va0, vb0, acc0);
        offset += 8;
    }

    let acc = _mm256_add_ps(acc0, acc1);
    let mut arr = [0.0f32; 8];
    _mm256_storeu_ps(arr.as_mut_ptr(), acc);
    let mut sum = arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7];

    for i in offset..len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
    }

    sum
}

/// Resolves a metric dispatch handler for `DistanceFunction`.
pub fn resolve_metric(func: DistanceFunction) -> &'static dyn ExactVectorMetric {
    match func {
        DistanceFunction::Cosine => &CosineMetric,
        DistanceFunction::Euclidean => &EuclideanMetric,
        DistanceFunction::ProjectiveOverlap
        | DistanceFunction::ProjectiveSineDistance
        | DistanceFunction::PhaseAlignedChordalDistance => &ProjectiveOverlapMetric,
    }
}
