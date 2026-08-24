/* holosphere/src/retrieval/performance_trial.rs */
//!▫~•◦-------------------------------‣
//! # Performance Track P0: Per-Query Retrieval Trial & Admission Harness
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the primary per-query trial data structure, hardware metadata capture,
//! admission gate evaluation, and machine-readable baseline serialization for the
//! HoloSphere Performance Research Track.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Detailed per-query retrieval trial recording full comparison against the Exact SIMD oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalTrial {
    pub query_id: u64,
    pub reference_top_k: Vec<u64>,
    pub candidate_top_k: Vec<u64>,
    pub recall_at_k_q32: u64, // Recall formatted in Q32 fixed-point (65536 = 1.0)
    pub reference_latency_ns: u64,
    pub candidate_latency_ns: u64,
    pub reference_exact_scores: u64,
    pub candidate_exact_scores: u64,
    pub candidate_nodes_visited: u64,
    pub candidate_bytes_read: u64,
}

impl RetrievalTrial {
    /// Computes trial metrics comparing candidate results to reference truth.
    pub fn compute(
        query_id: u64,
        reference_top_k: &[u64],
        candidate_top_k: &[u64],
        reference_latency_ns: u64,
        candidate_latency_ns: u64,
        reference_exact_scores: u64,
        candidate_exact_scores: u64,
        candidate_nodes_visited: u64,
        candidate_bytes_read: u64,
    ) -> Self {
        let ref_set: HashSet<u64> = reference_top_k.iter().copied().collect();
        let matched = candidate_top_k
            .iter()
            .filter(|id| ref_set.contains(id))
            .count();

        let recall_float = if reference_top_k.is_empty() {
            1.0
        } else {
            matched as f64 / reference_top_k.len() as f64
        };

        let recall_at_k_q32 = (recall_float * 65536.0).round() as u64;

        Self {
            query_id,
            reference_top_k: reference_top_k.to_vec(),
            candidate_top_k: candidate_top_k.to_vec(),
            recall_at_k_q32,
            reference_latency_ns,
            candidate_latency_ns,
            reference_exact_scores,
            candidate_exact_scores,
            candidate_nodes_visited,
            candidate_bytes_read,
        }
    }

    /// Returns floating-point recall in $[0.0, 1.0]$.
    #[inline(always)]
    pub fn recall(&self) -> f64 {
        self.recall_at_k_q32 as f64 / 65536.0
    }
}

use std::sync::Arc;

/// Hard admission gate status for candidates aiming to become production options.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionGateStatus {
    /// Satisfies the initial survival threshold (Recall@10 >= 95%).
    SurvivalPassed,
    /// Satisfies the full production candidate threshold (Recall@10 >= 99% AND Speedup_p50 >= 2x).
    ProductionCandidateApproved,
    /// Certified exact candidate (Recall == 100% strictly AND Latency < Exact).
    CertifiedProductionApproved,
    /// Rejected due to insufficient recall or insufficient speedup over Exact.
    Rejected { reason: Arc<str> },
}

/// Evaluates a candidate algorithm against the strict P0 admission gates.
pub fn evaluate_admission_gates(
    is_certified_mode: bool,
    recall_at_10: f64,
    exact_p50_ns: u64,
    candidate_p50_ns: u64,
) -> AdmissionGateStatus {
    if is_certified_mode {
        if recall_at_10 < 1.0 {
            return AdmissionGateStatus::Rejected {
                reason: Arc::from(
                    "Certified retrieval must achieve 100% recall strictly by construction",
                ),
            };
        }
        if candidate_p50_ns < exact_p50_ns {
            AdmissionGateStatus::CertifiedProductionApproved
        } else {
            AdmissionGateStatus::Rejected {
                reason: Arc::from(
                    "Certified retrieval did not beat Exact SIMD latency (research-only)",
                ),
            }
        }
    } else {
        if recall_at_10 < 0.95 {
            return AdmissionGateStatus::Rejected {
                reason: Arc::from("Recall@10 < 95% (failed initial survival gate)"),
            };
        }
        let speedup = exact_p50_ns as f64 / candidate_p50_ns.max(1) as f64;
        if recall_at_10 >= 0.99 && speedup >= 2.0 {
            AdmissionGateStatus::ProductionCandidateApproved
        } else if recall_at_10 >= 0.95 {
            AdmissionGateStatus::SurvivalPassed
        } else {
            AdmissionGateStatus::Rejected {
                reason: Arc::from("Speedup < 2x at 99% recall (insufficient performance margin)"),
            }
        }
    }
}

/// Machine-readable benchmark record for CI regression tracking and Pareto analysis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BenchmarkRecord {
    pub semantic_kernel_version: u32,
    pub benchmark_baseline_version: u32,
    pub dataset: String,
    pub algorithm: String,
    pub n_vectors: usize,
    pub dimension: usize,
    pub recall_at_10: f64,
    pub exact_p50_ns: u64,
    pub candidate_p50_ns: u64,
    pub candidate_p95_ns: u64,
    pub candidate_p99_ns: u64,
    pub candidate_exact_scores: u64,
    pub candidate_fraction: f64,
    pub proof_efficiency: f64,
    pub admission_status: AdmissionGateStatus,
}
