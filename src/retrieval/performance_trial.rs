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
use std::sync::Arc;
use thiserror::Error;

use crate::DistanceFunction;

/// Errors during validation of per-query retrieval trials.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Error)]
pub enum TrialValidationError {
    #[error("reference ground truth is empty for requested k = {k}")]
    EmptyReference { k: usize },
    #[error("reference top-k contains duplicate node ID: {id}")]
    DuplicateReferenceId { id: u64 },
    #[error("candidate top-k contains duplicate node ID: {id}")]
    DuplicateCandidateId { id: u64 },
    #[error("candidate top-k length ({actual}) exceeds requested k ({k})")]
    CandidateLenExceedsK { actual: usize, k: usize },
}

/// Detailed per-query retrieval trial recording full comparison against the Exact SIMD oracle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RetrievalTrial {
    pub query_id: u64,
    pub reference_top_k: Vec<u64>,
    pub candidate_top_k: Vec<u64>,
    pub recall_at_k_q32: u64, // Recall formatted in Q32.32 fixed-point (4_294_967_296 = 1.0)
    pub reference_latency_ns: u64,
    pub candidate_latency_ns: u64,
    pub reference_exact_scores: u64,
    pub candidate_exact_scores: u64,
    pub candidate_nodes_visited: u64,
    pub candidate_bytes_read: u64,
}

impl RetrievalTrial {
    /// Computes trial metrics comparing candidate results to reference truth with strict validation.
    pub fn compute(
        query_id: u64,
        k: usize,
        reference_top_k: &[u64],
        candidate_top_k: &[u64],
        reference_latency_ns: u64,
        candidate_latency_ns: u64,
        reference_exact_scores: u64,
        candidate_exact_scores: u64,
        candidate_nodes_visited: u64,
        candidate_bytes_read: u64,
    ) -> Result<Self, TrialValidationError> {
        if k > 0 && reference_top_k.is_empty() {
            return Err(TrialValidationError::EmptyReference { k });
        }
        if candidate_top_k.len() > k {
            return Err(TrialValidationError::CandidateLenExceedsK {
                actual: candidate_top_k.len(),
                k,
            });
        }

        let mut ref_set = HashSet::with_capacity(reference_top_k.len());
        for &id in reference_top_k {
            if !ref_set.insert(id) {
                return Err(TrialValidationError::DuplicateReferenceId { id });
            }
        }

        let mut cand_set = HashSet::with_capacity(candidate_top_k.len());
        for &id in candidate_top_k {
            if !cand_set.insert(id) {
                return Err(TrialValidationError::DuplicateCandidateId { id });
            }
        }

        let matched = candidate_top_k
            .iter()
            .filter(|id| ref_set.contains(id))
            .count();

        let recall_float = if reference_top_k.is_empty() {
            0.0
        } else {
            matched as f64 / reference_top_k.len() as f64
        };

        let recall_at_k_q32 = (recall_float * 4_294_967_296.0).round() as u64;

        Ok(Self {
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
        })
    }

    /// Returns floating-point recall in $[0.0, 1.0]$.
    #[inline(always)]
    pub fn recall(&self) -> f64 {
        self.recall_at_k_q32 as f64 / 4_294_967_296.0
    }
}

/// Epistemic evidence for Certified admission mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CertifiedEvidence {
    /// Every evaluated query reached proof-complete branch-and-bound termination with zero unpruned threats.
    pub all_queries_proof_complete: bool,
    /// Every evaluated query achieved global Cauchy-Schwarz exactness certified by the proof engine.
    pub all_queries_globally_exact: bool,
}

/// Hard admission gate status for candidates aiming to become production options.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionGateStatus {
    /// Satisfies the initial survival threshold (Recall@10 >= 95%).
    SurvivalPassed,
    /// Satisfies the full production candidate threshold (Recall@10 >= 99% AND Speedup_p50 >= 2x).
    ProductionCandidateApproved,
    /// Certified exact candidate (Proof-complete globally exact AND empirical recall == 100% AND Latency < Exact).
    CertifiedProductionApproved,
    /// Rejected due to insufficient recall, missing proof, or insufficient speedup over Exact.
    Rejected { reason: Arc<str> },
}

/// Evaluates a candidate algorithm against the strict P0 admission gates.
pub fn evaluate_admission_gates(
    certified_evidence: Option<CertifiedEvidence>,
    recall_at_10: f64,
    exact_p50_ns: u64,
    candidate_p50_ns: u64,
) -> AdmissionGateStatus {
    if let Some(evidence) = certified_evidence {
        if !evidence.all_queries_proof_complete || !evidence.all_queries_globally_exact {
            return AdmissionGateStatus::Rejected {
                reason: Arc::from(
                    "Certified admission requires proof-complete global exactness across all queries",
                ),
            };
        }
        if recall_at_10 < 1.0 {
            return AdmissionGateStatus::Rejected {
                reason: Arc::from(
                    "Certified retrieval failed empirical 100% recall sanity check",
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

/// Immutable descriptor of persistent HNSW construction parameters for experiment provenance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswBuildDescriptor {
    pub m: u16,
    pub m0: u16,
    pub ef_construction: u32,
    pub metric: DistanceFunction,
    pub dataset_digest: [u8; 32],
    pub corpus_rows: u64,
}

/// Query-time search descriptor for HNSW sweeps.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HnswSearchDescriptor {
    pub ef_search: u32,
    pub algorithm: String, // e.g. "HnswClassicalV1" or "HoloGraphSuperpositionV1"
}

/// Complete provenance and environment fingerprint for reproducible frozen performance baselines.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkRunIdentity {
    pub semantic_kernel_version: u32,
    pub benchmark_schema_version: u32,
    pub dataset_sha256: String,
    pub query_set_sha256: String,
    pub index_snapshot_sha256: String,
    pub metric: String,
    pub exact_scorer_fingerprint: String,
    pub git_commit: String,
    pub rustc_version: String,
    pub target_triple: String,
    pub rustflags: String,
    pub cpu_model: String,
    pub isa_features: String,
    pub logical_cpus: usize,
    pub benchmark_threads: usize,
    pub hnsw_build_descriptor: Option<HnswBuildDescriptor>,
    pub hnsw_search_descriptor: Option<HnswSearchDescriptor>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_identity: Option<BenchmarkRunIdentity>,
}
