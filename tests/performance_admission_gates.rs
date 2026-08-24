/* holosphere/tests/performance_admission_gates.rs */
//!▫~•◦-------------------------------‣
//! # Performance Track P0: Admission Gates & Oracle Invariants
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates the strict admission gates for retrieval candidates:
//!   1. Exact SIMD is the frozen oracle.
//!   2. Approximate candidates require Recall@10 >= 95% (survival) and Recall@10 >= 99% + Speedup >= 2x (production).
//!   3. Certified candidates require 100% exact recall by construction AND Latency < Exact.
//!   4. Production default remains unconditionally Exact.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::retrieval::performance_trial::{
    AdmissionGateStatus, BenchmarkRecord, RetrievalTrial, evaluate_admission_gates,
};

/// P0.1: Exact SIMD as Oracle & RetrievalTrial metric calculations
#[test]
fn test_performance_p0_retrieval_trial_accounting() {
    let ref_top10 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    // Candidate recalls 9 of 10
    let cand_top10 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 99];

    let trial = RetrievalTrial::compute(
        1,
        &ref_top10,
        &cand_top10,
        100_000, // Exact: 100 us
        30_000,  // Candidate: 30 us (3.33x speedup)
        10_000,  // Exact scored all 10K
        500,     // Candidate scored 500
        120,     // Candidate visited 120 nodes
        4096,    // Candidate read 4KB
    );

    assert_eq!(trial.query_id, 1);
    assert_eq!(trial.recall_at_k_q32, 58982); // 0.90 * 65536 = 58982.4
    let recall = trial.recall();
    assert!((recall - 0.90).abs() < 0.001);
}

/// P0.4: Admission Gate Evaluation Rules
#[test]
fn test_performance_p0_admission_gate_evaluations() {
    // 1. Approximate candidate: Recall 94% -> REJECTED (failed 95% survival)
    let status_low_recall = evaluate_admission_gates(false, 0.94, 100_000, 20_000);
    assert_eq!(
        status_low_recall,
        AdmissionGateStatus::Rejected {
            reason: std::sync::Arc::from("Recall@10 < 95% (failed initial survival gate)"),
        }
    );

    // 2. Approximate candidate: Recall 96% -> SURVIVAL PASSED
    let status_survival = evaluate_admission_gates(false, 0.96, 100_000, 30_000);
    assert_eq!(status_survival, AdmissionGateStatus::SurvivalPassed);

    // 3. Approximate candidate: Recall 99.2% + Speedup 1.5x (66us vs 100us) -> SURVIVAL PASSED (not 2x speedup)
    let status_low_speedup = evaluate_admission_gates(false, 0.992, 100_000, 66_000);
    assert_eq!(status_low_speedup, AdmissionGateStatus::SurvivalPassed);

    // 4. Approximate candidate: Recall 99.5% + Speedup 2.5x (40us vs 100us) -> APPROVED
    let status_approved = evaluate_admission_gates(false, 0.995, 100_000, 40_000);
    assert_eq!(
        status_approved,
        AdmissionGateStatus::ProductionCandidateApproved
    );

    // 5. Certified candidate: Recall 99.9% -> REJECTED (must be 100% by semantics)
    let status_cert_imperfect = evaluate_admission_gates(true, 0.999, 100_000, 50_000);
    assert_eq!(
        status_cert_imperfect,
        AdmissionGateStatus::Rejected {
            reason: std::sync::Arc::from(
                "Certified retrieval must achieve 100% recall strictly by construction"
            ),
        }
    );

    // 6. Certified candidate: Recall 100% + Slower than Exact (150us vs 100us) -> REJECTED
    let status_cert_slow = evaluate_admission_gates(true, 1.0, 100_000, 150_000);
    assert_eq!(
        status_cert_slow,
        AdmissionGateStatus::Rejected {
            reason: std::sync::Arc::from(
                "Certified retrieval did not beat Exact SIMD latency (research-only)"
            ),
        }
    );

    // 7. Certified candidate: Recall 100% + Faster than Exact (60us vs 100us) -> APPROVED
    let status_cert_approved = evaluate_admission_gates(true, 1.0, 100_000, 60_000);
    assert_eq!(
        status_cert_approved,
        AdmissionGateStatus::CertifiedProductionApproved
    );
}

/// P0.10: Machine-readable benchmark record serialization
#[test]
fn test_performance_p0_benchmark_record_serialization() {
    let record = BenchmarkRecord {
        semantic_kernel_version: 1,
        benchmark_baseline_version: 1,
        dataset: "SIFT-1M".into(),
        algorithm: "HNSW-M32-ef128".into(),
        n_vectors: 1_000_000,
        dimension: 128,
        recall_at_10: 0.994,
        exact_p50_ns: 38_500_000,    // 38.5 ms
        candidate_p50_ns: 1_250_000, // 1.25 ms (30.8x speedup)
        candidate_p95_ns: 1_850_000,
        candidate_p99_ns: 2_400_000,
        candidate_exact_scores: 1_450,
        candidate_fraction: 0.00145,
        proof_efficiency: 0.0,
        admission_status: AdmissionGateStatus::ProductionCandidateApproved,
    };

    let serialized = serde_json::to_string_pretty(&record).expect("serialization");
    assert!(serialized.contains("\"semantic_kernel_version\": 1"));
    assert!(serialized.contains("\"algorithm\": \"HNSW-M32-ef128\""));
    assert!(serialized.contains("\"admission_status\": \"ProductionCandidateApproved\""));

    let deserialized: BenchmarkRecord = serde_json::from_str(&serialized).expect("deserialization");
    assert_eq!(record, deserialized);
}
