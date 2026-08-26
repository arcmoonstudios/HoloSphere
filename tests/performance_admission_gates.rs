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

use hnsqr::DistanceFunction;
use hnsqr::retrieval::performance_trial::{
    AdmissionGateStatus, BenchmarkRecord, BenchmarkRunIdentity, CertifiedEvidence,
    HnswBuildDescriptor, HnswSearchDescriptor, RetrievalTrial, TrialValidationError,
    evaluate_admission_gates,
};

/// P0.1: Exact SIMD as Oracle & RetrievalTrial metric calculations with Q32.32 scaling
#[test]
fn test_performance_p0_retrieval_trial_accounting() {
    let ref_top10 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    // Candidate recalls 9 of 10
    let cand_top10 = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 99];

    let trial = RetrievalTrial::compute(
        1,
        10,
        &ref_top10,
        &cand_top10,
        100_000, // Exact: 100 us
        30_000,  // Candidate: 30 us (3.33x speedup)
        10_000,  // Exact scored all 10K
        500,     // Candidate scored 500
        120,     // Candidate visited 120 nodes
        4096,    // Candidate read 4KB
    )
    .expect("valid trial computation");

    assert_eq!(trial.query_id, 1);
    // True Q32.32 scaling: 0.90 * 4_294_967_296 = 3_865_470_566
    assert_eq!(trial.recall_at_k_q32, 3_865_470_566);
    let recall = trial.recall();
    assert!((recall - 0.90).abs() < 1e-6);
}

/// P0.2: Fail closed on invalid / fabricated trial inputs
#[test]
fn test_performance_p0_retrieval_trial_strict_validation() {
    let ref_top2 = vec![1, 2];

    // 1. Candidate with duplicate IDs [1, 1] must fail closed with DuplicateCandidateId
    let cand_duplicates = vec![1, 1];
    let err_cand_dup = RetrievalTrial::compute(
        1,
        2,
        &ref_top2,
        &cand_duplicates,
        100,
        50,
        100,
        50,
        10,
        1024,
    );
    assert_eq!(
        err_cand_dup,
        Err(TrialValidationError::DuplicateCandidateId { id: 1 })
    );

    // 2. Reference with duplicate IDs [1, 1] must fail closed with DuplicateReferenceId
    let ref_duplicates = vec![1, 1];
    let cand_valid = vec![1, 2];
    let err_ref_dup = RetrievalTrial::compute(
        1,
        2,
        &ref_duplicates,
        &cand_valid,
        100,
        50,
        100,
        50,
        10,
        1024,
    );
    assert_eq!(
        err_ref_dup,
        Err(TrialValidationError::DuplicateReferenceId { id: 1 })
    );

    // 3. Unexpected empty reference for k > 0 must fail closed with EmptyReference
    let ref_empty: Vec<u64> = vec![];
    let err_empty =
        RetrievalTrial::compute(1, 10, &ref_empty, &cand_valid, 100, 50, 100, 50, 10, 1024);
    assert_eq!(
        err_empty,
        Err(TrialValidationError::EmptyReference { k: 10 })
    );

    // 4. Candidate length exceeding requested k must fail closed
    let cand_oversized = vec![1, 2, 3];
    let err_oversized =
        RetrievalTrial::compute(1, 2, &ref_top2, &cand_oversized, 100, 50, 100, 50, 10, 1024);
    assert_eq!(
        err_oversized,
        Err(TrialValidationError::CandidateLenExceedsK { actual: 3, k: 2 })
    );
}

/// P0.4: Admission Gate Evaluation Rules with CertifiedEvidence
#[test]
fn test_performance_p0_admission_gate_evaluations() {
    // 1. Approximate candidate: Recall 94% -> REJECTED (failed 95% survival)
    let status_low_recall = evaluate_admission_gates(None, 0.94, 100_000, 20_000);
    assert_eq!(
        status_low_recall,
        AdmissionGateStatus::Rejected {
            reason: std::sync::Arc::from("Recall@10 < 95% (failed initial survival gate)"),
        }
    );

    // 2. Approximate candidate: Recall 96% -> SURVIVAL PASSED
    let status_survival = evaluate_admission_gates(None, 0.96, 100_000, 30_000);
    assert_eq!(status_survival, AdmissionGateStatus::SurvivalPassed);

    // 3. Approximate candidate: Recall 99.2% + Speedup 1.5x (66us vs 100us) -> SURVIVAL PASSED (not 2x speedup)
    let status_low_speedup = evaluate_admission_gates(None, 0.992, 100_000, 66_000);
    assert_eq!(status_low_speedup, AdmissionGateStatus::SurvivalPassed);

    // 4. Approximate candidate: Recall 99.5% + Speedup 2.5x (40us vs 100us) -> APPROVED
    let status_approved = evaluate_admission_gates(None, 0.995, 100_000, 40_000);
    assert_eq!(
        status_approved,
        AdmissionGateStatus::ProductionCandidateApproved
    );

    // 5. Certified candidate: missing proof completeness -> REJECTED even if empirical recall is 100%
    let status_cert_no_proof = evaluate_admission_gates(
        Some(CertifiedEvidence {
            all_queries_proof_complete: false,
            all_queries_globally_exact: true,
        }),
        1.0,
        100_000,
        50_000,
    );
    assert_eq!(
        status_cert_no_proof,
        AdmissionGateStatus::Rejected {
            reason: std::sync::Arc::from(
                "Certified admission requires proof-complete global exactness across all queries"
            ),
        }
    );

    // 6. Certified candidate: empirical recall < 100% -> REJECTED
    let status_cert_imperfect = evaluate_admission_gates(
        Some(CertifiedEvidence {
            all_queries_proof_complete: true,
            all_queries_globally_exact: true,
        }),
        0.999,
        100_000,
        50_000,
    );
    assert_eq!(
        status_cert_imperfect,
        AdmissionGateStatus::Rejected {
            reason: std::sync::Arc::from(
                "Certified retrieval failed empirical 100% recall sanity check"
            ),
        }
    );

    // 7. Certified candidate: Slower than Exact (150us vs 100us) -> REJECTED
    let status_cert_slow = evaluate_admission_gates(
        Some(CertifiedEvidence {
            all_queries_proof_complete: true,
            all_queries_globally_exact: true,
        }),
        1.0,
        100_000,
        150_000,
    );
    assert_eq!(
        status_cert_slow,
        AdmissionGateStatus::Rejected {
            reason: std::sync::Arc::from(
                "Certified retrieval did not beat Exact SIMD latency (research-only)"
            ),
        }
    );

    // 8. Certified candidate: Proof-complete globally exact + Recall 100% + Faster than Exact (60us vs 100us) -> APPROVED
    let status_cert_approved = evaluate_admission_gates(
        Some(CertifiedEvidence {
            all_queries_proof_complete: true,
            all_queries_globally_exact: true,
        }),
        1.0,
        100_000,
        60_000,
    );
    assert_eq!(
        status_cert_approved,
        AdmissionGateStatus::CertifiedProductionApproved
    );
}

/// P0.10: Machine-readable benchmark record serialization with provenance
#[test]
fn test_performance_p0_benchmark_record_serialization() {
    let identity = BenchmarkRunIdentity {
        semantic_kernel_version: 1,
        benchmark_schema_version: 1,
        dataset_sha256: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".into(),
        query_set_sha256: "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb".into(),
        index_snapshot_sha256: "3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            .into(),
        metric: "Cosine".into(),
        exact_scorer_fingerprint: "AVX2-FMA-DualAccComplex".into(),
        git_commit: "deadbeef2026".into(),
        rustc_version: "rustc 1.85.0".into(),
        target_triple: "x86_64-pc-windows-msvc".into(),
        rustflags: "-C target-cpu=native".into(),
        cpu_model: "AMD Ryzen 9 7950X".into(),
        isa_features: "AVX2,FMA,BMI2".into(),
        logical_cpus: 32,
        benchmark_threads: 1,
        hnsw_build_descriptor: Some(HnswBuildDescriptor {
            m: 32,
            m0: 64,
            ef_construction: 128,
            metric: DistanceFunction::Cosine,
            dataset_digest: [0u8; 32],
            corpus_rows: 1_000_000,
        }),
        hnsw_search_descriptor: Some(HnswSearchDescriptor {
            ef_search: 64,
            algorithm: "HnswClassicalV1".into(),
        }),
    };

    let record = BenchmarkRecord {
        semantic_kernel_version: 1,
        benchmark_baseline_version: 1,
        dataset: "SIFT-1M".into(),
        algorithm: "HnswClassicalV1-M32-ef128".into(),
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
        run_identity: Some(identity),
    };

    let serialized = serde_json::to_string_pretty(&record).expect("serialization");
    assert!(serialized.contains("\"semantic_kernel_version\": 1"));
    assert!(serialized.contains("\"algorithm\": \"HnswClassicalV1-M32-ef128\""));
    assert!(serialized.contains("\"admission_status\": \"ProductionCandidateApproved\""));
    assert!(serialized.contains("\"cpu_model\": \"AMD Ryzen 9 7950X\""));

    let deserialized: BenchmarkRecord = serde_json::from_str(&serialized).expect("deserialization");
    assert_eq!(record, deserialized);
}
