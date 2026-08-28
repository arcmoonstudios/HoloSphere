/* holosphere/tests/scorecard_primitives_invariants.rs */
//!▫~•◦-------------------------------‣
//! # Scorecard Primitives & Systems Invariants Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Exercises the mathematical and systems invariants of:
//!   1. `LsmSegmentConcurrencyHarness` (Concurrent read/write/compact, tombstone immunity)
//!   2. `PlannerRegretOracle` (Decision regret non-negativity, oracle optimality, admissibility gating)
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;
use std::time::Duration;

use hnsqr::VectorEmbedding;
use hnsqr::planning::planner::ExecutionPlan;
use hnsqr::planning::regret::{PlanExecutionMetrics, PlannerRegretOracle};
use hnsqr::rivero::RiveroProfile;
use hnsqr::storage::concurrency::{LsmConcurrencyConfig, LsmSegmentConcurrencyHarness};
use hnsqr::storage::segment::SegmentedEngine;

// ════════════════════════════════════════════════════════════════════════════════
// 1. LSM SEGMENT CONCURRENCY INVARIANTS
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_lsm_concurrency_invariants_and_zero_tombstone_leakage() {
    let dim = 16;
    let engine = Arc::new(SegmentedEngine::new(dim, 25));

    let vectors: Vec<VectorEmbedding> = (0..100)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| num_complex::Complex32::new((i * 3 + d) as f32, (i + d) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let config = LsmConcurrencyConfig {
        num_readers: 4,
        num_writers: 2,
        duration: Duration::from_millis(300),
        read_k: 10,
    };

    let report = LsmSegmentConcurrencyHarness::run(engine, vectors, config);

    // Invariant 1: Multi-threaded progress
    assert!(
        report.total_writes > 0,
        "Concurrent writers must produce writes"
    );
    assert!(
        report.total_reads > 0,
        "Concurrent readers must produce reads"
    );
    assert!(report.write_qps > 0.0);
    assert!(report.read_qps > 0.0);

    // Invariant 2: Zero tombstone or data violations under high contention
    assert_eq!(
        report.tombstone_violations, 0,
        "No tombstoned/corrupt data allowed in reader queries"
    );

    // Invariant 3: Bounded read latencies
    assert!(report.read_p50_us > 0.0);
    assert!(report.read_p99_us >= report.read_p50_us);
}

// ════════════════════════════════════════════════════════════════════════════════
// 2. PLANNER REGRET ORACLE INVARIANTS
// ════════════════════════════════════════════════════════════════════════════════

#[test]
fn test_planner_regret_oracle_invariants() {
    let fast_plan = PlanExecutionMetrics {
        plan: ExecutionPlan::RiveroRetrieval {
            profile: RiveroProfile::Balanced,
            candidate_cap: 1024,
        },
        plan_name: "RiveroBalanced".into(),
        recall_at_k: 0.998,
        latency_us: 1200.0,
        throughput_qps: 833.0,
        admissible: true,
    };

    let slow_plan = PlanExecutionMetrics {
        plan: ExecutionPlan::ExactScan { effective_n: 50000 },
        plan_name: "ExactScan".into(),
        recall_at_k: 1.000,
        latency_us: 15000.0,
        throughput_qps: 66.0,
        admissible: true,
    };

    let inadmissible_fast_plan = PlanExecutionMetrics {
        plan: ExecutionPlan::RiveroRetrieval {
            profile: RiveroProfile::Fast,
            candidate_cap: 256,
        },
        plan_name: "RiveroFastInadmissible".into(),
        recall_at_k: 0.850, // Fails 0.99 recall target
        latency_us: 300.0,
        throughput_qps: 3333.0,
        admissible: false,
    };

    let candidate_pool = vec![
        fast_plan.clone(),
        slow_plan.clone(),
        inadmissible_fast_plan.clone(),
    ];

    // Invariant 1: Inadmissible plans are NOT chosen as oracle optimal despite lower latency
    let eval_slow = PlannerRegretOracle::evaluate(&slow_plan, &candidate_pool);
    assert_eq!(
        eval_slow.optimal_plan_name, "RiveroBalanced",
        "Oracle must choose the fastest ADMISSIBLE plan, ignoring faster inadmissible candidates"
    );

    // Invariant 2: Non-negative regret
    assert!(
        eval_slow.regret_us >= 0.0,
        "Regret must be non-negative! got {}",
        eval_slow.regret_us
    );
    assert_eq!(eval_slow.regret_us, 15000.0 - 1200.0);
    assert!((eval_slow.optimality_ratio - (1200.0 / 15000.0)).abs() < 1e-5);

    // Invariant 3: Identity of Optimality (Zero Regret & 100% Optimality Ratio on exact optimal choice)
    let eval_fast = PlannerRegretOracle::evaluate(&fast_plan, &candidate_pool);
    assert_eq!(eval_fast.selected_plan_name, eval_fast.optimal_plan_name);
    assert_eq!(eval_fast.regret_us, 0.0);
    assert_eq!(eval_fast.optimality_ratio, 1.0);
    assert_eq!(eval_fast.relative_regret_pct, 0.0);

    // Invariant 4: Workload Aggregation Correctness
    let summary = PlannerRegretOracle::aggregate(&[eval_slow, eval_fast]);
    assert_eq!(summary.total_evaluations, 2);
    assert_eq!(summary.oracle_agreement_pct, 50.0);
    assert_eq!(summary.mean_regret_us, (13800.0 + 0.0) / 2.0);
}
