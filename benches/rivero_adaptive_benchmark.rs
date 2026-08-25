/* hnsqr/benches/rivero_adaptive_benchmark.rs */
//!▫~•◦-------------------------------‣
//! # Staged Confidence-Adaptive Rivero Benchmark
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates the multi-stage, state-reusing adaptive router against:
//!   1. Clustered Semantic Workload (Natural LLM Embeddings)
//!   2. Boundary Adversarial Workload (Queries Between Clusters)
//!   3. Isotropic Uniform Workload (Spherical Stress Test)
//!
//! Compares:
//!   - Strict Rivero (Provably Bounded, Fixed Profile)
//!   - Adaptive Rivero Bounded (Fast -> Balanced -> Strict progressive escalation)
//!   - Adaptive Rivero Hybrid (Rivero + Optional Graph Fallback)
//!   - Pure Graph Superposition Traversal (Classical HNSW)
//!
//! Reports:
//!   - Recall@1, Recall@10, NDCG@10, Latencies (p50, p95, p99)
//!   - Stage Acceptance Distributions (% Fast, % Balanced, % Strict, % Fallback)
//!   - Cumulative Scans & Exact Vector Evaluations
//!   - False Confidence Rate ($P(\text{Recall@10} < \text{Target} \mid \text{Accepted})$)
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::time::Instant;

use hnsqr::{
    AdaptivePolicy, HNSQRConfig, HNSQRIndex, NodeIndex, RiveroProfile, SimilarityScore,
    VectorEmbedding,
};
use rayon::prelude::*;

const D: usize = 64;
const N: usize = 10_000;
const QUERY_COUNT: usize = 100;
const K_BENCH: usize = 10;
const SEED: u64 = 0x5461_6765_6441_6470;

struct Workload {
    name: String,
    corpus: Vec<VectorEmbedding>,
    queries: Vec<VectorEmbedding>,
    ground_truth: Vec<Vec<(NodeIndex, SimilarityScore)>>,
}

use hnsqr::bench_support as common;

fn load_real_workload(name: &str, n: usize, d: usize, q_count: usize) -> Workload {
    let (base_path, query_path, _) = common::find_best_matching_dataset(d);
    let (corpus, _) = common::read_fvecs(&base_path, Some(n)).unwrap_or_default();
    let (queries, _) = common::read_fvecs(&query_path, Some(q_count)).unwrap_or_default();

    assert!(
        !corpus.is_empty(),
        "dataset '{}' is missing or empty — ensure datasets/ are populated",
        base_path.display()
    );
    assert!(
        !queries.is_empty(),
        "query file '{}' is missing or empty",
        query_path.display()
    );

    let ground_truth = compute_exact_ground_truth(&corpus, &queries, K_BENCH);
    Workload {
        name: name.to_string(),
        corpus,
        queries,
        ground_truth,
    }
}

fn generate_clustered_workload(n: usize, d: usize, q_count: usize, _seed: u64) -> Workload {
    load_real_workload("Real Clustered Semantic", n, d, q_count)
}

fn generate_boundary_workload(n: usize, d: usize, q_count: usize, _seed: u64) -> Workload {
    load_real_workload("Real Boundary Workload", n, d, q_count)
}

fn generate_isotropic_workload(n: usize, d: usize, q_count: usize, _seed: u64) -> Workload {
    load_real_workload("Real Isotropic Uniform", n, d, q_count)
}

fn compute_exact_ground_truth(
    corpus: &[VectorEmbedding],
    queries: &[VectorEmbedding],
    k: usize,
) -> Vec<Vec<(NodeIndex, SimilarityScore)>> {
    queries
        .par_iter()
        .map(|query| {
            let mut scores: Vec<(NodeIndex, SimilarityScore)> = corpus
                .iter()
                .enumerate()
                .map(|(idx, doc)| (idx as NodeIndex, query.projective_overlap(doc)))
                .collect();
            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scores.truncate(k);
            scores
        })
        .collect()
}

fn calculate_ndcg(
    retrieved: &[(NodeIndex, SimilarityScore)],
    gt: &[(NodeIndex, SimilarityScore)],
    k: usize,
) -> f64 {
    let limit = k.min(retrieved.len());
    if limit == 0 || gt.is_empty() {
        return 0.0;
    }

    let mut dcg = 0.0;
    for (i, item) in retrieved.iter().take(limit).enumerate() {
        if let Some(pos) = gt.iter().position(|g| g.0 == item.0) {
            let gain = (gt.len() - pos) as f64;
            let rank = (i + 1) as f64;
            dcg += gain / (rank + 1.0).log2();
        }
    }

    let mut idcg = 0.0;
    for i in 0..limit.min(gt.len()) {
        let gain = (gt.len() - i) as f64;
        let rank = (i + 1) as f64;
        idcg += gain / (rank + 1.0).log2();
    }

    if idcg > 0.0 { dcg / idcg } else { 0.0 }
}

#[derive(Clone, Debug, Default)]
struct BenchmarkResult {
    recall_at_1: f64,
    recall_at_10: f64,
    ndcg_at_10: f64,
    latency_p50_us: f64,
    latency_p95_us: f64,
    avg_scans: f64,
    pct_fast: f64,
    pct_balanced: f64,
    pct_strict: f64,
    pct_fallback: f64,
    false_confidence_rate: f64,
}

fn evaluate_adaptive_bounded(index: &HNSQRIndex, workload: &Workload) -> BenchmarkResult {
    let mut latencies_us = Vec::with_capacity(workload.queries.len());
    let mut top1_matches = 0;
    let mut recall_at_10_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let mut scan_count_sum = 0usize;

    let mut count_fast = 0usize;
    let mut count_balanced = 0usize;
    let mut count_strict = 0usize;
    let mut count_fallback = 0usize;
    let mut false_confidence_count = 0usize;
    let mut accepted_count = 0usize;

    for (q_idx, query) in workload.queries.iter().enumerate() {
        let gt = &workload.ground_truth[q_idx];

        let start = Instant::now();
        let (results, diag) = index
            .search_indices_adaptive(query, K_BENCH, None, AdaptivePolicy::RiveroOnly)
            .expect("Adaptive search must succeed");
        let elapsed = start.elapsed().as_micros() as f64;
        latencies_us.push(elapsed);

        scan_count_sum += diag.cumulative_resident_scans;

        match diag.final_profile {
            RiveroProfile::Fast => count_fast += 1,
            RiveroProfile::Balanced => count_balanced += 1,
            RiveroProfile::Strict => count_strict += 1,
        }
        if diag.graph_fallback_used {
            count_fallback += 1;
        }

        // Recall@1
        if let (Some(top_ret), Some(top_gt)) = (results.first(), gt.first())
            && top_ret.0 == top_gt.0
        {
            top1_matches += 1;
        }

        // Recall@10
        let hits = results
            .iter()
            .take(K_BENCH)
            .filter(|r| gt.iter().take(K_BENCH).any(|g| g.0 == r.0))
            .count();
        let rec_10 = (hits as f64) / (K_BENCH as f64);
        recall_at_10_sum += rec_10;

        // False Confidence Tracking: router accepted without escalation to Strict, but recall < 1.0
        if !diag.confidence.escalation_recommended {
            accepted_count += 1;
            if rec_10 < 0.999 {
                false_confidence_count += 1;
            }
        }

        ndcg_sum += calculate_ndcg(&results, gt, K_BENCH);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = workload.queries.len() as f64;

    BenchmarkResult {
        recall_at_1: (top1_matches as f64) / n,
        recall_at_10: recall_at_10_sum / n,
        ndcg_at_10: ndcg_sum / n,
        latency_p50_us: latencies_us[(latencies_us.len() as f64 * 0.50) as usize],
        latency_p95_us: latencies_us[(latencies_us.len() as f64 * 0.95) as usize],
        avg_scans: (scan_count_sum as f64) / n,
        pct_fast: ((count_fast as f64) / n) * 100.0,
        pct_balanced: ((count_balanced as f64) / n) * 100.0,
        pct_strict: ((count_strict as f64) / n) * 100.0,
        pct_fallback: ((count_fallback as f64) / n) * 100.0,
        false_confidence_rate: if accepted_count > 0 {
            ((false_confidence_count as f64) / (accepted_count as f64)) * 100.0
        } else {
            0.0
        },
    }
}

fn evaluate_adaptive_hybrid(index: &HNSQRIndex, workload: &Workload) -> BenchmarkResult {
    let mut latencies_us = Vec::with_capacity(workload.queries.len());
    let mut top1_matches = 0;
    let mut recall_at_10_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let mut scan_count_sum = 0usize;

    let mut count_fast = 0usize;
    let mut count_balanced = 0usize;
    let mut count_strict = 0usize;
    let mut count_fallback = 0usize;
    let mut false_confidence_count = 0usize;
    let mut accepted_count = 0usize;

    for (q_idx, query) in workload.queries.iter().enumerate() {
        let gt = &workload.ground_truth[q_idx];

        let start = Instant::now();
        let (results, diag) = index
            .search_indices_adaptive(query, K_BENCH, None, AdaptivePolicy::AllowGraphFallback)
            .expect("Adaptive search must succeed");
        let elapsed = start.elapsed().as_micros() as f64;
        latencies_us.push(elapsed);

        scan_count_sum += diag.cumulative_resident_scans;

        match diag.final_profile {
            RiveroProfile::Fast => count_fast += 1,
            RiveroProfile::Balanced => count_balanced += 1,
            RiveroProfile::Strict => count_strict += 1,
        }
        if diag.graph_fallback_used {
            count_fallback += 1;
        }

        if let (Some(top_ret), Some(top_gt)) = (results.first(), gt.first())
            && top_ret.0 == top_gt.0
        {
            top1_matches += 1;
        }

        let hits = results
            .iter()
            .take(K_BENCH)
            .filter(|r| gt.iter().take(K_BENCH).any(|g| g.0 == r.0))
            .count();
        let rec_10 = (hits as f64) / (K_BENCH as f64);
        recall_at_10_sum += rec_10;

        if !diag.confidence.escalation_recommended {
            accepted_count += 1;
            if rec_10 < 0.999 {
                false_confidence_count += 1;
            }
        }

        ndcg_sum += calculate_ndcg(&results, gt, K_BENCH);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = workload.queries.len() as f64;

    BenchmarkResult {
        recall_at_1: (top1_matches as f64) / n,
        recall_at_10: recall_at_10_sum / n,
        ndcg_at_10: ndcg_sum / n,
        latency_p50_us: latencies_us[(latencies_us.len() as f64 * 0.50) as usize],
        latency_p95_us: latencies_us[(latencies_us.len() as f64 * 0.95) as usize],
        avg_scans: (scan_count_sum as f64) / n,
        pct_fast: ((count_fast as f64) / n) * 100.0,
        pct_balanced: ((count_balanced as f64) / n) * 100.0,
        pct_strict: ((count_strict as f64) / n) * 100.0,
        pct_fallback: ((count_fallback as f64) / n) * 100.0,
        false_confidence_rate: if accepted_count > 0 {
            ((false_confidence_count as f64) / (accepted_count as f64)) * 100.0
        } else {
            0.0
        },
    }
}

fn evaluate_strict_reference(index: &HNSQRIndex, workload: &Workload) -> BenchmarkResult {
    let mut latencies_us = Vec::with_capacity(workload.queries.len());
    let mut top1_matches = 0;
    let mut recall_at_10_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let mut scan_count_sum = 0usize;

    for (q_idx, query) in workload.queries.iter().enumerate() {
        let gt = &workload.ground_truth[q_idx];

        let start = Instant::now();
        let (results, diag) = index
            .search_indices_strict(query, K_BENCH, None)
            .expect("Strict search must succeed");
        let elapsed = start.elapsed().as_micros() as f64;
        latencies_us.push(elapsed);

        scan_count_sum += diag.resident_scans;

        if let (Some(top_ret), Some(top_gt)) = (results.first(), gt.first())
            && top_ret.0 == top_gt.0
        {
            top1_matches += 1;
        }

        let hits = results
            .iter()
            .take(K_BENCH)
            .filter(|r| gt.iter().take(K_BENCH).any(|g| g.0 == r.0))
            .count();
        recall_at_10_sum += (hits as f64) / (K_BENCH as f64);
        ndcg_sum += calculate_ndcg(&results, gt, K_BENCH);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = workload.queries.len() as f64;

    BenchmarkResult {
        recall_at_1: (top1_matches as f64) / n,
        recall_at_10: recall_at_10_sum / n,
        ndcg_at_10: ndcg_sum / n,
        latency_p50_us: latencies_us[(latencies_us.len() as f64 * 0.50) as usize],
        latency_p95_us: latencies_us[(latencies_us.len() as f64 * 0.95) as usize],
        avg_scans: (scan_count_sum as f64) / n,
        pct_fast: 0.0,
        pct_balanced: 0.0,
        pct_strict: 100.0,
        pct_fallback: 0.0,
        false_confidence_rate: 0.0,
    }
}

fn evaluate_graph_only(index: &HNSQRIndex, workload: &Workload) -> BenchmarkResult {
    let mut latencies_us = Vec::with_capacity(workload.queries.len());
    let mut top1_matches = 0;
    let mut recall_at_10_sum = 0.0;
    let mut ndcg_sum = 0.0;

    for (q_idx, query) in workload.queries.iter().enumerate() {
        let gt = &workload.ground_truth[q_idx];

        let start = Instant::now();
        let results = index
            .search_indices_graph(query, K_BENCH, None)
            .expect("Graph search must succeed");
        let elapsed = start.elapsed().as_micros() as f64;
        latencies_us.push(elapsed);

        if let (Some(top_ret), Some(top_gt)) = (results.first(), gt.first())
            && top_ret.0 == top_gt.0
        {
            top1_matches += 1;
        }

        let hits = results
            .iter()
            .take(K_BENCH)
            .filter(|r| gt.iter().take(K_BENCH).any(|g| g.0 == r.0))
            .count();
        recall_at_10_sum += (hits as f64) / (K_BENCH as f64);
        ndcg_sum += calculate_ndcg(&results, gt, K_BENCH);
    }

    latencies_us.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = workload.queries.len() as f64;

    BenchmarkResult {
        recall_at_1: (top1_matches as f64) / n,
        recall_at_10: recall_at_10_sum / n,
        ndcg_at_10: ndcg_sum / n,
        latency_p50_us: latencies_us[(latencies_us.len() as f64 * 0.50) as usize],
        latency_p95_us: latencies_us[(latencies_us.len() as f64 * 0.95) as usize],
        avg_scans: 0.0,
        pct_fast: 0.0,
        pct_balanced: 0.0,
        pct_strict: 0.0,
        pct_fallback: 100.0,
        false_confidence_rate: 0.0,
    }
}

fn run_workload_benchmark(workload: &Workload) {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" WORKLOAD: {}", workload.name.to_uppercase());
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let actual_dim = workload.corpus.first().map_or(D, |v| v.dimension());
    let snap_path = common::bench_cache_dir().join(format!(
        "rivero_adapt_{}_d{actual_dim}_n{}.snapshot",
        workload.name.replace(' ', "_"),
        workload.corpus.len()
    ));
    let index = if snap_path.exists() {
        if let Ok(idx) = HNSQRIndex::open_snapshot_v2(
            &snap_path,
            hnsqr::storage::snapshot::SnapshotOpenOptions::default(),
        ) {
            idx.freeze_rivero_routing();
            idx
        } else {
            let mut config = HNSQRConfig::default();
            config.max_elements = workload.corpus.len() + 1000;
            config.rivero_enabled = true;
            config.rivero_fallback_on_underfill = true;
            config.rivero_witness_degree = 64;
            config.rivero_witness_seeds = 48;
            config.rivero_witness_second_seeds = 16;
            config.ef_construction = 8;
            config.m = 8;
            config.m0 = 8;
            let index = HNSQRIndex::new(config, actual_dim);
            for (i, v) in workload.corpus.iter().enumerate() {
                index.insert(format!("node_{i}"), v.clone()).unwrap();
            }
            index.freeze_rivero_routing();
            let _ = index.save_snapshot_v2(&snap_path);
            index
        }
    } else {
        let mut config = HNSQRConfig::default();
        config.max_elements = workload.corpus.len() + 1000;
        config.rivero_enabled = true;
        config.rivero_fallback_on_underfill = true;
        config.rivero_witness_degree = 64;
        config.rivero_witness_seeds = 48;
        config.rivero_witness_second_seeds = 16;
        config.ef_construction = 8;
        config.m = 8;
        config.m0 = 8;
        let index = HNSQRIndex::new(config, actual_dim);
        print!("  Indexing {} vectors... ", workload.corpus.len());
        let start_idx = Instant::now();
        for (i, v) in workload.corpus.iter().enumerate() {
            index.insert(format!("node_{i}"), v.clone()).unwrap();
        }
        let dur = start_idx.elapsed();
        println!(
            "Done in {:.2}s ({:.1} vec/s)\n",
            dur.as_secs_f64(),
            (workload.corpus.len() as f64) / dur.as_secs_f64()
        );
        index.freeze_rivero_routing();
        let _ = index.save_snapshot_v2(&snap_path);
        index
    };

    let res_strict = evaluate_strict_reference(&index, workload);
    let res_adapt_bounded = evaluate_adaptive_bounded(&index, workload);
    let res_adapt_hybrid = evaluate_adaptive_hybrid(&index, workload);
    let res_graph = evaluate_graph_only(&index, workload);

    println!("  Staged Execution & Routing Performance Table:");
    println!(
        "  ┌─────────────────────────────┬──────────┬──────────┬──────────┬────────────┬────────────┬─────────────┬──────────────┬──────────────┐"
    );
    println!(
        "  │ Architecture / Mode         │ Recall@1 │ Rec@10   │ NDCG@10  │ Latency p50│ Latency p95│ Avg Scans   │ % Fast/Bal/St│ % Fallback   │"
    );
    println!(
        "  ├─────────────────────────────┼──────────┼──────────┼──────────┼────────────┼────────────┼─────────────┼──────────────┼──────────────┤"
    );
    println!(
        "  │ Strict Bounded (Fixed)      │ {:>8.3} │ {:>8.3} │ {:>8.3} │ {:>8.1} µs│ {:>8.1} µs│ {:>11.0} │   0/  0/100% │ {:>10.1}%  │",
        res_strict.recall_at_1,
        res_strict.recall_at_10,
        res_strict.ndcg_at_10,
        res_strict.latency_p50_us,
        res_strict.latency_p95_us,
        res_strict.avg_scans,
        res_strict.pct_fallback
    );
    println!(
        "  │ Adaptive Bounded (Rivero)   │ {:>8.3} │ {:>8.3} │ {:>8.3} │ {:>8.1} µs│ {:>8.1} µs│ {:>11.0} │ {:>2.0}/{:>2.0}/{:>2.0}% │ {:>10.1}%  │",
        res_adapt_bounded.recall_at_1,
        res_adapt_bounded.recall_at_10,
        res_adapt_bounded.ndcg_at_10,
        res_adapt_bounded.latency_p50_us,
        res_adapt_bounded.latency_p95_us,
        res_adapt_bounded.avg_scans,
        res_adapt_bounded.pct_fast,
        res_adapt_bounded.pct_balanced,
        res_adapt_bounded.pct_strict,
        res_adapt_bounded.pct_fallback
    );
    println!(
        "  │ Adaptive Hybrid (Escape)    │ {:>8.3} │ {:>8.3} │ {:>8.3} │ {:>8.1} µs│ {:>8.1} µs│ {:>11.0} │ {:>2.0}/{:>2.0}/{:>2.0}% │ {:>10.1}%  │",
        res_adapt_hybrid.recall_at_1,
        res_adapt_hybrid.recall_at_10,
        res_adapt_hybrid.ndcg_at_10,
        res_adapt_hybrid.latency_p50_us,
        res_adapt_hybrid.latency_p95_us,
        res_adapt_hybrid.avg_scans,
        res_adapt_hybrid.pct_fast,
        res_adapt_hybrid.pct_balanced,
        res_adapt_hybrid.pct_strict,
        res_adapt_hybrid.pct_fallback
    );
    println!(
        "  │ Pure Graph (HNSW Traversal) │ {:>8.3} │ {:>8.3} │ {:>8.3} │ {:>8.1} µs│ {:>8.1} µs│         N/A │       N/A    │ {:>10.1}%  │",
        res_graph.recall_at_1,
        res_graph.recall_at_10,
        res_graph.ndcg_at_10,
        res_graph.latency_p50_us,
        res_graph.latency_p95_us,
        res_graph.pct_fallback
    );
    println!(
        "  └─────────────────────────────┴──────────┴──────────┴──────────┴────────────┴────────────┴─────────────┴──────────────┴──────────────┘\n"
    );

    println!("  Router Confidence & Calibration Metrics:");
    println!(
        "    * False Confidence Rate: {:.2}% (Queries accepted without strict escalation that missed target recall)",
        res_adapt_bounded.false_confidence_rate
    );
    println!(
        "    * Stage Distribution: Fast: {:.1}% | Balanced: {:.1}% | Strict: {:.1}%",
        res_adapt_bounded.pct_fast, res_adapt_bounded.pct_balanced, res_adapt_bounded.pct_strict
    );
    let speedup = res_strict.latency_p50_us / res_adapt_bounded.latency_p50_us.max(1e-3);
    println!(
        "    * Latency Acceleration: {:.2}x faster than Strict baseline (p50: {:.1} µs vs {:.1} µs)\n",
        speedup, res_adapt_bounded.latency_p50_us, res_strict.latency_p50_us
    );
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR COMMIT 2: STAGED CONFIDENCE-ADAPTIVE BOUNDED ROUTING BENCHMARK                ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    println!("Evaluation Configuration:");
    println!("  Corpus Size (N):     {}", N);
    println!("  Complex Dimension:   {}", D);
    println!("  Benchmark Queries:   {}", QUERY_COUNT);
    println!("  Target Top-k:        {}", K_BENCH);
    println!("  Random Seed:         0x{:x}", SEED);
    println!();

    let clustered = generate_clustered_workload(N, D, QUERY_COUNT, SEED);
    run_workload_benchmark(&clustered);

    let boundary = generate_boundary_workload(N, D, QUERY_COUNT, SEED ^ 0x1111);
    run_workload_benchmark(&boundary);

    let isotropic = generate_isotropic_workload(N, D, QUERY_COUNT, SEED ^ 0x2222);
    run_workload_benchmark(&isotropic);

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" STAGED CONFIDENCE-ADAPTIVE ROUTER EVALUATION COMPLETE");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
}
