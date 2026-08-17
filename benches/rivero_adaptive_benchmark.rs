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
use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

fn normalize_complex(data: Vec<Complex32>) -> VectorEmbedding {
    VectorEmbedding::from_complex(data).into_normalized()
}

fn generate_clustered_workload(n: usize, d: usize, q_count: usize, seed: u64) -> Workload {
    let mut rng = StdRng::seed_from_u64(seed);
    let num_clusters = 50;
    let cluster_centers: Vec<Vec<Complex32>> = (0..num_clusters)
        .map(|_| {
            (0..d)
                .map(|_| Complex32::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
                .collect()
        })
        .collect();

    let mut corpus = Vec::with_capacity(n);
    for i in 0..n {
        let c_idx = i % num_clusters;
        let center = &cluster_centers[c_idx];
        let noise_scale = rng.gen_range(0.04..0.15f32);
        let vec: Vec<Complex32> = center
            .iter()
            .map(|&z| {
                z + Complex32::new(
                    rng.gen_range(-noise_scale..noise_scale),
                    rng.gen_range(-noise_scale..noise_scale),
                )
            })
            .collect();
        corpus.push(normalize_complex(vec));
    }

    let mut queries = Vec::with_capacity(q_count);
    for q in 0..q_count {
        let c_idx = q % num_clusters;
        let center = &cluster_centers[c_idx];
        let vec: Vec<Complex32> = center
            .iter()
            .map(|&z| z + Complex32::new(rng.gen_range(-0.1..0.1), rng.gen_range(-0.1..0.1)))
            .collect();
        queries.push(normalize_complex(vec));
    }

    let ground_truth = compute_exact_ground_truth(&corpus, &queries, K_BENCH);
    Workload {
        name: "Clustered Semantic".to_string(),
        corpus,
        queries,
        ground_truth,
    }
}

fn generate_boundary_workload(n: usize, d: usize, q_count: usize, seed: u64) -> Workload {
    let mut rng = StdRng::seed_from_u64(seed);
    let num_clusters = 50;
    let cluster_centers: Vec<Vec<Complex32>> = (0..num_clusters)
        .map(|_| {
            (0..d)
                .map(|_| Complex32::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
                .collect()
        })
        .collect();

    let mut corpus = Vec::with_capacity(n);
    for i in 0..n {
        let c_idx = i % num_clusters;
        let center = &cluster_centers[c_idx];
        let noise_scale = rng.gen_range(0.05..0.20f32);
        let vec: Vec<Complex32> = center
            .iter()
            .map(|&z| {
                z + Complex32::new(
                    rng.gen_range(-noise_scale..noise_scale),
                    rng.gen_range(-noise_scale..noise_scale),
                )
            })
            .collect();
        corpus.push(normalize_complex(vec));
    }

    let mut queries = Vec::with_capacity(q_count);
    for q in 0..q_count {
        let c1 = q % num_clusters;
        let c2 = (q + 1) % num_clusters;
        let alpha = rng.gen_range(0.4..0.6f32);
        let vec: Vec<Complex32> = cluster_centers[c1]
            .iter()
            .zip(cluster_centers[c2].iter())
            .map(|(&z1, &z2)| z1 * alpha + z2 * (1.0 - alpha))
            .collect();
        queries.push(normalize_complex(vec));
    }

    let ground_truth = compute_exact_ground_truth(&corpus, &queries, K_BENCH);
    Workload {
        name: "Boundary Adversarial".to_string(),
        corpus,
        queries,
        ground_truth,
    }
}

fn generate_isotropic_workload(n: usize, d: usize, q_count: usize, seed: u64) -> Workload {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut corpus = Vec::with_capacity(n);
    for _ in 0..n {
        let vec: Vec<Complex32> = (0..d)
            .map(|_| Complex32::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
            .collect();
        corpus.push(normalize_complex(vec));
    }

    let mut queries = Vec::with_capacity(q_count);
    for _ in 0..q_count {
        let vec: Vec<Complex32> = (0..d)
            .map(|_| Complex32::new(rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)))
            .collect();
        queries.push(normalize_complex(vec));
    }

    let ground_truth = compute_exact_ground_truth(&corpus, &queries, K_BENCH);
    Workload {
        name: "Isotropic Uniform".to_string(),
        corpus,
        queries,
        ground_truth,
    }
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
                .map(|(idx, doc)| (idx as NodeIndex, query.quantum_fidelity(doc)))
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

#[allow(dead_code)]
#[derive(Clone, Debug, Default)]
struct BenchmarkResult {
    recall_at_1: f64,
    recall_at_10: f64,
    ndcg_at_10: f64,
    latency_p50_us: f64,
    latency_p95_us: f64,
    latency_p99_us: f64,
    avg_scans: f64,
    avg_exact_scores: f64,
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
    let mut exact_eval_sum = 0usize;

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
        exact_eval_sum += diag.cumulative_exact_scores;

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
        latency_p99_us: latencies_us[(latencies_us.len() as f64 * 0.99) as usize],
        avg_scans: (scan_count_sum as f64) / n,
        avg_exact_scores: (exact_eval_sum as f64) / n,
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
    let mut exact_eval_sum = 0usize;

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
        exact_eval_sum += diag.cumulative_exact_scores;

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
        latency_p99_us: latencies_us[(latencies_us.len() as f64 * 0.99) as usize],
        avg_scans: (scan_count_sum as f64) / n,
        avg_exact_scores: (exact_eval_sum as f64) / n,
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
    let mut exact_eval_sum = 0usize;

    for (q_idx, query) in workload.queries.iter().enumerate() {
        let gt = &workload.ground_truth[q_idx];

        let start = Instant::now();
        let (results, diag) = index
            .search_indices_strict(query, K_BENCH, None)
            .expect("Strict search must succeed");
        let elapsed = start.elapsed().as_micros() as f64;
        latencies_us.push(elapsed);

        scan_count_sum += diag.resident_scans;
        exact_eval_sum += diag.exact_score_evaluations;

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
        latency_p99_us: latencies_us[(latencies_us.len() as f64 * 0.99) as usize],
        avg_scans: (scan_count_sum as f64) / n,
        avg_exact_scores: (exact_eval_sum as f64) / n,
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
        latency_p99_us: latencies_us[(latencies_us.len() as f64 * 0.99) as usize],
        avg_scans: 0.0,
        avg_exact_scores: 0.0,
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

    let mut config = HNSQRConfig::default();
    config.max_elements = workload.corpus.len() + 1000;
    config.rivero_enabled = true;
    config.rivero_fallback_on_underfill = true;
    config.rivero_witness_degree = 64;
    config.rivero_witness_seeds = 48;
    config.rivero_witness_second_seeds = 16;

    let index = HNSQRIndex::new(config, D);

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
