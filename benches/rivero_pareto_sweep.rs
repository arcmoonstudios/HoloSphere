/* hnsqr/benches/rivero_pareto_sweep.rs */
//!▫~•◦-------------------------------‣
//! # Rivero Parameter Pareto Frontier Sweep Benchmark
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Systematically evaluates Rivero configuration parameter space:
//!   - Foundations: 8, 12, 16, 24
//!   - SimHash Query Probes: 8, 16, 24, 32
//!   - Cell Capacity: 32, 48, 64
//!   - Cell Budget: 8, 12, 16
//!   - Candidate Cap: 512, 1024, 2048
//!
//! Computes exact exhaustive ground-truth and measures:
//!   - Recall@1, Recall@10, Recall@100
//!   - MRR (Mean Reciprocal Rank) & NDCG@10
//!   - Resident Scans (mean & peak) vs Theoretical bounds
//!   - Admissions & Candidate Deduplication
//!   - Latency distributions (mean, p50, p95, p99)
//! across Clustered, Isotropic Uniform, and Boundary Adversarial query workloads.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashSet;
use std::time::Instant;

use hnsqr::{
    HNSQRConfig, HNSQRIndex, NodeIndex, RiveroAddress, RiveroConfig, SimilarityScore,
    VectorEmbedding,
};
use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

const SWEEP_SEED: u64 = 0x5249_5645_524f_5357;
const DIMENSION: usize = 64;
const CORPUS_SIZE: usize = 10_000;
const QUERY_COUNT: usize = 100;
const K_BENCH: usize = 10;

#[derive(Clone, Debug)]
struct Workload {
    name: String,
    corpus: Vec<VectorEmbedding>,
    queries: Vec<VectorEmbedding>,
    ground_truth: Vec<Vec<(NodeIndex, SimilarityScore)>>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct SweepResult {
    config: RiveroConfig,
    workload_name: String,
    recall_at_1: f64,
    recall_at_10: f64,
    recall_at_100: f64,
    mrr: f64,
    ndcg_at_10: f64,
    avg_scans: f64,
    max_scans: usize,
    scan_ceiling: usize,
    scan_saturation_pct: f64,
    avg_admissions: f64,
    avg_candidates: f64,
    latency_mean_us: f64,
    latency_p50_us: f64,
    latency_p95_us: f64,
    latency_p99_us: f64,
}

fn normalize_complex(data: Vec<Complex32>) -> VectorEmbedding {
    VectorEmbedding::from_complex(data).into_normalized()
}

fn generate_clustered_workload(
    count: usize,
    dim: usize,
    query_count: usize,
    clusters: usize,
    seed: u64,
) -> Workload {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<VectorEmbedding> = (0..clusters)
        .map(|_| {
            normalize_complex(
                (0..dim)
                    .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            )
        })
        .collect();

    let corpus: Vec<VectorEmbedding> = (0..count)
        .map(|idx| {
            let center = &centers[idx % clusters];
            normalize_complex(
                center
                    .complex_data()
                    .iter()
                    .map(|&z| {
                        z + Complex32::new(rng.random_range(-0.04..0.04), rng.random_range(-0.04..0.04))
                    })
                    .collect(),
            )
        })
        .collect();

    let queries: Vec<VectorEmbedding> = (0..query_count)
        .map(|idx| {
            let center = &centers[idx % clusters];
            normalize_complex(
                center
                    .complex_data()
                    .iter()
                    .map(|&z| {
                        z + Complex32::new(rng.random_range(-0.03..0.03), rng.random_range(-0.03..0.03))
                    })
                    .collect(),
            )
        })
        .collect();

    let ground_truth = compute_ground_truth(&corpus, &queries, 100);

    Workload {
        name: "Clustered Semantic".to_string(),
        corpus,
        queries,
        ground_truth,
    }
}

fn generate_isotropic_workload(
    count: usize,
    dim: usize,
    query_count: usize,
    seed: u64,
) -> Workload {
    let mut rng = StdRng::seed_from_u64(seed);
    let corpus: Vec<VectorEmbedding> = (0..count)
        .map(|_| {
            normalize_complex(
                (0..dim)
                    .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            )
        })
        .collect();

    let queries: Vec<VectorEmbedding> = (0..query_count)
        .map(|_| {
            normalize_complex(
                (0..dim)
                    .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            )
        })
        .collect();

    let ground_truth = compute_ground_truth(&corpus, &queries, 100);

    Workload {
        name: "Isotropic Uniform".to_string(),
        corpus,
        queries,
        ground_truth,
    }
}

fn generate_boundary_workload(
    count: usize,
    dim: usize,
    query_count: usize,
    clusters: usize,
    seed: u64,
) -> Workload {
    let mut rng = StdRng::seed_from_u64(seed);
    let centers: Vec<VectorEmbedding> = (0..clusters)
        .map(|_| {
            normalize_complex(
                (0..dim)
                    .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                    .collect(),
            )
        })
        .collect();

    let corpus: Vec<VectorEmbedding> = (0..count)
        .map(|idx| {
            let center = &centers[idx % clusters];
            normalize_complex(
                center
                    .complex_data()
                    .iter()
                    .map(|&z| {
                        z + Complex32::new(rng.random_range(-0.03..0.03), rng.random_range(-0.03..0.03))
                    })
                    .collect(),
            )
        })
        .collect();

    // Queries interpolated midway between adjacent cluster centers
    let queries: Vec<VectorEmbedding> = (0..query_count)
        .map(|idx| {
            let c1 = &centers[idx % clusters];
            let c2 = &centers[(idx + 1) % clusters];
            let interp: Vec<Complex32> = c1
                .complex_data()
                .iter()
                .zip(c2.complex_data().iter())
                .map(|(&z1, &z2)| {
                    (z1 + z2) * 0.5
                        + Complex32::new(rng.random_range(-0.01..0.01), rng.random_range(-0.01..0.01))
                })
                .collect();
            normalize_complex(interp)
        })
        .collect();

    let ground_truth = compute_ground_truth(&corpus, &queries, 100);

    Workload {
        name: "Boundary Adversarial".to_string(),
        corpus,
        queries,
        ground_truth,
    }
}

fn compute_ground_truth(
    corpus: &[VectorEmbedding],
    queries: &[VectorEmbedding],
    top_k: usize,
) -> Vec<Vec<(NodeIndex, SimilarityScore)>> {
    queries
        .par_iter()
        .map(|query| {
            let mut scores: Vec<(NodeIndex, SimilarityScore)> = corpus
                .iter()
                .enumerate()
                .map(|(idx, doc)| (idx as NodeIndex, query.projective_overlap(doc)))
                .collect();
            scores.sort_unstable_by(|lhs, rhs| {
                rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
            });
            scores.truncate(top_k);
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

fn evaluate_configuration(
    index: &HNSQRIndex,
    workload: &Workload,
    rivero_cfg: &RiveroConfig,
    _addresses: &[RiveroAddress],
) -> SweepResult {
    let mut latencies_us: Vec<f64> = Vec::with_capacity(workload.queries.len());
    let mut top1_matches = 0;
    let mut recall_at_10_sum = 0.0;
    let mut recall_at_100_sum = 0.0;
    let mut mrr_sum = 0.0;
    let mut ndcg_sum = 0.0;
    let mut scan_count_sum = 0usize;
    let mut max_scans = 0usize;
    let mut admission_count_sum = 0usize;
    let mut candidate_count_sum = 0usize;

    for (q_idx, query) in workload.queries.iter().enumerate() {
        let gt = &workload.ground_truth[q_idx];

        let start = Instant::now();
        let (results, diag) = index
            .search_indices_o1_with_config(query, K_BENCH, None, rivero_cfg)
            .expect("Rivero query must succeed");
        let elapsed_us = start.elapsed().as_micros() as f64;
        latencies_us.push(elapsed_us);

        scan_count_sum += diag.resident_scans;
        max_scans = max_scans.max(diag.resident_scans);
        admission_count_sum += diag.resident_reads;
        candidate_count_sum += diag.unique_candidates;

        // Recall@1
        if let (Some(top_ret), Some(top_gt)) = (results.first(), gt.first())
            && top_ret.0 == top_gt.0
        {
            top1_matches += 1;
        }

        // MRR of top GT
        if let Some(top_gt) = gt.first()
            && let Some(pos) = results.iter().position(|r| r.0 == top_gt.0)
        {
            mrr_sum += 1.0 / ((pos + 1) as f64);
        }

        // Recall@10
        let gt_10_set: HashSet<NodeIndex> = gt.iter().take(K_BENCH).map(|g| g.0).collect();
        let ret_10_hits = results
            .iter()
            .take(K_BENCH)
            .filter(|r| gt_10_set.contains(&r.0))
            .count();
        recall_at_10_sum += (ret_10_hits as f64) / (gt_10_set.len().max(1) as f64);

        // Recall@100 (in candidate set)
        let gt_100_set: HashSet<NodeIndex> = gt.iter().take(100).map(|g| g.0).collect();
        let ret_100_hits = results.iter().filter(|r| gt_100_set.contains(&r.0)).count();
        recall_at_100_sum += (ret_100_hits as f64) / (gt_100_set.len().max(1) as f64);

        // NDCG@10
        ndcg_sum += calculate_ndcg(&results, gt, K_BENCH);
    }

    latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q_count = workload.queries.len() as f64;
    let scan_ceiling = rivero_cfg.resident_scan_bound();
    let avg_scans = (scan_count_sum as f64) / q_count;

    SweepResult {
        config: *rivero_cfg,
        workload_name: workload.name.clone(),
        recall_at_1: (top1_matches as f64) / q_count,
        recall_at_10: recall_at_10_sum / q_count,
        recall_at_100: recall_at_100_sum / q_count,
        mrr: mrr_sum / q_count,
        ndcg_at_10: ndcg_sum / q_count,
        avg_scans,
        max_scans,
        scan_ceiling,
        scan_saturation_pct: (avg_scans / (scan_ceiling.max(1) as f64)) * 100.0,
        avg_admissions: (admission_count_sum as f64) / q_count,
        avg_candidates: (candidate_count_sum as f64) / q_count,
        latency_mean_us: latencies_us.iter().sum::<f64>() / q_count,
        latency_p50_us: latencies_us[(latencies_us.len() * 50) / 100],
        latency_p95_us: latencies_us[(latencies_us.len() * 95) / 100],
        latency_p99_us: latencies_us[(latencies_us.len() * 99) / 100],
    }
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR / RIVERO PARAMETER PARETO SWEEP & FIXED-WORK BOUND ANALYSIS                    ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    println!(
        "Building test workloads (N={}, Dim={}, Queries={})...",
        CORPUS_SIZE, DIMENSION, QUERY_COUNT
    );
    let clustered =
        generate_clustered_workload(CORPUS_SIZE, DIMENSION, QUERY_COUNT, 32, SWEEP_SEED);
    let isotropic =
        generate_isotropic_workload(CORPUS_SIZE, DIMENSION, QUERY_COUNT, SWEEP_SEED ^ 0x1234);
    let boundary =
        generate_boundary_workload(CORPUS_SIZE, DIMENSION, QUERY_COUNT, 32, SWEEP_SEED ^ 0x5678);

    let workloads = vec![clustered, isotropic, boundary];

    // Parameter Grid Definition
    let foundation_options = [8, 12, 16, 24];
    let probe_options = [8, 16, 24, 32];
    let capacity_options = [32, 48, 64];
    let budget_options = [8, 12, 16];
    let candidate_cap_options = [512, 1024, 2048];

    println!("Parameter Search Grid:");
    println!("  Foundations (F):       {:?}", foundation_options);
    println!("  SimHash Probes (P):    {:?}", probe_options);
    println!("  Cell Capacity (C):     {:?}", capacity_options);
    println!("  Cell Budget (B):       {:?}", budget_options);
    println!("  Candidate Cap (K_cap): {:?}", candidate_cap_options);
    println!();

    for workload in &workloads {
        println!(
            "════════════════════════════════════════════════════════════════════════════════════════"
        );
        println!(" WORKLOAD: {}", workload.name.to_uppercase());
        println!(
            "════════════════════════════════════════════════════════════════════════════════════════"
        );

        // Build Index with maximum capacity profile
        let mut base_config = HNSQRConfig::strict_rivero_for_dim(DIMENSION);
        base_config.max_elements = CORPUS_SIZE + 1000;
        base_config.rivero_enabled = true;
        base_config.rivero_fallback_on_underfill = false;

        let index = HNSQRIndex::new(base_config, DIMENSION);
        print!("  Indexing {} vectors... ", workload.corpus.len());
        let build_start = Instant::now();
        for (idx, doc) in workload.corpus.iter().enumerate() {
            index
                .insert(format!("node_{}", idx), doc.clone())
                .expect("Insert succeeded");
        }
        let build_time = build_start.elapsed().as_secs_f64();
        println!(
            "Done in {:.2}s ({:.1} vec/s)",
            build_time,
            (workload.corpus.len() as f64) / build_time
        );

        // Precompile query addresses
        let addresses: Vec<RiveroAddress> = workload
            .queries
            .iter()
            .map(|q| index.compile_rivero_address(q).unwrap())
            .collect();

        // Selected representative grid points for clear analysis
        let test_profiles = vec![
            ("Baseline Strict", RiveroConfig::strict_default()),
            ("Balanced Fast", RiveroConfig::fast_balanced()),
            ("Ultra-Lean", RiveroConfig::custom(8, 8, 32, 8, 512)),
            ("Medium Lean", RiveroConfig::custom(12, 16, 32, 8, 1024)),
            ("High Density", RiveroConfig::custom(16, 24, 48, 12, 1024)),
            (
                "Full Foundations-Reduced Probes",
                RiveroConfig::custom(24, 16, 48, 12, 1024),
            ),
            (
                "Reduced Foundations-Max Capacity",
                RiveroConfig::custom(12, 32, 64, 16, 2048),
            ),
        ];

        println!("\n  Representative Configuration Sweep Table:");
        println!(
            "  ┌─────────────────────────────────┬──────┬──────┬──────┬──────┬─────────┬──────────┬──────────┬──────────┬────────────┬─────────────┐"
        );
        println!(
            "  │ Profile Name                    │   F  │   P  │   C  │   B  │  K_cap  │ Recall@1 │ Rec@10   │ NDCG@10  │ Avg Scans  │ Latency p50 │"
        );
        println!(
            "  ├─────────────────────────────────┼──────┼──────┼──────┼──────┼─────────┼──────────┼──────────┼──────────┼────────────┼─────────────┤"
        );

        let mut results = Vec::new();
        for (name, cfg) in &test_profiles {
            let res = evaluate_configuration(&index, workload, cfg, &addresses);
            println!(
                "  │ {:<31} │ {:>4} │ {:>4} │ {:>4} │ {:>4} │ {:>7} │ {:>8.3} │ {:>8.3} │ {:>8.3} │ {:>10.0} │ {:>9.1} µs │",
                name,
                cfg.foundations,
                cfg.simhash_query_probes,
                cfg.cell_capacity,
                cfg.cell_budget,
                cfg.query_candidate_cap,
                res.recall_at_1,
                res.recall_at_10,
                res.ndcg_at_10,
                res.avg_scans,
                res.latency_p50_us,
            );
            results.push((name.to_string(), res));
        }
        println!(
            "  └─────────────────────────────────┴──────┴──────┴──────┴──────┴─────────┴──────────┴──────────┴──────────┴────────────┴─────────────┘\n"
        );

        // Print Pareto Frontier Insights
        println!("  Pareto Frontier Analysis for {}:", workload.name);
        if let Some((_, baseline)) = results.iter().find(|(name, _)| name.contains("Baseline"))
            && let Some((_, fast)) = results
                .iter()
                .find(|(name, _)| name.contains("Balanced Fast"))
        {
            let scan_reduction = (1.0 - (fast.avg_scans / baseline.avg_scans.max(1.0))) * 100.0;
            let lat_speedup = baseline.latency_p50_us / fast.latency_p50_us.max(1e-3);
            println!("    * 'Balanced Fast' vs 'Baseline Strict':");
            println!(
                "      - Resident Scans: {:.0} vs {:.0} ({:.1}% reduction)",
                fast.avg_scans, baseline.avg_scans, scan_reduction
            );
            println!(
                "      - Recall@10:      {:.3} vs {:.3}",
                fast.recall_at_10, baseline.recall_at_10
            );
            println!(
                "      - Latency p50:    {:.1} µs vs {:.1} µs ({:.2}x faster)",
                fast.latency_p50_us, baseline.latency_p50_us, lat_speedup
            );
        }
        println!();
    }

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" RIVERO PARAMETER PARETO SWEEP COMPLETE");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
}
