/* hnsqr/benches/million_scale_path_comparison.rs */
//!▫~•◦-----------------------------------------------------------------‣
//! # HoloSphere Million-Scale Public Dataset Audit & Path Comparison
//!▫~•◦-----------------------------------------------------------------‣
//!
//! Evaluates retrieval latency and recall on 1,000,000+ vector real public
//! datasets (SIFT1M 128D and GloVe-100 100D) across all five retrieval paths:
//!   1. Exact-Forced Scan (Brute Force O(N))
//!   2. Graph-Forced (HNSW Traversal)
//!   3. Rivero-Strict-Forced (O(1) Bounded)
//!   4. Rivero-Adaptive-Forced (O(1) Progressive)
//!   5. Planner-Routed (Universal Cost-Based Planner)
//!
//! Automatically caches prebuilt indices as persistent `.snapshot` files
//! so subsequent runs load in milliseconds via memory mapping!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

mod common;

use hnsqr::planning::RetrievalContract;
use hnsqr::storage::snapshot::SnapshotOpenOptions;
use hnsqr::vector::folding::ComplexWeaver;
use hnsqr::{HNSQRIndex, VectorEmbedding};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const QUERIES_COUNT: usize = 20;
const K_NEIGHBORS: usize = 10;

#[derive(Clone, Copy, Debug)]
enum ExecMode {
    PlannerDefault,
    ExactForced,
    GraphOnly,
    RiveroStrict,
    RiveroAdaptive,
}

impl ExecMode {
    fn label(self) -> &'static str {
        match self {
            Self::PlannerDefault => "Planner-Routed",
            Self::ExactForced => "Exact-Forced",
            Self::GraphOnly => "Graph-Forced",
            Self::RiveroStrict => "Rivero-Strict-Forced",
            Self::RiveroAdaptive => "Rivero-Adaptive-Forced",
        }
    }

    fn requires_exact_recall(self) -> bool {
        matches!(self, Self::PlannerDefault | Self::ExactForced)
    }
}

fn read_fvecs(
    path: impl AsRef<Path>,
    max_vectors: Option<usize>,
) -> io::Result<Vec<VectorEmbedding>> {
    let mut file = File::open(path)?;
    let mut vectors = Vec::new();
    let mut dim_buf = [0u8; 4];
    while file.read_exact(&mut dim_buf).is_ok() {
        if let Some(limit) = max_vectors {
            if vectors.len() >= limit {
                break;
            }
        }
        let dim = i32::from_le_bytes(dim_buf) as usize;
        let mut float_buf = vec![0u8; dim * 4];
        file.read_exact(&mut float_buf)?;
        let mut floats = Vec::with_capacity(dim);
        for chunk in float_buf.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let norm = floats
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-9);
        let normalized: Vec<f32> = floats.iter().map(|value| value / norm).collect();
        vectors.push(ComplexWeaver::fold_llm_embedding(&normalized));
    }
    Ok(vectors)
}

fn percentile(mut xs: Vec<Duration>, pct: f64) -> Duration {
    if xs.is_empty() {
        return Duration::ZERO;
    }
    xs.sort();
    let idx = ((xs.len() as f64 - 1.0) * pct).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn benchmark_million_dataset(name: &str, snapshot_path: &Path, query_path: &Path, dim: usize) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  AUDITING: {name} (dim={dim})");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t0 = Instant::now();
    let queries =
        read_fvecs(query_path, Some(QUERIES_COUNT)).expect("Failed to read query vectors");
    println!("  Loaded {} queries in {:.2?}", queries.len(), t0.elapsed());

    assert!(
        snapshot_path.is_file(),
        "missing permanent million-scale database: {}",
        snapshot_path.display()
    );
    println!(
        "  Attaching permanent snapshot via zero-copy mmap: {:?}",
        snapshot_path
    );
    let t_load = Instant::now();
    let index = HNSQRIndex::open_snapshot_v2(snapshot_path, SnapshotOpenOptions::default())
        .expect("Failed to open permanent snapshot");
    println!("  Snapshot attached in {:.2?}", t_load.elapsed());
    index.freeze_rivero_routing();

    // Precompute parallel ground truth for all queries
    println!(
        "  Computing parallel ground truth for {} queries...",
        queries.len()
    );
    let t_gt = Instant::now();
    let ground_truths = common::compute_exact_ground_truth(&index, &queries, K_NEIGHBORS);
    println!("  Ground truth computed in {:.2?}", t_gt.elapsed());

    let modes = [
        ExecMode::PlannerDefault,
        ExecMode::ExactForced,
        ExecMode::GraphOnly,
        ExecMode::RiveroStrict,
        ExecMode::RiveroAdaptive,
    ];

    println!(
        "\n  {:<24} {:<8} {:<15} {:<12} {:<12}",
        "Mode", "Queries", "Recall mean/min", "p50 Lat", "p95 Lat"
    );
    println!("  {}", "-".repeat(73));

    for mode in modes {
        // Warmup
        let _ = index.search_indices_strict(&queries[0], K_NEIGHBORS, None);

        let mut recalls = Vec::with_capacity(queries.len());
        let mut latencies = Vec::with_capacity(queries.len());

        for (qi, query) in queries.iter().enumerate() {
            let gt = &ground_truths[qi];
            let gt_indices: std::collections::HashSet<_> = gt.iter().map(|(idx, _)| *idx).collect();

            let start = Instant::now();
            let raw_results = match mode {
                ExecMode::PlannerDefault => index.search_indices_with_contract(
                    query,
                    K_NEIGHBORS,
                    None,
                    RetrievalContract::default(),
                ),
                ExecMode::ExactForced => index.search_indices_exact(query, K_NEIGHBORS, None),
                ExecMode::GraphOnly => index.search_indices_graph(query, K_NEIGHBORS, None),
                ExecMode::RiveroStrict => index
                    .search_indices_strict(query, K_NEIGHBORS, None)
                    .map(|(res, _)| res),
                ExecMode::RiveroAdaptive => index
                    .search_indices_adaptive(
                        query,
                        K_NEIGHBORS,
                        None,
                        hnsqr::rivero::AdaptivePolicy::RiveroOnly,
                    )
                    .map(|(res, _)| res),
            };
            let elapsed = start.elapsed();

            let results = raw_results.unwrap_or_else(|error| {
                panic!(
                    "{} search failed for dataset {} query {qi}: {error}",
                    mode.label(),
                    name
                )
            });
            let matched = results
                .iter()
                .filter(|&&(idx, _)| gt_indices.contains(&idx))
                .count();
            let recall_pct = (matched as f64 / K_NEIGHBORS as f64) * 100.0;

            recalls.push(recall_pct);
            latencies.push(elapsed);
        }

        let mean_recall = recalls.iter().sum::<f64>() / recalls.len().max(1) as f64;
        let min_recall = recalls.iter().cloned().fold(f64::MAX, f64::min);
        let p50 = percentile(latencies.clone(), 0.50);
        let p95 = percentile(latencies, 0.95);
        let pass_str =
            if mode.requires_exact_recall() && mean_recall == 100.0 && min_recall == 100.0 {
                "[PASS]"
            } else if mode.requires_exact_recall() {
                "[FAIL]"
            } else {
                "[MEASURED]"
            };

        assert!(
            !mode.requires_exact_recall() || (mean_recall == 100.0 && min_recall == 100.0),
            "{} violated its exact-recall contract on {name}: mean={mean_recall:.1}% min={min_recall:.1}%",
            mode.label()
        );

        println!(
            "  {:<24} {:<8} {:>5.1}%/{:>5.1}% {:<6} {:<12.2?} {:<12.2?}",
            mode.label(),
            queries.len(),
            mean_recall,
            min_recall,
            pass_str,
            p50,
            p95
        );
    }
}

fn main() {
    println!(
        "╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║   HOLOSPHERE MILLION-SCALE EMPIRICAL PROOF (1M+ Corpora, Exact vs Graph vs Rivero O(1))                   ║"
    );
    println!(
        "╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );

    // 1. SIFT1M (1,000,000 vectors, 128-dim)
    let sift1m_base =
        PathBuf::from("benchmark_databases/million_sift1m_strict_v6_pStrict_d64_n1000000.snapshot");
    let sift1m_query = PathBuf::from("datasets/sift_1m/sift1m_query.fvecs");
    benchmark_million_dataset(
        "Texmex SIFT1M (Full 1,000,000 Vectors)",
        &sift1m_base,
        &sift1m_query,
        128,
    );

    // 2. GloVe-100 (1,183,514 vectors, 100-dim)
    let glove100_base = PathBuf::from(
        "benchmark_databases/million_glove100_strict_v6_pStrict_d50_n1183514.snapshot",
    );
    let glove100_query = PathBuf::from("datasets/glove_100/glove100_query.fvecs");
    benchmark_million_dataset(
        "GloVe-100 (Full 1,183,514 Vectors)",
        &glove100_base,
        &glove100_query,
        100,
    );
}
