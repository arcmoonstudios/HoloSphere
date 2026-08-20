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

use hnsqr::planning::RetrievalContract;
use hnsqr::rivero::{RiveroBulkBuilder, RiveroProfile};
use hnsqr::storage::snapshot::SnapshotOpenOptions;
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, VectorEmbedding};
use rayon::prelude::*;
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
}

fn read_fvecs(path: impl AsRef<Path>, max_vectors: Option<usize>) -> io::Result<Vec<VectorEmbedding>> {
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
        vectors.push(VectorEmbedding::from_reals(&floats).into_normalized());
    }
    Ok(vectors)
}

fn compute_brute_force_ground_truth_parallel(
    corpus: &[VectorEmbedding],
    query: &VectorEmbedding,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = corpus
        .par_iter()
        .enumerate()
        .map(|(idx, v)| (idx, v.dot_product_complex(query).re))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

fn percentile(mut xs: Vec<Duration>, pct: f64) -> Duration {
    if xs.is_empty() {
        return Duration::ZERO;
    }
    xs.sort();
    let idx = ((xs.len() as f64 - 1.0) * pct).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn benchmark_million_dataset(name: &str, base_path: &Path, query_path: &Path, dim: usize, limit_n: Option<usize>) {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  AUDITING: {name} (dim={dim})");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let t0 = Instant::now();
    let corpus = read_fvecs(base_path, limit_n).expect("Failed to read base vectors");
    let queries = read_fvecs(query_path, Some(QUERIES_COUNT)).expect("Failed to read query vectors");
    println!("  Loaded {} vectors and {} queries in {:.2?}", corpus.len(), queries.len(), t0.elapsed());

    let snapshot_path = base_path.with_extension("snapshot");
    let index = if snapshot_path.exists() {
        println!("  Found prebuilt snapshot on disk: {:?}. Loading via zero-copy mmap...", snapshot_path);
        let t_load = Instant::now();
        let index = HNSQRIndex::open_snapshot_v2(&snapshot_path, SnapshotOpenOptions::default())
            .expect("Failed to open prebuilt snapshot");
        println!("  Snapshot attached in {:.2?}", t_load.elapsed());
        index.freeze_rivero_routing();
        index
    } else {
        println!("  No prebuilt snapshot found. Building with parallel Rayon pipeline...");
        let mut config = HNSQRConfig::strict_rivero_for_dim(dim);
        config.distance_function = DistanceFunction::Cosine;
        config.max_elements = corpus.len() + 10_000;
        config.rivero_fallback_on_underfill = false;
        config.rivero_witness_degree = 32;
        let index = HNSQRIndex::new(config, dim);

        println!("  1/3: Populating Arena with {} vectors in deterministic slot order...", corpus.len());
        let t_pop = Instant::now();
        for (i, v) in corpus.iter().enumerate() {
            let doc_id = format!("doc_{i}");
            index.insert(doc_id.as_str(), v.clone()).unwrap();
        }
        println!("  Arena populated in {:.2?}", t_pop.elapsed());

        println!("  2/3: Building Parallel Rivero E8 State across all CPU threads via Rayon...");
        let t_bulk = Instant::now();
        let mut addr_cfg = index.config().rivero_address_config;
        addr_cfg.geometry = hnsqr::rivero::VectorGeometry::Real;
        let builder = RiveroBulkBuilder::with_profile(RiveroProfile::Strict)
            .with_address_config(addr_cfg)
            .with_witness_params(32, 16, 8);
        let built_state = builder.build(&corpus).expect("Bulk build failed");
        println!("  Rivero Bulk State computed in {:.2?}", t_bulk.elapsed());

        index.install_rivero_state(built_state).expect("Failed to install rivero state");
        index.freeze_rivero_routing();

        println!("  3/3: Saving prebuilt snapshot to disk for instant future loading...");
        let t_snap = Instant::now();
        index.save_snapshot_v2(&snapshot_path).expect("Failed to save snapshot");
        println!("  Snapshot saved in {:.2?} ({:?})", t_snap.elapsed(), snapshot_path);

        index
    };

    // Precompute parallel ground truth for all queries
    println!("  Computing parallel ground truth for {} queries...", queries.len());
    let t_gt = Instant::now();
    let ground_truths: Vec<Vec<(usize, f32)>> = queries
        .iter()
        .map(|q| compute_brute_force_ground_truth_parallel(&corpus, q, K_NEIGHBORS))
        .collect();
    println!("  Ground truth computed in {:.2?}", t_gt.elapsed());

    let modes = [
        ExecMode::PlannerDefault,
        ExecMode::ExactForced,
        ExecMode::GraphOnly,
        ExecMode::RiveroStrict,
        ExecMode::RiveroAdaptive,
    ];

    println!("\n  {:<24} {:<8} {:<15} {:<12} {:<12}", "Mode", "Queries", "Recall mean/min", "p50 Lat", "p95 Lat");
    println!("  {}", "-".repeat(73));

    for mode in modes {
        // Warmup
        let _ = index.search_indices_strict(&queries[0], K_NEIGHBORS, None);

        let mut recalls = Vec::with_capacity(queries.len());
        let mut latencies = Vec::with_capacity(queries.len());

        for (qi, query) in queries.iter().enumerate() {
            let gt = &ground_truths[qi];
            let gt_indices: std::collections::HashSet<usize> = gt.iter().map(|(idx, _)| *idx).collect();

            let start = Instant::now();
            let raw_results = match mode {
                ExecMode::PlannerDefault => {
                    index.search_indices_with_contract(query, K_NEIGHBORS, None, RetrievalContract::Certified)
                }
                ExecMode::ExactForced => {
                    index.search_indices_exact(query, K_NEIGHBORS, None)
                }
                ExecMode::GraphOnly => {
                    index.search_indices_graph(query, K_NEIGHBORS, None)
                }
                ExecMode::RiveroStrict => {
                    index.search_indices_strict(query, K_NEIGHBORS, None).map(|(res, _)| res)
                }
                ExecMode::RiveroAdaptive => {
                    index.search_indices_adaptive(query, K_NEIGHBORS, None, hnsqr::rivero::AdaptivePolicy::RiveroOnly).map(|(res, _)| res)
                }
            };
            let elapsed = start.elapsed();

            let results = raw_results.unwrap_or_default();
            let matched = results
                .iter()
                .filter(|&&(idx, _)| gt_indices.contains(&(idx as usize)))
                .count();
            let recall_pct = (matched as f64 / K_NEIGHBORS as f64) * 100.0;

            recalls.push(recall_pct);
            latencies.push(elapsed);
        }

        let mean_recall = recalls.iter().sum::<f64>() / recalls.len().max(1) as f64;
        let min_recall = recalls.iter().cloned().fold(f64::MAX, f64::min);
        let p50 = percentile(latencies.clone(), 0.50);
        let p95 = percentile(latencies, 0.95);
        let pass_str = if mean_recall >= 99.0 { "[PASS]" } else { "[FAIL]" };

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
    println!("╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║   HOLOSPHERE MILLION-SCALE EMPIRICAL PROOF (1M+ Corpora, Exact vs Graph vs Rivero O(1))                   ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // 1. SIFT1M (1,000,000 vectors, 128-dim)
    let sift1m_base = PathBuf::from("datasets/sift_1m/sift1m_base.fvecs");
    let sift1m_query = PathBuf::from("datasets/sift_1m/sift1m_query.fvecs");
    if sift1m_base.exists() && sift1m_query.exists() {
        benchmark_million_dataset("Texmex SIFT1M (Full 1,000,000 Vectors)", &sift1m_base, &sift1m_query, 128, Some(1_000_000));
    }

    // 2. GloVe-100 (1,183,514 vectors, 100-dim)
    let glove100_base = PathBuf::from("datasets/glove_100/glove100_base.fvecs");
    let glove100_query = PathBuf::from("datasets/glove_100/glove100_query.fvecs");
    if glove100_base.exists() && glove100_query.exists() {
        benchmark_million_dataset("GloVe-100 (Full 1,183,514 Vectors)", &glove100_base, &glove100_query, 100, Some(1_183_514));
    }
}
