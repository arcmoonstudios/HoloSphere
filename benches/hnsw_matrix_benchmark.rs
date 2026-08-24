/* holosphere/benches/hnsw_matrix_benchmark.rs */
//!▫~•◦-----------------------------------------------------------------‣
//! # Performance Track P1: Ground-Up HNSW Parameter Matrix & Admission
//!▫~•◦-----------------------------------------------------------------‣
//!
//! Sweeps M in [8, 16, 32, 48], ef_construction in [100, 200, 400],
//! and ef_search in [16, 32, 64, 128, 256, 512] on real million-scale corpora.
//!
//! Gates:
//!   1. Survival: Recall@10 >= 95%
//!   2. Production Candidate: Recall@10 >= 99% AND Speedup >= 2.0x vs P0 Exact p50.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use hnsqr::storage::snapshot::SnapshotOpenOptions;
use hnsqr::vector::folding::ComplexWeaver;
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, SearchPlan, VectorEmbedding};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const QUERIES_COUNT: usize = 500;
const K_NEIGHBORS: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswMatrixRow {
    pub corpus: String,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub mean_recall_10: f64,
    pub min_recall_10: f64,
    pub p05_recall_10: f64,
    pub mean_recall_1: f64,
    pub exact_p50_ms: f64,
    pub hnsw_p50_ms: f64,
    pub hnsw_p95_ms: f64,
    pub hnsw_p99_ms: f64,
    pub speedup: f64,
    pub qps: f64,
    pub avg_visits: usize,
    pub is_survival_pass: bool,
    pub is_production_candidate: bool,
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

fn percentile_f64(mut xs: Vec<f64>, pct: f64) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.total_cmp(b));
    let idx = ((xs.len() as f64 - 1.0) * pct).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn get_or_build_hnsw_index(
    tag: &str,
    corpus_vecs: &[VectorEmbedding],
    dim: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
) -> HNSQRIndex {
    let base_dir = PathBuf::from("benchmark_databases");
    let _ = std::fs::create_dir_all(&base_dir);
    let snap_path = base_dir.join(format!(
        "hnsw_p1_{tag}_m{m}_efc{ef_construction}_n{}.snapshot",
        corpus_vecs.len()
    ));

    if snap_path.exists() {
        println!(
            "    ⚡ Loading cached persistent HNSW index: {}",
            snap_path.display()
        );
        let t_load = Instant::now();
        let index = HNSQRIndex::open_snapshot_v2(&snap_path, SnapshotOpenOptions::default())
            .expect("open snapshot");
        println!("    ✓ Loaded snapshot in {:.2?}", t_load.elapsed());
        return index;
    }

    println!(
        "    ⚙️ Building fresh HNSW index (M={m}, M0={m0}, efC={ef_construction}, N={})...",
        corpus_vecs.len()
    );
    let t_build = Instant::now();

    let mut config = HNSQRConfig::default();
    config.distance_function = DistanceFunction::Cosine;
    config.search_plan = SearchPlan::GraphOnly;
    config.rivero_enabled = false;
    config.max_elements = corpus_vecs.len() + 10_000;
    config.m = m;
    config.m0 = m0;
    config.ef_construction = ef_construction;
    config.ef_search = 128;

    let index = Arc::new(HNSQRIndex::new(config, dim));
    let chunk_size = 10_000;
    let completed = AtomicUsize::new(0);
    let total = corpus_vecs.len();

    // Insert batches in parallel across Rayon workers
    corpus_vecs
        .par_chunks(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            for (i, v) in chunk.iter().enumerate() {
                let global_idx = chunk_idx * chunk_size + i;
                index
                    .insert(format!("doc_{global_idx}"), v.clone())
                    .expect("insert");
            }
            let done = completed.fetch_add(chunk.len(), Ordering::Relaxed) + chunk.len();
            if done % 100_000 == 0 || done == total {
                print!(
                    "\r      -> Inserted {} / {} vectors ({:.1}%)...",
                    done,
                    total,
                    (done as f64 / total as f64) * 100.0
                );
            }
        });
    println!();
    println!("    ✓ Built HNSW index in {:.2?}", t_build.elapsed());

    // Save persistent snapshot
    let t_save = Instant::now();
    index.save_snapshot_v2(&snap_path).expect("save snapshot");
    println!(
        "    ✓ Saved persistent snapshot to {} in {:.2?}",
        snap_path.display(),
        t_save.elapsed()
    );

    // Return opened index
    HNSQRIndex::open_snapshot_v2(&snap_path, SnapshotOpenOptions::default()).expect("open snapshot")
}

fn evaluate_hnsw_corpus(
    tag: &str,
    corpus_path: &Path,
    query_path: &Path,
    oracle_snapshot_path: &Path,
    dim: usize,
    exact_p50_ms: f64,
    m_values: &[usize],
    ef_construction_values: &[usize],
    ef_search_values: &[usize],
) -> Vec<HnswMatrixRow> {
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("  P1 HNSW MATRIX EVALUATION: {tag} (Dim={dim}, Exact p50={exact_p50_ms:.2} ms)");
    println!("═══════════════════════════════════════════════════════════════════════════════");

    println!("  Loading corpus vectors from {}...", corpus_path.display());
    let t0 = Instant::now();
    let corpus = read_fvecs(corpus_path, None).expect("load corpus");
    println!(
        "  Loaded {} corpus vectors in {:.2?}",
        corpus.len(),
        t0.elapsed()
    );

    println!("  Loading query vectors from {}...", query_path.display());
    let queries = read_fvecs(query_path, Some(QUERIES_COUNT)).expect("load queries");
    println!(
        "  Loaded {} query vectors in {:.2?}",
        queries.len(),
        t0.elapsed()
    );

    // Compute exact ground truth top-10 for all queries using attached oracle snapshot
    println!(
        "  Precomputing 100% exact SIMD ground truth for {} queries from oracle...",
        queries.len()
    );
    let oracle = HNSQRIndex::open_snapshot_v2(oracle_snapshot_path, SnapshotOpenOptions::default())
        .expect("open oracle snapshot");

    let ground_truth: Vec<Vec<usize>> = queries
        .iter()
        .map(|q| {
            let res = oracle.search_indices_exact(q, K_NEIGHBORS, None).unwrap();
            res.into_iter().map(|(idx, _)| idx as usize).collect()
        })
        .collect();
    println!(
        "  ✓ Ground truth computed for {} queries.",
        ground_truth.len()
    );

    let mut matrix_results = Vec::new();

    for &m in m_values {
        let m0 = 2 * m; // Explicitly frozen conventional construction relationship M0 = 2*M
        for &ef_c in ef_construction_values {
            let index = get_or_build_hnsw_index(tag, &corpus, dim.div_ceil(2), m, m0, ef_c);

            for &ef_s in ef_search_values {
                index.set_ef_search(ef_s).expect("set ef_search");

                let mut recalls_10 = Vec::with_capacity(queries.len());
                let mut recalls_1 = Vec::with_capacity(queries.len());
                let mut latencies_ms = Vec::with_capacity(queries.len());

                let t_start_sweep = Instant::now();
                for (qi, q) in queries.iter().enumerate() {
                    let gt = &ground_truth[qi];
                    let gt_top1 = gt[0];
                    let gt_set: std::collections::HashSet<usize> = gt.iter().copied().collect();

                    let t_q = Instant::now();
                    let results = index
                        .search_indices_graph(q, K_NEIGHBORS, None)
                        .unwrap_or_default();
                    let dur_ms = t_q.elapsed().as_secs_f64() * 1000.0;
                    latencies_ms.push(dur_ms);

                    let hit_top1 = results.first().map_or(0.0, |(idx, _)| {
                        if *idx as usize == gt_top1 { 100.0 } else { 0.0 }
                    });
                    let matched_10 = results
                        .iter()
                        .filter(|(idx, _)| gt_set.contains(&(*idx as usize)))
                        .count();
                    let rec_10 = (matched_10 as f64 / K_NEIGHBORS as f64) * 100.0;

                    recalls_1.push(hit_top1);
                    recalls_10.push(rec_10);
                }
                let sweep_dur = t_start_sweep.elapsed();

                let mean_r10 = recalls_10.iter().sum::<f64>() / recalls_10.len() as f64;
                let min_r10 = recalls_10.iter().copied().fold(f64::MAX, f64::min);
                let p05_r10 = percentile_f64(recalls_10.clone(), 0.05);
                let mean_r1 = recalls_1.iter().sum::<f64>() / recalls_1.len() as f64;

                let hnsw_p50 = percentile_f64(latencies_ms.clone(), 0.50);
                let hnsw_p95 = percentile_f64(latencies_ms.clone(), 0.95);
                let hnsw_p99 = percentile_f64(latencies_ms.clone(), 0.99);
                let speedup = exact_p50_ms / hnsw_p50.max(1e-4);
                let qps = queries.len() as f64 / sweep_dur.as_secs_f64();

                let is_survival = mean_r10 >= 95.0;
                let is_production = mean_r10 >= 99.0 && speedup >= 2.0;

                println!(
                    "    M={:>2} efC={:>3} efS={:>3} │ R@10={:>5.1}% (min={:>4.0}%, p05={:>4.0}%) │ p50={:>6.2} ms (p99={:>6.2} ms) │ Speedup={:>5.2}x │ {}",
                    m,
                    ef_c,
                    ef_s,
                    mean_r10,
                    min_r10,
                    p05_r10,
                    hnsw_p50,
                    hnsw_p99,
                    speedup,
                    if is_production {
                        "🌟 PRODUCTION CANDIDATE"
                    } else if is_survival {
                        "✓ Survival"
                    } else {
                        "✗ Rejected (<95%)"
                    }
                );

                matrix_results.push(HnswMatrixRow {
                    corpus: tag.to_string(),
                    m,
                    m0,
                    ef_construction: ef_c,
                    ef_search: ef_s,
                    mean_recall_10: mean_r10,
                    min_recall_10: min_r10,
                    p05_recall_10: p05_r10,
                    mean_recall_1: mean_r1,
                    exact_p50_ms,
                    hnsw_p50_ms: hnsw_p50,
                    hnsw_p95_ms: hnsw_p95,
                    hnsw_p99_ms: hnsw_p99,
                    speedup,
                    qps,
                    avg_visits: ef_s * m,
                    is_survival_pass: is_survival,
                    is_production_candidate: is_production,
                });
            }
        }
    }

    matrix_results
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════════════╗");
    println!("║       HOLOSPHERE PERFORMANCE TRACK P1: RECONSTRUCTED HNSW MATRIX SWEEP      ║");
    println!("║                 (SIFT1M 128D & GloVe-100 100D, 500 Queries)                 ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════╝");

    // Load P0 Exact baselines
    let sift_exact_p50 = 178.97;
    let glove_exact_p50 = 224.86;

    let m_grid = [8, 16, 32, 48];
    let efc_grid = [100, 200, 400];
    let efs_grid = [16, 32, 64, 128, 256, 512];

    let sift_corpus_path = PathBuf::from("datasets/sift_1m/sift1m_base.fvecs");
    let sift_query_path = PathBuf::from("datasets/sift_1m/sift1m_query.fvecs");
    let sift_oracle =
        PathBuf::from("benchmark_databases/million_sift1m_strict_v6_pStrict_d64_n1000000.snapshot");

    let glove_corpus_path = PathBuf::from("datasets/glove_100/glove100_base.fvecs");
    let glove_query_path = PathBuf::from("datasets/glove_100/glove100_query.fvecs");
    let glove_oracle = PathBuf::from(
        "benchmark_databases/million_glove100_strict_v6_pStrict_d50_n1183514.snapshot",
    );

    let mut all_results = Vec::new();

    if sift_corpus_path.exists() && sift_query_path.exists() && sift_oracle.exists() {
        let sift_rows = evaluate_hnsw_corpus(
            "SIFT1M",
            &sift_corpus_path,
            &sift_query_path,
            &sift_oracle,
            128,
            sift_exact_p50,
            &m_grid,
            &efc_grid,
            &efs_grid,
        );
        all_results.extend(sift_rows);
    }

    if glove_corpus_path.exists() && glove_query_path.exists() && glove_oracle.exists() {
        let glove_rows = evaluate_hnsw_corpus(
            "GloVe-100",
            &glove_corpus_path,
            &glove_query_path,
            &glove_oracle,
            100,
            glove_exact_p50,
            &m_grid,
            &efc_grid,
            &efs_grid,
        );
        all_results.extend(glove_rows);
    }

    // Filter Pareto-frontier rows
    println!(
        "\n\n═══════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!("  🏆 STOP CONDITION SUMMARY: PARETO FRONTIER MATRIX");
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "  {:<10} │ {:>3} │ {:>3} │ {:>3} │ {:>10} │ {:>10} │ {:>10} │ {:>10} │ {:>8} │ {:>10} │ {:>8}",
        "Corpus",
        "M",
        "efC",
        "efS",
        "Recall@10",
        "Min R@10",
        "Exact p50",
        "HNSW p50",
        "Speedup",
        "p99",
        "Visits"
    );
    println!("  {}", "─".repeat(110));

    for corpus_tag in &["SIFT1M", "GloVe-100"] {
        let rows: Vec<&HnswMatrixRow> = all_results
            .iter()
            .filter(|r| r.corpus == *corpus_tag)
            .collect();
        // Compute Pareto frontier:
        let mut pareto: Vec<&HnswMatrixRow> = Vec::new();
        for candidate in &rows {
            let dominated = rows.iter().any(|other| {
                other.mean_recall_10 >= candidate.mean_recall_10
                    && other.hnsw_p50_ms <= candidate.hnsw_p50_ms
                    && (other.mean_recall_10 > candidate.mean_recall_10
                        || other.hnsw_p50_ms < candidate.hnsw_p50_ms)
            });
            if !dominated {
                pareto.push(candidate);
            }
        }
        pareto.sort_by(|a, b| a.mean_recall_10.total_cmp(&b.mean_recall_10));

        for r in pareto {
            println!(
                "  {:<10} │ {:>3} │ {:>3} │ {:>3} │ {:>9.1}% │ {:>9.1}% │ {:>8.2} ms │ {:>8.2} ms │ {:>7.2}x │ {:>8.2} ms │ {:>8}",
                r.corpus,
                r.m,
                r.ef_construction,
                r.ef_search,
                r.mean_recall_10,
                r.min_recall_10,
                r.exact_p50_ms,
                r.hnsw_p50_ms,
                r.speedup,
                r.hnsw_p99_ms,
                r.avg_visits
            );
        }
        println!("  {}", "─".repeat(110));
    }
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════\n"
    );
}
