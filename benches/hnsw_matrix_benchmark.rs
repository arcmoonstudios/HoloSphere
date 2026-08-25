/* holosphere/benches/hnsw_matrix_benchmark.rs */
//!▫~•◦-----------------------------------------------------------------‣
//! # Performance Track P1: Ground-Up HNSW Parameter Matrix & Admission
//!▫~•◦-----------------------------------------------------------------‣
//!
//! Sweeps M in [8, 16, 32, 48], ef_construction in [100, 200, 400],
//! and ef_search in [16, 32, 64, 128, 256, 512] on real million-scale corpora.
//!
//! Protocol:
//!   1. Stage 1: Full matrix sweep on 20% Tuning queries (100 queries) to extract Pareto frontier.
//!   2. Stage 2: Rerun Pareto frontier on 80% Held-out Admission queries (400 queries).
//!   3. Stage 3: Rebuild production finalists under multiple deterministic construction seeds.
//!
//! Gates:
//!   1. Survival: Recall@10 >= 95%
//!   2. Production Candidate: Recall@10 >= 99% AND Speedup >= 2.0x vs P0 Exact p50.

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use hnsqr::storage::snapshot::SnapshotOpenOptions;
use hnsqr::vector::folding::ComplexWeaver;
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, SearchPlan, VectorEmbedding};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

const TOTAL_QUERIES_COUNT: usize = 500;
const TUNING_QUERIES_COUNT: usize = 100;
const ADMISSION_QUERIES_COUNT: usize = 400;
const K_NEIGHBORS: usize = 10;
const DEFAULT_CONSTRUCTION_SEED: u64 = 42;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswEvaluationRow {
    pub corpus: String,
    pub stage: String, // "tuning" or "held_out_admission"
    pub construction_seed: u64,
    pub m: usize,
    pub m0: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub queries_count: usize,
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
    pub avg_visited_nodes: usize,
    pub avg_distance_evaluations: usize,
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
    seed: u64,
) -> HNSQRIndex {
    let base_dir = PathBuf::from("benchmark_databases");
    let _ = std::fs::create_dir_all(&base_dir);
    let snap_path = base_dir.join(format!(
        "hnsw_p1_{tag}_s{seed}_m{m}_efc{ef_construction}_n{}.snapshot",
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
        "    ⚙️ Building fresh HNSW index (Seed={seed}, M={m}, M0={m0}, efC={ef_construction}, N={})...",
        corpus_vecs.len()
    );
    let t_build = Instant::now();

    let mut config = HNSQRConfig::default();
    config.distance_function = DistanceFunction::Cosine;
    config.search_plan = SearchPlan::HnswClassical;
    config.rivero_enabled = false;
    config.construction_seed = Some(seed);
    config.max_elements = corpus_vecs.len() + 10_000;
    config.m = m;
    config.m0 = m0;
    config.ef_construction = ef_construction;
    config.ef_search = 128;

    let index = HNSQRIndex::new(config, dim);
    let total = corpus_vecs.len();

    for (i, v) in corpus_vecs.iter().enumerate() {
        index
            .insert(format!("doc_{i}"), v.clone())
            .expect("insert");
        if (i + 1) % 100_000 == 0 || i + 1 == total {
            print!(
                "\r      -> Inserted {} / {} vectors ({:.1}%)...",
                i + 1,
                total,
                ((i + 1) as f64 / total as f64) * 100.0
            );
            let _ = std::io::stdout().flush();
        }
    }
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

fn evaluate_hnsw_subset(
    corpus_tag: &str,
    stage_tag: &str,
    index: &HNSQRIndex,
    queries: &[VectorEmbedding],
    ground_truth: &[Vec<usize>],
    m: usize,
    m0: usize,
    ef_c: usize,
    ef_s: usize,
    seed: u64,
    exact_p50_ms: f64,
) -> HnswEvaluationRow {
    index.set_ef_search(ef_s).expect("set ef_search");

    let mut recalls_10 = Vec::with_capacity(queries.len());
    let mut recalls_1 = Vec::with_capacity(queries.len());
    let mut latencies_ms = Vec::with_capacity(queries.len());
    let mut total_visits = 0usize;
    let mut total_dists = 0usize;

    let t_start_sweep = Instant::now();
    for (qi, q) in queries.iter().enumerate() {
        let gt = &ground_truth[qi];
        let gt_top1 = gt[0];
        let gt_set: std::collections::HashSet<usize> = gt.iter().copied().collect();

        let t_q = Instant::now();
        let (results, diagnostics) = index
            .search_indices_hnsw_classical_diagnostics(q, K_NEIGHBORS, None)
            .unwrap_or_default();
        let dur_ms = t_q.elapsed().as_secs_f64() * 1000.0;
        latencies_ms.push(dur_ms);
        total_visits += diagnostics.visited_nodes as usize;
        total_dists += diagnostics.distance_evaluations as usize;

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
    let avg_visits = total_visits / queries.len().max(1);
    let avg_dists = total_dists / queries.len().max(1);

    let is_survival = mean_r10 >= 95.0;
    let is_production = mean_r10 >= 99.0 && speedup >= 2.0;

    HnswEvaluationRow {
        corpus: corpus_tag.to_string(),
        stage: stage_tag.to_string(),
        construction_seed: seed,
        m,
        m0,
        ef_construction: ef_c,
        ef_search: ef_s,
        queries_count: queries.len(),
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
        avg_visited_nodes: avg_visits,
        avg_distance_evaluations: avg_dists,
        is_survival_pass: is_survival,
        is_production_candidate: is_production,
    }
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════════════╗");
    println!("║       HOLOSPHERE PERFORMANCE TRACK P1: CLASSICAL HNSW PARETO CHARACTERIZATION║");
    println!("║                 (SIFT1M 128D & GloVe-100 100D, Two-Stage Admission)          ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════╝");

    // 1. Load P0 Exact Baseline values from performance-baseline-v1
    let sift_baseline_path = PathBuf::from("performance-baseline-v1/sift1m_exact.json");
    let glove_baseline_path = PathBuf::from("performance-baseline-v1/glove100_exact.json");

    let sift_exact_p50 = if sift_baseline_path.exists() {
        let file = File::open(&sift_baseline_path).expect("open sift1m baseline");
        let data: serde_json::Value = serde_json::from_reader(file).expect("parse sift1m baseline");
        data["aggregate_held_out_admission"]["p50_ms"].as_f64().unwrap_or(33.74)
    } else {
        33.74
    };

    let glove_exact_p50 = if glove_baseline_path.exists() {
        let file = File::open(&glove_baseline_path).expect("open glove100 baseline");
        let data: serde_json::Value = serde_json::from_reader(file).expect("parse glove100 baseline");
        data["aggregate_held_out_admission"]["p50_ms"].as_f64().unwrap_or(34.85)
    } else {
        34.85
    };

    println!("  P0 Frozen Exact Held-out p50: SIFT1M = {:.2} ms | GloVe-100 = {:.2} ms\n", sift_exact_p50, glove_exact_p50);

    let m_grid = [8, 16, 32, 48];
    let efc_grid = [100, 200, 400];
    let efs_grid = [16, 32, 64, 128, 256, 512];

    let datasets = [
        (
            "SIFT1M",
            PathBuf::from("datasets/sift_1m/sift1m_base.fvecs"),
            PathBuf::from("datasets/sift_1m/sift1m_query.fvecs"),
            PathBuf::from("benchmark_databases/million_sift1m_strict_v6_pStrict_d64_n1000000.snapshot"),
            128usize,
            sift_exact_p50,
        ),
        (
            "GloVe-100",
            PathBuf::from("datasets/glove_100/glove100_base.fvecs"),
            PathBuf::from("datasets/glove_100/glove100_query.fvecs"),
            PathBuf::from("benchmark_databases/million_glove100_strict_v6_pStrict_d50_n1183514.snapshot"),
            100usize,
            glove_exact_p50,
        ),
    ];

    let mut all_tuning_results = Vec::new();
    let mut all_admission_results = Vec::new();
    let mut all_seed_variance_results = Vec::new();

    for (corpus_tag, corpus_path, query_path, oracle_path, dim, exact_p50) in &datasets {
        println!("\n═══════════════════════════════════════════════════════════════════════════════");
        println!("  STAGE 1: FULL TUNING MATRIX (100 Queries, 20%) FOR {corpus_tag}");
        println!("═══════════════════════════════════════════════════════════════════════════════");

        let corpus = read_fvecs(corpus_path, None).expect("load corpus");
        let all_queries = read_fvecs(query_path, Some(TOTAL_QUERIES_COUNT)).expect("load queries");
        let tuning_queries = &all_queries[..TUNING_QUERIES_COUNT];
        let admission_queries =
            &all_queries[TUNING_QUERIES_COUNT..TUNING_QUERIES_COUNT + ADMISSION_QUERIES_COUNT];

        let oracle = HNSQRIndex::open_snapshot_v2(oracle_path, SnapshotOpenOptions::default())

            .expect("open oracle snapshot");

        println!("  Precomputing exact ground truth for 100 tuning queries...");
        let tuning_gt: Vec<Vec<usize>> = tuning_queries
            .iter()
            .map(|q| {
                let res = oracle.search_indices_exact(q, K_NEIGHBORS, None).unwrap();
                res.into_iter().map(|(idx, _)| idx as usize).collect()
            })
            .collect();

        println!("  Precomputing exact ground truth for 400 held-out admission queries...");
        let admission_gt: Vec<Vec<usize>> = admission_queries
            .iter()
            .map(|q| {
                let res = oracle.search_indices_exact(q, K_NEIGHBORS, None).unwrap();
                res.into_iter().map(|(idx, _)| idx as usize).collect()
            })
            .collect();

        // Stage 1 Sweep
        let mut corpus_tuning_rows = Vec::new();
        for &m in &m_grid {
            let m0 = 2 * m;
            for &ef_c in &efc_grid {
                let index = get_or_build_hnsw_index(corpus_tag, &corpus, dim.div_ceil(2), m, m0, ef_c, DEFAULT_CONSTRUCTION_SEED);
                for &ef_s in &efs_grid {
                    let row = evaluate_hnsw_subset(
                        corpus_tag,
                        "tuning",
                        &index,
                        tuning_queries,
                        &tuning_gt,
                        m,
                        m0,
                        ef_c,
                        ef_s,
                        DEFAULT_CONSTRUCTION_SEED,
                        *exact_p50,
                    );
                    println!(
                        "    [Tuning] M={:>2} efC={:>3} efS={:>3} │ R@10={:>5.1}% │ p50={:>6.2} ms │ Speedup={:>5.2}x │ Visits={:>4}",
                        row.m, row.ef_construction, row.ef_search, row.mean_recall_10, row.hnsw_p50_ms, row.speedup, row.avg_visited_nodes
                    );
                    corpus_tuning_rows.push(row);
                }
            }
        }

        // Identify Pareto frontier on Tuning set
        let mut tuning_pareto = Vec::new();
        for candidate in &corpus_tuning_rows {
            let dominated = corpus_tuning_rows.iter().any(|other| {
                other.mean_recall_10 >= candidate.mean_recall_10
                    && other.hnsw_p50_ms <= candidate.hnsw_p50_ms
                    && (other.mean_recall_10 > candidate.mean_recall_10
                        || other.hnsw_p50_ms < candidate.hnsw_p50_ms)
            });
            if !dominated {
                tuning_pareto.push(candidate.clone());
            }
        }
        tuning_pareto.sort_by(|a, b| a.mean_recall_10.total_cmp(&b.mean_recall_10));
        all_tuning_results.extend(corpus_tuning_rows);

        println!("\n═══════════════════════════════════════════════════════════════════════════════");
        println!("  STAGE 2: HELD-OUT ADMISSION (400 Queries, 80%) ON TUNING PARETO FRONTIER FOR {corpus_tag}");
        println!("═══════════════════════════════════════════════════════════════════════════════");

        let mut corpus_admission_rows = Vec::new();
        for frontier_point in &tuning_pareto {
            let index = get_or_build_hnsw_index(
                corpus_tag,
                &corpus,
                dim.div_ceil(2),
                frontier_point.m,
                frontier_point.m0,
                frontier_point.ef_construction,
                DEFAULT_CONSTRUCTION_SEED,
            );
            let adm_row = evaluate_hnsw_subset(
                corpus_tag,
                "held_out_admission",
                &index,
                admission_queries,
                &admission_gt,
                frontier_point.m,
                frontier_point.m0,
                frontier_point.ef_construction,
                frontier_point.ef_search,
                DEFAULT_CONSTRUCTION_SEED,
                *exact_p50,
            );
            println!(
                "  🎯 [Held-Out] M={:>2} efC={:>3} efS={:>3} │ R@10={:>5.1}% (min={:>4.0}%, p05={:>4.0}%) │ p50={:>6.2} ms (p99={:>6.2} ms) │ Speedup={:>5.2}x │ {}",
                adm_row.m, adm_row.ef_construction, adm_row.ef_search,
                adm_row.mean_recall_10, adm_row.min_recall_10, adm_row.p05_recall_10,
                adm_row.hnsw_p50_ms, adm_row.hnsw_p99_ms, adm_row.speedup,
                if adm_row.is_production_candidate {
                    "🌟 PRODUCTION CANDIDATE"
                } else if adm_row.is_survival_pass {
                    "✓ Survival"
                } else {
                    "✗ Rejected (<95%)"
                }
            );
            corpus_admission_rows.push(adm_row);
        }

        // Check if any frontier points passed Production Candidate gate
        let production_finalists: Vec<HnswEvaluationRow> = corpus_admission_rows
            .iter()
            .filter(|r| r.is_production_candidate)
            .cloned()
            .collect();

        if !production_finalists.is_empty() {
            println!("\n  STAGE 3: SEED INVARIANCE CHECK FOR PRODUCTION CANDIDATES ON {corpus_tag}");
            let additional_seeds = [1337u64, 2026u64, 9999u64];
            for finalist in &production_finalists {
                for &seed in &additional_seeds {
                    let index = get_or_build_hnsw_index(
                        corpus_tag,
                        &corpus,
                        dim.div_ceil(2),
                        finalist.m,
                        finalist.m0,
                        finalist.ef_construction,
                        seed,
                    );
                    let seed_row = evaluate_hnsw_subset(
                        corpus_tag,
                        "seed_invariance",
                        &index,
                        admission_queries,
                        &admission_gt,
                        finalist.m,
                        finalist.m0,
                        finalist.ef_construction,
                        finalist.ef_search,
                        seed,
                        *exact_p50,
                    );
                    println!(
                        "    🎲 [Seed={seed}] M={:>2} efC={:>3} efS={:>3} │ R@10={:>5.1}% (min={:>4.0}%) │ p50={:>6.2} ms │ Speedup={:>5.2}x",
                        seed_row.m, seed_row.ef_construction, seed_row.ef_search,
                        seed_row.mean_recall_10, seed_row.min_recall_10,
                        seed_row.hnsw_p50_ms, seed_row.speedup
                    );
                    all_seed_variance_results.push(seed_row);
                }
            }
        } else {
            println!("  ℹ️ No production candidates reached >=99% Recall@10 AND >=2.0x speedup on held-out {corpus_tag}.");
        }

        all_admission_results.extend(corpus_admission_rows);
    }

    // Save full machine readable JSON
    let out_dir = PathBuf::from("performance-baseline-v1");
    let p1_json_path = out_dir.join("hnsw_p1_matrix.json");
    let serialized = serde_json::to_string_pretty(&serde_json::json!({
        "semantic_kernel_version": 1,
        "benchmark_stage": "P1-Classical-HNSW-Pareto",
        "tuning_matrix": all_tuning_results,
        "held_out_admission_pareto": all_admission_results,
        "seed_variance_trials": all_seed_variance_results,
    })).expect("serialize p1 json");
    let mut file = File::create(&p1_json_path).expect("create p1 json");
    file.write_all(serialized.as_bytes()).expect("write p1 json");

    // Output Final Brutally Small Summary Table
    println!("\n\n═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!("  🏆 P1 HELD-OUT PARETO-FRONTIER ADMISSION SUMMARY TABLE");
    println!("═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════");
    println!(
        "  {:<10} │ {:>2} │ {:>2} │ {:>3} │ {:>3} │ {:>6} │ {:>9} │ {:>8} │ {:>9} │ {:>8} │ {:>7} │ {:>8} │ {:>6}",
        "Dataset", "M", "M0", "efC", "efS", "R@10", "p05 R@10", "Min R@10", "Exact p50", "HNSW p50", "Speedup", "HNSW p99", "Visits"
    );
    println!("  {}", "─".repeat(123));

    for r in &all_admission_results {
        println!(
            "  {:<10} │ {:>2} │ {:>2} │ {:>3} │ {:>3} │ {:>5.1}% │ {:>8.1}% │ {:>7.1}% │ {:>7.2} ms │ {:>6.2} ms │ {:>6.2}x │ {:>6.2} ms │ {:>6}",
            r.corpus, r.m, r.m0, r.ef_construction, r.ef_search,
            r.mean_recall_10, r.p05_recall_10, r.min_recall_10,
            r.exact_p50_ms, r.hnsw_p50_ms, r.speedup, r.hnsw_p99_ms, r.avg_visited_nodes
        );
    }
    println!("═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n");
}
