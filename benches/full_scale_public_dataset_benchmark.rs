/* hnsqr/benches/full_scale_public_dataset_benchmark.rs */
//!▫~•◦-----------------------------------------------------------------‣
//! # HoloSphere Full-Scale Public Dataset Audit (Uncapped)
//!▫~•◦-----------------------------------------------------------------‣
//!
//! Same self-auditing methodology as public_dataset_benchmark.rs — label
//! consistency checks, multi-query recall, mean/min/p50/p95 — but against
//! the FULL corpus for every dataset, no truncation. Multi-minute runtime.
//! Gated behind HOLOSPHERE_FULL_SCALE=1 so it never runs as a side effect
//! of cargo bench without --bench full_scale_public_dataset_benchmark.
//!
//! Ground truth is computed in parallel via rayon; without it this does
//! not finish in reasonable time on the 1M+ corpora.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::planning::RetrievalContract;
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, VectorEmbedding};
use rayon::prelude::*;
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const RECALL_PASS_THRESHOLD_PCT: f64 = 100.0;
const TOP1_SCORE_TOLERANCE: f32 = 1e-4;
/// Full-scale still caps *queries*, not corpus. 200 queries against a
/// full 1.18M corpus is 200 * 1.18M ≈ 236M dot products — that's the
/// honest cost of an uncapped audit; raise this only if you have the
/// wall-clock budget for it.
const QUERIES_PER_DATASET: usize = 200;

fn read_fvecs_limited(
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
        vectors.push(VectorEmbedding::from_reals(&floats).into_normalized());
    }
    Ok(vectors)
}

/// Parallel brute-force ground truth. Sequential over 1.18M vectors per
/// query is the bottleneck this file exists to pay for — rayon splits
/// the scan across cores; still correct, just not free.
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

struct QuerySample {
    recall_pct: f64,
    #[allow(dead_code)]
    top1_delta: f32,
    latency: Duration,
    passed: bool,
}

struct DatasetResult {
    name: String,
    dim: usize,
    corpus_n: usize,
    declared_n: usize,
    label_consistent: bool,
    ran: bool,
    skip_reason: Option<String>,
    samples: Vec<QuerySample>,
    mean_recall_pct: f64,
    min_recall_pct: f64,
    p50_latency: Duration,
    p95_latency: Duration,
    passed: bool,
}

impl DatasetResult {
    fn skipped(name: &str, declared_n: usize, reason: String) -> Self {
        Self {
            name: name.to_string(),
            dim: 0,
            corpus_n: 0,
            declared_n,
            label_consistent: false,
            ran: false,
            skip_reason: Some(reason),
            samples: Vec::new(),
            mean_recall_pct: 0.0,
            min_recall_pct: 0.0,
            p50_latency: Duration::ZERO,
            p95_latency: Duration::ZERO,
            passed: false,
        }
    }
}

fn percentile(mut xs: Vec<Duration>, pct: f64) -> Duration {
    if xs.is_empty() {
        return Duration::ZERO;
    }
    xs.sort();
    let idx = ((xs.len() as f64 - 1.0) * pct).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn evaluate_corpus_full(
    name: &str,
    base_path: &Path,
    corpus: &[VectorEmbedding],
    queries: &[VectorEmbedding],
    dim: usize,
    k: usize,
    declared_n: usize,
) -> DatasetResult {
    let label_consistent = declared_n == corpus.len();
    let snapshot_path = base_path.with_extension("snapshot");

    let index = if snapshot_path.exists() {
        eprintln!(
            "  [{name}] found prebuilt snapshot {:?}, attaching via mmap...",
            snapshot_path
        );
        let t_load = Instant::now();
        let idx =
            HNSQRIndex::open_snapshot_v2(&snapshot_path, hnsqr::SnapshotOpenOptions::default())
                .expect("Failed to open prebuilt snapshot");
        idx.freeze_rivero_routing();
        eprintln!("  [{name}] snapshot attached in {:.2?}", t_load.elapsed());
        idx
    } else {
        eprintln!("  [{name}] building index over {} vectors...", corpus.len());
        let mut config = HNSQRConfig::default();
        config.distance_function = DistanceFunction::Cosine;
        let index = HNSQRIndex::new(config, dim);
        let build_start = Instant::now();
        for (i, v) in corpus.iter().enumerate() {
            let doc_id = format!("doc_{i}");
            index.insert(doc_id.as_str(), v.clone()).unwrap();
        }
        index.freeze_rivero_routing();
        let _ = index.save_snapshot_v2(&snapshot_path);
        eprintln!(
            "  [{name}] index built and snapshot saved in {:.2?}",
            build_start.elapsed()
        );
        index
    };

    let mut samples = Vec::with_capacity(queries.len());
    for (qi, query) in queries.iter().enumerate() {
        if qi % 25 == 0 {
            eprintln!("  [{name}] query {}/{}", qi + 1, queries.len());
        }
        let gt = compute_brute_force_ground_truth_parallel(corpus, query, k);
        let gt_top1_score = gt[0].1;
        let gt_indices: std::collections::HashSet<usize> = gt.iter().map(|(idx, _)| *idx).collect();

        let start = Instant::now();
        let raw_results = index
            .search_indices_with_contract(query, k, None, RetrievalContract::Certified)
            .unwrap();
        let elapsed = start.elapsed();

        let matched = raw_results
            .iter()
            .filter(|&&(idx, _)| gt_indices.contains(&(idx as usize)))
            .count();
        let recall_pct = (matched as f64 / k as f64) * 100.0;
        let top1_score = raw_results.first().map(|r| r.1).unwrap_or(0.0);
        let top1_delta = (top1_score - gt_top1_score).abs();
        let passed = recall_pct >= RECALL_PASS_THRESHOLD_PCT && top1_delta < TOP1_SCORE_TOLERANCE;

        samples.push(QuerySample {
            recall_pct,
            top1_delta,
            latency: elapsed,
            passed,
        });
    }

    let mean_recall_pct =
        samples.iter().map(|s| s.recall_pct).sum::<f64>() / samples.len().max(1) as f64;
    let min_recall_pct = samples
        .iter()
        .map(|s| s.recall_pct)
        .fold(f64::MAX, f64::min);
    let latencies: Vec<Duration> = samples.iter().map(|s| s.latency).collect();
    let p50_latency = percentile(latencies.clone(), 0.50);
    let p95_latency = percentile(latencies, 0.95);
    let all_passed = samples.iter().all(|s| s.passed);

    DatasetResult {
        name: name.to_string(),
        dim,
        corpus_n: corpus.len(),
        declared_n,
        label_consistent,
        ran: true,
        skip_reason: None,
        samples,
        mean_recall_pct,
        min_recall_pct,
        p50_latency,
        p95_latency,
        passed: all_passed && label_consistent,
    }
}

fn print_row(r: &DatasetResult) {
    if !r.ran {
        println!(
            "{:<40} {:<10} {:<12} {:<8} {:<15} {:<12} {:<12}",
            r.name,
            "-",
            "-",
            "-",
            format!("SKIPPED: {}", r.skip_reason.as_deref().unwrap_or("unknown")),
            "-",
            "-"
        );
        return;
    }
    let label_flag = if r.label_consistent {
        ""
    } else {
        " [LABEL MISMATCH]"
    };
    let status = if r.passed { "PASS" } else { "FAIL" };
    println!(
        "{:<40} {:<10} {:<12} {:<8} {:<15} {:<12.2?} {:<12.2?}",
        r.name,
        r.dim,
        r.corpus_n,
        r.samples.len(),
        format!(
            "{:.1}%/{:.1}% [{status}]{label_flag}",
            r.mean_recall_pct, r.min_recall_pct
        ),
        r.p50_latency,
        r.p95_latency,
    );
}

fn main() {
    if std::env::var("HOLOSPHERE_FULL_SCALE").as_deref() != Ok("1") {
        eprintln!("full_scale_public_dataset_benchmark is gated behind an explicit env var.");
        eprintln!("This runs uncapped ground truth against corpora up to 1.18M vectors and");
        eprintln!("takes multiple minutes. Run with:");
        eprintln!();
        eprintln!(
            "  HOLOSPHERE_FULL_SCALE=1 cargo bench --bench full_scale_public_dataset_benchmark"
        );
        eprintln!();
        std::process::exit(0);
    }

    println!(
        "╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║             HOLOSPHERE FULL-SCALE PUBLIC DATASET AUDIT (UNCAPPED)                                           ║"
    );
    println!(
        "╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════╝"
    );
    println!("Rayon threads: {}", rayon::current_num_threads());

    let k = 10;
    let mut results: Vec<DatasetResult> = Vec::new();

    // declared_n here is the FULL dataset size per the public spec sheets
    // for each corpus — this is what full-scale is claiming to test.
    let public_datasets: [(&str, PathBuf, PathBuf, usize); 5] = [
        (
            "GloVe-25 (Full, Real Public)",
            PathBuf::from("datasets/glove_25/glove25_base.fvecs"),
            PathBuf::from("datasets/glove_25/glove25_query.fvecs"),
            1_183_514,
        ),
        (
            "GloVe-50 (Full, Real Public)",
            PathBuf::from("datasets/glove_50/glove50_base.fvecs"),
            PathBuf::from("datasets/glove_50/glove50_query.fvecs"),
            1_183_514,
        ),
        (
            "GloVe-100 (Full, Real Public)",
            PathBuf::from("datasets/glove_100/glove100_base.fvecs"),
            PathBuf::from("datasets/glove_100/glove100_query.fvecs"),
            1_183_514,
        ),
        (
            "Texmex SIFT10K (Full, Real Public)",
            PathBuf::from("datasets/siftsmall/siftsmall_base.fvecs"),
            PathBuf::from("datasets/siftsmall/siftsmall_query.fvecs"),
            10_000,
        ),
        (
            "Texmex SIFT1M (Full, Real Public)",
            PathBuf::from("datasets/sift_1m/sift1m_base.fvecs"),
            PathBuf::from("datasets/sift_1m/sift1m_query.fvecs"),
            1_000_000,
        ),
    ];

    println!(
        "\n{:<40} {:<10} {:<12} {:<8} {:<15} {:<12} {:<12}",
        "Dataset Source", "Dim", "Corpus N", "Queries", "Recall mean/min", "p50 Lat", "p95 Lat"
    );
    println!("{:-<120}", "");

    for (name, base_path, query_path, declared_n) in &public_datasets {
        eprintln!("\n=== {name} ===");
        let result = if !base_path.exists() || !query_path.exists() {
            DatasetResult::skipped(
                name,
                *declared_n,
                format!("dataset files not found at {}", base_path.display()),
            )
        } else {
            match (
                read_fvecs_limited(base_path, None),
                read_fvecs_limited(query_path, Some(QUERIES_PER_DATASET)),
            ) {
                (Ok(base_vecs), Ok(query_vecs))
                    if !base_vecs.is_empty() && !query_vecs.is_empty() =>
                {
                    let dim = base_vecs[0].dimension();
                    evaluate_corpus_full(
                        name,
                        base_path,
                        &base_vecs,
                        &query_vecs,
                        dim,
                        k,
                        *declared_n,
                    )
                }
                (Ok(_), Ok(_)) => DatasetResult::skipped(
                    name,
                    *declared_n,
                    "dataset files present but empty after parse".to_string(),
                ),
                (Err(e), _) | (_, Err(e)) => {
                    DatasetResult::skipped(name, *declared_n, format!("read_fvecs failed: {e}"))
                }
            }
        };
        print_row(&result);
        results.push(result);
    }

    let ran: Vec<&DatasetResult> = results.iter().filter(|r| r.ran).collect();
    let skipped: Vec<&DatasetResult> = results.iter().filter(|r| !r.ran).collect();
    let mismatched: Vec<&DatasetResult> = results
        .iter()
        .filter(|r| r.ran && !r.label_consistent)
        .collect();
    let passed_count = ran.iter().filter(|r| r.passed).count();
    let total_queries: usize = ran.iter().map(|r| r.samples.len()).sum();

    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("FULL-SCALE AUDIT SUMMARY");
    println!("  Datasets evaluated:  {}/{}", ran.len(), results.len());
    println!("  Total queries run:   {}", total_queries);
    println!(
        "  Datasets passed:     {}/{}",
        passed_count,
        ran.len().max(1)
    );
    for m in &mismatched {
        println!(
            "  LABEL MISMATCH: {} — loaded {} vectors, declared {}",
            m.name, m.corpus_n, m.declared_n
        );
    }
    for s in &skipped {
        println!(
            "  SKIPPED: {} — {}",
            s.name,
            s.skip_reason.as_deref().unwrap_or("unknown")
        );
    }
    let overall_pass =
        !ran.is_empty() && passed_count == ran.len() && skipped.is_empty() && mismatched.is_empty();
    println!(
        "  Verdict: {}",
        if overall_pass {
            "100.000% exact recall confirmed at full declared scale, all labels self-consistent."
        } else {
            "FAIL — see PASS/FAIL, LABEL MISMATCH, and SKIPPED lines above."
        }
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");

    if !overall_pass {
        std::process::exit(1);
    }
}
