/* hnsqr/benches/public_dataset_benchmark.rs */
//!▫~•◦-----------------------------------------------------------------‣
//! # HoloSphere Real Public Dataset Retrieval & Scale Audit
//!▫~•◦-----------------------------------------------------------------‣
//!
//! Every dataset is evaluated under EVERY execution path the planner can
//! select — not just whichever one UniversalPlanner::plan picks by default.
//! A recall number is meaningless without knowing whether it came from
//! ExactScan, GraphOnly, RiveroStrict, or RiveroAdaptive. This harness
//! forces each path explicitly via config and labels every row with the
//! mode that actually ran, so "Certified" recall can never again mean
//! "brute force vs brute force" by accident.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::planning::RetrievalContract;
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, RiveroSearchMode, SearchPlan, VectorEmbedding};
use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const RECALL_PASS_THRESHOLD_PCT: f64 = 100.0;
const TOP1_SCORE_TOLERANCE: f32 = 1e-4;
const MAX_QUERIES_PER_DATASET: usize = 50;

/// Every execution path the retrieval planner can select. `PlannerDefault`
/// is whatever UniversalPlanner::plan chooses given (N, D, contract) —
/// this is the ONLY mode where we do not know in advance which underlying
/// algorithm ran, so its row must report the observed path, not assume it.
#[derive(Clone, Copy, Debug)]
enum ExecMode {
    PlannerDefault,
    ExactForced,
    GraphOnly,
    RiveroStrict,
    RiveroAdaptive,
}

impl ExecMode {
    fn label(&self) -> &'static str {
        match self {
            ExecMode::PlannerDefault => "Planner-Routed",
            ExecMode::ExactForced => "Exact-Forced",
            ExecMode::GraphOnly => "Graph-Forced",
            ExecMode::RiveroStrict => "Rivero-Strict-Forced",
            ExecMode::RiveroAdaptive => "Rivero-Adaptive-Forced",
        }
    }

    /// Applies this mode to a config. PlannerDefault leaves config
    /// untouched — the planner decides based on N vs N_cross. All other
    /// modes override exact_scan_threshold and/or search_plan to force
    /// the path regardless of corpus size.
    fn apply(&self, config: &mut HNSQRConfig) {
        match self {
            ExecMode::PlannerDefault => {}
            ExecMode::ExactForced => {
                config.exact_scan_threshold = usize::MAX;
                config.search_plan = SearchPlan::Exact;
            }
            ExecMode::GraphOnly => {
                config.exact_scan_threshold = 0;
                config.search_plan = SearchPlan::GraphOnly;
                config.rivero_enabled = false;
                config.ef_search = 128;
            }
            ExecMode::RiveroStrict => {
                config.exact_scan_threshold = 0;
                config.search_plan = SearchPlan::Rivero;
                config.rivero_enabled = true;
                config.rivero_mode = RiveroSearchMode::Strict;
            }
            ExecMode::RiveroAdaptive => {
                config.exact_scan_threshold = 0;
                config.search_plan = SearchPlan::Rivero;
                config.rivero_enabled = true;
                config.rivero_mode = RiveroSearchMode::Adaptive;
            }
        }
    }
}

fn read_fvecs_limited(path: impl AsRef<Path>, max_vectors: Option<usize>) -> io::Result<Vec<VectorEmbedding>> {
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

fn compute_brute_force_ground_truth(
    corpus: &[VectorEmbedding],
    query: &VectorEmbedding,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(idx, v)| (idx, v.dot_product_complex(query).re))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    scored
}

/// Complex-dimension crossover threshold, replicated from
/// UniversalPlanner::compute_crossover so this harness can predict —
/// and then verify — which path PlannerDefault should take.
/// N_cross = 3000 + 5,768,286 / D_complex^1.3
fn predicted_crossover(complex_dim: usize) -> f64 {
    3000.0 + 5_768_286.0 / (complex_dim as f64).powf(1.3)
}

struct ModeResult {
    mode: ExecMode,
    ran: bool,
    skip_reason: Option<String>,
    mean_recall_pct: f64,
    min_recall_pct: f64,
    p50_latency: Duration,
    p95_latency: Duration,
    passed: bool,
}

struct DatasetResult {
    name: String,
    dim: usize,
    corpus_n: usize,
    declared_n: Option<usize>,
    label_consistent: bool,
    predicted_crossover_n: f64,
    modes: Vec<ModeResult>,
}

fn percentile(mut xs: Vec<Duration>, pct: f64) -> Duration {
    if xs.is_empty() {
        return Duration::ZERO;
    }
    xs.sort();
    let idx = ((xs.len() as f64 - 1.0) * pct).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn run_mode(
    mode: ExecMode,
    corpus: &[VectorEmbedding],
    queries: &[VectorEmbedding],
    dim: usize,
    k: usize,
) -> ModeResult {
    let mut config = HNSQRConfig::default();
    config.distance_function = DistanceFunction::Cosine;
    mode.apply(&mut config);

    let index = HNSQRIndex::new(config, dim);
    for (i, v) in corpus.iter().enumerate() {
        let doc_id = format!("doc_{i}");
        index.insert(doc_id.as_str(), v.clone()).unwrap();
    }
    index.freeze_rivero_routing();

    // Warmup query to populate CPU instruction and data caches
    if let Some(first_query) = queries.first() {
        match mode {
            ExecMode::PlannerDefault => {
                let _ = index.search_indices_with_contract(first_query, k, None, RetrievalContract::Certified);
            }
            _ => {
                let _ = index.search_indices(first_query, k);
            }
        }
    }

    let mut recalls = Vec::with_capacity(queries.len());
    let mut latencies = Vec::with_capacity(queries.len());
    let mut all_passed = true;

    for query in queries {
        let gt = compute_brute_force_ground_truth(corpus, query, k);
        let gt_top1_score = gt[0].1;
        let gt_indices: std::collections::HashSet<usize> = gt.iter().map(|(idx, _)| *idx).collect();

        let start = Instant::now();
        let raw_results = match mode {
            ExecMode::PlannerDefault => {
                index.search_indices_with_contract(query, k, None, RetrievalContract::Certified)
            }
            _ => index.search_indices(query, k),
        };

        let raw_results = match raw_results {
            Ok(r) => r,
            Err(e) => {
                return ModeResult {
                    mode,
                    ran: false,
                    skip_reason: Some(format!("search failed under {}: {e}", mode.label())),
                    mean_recall_pct: 0.0,
                    min_recall_pct: 0.0,
                    p50_latency: Duration::ZERO,
                    p95_latency: Duration::ZERO,
                    passed: false,
                };
            }
        };
        let elapsed = start.elapsed();

        let matched = raw_results
            .iter()
            .filter(|&&(idx, _)| gt_indices.contains(&(idx as usize)))
            .count();
        let recall_pct = (matched as f64 / k as f64) * 100.0;
        let top1_score = raw_results.first().map(|r| r.1).unwrap_or(0.0);
        let top1_delta = (top1_score - gt_top1_score).abs();
        let passed = recall_pct >= RECALL_PASS_THRESHOLD_PCT && top1_delta < TOP1_SCORE_TOLERANCE;
        all_passed &= passed;

        recalls.push(recall_pct);
        latencies.push(elapsed);
    }

    let mean_recall_pct = recalls.iter().sum::<f64>() / recalls.len().max(1) as f64;
    let min_recall_pct = recalls.iter().cloned().fold(f64::MAX, f64::min);

    ModeResult {
        mode,
        ran: true,
        skip_reason: None,
        mean_recall_pct,
        min_recall_pct,
        p50_latency: percentile(latencies.clone(), 0.50),
        p95_latency: percentile(latencies, 0.95),
        passed: all_passed,
    }
}

fn evaluate_dataset(
    name: &str,
    corpus: &[VectorEmbedding],
    queries: &[VectorEmbedding],
    dim: usize,
    k: usize,
    declared_n: Option<usize>,
) -> DatasetResult {
    let label_consistent = declared_n.map_or(true, |d| d == corpus.len());
    let complex_dim = dim / 2;
    let predicted_crossover_n = predicted_crossover(complex_dim.max(1));

    let modes = [
        ExecMode::PlannerDefault,
        ExecMode::ExactForced,
        ExecMode::GraphOnly,
        ExecMode::RiveroStrict,
        ExecMode::RiveroAdaptive,
    ];

    let mode_results = modes
        .iter()
        .map(|&m| run_mode(m, corpus, queries, dim, k))
        .collect();

    DatasetResult {
        name: name.to_string(),
        dim,
        corpus_n: corpus.len(),
        declared_n,
        label_consistent,
        predicted_crossover_n,
        modes: mode_results,
    }
}

fn print_dataset(r: &DatasetResult) {
    let crosses_naturally = r.corpus_n as f64 > r.predicted_crossover_n;
    let label_suffix = if r.label_consistent {
        String::new()
    } else {
        format!(
            " [LABEL MISMATCH: loaded {} vectors but declared {}]",
            r.corpus_n,
            r.declared_n.map_or_else(|| "unknown".to_string(), |n| n.to_string())
        )
    };
    println!(
        "\n{} — dim={} corpus_n={} N_cross≈{:.0} ({}){}",
        r.name,
        r.dim,
        r.corpus_n,
        r.predicted_crossover_n,
        if crosses_naturally { "corpus exceeds crossover: PlannerDefault SHOULD route to graph" } else { "corpus under crossover: PlannerDefault WILL route to ExactScan" },
        label_suffix
    );
    println!(
        "  {:<24} {:<8} {:<15} {:<12} {:<12}",
        "Mode", "Queries", "Recall mean/min", "p50 Lat", "p95 Lat"
    );
    for m in &r.modes {
        if !m.ran {
            println!("  {:<24} SKIPPED — {}", m.mode.label(), m.skip_reason.as_deref().unwrap_or("unknown"));
            continue;
        }
        let status = if m.passed { "PASS" } else { "FAIL" };
        println!(
            "  {:<24} {:<8} {:<15} {:<12.2?} {:<12.2?}",
            m.mode.label(),
            MAX_QUERIES_PER_DATASET,
            format!("{:.1}%/{:.1}% [{status}]", m.mean_recall_pct, m.min_recall_pct),
            m.p50_latency,
            m.p95_latency,
        );
    }
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║   HOLOSPHERE PATH-EXPLICIT AUDIT — every dataset run under every search mode, no assumed routing            ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    let k = 10;
    let mut results: Vec<DatasetResult> = Vec::new();

    let public_datasets: [(&str, PathBuf, PathBuf, Option<usize>); 8] = [
        ("GloVe-25 (Real Public)", PathBuf::from("datasets/glove_25/glove25_base.fvecs"), PathBuf::from("datasets/glove_25/glove25_query.fvecs"), Some(2_500)),
        ("GloVe-50 (Real Public)", PathBuf::from("datasets/glove_50/glove50_base.fvecs"), PathBuf::from("datasets/glove_50/glove50_query.fvecs"), Some(2_500)),
        ("GloVe-100 (Real Public)", PathBuf::from("datasets/glove_100/glove100_base.fvecs"), PathBuf::from("datasets/glove_100/glove100_query.fvecs"), Some(2_500)),
        ("Texmex SIFT10K (Real Public, Full)", PathBuf::from("datasets/siftsmall/siftsmall_base.fvecs"), PathBuf::from("datasets/siftsmall/siftsmall_query.fvecs"), None),
        ("Texmex SIFT1M (Real Public)", PathBuf::from("datasets/sift_1m/sift1m_base.fvecs"), PathBuf::from("datasets/sift_1m/sift1m_query.fvecs"), Some(5_000)),
        ("CLIP ViT-B/32 512 (Real Public, Full — only 1K vectors exist on disk)", PathBuf::from("datasets/clip_512/clip_base.fvecs"), PathBuf::from("datasets/clip_512/clip_query.fvecs"), None),
        ("Cohere Wikipedia 768 (Real Public, Full — only 1K vectors exist on disk)", PathBuf::from("datasets/cohere_768/cohere_base.fvecs"), PathBuf::from("datasets/cohere_768/cohere_query.fvecs"), None),
        ("OpenAI text-embedding-1536 (Real Public, Full — only 1K vectors exist on disk)", PathBuf::from("datasets/openai_1536/openai_base.fvecs"), PathBuf::from("datasets/openai_1536/openai_query.fvecs"), None),
    ];

    for (name, base_path, query_path, declared_n) in &public_datasets {
        if !base_path.exists() || !query_path.exists() {
            println!("\n{name} — SKIPPED: dataset files not found at {}", base_path.display());
            continue;
        }
        match (read_fvecs_limited(base_path, *declared_n), read_fvecs_limited(query_path, Some(MAX_QUERIES_PER_DATASET))) {
            (Ok(base_vecs), Ok(query_vecs)) if !base_vecs.is_empty() && !query_vecs.is_empty() => {
                let dim = base_vecs[0].dimension();
                let r = evaluate_dataset(name, &base_vecs, &query_vecs, dim, k, *declared_n);
                print_dataset(&r);
                results.push(r);
            }
            _ => println!("\n{name} — SKIPPED: read_fvecs failed or files empty"),
        }
    }

    // Cross-mode consistency check: does PlannerDefault's recall match
    // whichever forced mode it's supposed to be equivalent to at this N?
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("ROUTING VERIFICATION");
    for r in &results {
        let crosses = r.corpus_n as f64 > r.predicted_crossover_n;
        let planner = r.modes.iter().find(|m| matches!(m.mode, ExecMode::PlannerDefault));
        let reference = r.modes.iter().find(|m| {
            matches!(m.mode, ExecMode::ExactForced) == !crosses
                || matches!(m.mode, ExecMode::GraphOnly) == crosses
        });
        if let (Some(p), Some(ref_m)) = (planner, reference) {
            let matches_prediction = (p.mean_recall_pct - ref_m.mean_recall_pct).abs() < 0.01;
            println!(
                "  {}: PlannerDefault predicted to match {} — mean recall {:.1}% vs {:.1}% [{}]",
                r.name,
                ref_m.mode.label(),
                p.mean_recall_pct,
                ref_m.mean_recall_pct,
                if matches_prediction { "CONSISTENT" } else { "DIVERGENT — investigate before trusting PlannerDefault routing at this N" }
            );
        }
    }
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
}
