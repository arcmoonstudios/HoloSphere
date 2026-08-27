/* holosphere/benches/common.rs */
//!▫~•◦-------------------------------‣
//! # Benchmark Support & Dataset Fixtures
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides shared dataset fixtures, .fvecs loaders, and pre-built snapshot
//! attachments for HoloSphere benchmark targets.
//!
//! ## Key Capabilities
//! - **Multi-Tier Benchmarking Scale:** Supports Smoke (1k), Dev (5k), Validation (25k), and Scale (1M) tiers.
//! - **Public Corpus Ingestion:** Reads real `.fvecs` datasets from `datasets/`; never generates synthetic data.
//! - **Immutable Snapshot Attachments:** Pre-materializes read-only snapshots to isolate retrieval latency from indexing overhead.
//!
//! ## Contract
//! **No benchmark binary may build or reindex at run time.**
//! All bench binaries must load real vectors from `datasets/` (read-only .fvecs I/O) and,
//! when an index is required, attach a prebuilt snapshot from `benchmark_databases/`.
//! Use `hnsqr_build_bench_db` once to materialise any missing snapshot artifacts.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use hnsqr::metadata::index::MetadataValue;
use hnsqr::rivero::RiveroProfile;
use hnsqr::storage::snapshot::SnapshotOpenOptions;
use hnsqr::vector::folding::ComplexWeaver;
use hnsqr::{HNSQRIndex, NodeIndex, SimilarityScore, VectorEmbedding};
use num_complex::Complex32;
use serde::{Deserialize, Serialize};

pub const DEFAULT_BENCH_SEED: u64 = 0x484e_5351_525f_5632;
pub const CACHE_VERSION: u32 = 6;

/// Produces the canonical benchmark oracle using the index's own exhaustive
/// search implementation and configured metric.  Benchmarks must not recreate
/// this with a local similarity helper: doing so can silently grade the same
/// Rivero result against a different ranking contract.
pub fn compute_exact_ground_truth(
    index: &HNSQRIndex,
    queries: &[VectorEmbedding],
    k: usize,
) -> Vec<Vec<(NodeIndex, SimilarityScore)>> {
    queries
        .iter()
        .map(|query| {
            index
                .search_indices_exact(query, k, None)
                .expect("exact ground-truth search must succeed")
        })
        .collect()
}

/// Multi-tier benchmarking scale configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchScale {
    /// 1,000 vectors - Sub-second fast correctness and invariant checks on every edit.
    Smoke,
    /// 5,000 vectors - Rapid iteration and performance comparisons during development.
    Dev,
    /// 25,000 vectors - Realistic latency, high concurrency, and recall validation.
    Validation,
    /// 250,000 to 1,000,000 vectors - Release certification and asymptotic work ceiling checks.
    Scale,
}

impl BenchScale {
    /// Reads the active scale tier from the `HNSQR_BENCH_SCALE` environment variable. Defaults to `Dev`.
    #[must_use]
    pub fn from_env() -> Self {
        if cfg!(debug_assertions) {
            return Self::Smoke;
        }
        match std::env::var("HNSQR_BENCH_SCALE").as_deref() {
            Ok("smoke") | Ok("SMOKE") => Self::Smoke,
            Ok("validation") | Ok("VALIDATION") => Self::Validation,
            Ok("scale") | Ok("SCALE") => Self::Scale,
            _ => Self::Dev,
        }
    }

    #[must_use]
    pub const fn corpus_size(self) -> usize {
        match self {
            Self::Smoke => 1_000,
            Self::Dev => 5_000,
            Self::Validation => 25_000,
            Self::Scale => 1_000_000,
        }
    }

    #[must_use]
    pub const fn query_count(self) -> usize {
        match self {
            Self::Smoke => 16,
            Self::Dev => 64,
            Self::Validation => 200,
            Self::Scale => 500,
        }
    }

    #[must_use]
    pub const fn concurrency_clients(self) -> &'static [usize] {
        match self {
            Self::Smoke => &[1, 4, 8],
            Self::Dev => &[1, 4, 8, 16, 24, 32],
            Self::Validation => &[1, 2, 4, 8, 16, 24, 32, 48, 64],
            Self::Scale => &[1, 4, 8, 16, 24, 32, 48, 64],
        }
    }
}

/// A serialized realistic corpus for benchmarking loaded from real public datasets.
#[derive(Clone, Serialize, Deserialize)]
pub struct TextRetrievalCorpus {
    pub name: String,
    pub real_dim: usize,
    pub complex_dim: usize,
    pub corpus_raw: Vec<Vec<f32>>,
    pub folded_corpus: Vec<VectorEmbedding>,
    pub queries_raw: Vec<Vec<f32>>,
    pub folded_queries: Vec<VectorEmbedding>,
    /// Legacy field name: a disjoint, native query partition (B), not
    /// synthetically generated hard negatives.
    pub hard_negatives: Vec<VectorEmbedding>,
    /// Legacy field name: a disjoint, native query partition (C), not a
    /// fabricated out-of-distribution workload.
    pub ood_queries: Vec<VectorEmbedding>,
    /// Legacy field name: a disjoint, native query partition (D), not an
    /// unlabelled dataset being asserted to be isotropic.
    pub isotropic_queries: Vec<VectorEmbedding>,
    pub relevance_ground_truth: Vec<Vec<(usize, u32)>>,
}

/// Directory containing durable, prebuilt benchmark databases.
///
/// This deliberately lives outside `target/`: `target` is disposable and caused
/// benchmarks to quietly include index construction after a clean build.  Override
/// the location with `HNSQR_BENCH_DATABASE_DIR` for a shared benchmark volume.
pub fn bench_cache_dir() -> PathBuf {
    std::env::var_os("HNSQR_BENCH_DATABASE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new("benchmark_databases").to_path_buf())
}

fn require_prebuilt_snapshot(path: &Path) {
    assert!(
        path.is_file(),
        "prebuilt benchmark database is missing: {}\n\
         Build it once from the checked-in datasets with:\n\
           cargo run --release --bin hnsqr_build_bench_db -- --help\n\
         Benchmark processes never build or overwrite database artifacts.",
        path.display()
    );
}

/// High-performance binary `.fvecs` parser.
pub fn read_fvecs<P: AsRef<Path>>(
    path: P,
    limit: Option<usize>,
) -> std::io::Result<(Vec<VectorEmbedding>, usize)> {
    read_fvecs_slice(path, 0, limit)
}

/// Reads a bounded, native query partition from an `.fvecs` file.
///
/// The benchmark suite uses this instead of synthesizing query distributions:
/// each workload must identify the real query rows it evaluates.  `start` is a
/// vector row offset, not a byte offset.
pub fn read_fvecs_slice<P: AsRef<Path>>(
    path: P,
    start: usize,
    limit: Option<usize>,
) -> std::io::Result<(Vec<VectorEmbedding>, usize)> {
    let mut file = File::open(path)?;
    let mut dim_buf = [0u8; 4];
    let mut vectors = Vec::new();
    let mut dim = 0usize;
    let mut row = 0usize;

    while let Ok(()) = file.read_exact(&mut dim_buf) {
        let current_dim = u32::from_le_bytes(dim_buf) as usize;
        if dim == 0 {
            dim = current_dim;
        }
        let mut float_buf = vec![0u8; current_dim * 4];
        file.read_exact(&mut float_buf)?;
        if row < start {
            row += 1;
            continue;
        }
        let mut floats = Vec::with_capacity(current_dim);
        for chunk in float_buf.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        vectors.push(ComplexWeaver::fold_llm_embedding(&floats));
        row += 1;
        if let Some(max) = limit {
            if vectors.len() >= max {
                break;
            }
        }
    }
    Ok((vectors, dim))
}

/// High-performance binary `.fvecs` parser returning raw floats and folded embeddings.
pub fn read_fvecs_raw<P: AsRef<Path>>(
    path: P,
    limit: Option<usize>,
) -> std::io::Result<(Vec<Vec<f32>>, Vec<VectorEmbedding>, usize)> {
    read_fvecs_raw_slice(path, 0, limit)
}

/// Reads raw and folded vectors from a native `.fvecs` row partition.
pub fn read_fvecs_raw_slice<P: AsRef<Path>>(
    path: P,
    start: usize,
    limit: Option<usize>,
) -> std::io::Result<(Vec<Vec<f32>>, Vec<VectorEmbedding>, usize)> {
    let mut file = File::open(path)?;
    let mut dim_buf = [0u8; 4];
    let mut raw_vectors = Vec::new();
    let mut folded_vectors = Vec::new();
    let mut dim = 0usize;
    let mut row = 0usize;

    while let Ok(()) = file.read_exact(&mut dim_buf) {
        let current_dim = u32::from_le_bytes(dim_buf) as usize;
        if dim == 0 {
            dim = current_dim;
        }
        let mut float_buf = vec![0u8; current_dim * 4];
        file.read_exact(&mut float_buf)?;
        if row < start {
            row += 1;
            continue;
        }
        let mut floats = Vec::with_capacity(current_dim);
        for chunk in float_buf.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let norm: f32 = floats.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let normalized_floats: Vec<f32> = floats.iter().map(|&x| x / norm).collect();
        folded_vectors.push(ComplexWeaver::fold_llm_embedding(&normalized_floats));
        raw_vectors.push(normalized_floats);
        row += 1;
        if let Some(max) = limit {
            if raw_vectors.len() >= max {
                break;
            }
        }
    }
    Ok((raw_vectors, folded_vectors, dim))
}

/// Finds the closest matching real dataset in `datasets/` for the requested dimensionality.
pub fn find_best_matching_dataset(target_dim: usize) -> (PathBuf, PathBuf, usize) {
    let base_dir = Path::new("datasets");
    if target_dim <= 35 {
        (
            base_dir.join("glove_25/glove25_base.fvecs"),
            base_dir.join("glove_25/glove25_query.fvecs"),
            25,
        )
    } else if target_dim <= 75 {
        (
            base_dir.join("glove_50/glove50_base.fvecs"),
            base_dir.join("glove_50/glove50_query.fvecs"),
            50,
        )
    } else if target_dim <= 115 {
        (
            base_dir.join("glove_100/glove100_base.fvecs"),
            base_dir.join("glove_100/glove100_query.fvecs"),
            100,
        )
    } else if target_dim <= 300 {
        let sift1m = base_dir.join("sift_1m/sift1m_base.fvecs");
        let siftsmall = base_dir.join("siftsmall/siftsmall_base.fvecs");
        if sift1m.exists() {
            (sift1m, base_dir.join("sift_1m/sift1m_query.fvecs"), 128)
        } else {
            (
                siftsmall,
                base_dir.join("siftsmall/siftsmall_query.fvecs"),
                128,
            )
        }
    } else if target_dim <= 600 {
        (
            base_dir.join("clip_512/clip_base.fvecs"),
            base_dir.join("clip_512/clip_query.fvecs"),
            512,
        )
    } else if target_dim <= 1000 {
        let cohere_large = base_dir.join("cohere_768_large/cohere_100k_base.fvecs");
        if cohere_large.exists() {
            (
                cohere_large,
                base_dir.join("cohere_768/cohere_query.fvecs"),
                768,
            )
        } else {
            (
                base_dir.join("cohere_768/cohere_base.fvecs"),
                base_dir.join("cohere_768/cohere_query.fvecs"),
                768,
            )
        }
    } else if target_dim <= 2500 {
        let openai_large = base_dir.join("openai_1536_large/openai_1m_base.fvecs");
        if openai_large.exists() {
            (
                openai_large,
                base_dir.join("openai_1536/openai_query.fvecs"),
                1536,
            )
        } else {
            (
                base_dir.join("openai_1536/openai_base.fvecs"),
                base_dir.join("openai_1536/openai_query.fvecs"),
                1536,
            )
        }
    } else {
        (
            base_dir.join("arxiv_4096/database_vectors.fvecs"),
            base_dir.join("arxiv_4096/query_vectors.fvecs"),
            4096,
        )
    }
}

/// Returns the number of vectors available in the best-matching dataset for `target_dim`.
///
/// Reads only the file size and the first header word; never allocates vector data.
/// Returns 0 if no matching file exists on disk.
/// Sweep benchmarks should call this to clamp their probe lists before requesting a
/// corpus slice, so they never request more vectors than the `.fvecs` file contains.
pub fn corpus_available_count(target_dim: usize) -> usize {
    let (base_path, _, _) = find_best_matching_dataset(target_dim);
    if !base_path.exists() {
        return 0;
    }
    let Ok(mut file) = std::fs::File::open(&base_path) else {
        return 0;
    };
    let mut dim_buf = [0u8; 4];
    let Ok(()) = std::io::Read::read_exact(&mut file, &mut dim_buf) else {
        return 0;
    };
    let dim = u32::from_le_bytes(dim_buf) as usize;
    if dim == 0 {
        return 0;
    }
    let Ok(meta) = std::fs::metadata(&base_path) else {
        return 0;
    };
    let bytes_per_vec = 4 + dim * 4; // 4-byte header + dim floats
    (meta.len() as usize) / bytes_per_vec
}

/// Loads a real public dataset best matching the requested scale and dimensionality.
pub fn load_real_dataset_corpus(
    n: usize,
    num_queries: usize,
    real_dim: usize,
    _seed: u64,
) -> TextRetrievalCorpus {
    let (base_path, query_path, actual_dim) = find_best_matching_dataset(real_dim);

    let (corpus_raw, folded_corpus, dim_loaded) = if base_path.exists() {
        read_fvecs_raw(&base_path, Some(n)).unwrap_or_else(|_| (Vec::new(), Vec::new(), actual_dim))
    } else {
        (Vec::new(), Vec::new(), actual_dim)
    };

    let (queries_raw, folded_queries, _) = if query_path.exists() {
        read_fvecs_raw_slice(&query_path, 0, Some(num_queries))
            .unwrap_or_else(|_| (Vec::new(), Vec::new(), actual_dim))
    } else {
        (Vec::new(), Vec::new(), actual_dim)
    };

    // If the dataset file holds fewer vectors than requested, load what is available.
    // Callers that require an exact cardinality must either pre-check corpus_available_count()
    // or use open_prebuilt_index(), which asserts against the snapshot row count.
    if folded_corpus.len() < n && !folded_corpus.is_empty() {
        eprintln!(
            "[bench_common] warning: requested {n} corpus vectors from '{}', \
             only {loaded} available — proceeding with {loaded}",
            base_path.display(),
            loaded = folded_corpus.len()
        );
    }

    if folded_queries.len() < num_queries && !folded_queries.is_empty() {
        eprintln!(
            "[bench_common] warning: requested {num_queries} query vectors from '{}', \
             only {loaded} available — proceeding with {loaded}",
            query_path.display(),
            loaded = folded_queries.len()
        );
    }

    let (_, hard_negatives, _) = if query_path.exists() {
        read_fvecs_raw_slice(&query_path, num_queries, Some(num_queries))
            .unwrap_or_else(|_| (Vec::new(), Vec::new(), actual_dim))
    } else {
        (Vec::new(), Vec::new(), actual_dim)
    };
    let (_, ood_queries, _) = if query_path.exists() {
        read_fvecs_raw_slice(&query_path, num_queries * 2, Some(num_queries))
            .unwrap_or_else(|_| (Vec::new(), Vec::new(), actual_dim))
    } else {
        (Vec::new(), Vec::new(), actual_dim)
    };
    let (_, isotropic_queries, _) = if query_path.exists() {
        read_fvecs_raw_slice(&query_path, num_queries * 3, Some(num_queries))
            .unwrap_or_else(|_| (Vec::new(), Vec::new(), actual_dim))
    } else {
        (Vec::new(), Vec::new(), actual_dim)
    };

    // Grade relevance ground truth for production validation benchmarks
    let mut relevance_ground_truth = Vec::with_capacity(queries_raw.len());
    if !corpus_raw.is_empty() && !queries_raw.is_empty() {
        for q_vec in &queries_raw {
            let mut doc_scores: Vec<(usize, f32)> = corpus_raw
                .iter()
                .enumerate()
                .map(|(d_idx, d_vec)| {
                    let dot: f32 = q_vec.iter().zip(d_vec.iter()).map(|(a, b)| a * b).sum();
                    (d_idx, dot)
                })
                .collect();
            doc_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

            let rel: Vec<(usize, u32)> = doc_scores
                .iter()
                .take(100)
                .enumerate()
                .map(|(rank, &(d_idx, score))| {
                    let grade = if rank < 3 && score > 0.85 {
                        3
                    } else if rank < 10 && score > 0.70 {
                        2
                    } else if score > 0.50 {
                        1
                    } else {
                        0
                    };
                    (d_idx, grade)
                })
                .collect();
            relevance_ground_truth.push(rel);
        }
    }

    TextRetrievalCorpus {
        name: format!("real-dataset-dim{dim_loaded}-n{n}"),
        real_dim: dim_loaded,
        complex_dim: dim_loaded.div_ceil(2),
        corpus_raw,
        folded_corpus,
        queries_raw,
        folded_queries,
        hard_negatives,
        ood_queries,
        isotropic_queries,
        relevance_ground_truth,
    }
}

/// Locates a canonical prebuilt Snapshot V2 index.
///
/// `seed` remains part of the API for call-site compatibility.  Dataset-backed
/// artifacts are deterministic, so it does not participate in the filename.
pub fn open_prebuilt_snapshot_v2(
    scale: BenchScale,
    profile: RiveroProfile,
    seed: u64,
) -> (PathBuf, TextRetrievalCorpus) {
    let n = scale.corpus_size();
    let q_count = scale.query_count();
    let corpus = load_real_dataset_corpus(n, q_count, 128, seed);

    let snap_path = bench_cache_dir().join(format!(
        "snapshot-v{CACHE_VERSION}-n{n}-p{:?}-d{}.hnsqr",
        profile, corpus.real_dim
    ));

    let _ = seed;
    require_prebuilt_snapshot(&snap_path);
    (snap_path, corpus)
}

/// Zero-copy attaches a prebuilt snapshot-backed HNSQRIndex for any dataset.
pub fn open_prebuilt_index(
    dataset_tag: &str,
    corpus: &[VectorEmbedding],
    dim: usize,
    profile: RiveroProfile,
) -> HNSQRIndex {
    let snap_path = bench_cache_dir().join(format!(
        "{dataset_tag}_v{CACHE_VERSION}_p{:?}_d{dim}_n{}.snapshot",
        profile,
        corpus.len()
    ));
    require_prebuilt_snapshot(&snap_path);
    let index = HNSQRIndex::open_snapshot_v2(&snap_path, SnapshotOpenOptions::default())
        .unwrap_or_else(|error| {
            panic!(
                "invalid prebuilt benchmark database {}: {error}",
                snap_path.display()
            )
        });
    assert_eq!(
        index.dimension(),
        dim,
        "prebuilt index dimension mismatch: expected {dim}, got {}",
        index.dimension()
    );
    assert_eq!(
        index.size(),
        corpus.len(),
        "prebuilt index row count mismatch: expected {}, got {}",
        corpus.len(),
        index.size()
    );
    index.freeze_rivero_routing();
    index
}

/// Attaches an immutable snapshot keyed by `cache_key`.
///
/// Benchmark processes must never construct an index. Materialize this artifact
/// beforehand with `hnsqr_build_bench_db`; a missing snapshot is a configuration
/// error rather than permission to rebuild from the benchmark process.
pub fn open_prebuilt_snapshot(cache_key: &str) -> HNSQRIndex {
    let snap_path = bench_cache_dir().join(format!("{cache_key}.snapshot"));
    require_prebuilt_snapshot(&snap_path);
    HNSQRIndex::open_snapshot_v2(&snap_path, SnapshotOpenOptions::default()).unwrap_or_else(|e| {
        panic!(
            "prebuilt snapshot '{}' failed to open: {e}",
            snap_path.display()
        )
    })
}

/// Fixed Adversarial Regression Corpus ($N=2,000$).
#[derive(Clone)]
pub struct AdversarialRegressionCorpus {
    pub corpus: Vec<VectorEmbedding>,
    pub metadata: Vec<HashMap<String, MetadataValue>>,
    pub in_domain_queries: Vec<VectorEmbedding>,
    pub hard_negatives: Vec<VectorEmbedding>,
    pub ood_noise_queries: Vec<VectorEmbedding>,
    pub phase_adversaries: Vec<(VectorEmbedding, VectorEmbedding, f32)>,
    pub exact_duplicates: Vec<(NodeIndex, NodeIndex)>,
    pub in_domain_ground_truth: Vec<Vec<NodeIndex>>,
    pub hard_negatives_ground_truth: Vec<Vec<NodeIndex>>,
}

pub fn load_adversarial_regression_corpus() -> AdversarialRegressionCorpus {
    let n = 2_000;
    let real_dim = 64;
    let dataset = load_real_dataset_corpus(n, 96, real_dim, DEFAULT_BENCH_SEED);
    assert_eq!(
        dataset.folded_corpus.len(),
        n,
        "real benchmark corpus is incomplete"
    );
    assert!(
        dataset.folded_queries.len() >= 96,
        "real benchmark query set is incomplete"
    );

    let corpus = dataset.folded_corpus;
    let metadata = corpus
        .iter()
        .enumerate()
        .map(|(i, _)| {
            let mut meta = HashMap::new();
            meta.insert(
                "category".to_string(),
                MetadataValue::String(
                    if i % 2 == 0 { "finance" } else { "technology" }.to_string(),
                ),
            );
            meta.insert(
                "year".to_string(),
                MetadataValue::Integer((2020 + (i % 6)) as i64),
            );
            meta
        })
        .collect();
    let in_domain_queries = dataset.folded_queries[..32].to_vec();
    let hard_negatives = dataset.folded_queries[32..64].to_vec();
    let ood_noise_queries = dataset.folded_queries[64..96].to_vec();
    let mut phase_adversaries = Vec::new();
    // Use indexed corpus vectors, not holdout queries: the phase-adversary
    // assertion verifies self-alignment before applying a global phase shift.
    // A real holdout query has no guaranteed near-identical corpus member.
    for folded in corpus.iter().take(32) {
        let phase = std::f32::consts::PI;
        let rotated = folded
            .complex_data()
            .iter()
            .map(|z| z * Complex32::from_polar(1.0, phase))
            .collect();
        phase_adversaries.push((
            folded.clone(),
            VectorEmbedding::from_complex(rotated),
            phase,
        ));
    }

    // Exact ground truth for in-domain queries
    let in_domain_ground_truth = compute_ground_truth(&in_domain_queries, &corpus);
    let hard_negatives_ground_truth = compute_ground_truth(&hard_negatives, &corpus);

    AdversarialRegressionCorpus {
        corpus,
        metadata,
        in_domain_queries,
        hard_negatives,
        ood_noise_queries,
        phase_adversaries,
        exact_duplicates: Vec::new(),
        in_domain_ground_truth,
        hard_negatives_ground_truth,
    }
}

fn compute_ground_truth(
    queries: &[VectorEmbedding],
    corpus: &[VectorEmbedding],
) -> Vec<Vec<NodeIndex>> {
    let mut ground_truth = Vec::with_capacity(queries.len());
    for q in queries {
        let mut scored: Vec<(NodeIndex, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(idx, doc): (usize, &VectorEmbedding)| {
                let dot: f32 = q
                    .complex_data()
                    .iter()
                    .zip(doc.complex_data().iter())
                    .map(|(a, b): (&Complex32, &Complex32)| (a * b.conj()).re)
                    .sum();
                (idx as NodeIndex, dot)
            })
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let top10: Vec<NodeIndex> = scored.iter().take(10).map(|s| s.0).collect();
        ground_truth.push(top10);
    }
    ground_truth
}
