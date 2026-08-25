/* holosphere/src/bench_support/mod.rs */
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

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::metadata::index::MetadataValue;
use crate::rivero::RiveroProfile;
use crate::storage::snapshot::SnapshotOpenOptions;
use crate::vector::folding::ComplexWeaver;
use crate::{HNSQRIndex, NodeIndex, VectorEmbedding};
use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};

pub const DEFAULT_BENCH_SEED: u64 = 0x484e_5351_525f_5632;
const CACHE_VERSION: u32 = 6;

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
    pub hard_negatives: Vec<VectorEmbedding>,
    pub ood_queries: Vec<VectorEmbedding>,
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
    let mut file = File::open(path)?;
    let mut dim_buf = [0u8; 4];
    let mut vectors = Vec::new();
    let mut dim = 0usize;

    while let Ok(()) = file.read_exact(&mut dim_buf) {
        let current_dim = u32::from_le_bytes(dim_buf) as usize;
        if dim == 0 {
            dim = current_dim;
        }
        let mut float_buf = vec![0u8; current_dim * 4];
        file.read_exact(&mut float_buf)?;
        let mut floats = Vec::with_capacity(current_dim);
        for chunk in float_buf.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        vectors.push(ComplexWeaver::fold_llm_embedding(&floats));
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
    let mut file = File::open(path)?;
    let mut dim_buf = [0u8; 4];
    let mut raw_vectors = Vec::new();
    let mut folded_vectors = Vec::new();
    let mut dim = 0usize;

    while let Ok(()) = file.read_exact(&mut dim_buf) {
        let current_dim = u32::from_le_bytes(dim_buf) as usize;
        if dim == 0 {
            dim = current_dim;
        }
        let mut float_buf = vec![0u8; current_dim * 4];
        file.read_exact(&mut float_buf)?;
        let mut floats = Vec::with_capacity(current_dim);
        for chunk in float_buf.chunks_exact(4) {
            floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
        }
        let norm: f32 = floats.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        let normalized_floats: Vec<f32> = floats.iter().map(|&x| x / norm).collect();
        folded_vectors.push(ComplexWeaver::fold_llm_embedding(&normalized_floats));
        raw_vectors.push(normalized_floats);
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
pub fn generate_realistic_text_corpus(
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
        read_fvecs_raw(&query_path, Some(num_queries))
            .unwrap_or_else(|_| (Vec::new(), Vec::new(), actual_dim))
    } else {
        (Vec::new(), Vec::new(), actual_dim)
    };

    // If the dataset file holds fewer vectors than requested, load what is available.
    // Callers that require an exact cardinality must either pre-check corpus_available_count()
    // or use open_prebuilt_index(), which asserts against the snapshot row count.
    if folded_corpus.len() < n && !folded_corpus.is_empty() {
        eprintln!(
            "[bench_support] warning: requested {n} corpus vectors from '{}', \
             only {loaded} available — proceeding with {loaded}",
            base_path.display(),
            loaded = folded_corpus.len()
        );
    }

    if folded_queries.len() < num_queries && !folded_queries.is_empty() {
        eprintln!(
            "[bench_support] warning: requested {num_queries} query vectors from '{}', \
             only {loaded} available — proceeding with {loaded}",
            query_path.display(),
            loaded = folded_queries.len()
        );
    }

    let hard_negatives = folded_queries.clone();
    let ood_queries = folded_queries.clone();
    let isotropic_queries = folded_queries.clone();

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
    let corpus = generate_realistic_text_corpus(n, q_count, 128, seed);

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

pub fn generate_adversarial_regression_corpus() -> AdversarialRegressionCorpus {
    let n = 2_000;
    let real_dim = 64;
    let seed = 0xadfe_2026_beef_0001;
    let mut rng = StdRng::seed_from_u64(seed);

    let (base_path, _, _) = find_best_matching_dataset(real_dim);
    let (raw_corpus, _) = if base_path.exists() {
        read_fvecs(&base_path, Some(n)).unwrap_or_default()
    } else {
        (Vec::new(), real_dim)
    };

    let num_clusters = 20;
    let mut centers = Vec::with_capacity(num_clusters);
    for i in 0..num_clusters {
        if !raw_corpus.is_empty() {
            let idx = (i * 73) % raw_corpus.len();
            let c_vec: Vec<f32> = raw_corpus[idx]
                .complex_data()
                .iter()
                .flat_map(|z| [z.re, z.im])
                .take(real_dim)
                .collect();
            let mut c = vec![0.0f32; real_dim];
            for (d, &val) in c_vec.iter().enumerate().take(real_dim) {
                c[d] = val;
            }
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            for v in &mut c {
                *v /= norm;
            }
            centers.push(c);
        } else {
            let mut c = vec![0.0f32; real_dim];
            for v in &mut c {
                *v = rng.random_range(-1.0..1.0);
            }
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
            for v in &mut c {
                *v /= norm;
            }
            centers.push(c);
        }
    }

    let mut corpus = Vec::with_capacity(n);
    let mut metadata = Vec::with_capacity(n);

    // 1. Normal semantic cluster nodes (0..1600)
    for i in 0..1600 {
        let c = &centers[i % num_clusters];
        let mut v = c.clone();
        for val in &mut v {
            *val += rng.random_range(-0.10..0.10);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut v {
            *val /= norm;
        }
        let folded = ComplexWeaver::fold_llm_embedding(&v);
        corpus.push(folded);

        let mut meta = HashMap::new();
        let cat = if i % 2 == 0 { "finance" } else { "technology" };
        meta.insert(
            "category".to_string(),
            MetadataValue::String(cat.to_string()),
        );
        meta.insert(
            "year".to_string(),
            MetadataValue::Integer((2020 + (i % 6)) as i64),
        );
        metadata.push(meta);
    }

    // 2. Exact ties & duplicates (1600..1700)
    let mut exact_duplicates = Vec::new();
    for i in 1600..1700 {
        let source_slot = i - 800;
        let dup = corpus[source_slot].clone();
        corpus.push(dup);
        exact_duplicates.push((source_slot as NodeIndex, i as NodeIndex));

        let mut meta = HashMap::new();
        meta.insert(
            "category".to_string(),
            MetadataValue::String("duplicate".to_string()),
        );
        meta.insert("year".to_string(), MetadataValue::Integer(2026));
        metadata.push(meta);
    }

    // 3. Cluster boundary & hard negative vectors (1700..1900)
    for i in 1700..1900 {
        let c1 = &centers[i % num_clusters];
        let c2 = &centers[(i + 1) % num_clusters];
        let mut v = vec![0.0f32; real_dim];
        for d in 0..real_dim {
            v[d] = 0.5 * c1[d] + 0.5 * c2[d] + rng.random_range(-0.02..0.02);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut v {
            *val /= norm;
        }
        corpus.push(ComplexWeaver::fold_llm_embedding(&v));

        let mut meta = HashMap::new();
        meta.insert(
            "category".to_string(),
            MetadataValue::String("boundary".to_string()),
        );
        meta.insert("year".to_string(), MetadataValue::Integer(2024));
        metadata.push(meta);
    }

    // 4. Random Isotropic spherical vectors (1900..2000)
    for _ in 1900..2000 {
        let mut v = vec![0.0f32; real_dim];
        for val in &mut v {
            *val = rng.random_range(-1.0..1.0);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut v {
            *val /= norm;
        }
        corpus.push(ComplexWeaver::fold_llm_embedding(&v));

        let mut meta = HashMap::new();
        meta.insert(
            "category".to_string(),
            MetadataValue::String("noise".to_string()),
        );
        meta.insert("year".to_string(), MetadataValue::Integer(2025));
        metadata.push(meta);
    }

    // 5. Queries & Adversaries
    let mut in_domain_queries = Vec::new();
    let mut hard_negatives = Vec::new();
    let mut ood_noise_queries = Vec::new();
    let mut phase_adversaries = Vec::new();

    for i in 0..32 {
        let c = &centers[i % num_clusters];
        let mut v = c.clone();
        for val in &mut v {
            *val += rng.random_range(-0.05..0.05);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut v {
            *val /= norm;
        }
        let folded = ComplexWeaver::fold_llm_embedding(&v);
        in_domain_queries.push(folded.clone());

        // Phase adversary
        let phase = std::f32::consts::PI;
        let mut rotated_complex = Vec::new();
        for z in folded.complex_data() {
            let rot = Complex32::from_polar(1.0, phase);
            rotated_complex.push(z * rot);
        }
        let adversary = VectorEmbedding::from_complex(rotated_complex);
        phase_adversaries.push((folded, adversary, phase));
    }

    for i in 0..32 {
        let c1 = &centers[i % num_clusters];
        let c2 = &centers[(i + 7) % num_clusters];
        let mut v = vec![0.0f32; real_dim];
        for d in 0..real_dim {
            v[d] = 0.5 * c1[d] + 0.5 * c2[d] + rng.random_range(-0.01..0.01);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut v {
            *val /= norm;
        }
        hard_negatives.push(ComplexWeaver::fold_llm_embedding(&v));

        let mut ood_v = vec![0.0f32; real_dim];
        for d in 0..real_dim {
            ood_v[d] = rng.random_range(-1.0..1.0);
        }
        let norm_ood: f32 = ood_v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut ood_v {
            *val /= norm_ood;
        }
        ood_noise_queries.push(ComplexWeaver::fold_llm_embedding(&ood_v));
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
        exact_duplicates,
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
