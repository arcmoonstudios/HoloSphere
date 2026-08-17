#![allow(dead_code)]

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use hnsqr::gateway::ComplexWeaver;
use hnsqr::metadata_index::MetadataValue;
use hnsqr::rivero::RiveroProfile;
use hnsqr::rivero_bulk::RiveroBulkBuilder;
use hnsqr::{HNSQRConfig, HNSQRIndex, NodeIndex, VectorEmbedding};
use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

pub const DEFAULT_BENCH_SEED: u64 = 0x484e_5351_525f_5632;
const CACHE_VERSION: u32 = 4;

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

/// A serialized realistic synthetic corpus for benchmarking.
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
}

/// Directory path for cached benchmark datasets and snapshot indices.
pub fn bench_cache_dir() -> PathBuf {
    let dir = Path::new("target").join("hnsqr-bench-data");
    let _ = fs::create_dir_all(&dir);
    dir
}

/// Generates a realistic clustered semantic text corpus.
pub fn generate_realistic_text_corpus(
    n: usize,
    num_queries: usize,
    real_dim: usize,
    seed: u64,
) -> TextRetrievalCorpus {
    let cache_file = bench_cache_dir().join(format!(
        "corpus-v{CACHE_VERSION}-n{n}-q{num_queries}-d{real_dim}-s{seed:x}.bin"
    ));

    if let Ok(mut file) = File::open(&cache_file) {
        let mut bytes = Vec::new();
        if file.read_to_end(&mut bytes).is_ok()
            && let Ok(corpus) = bincode::deserialize::<TextRetrievalCorpus>(&bytes)
        {
            return corpus;
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let complex_dim = real_dim / 2;

    let num_clusters = (n / 50).clamp(4, 500);
    let mut cluster_centers: Vec<Vec<f32>> = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let mut center = vec![0.0f32; real_dim];
        for val in &mut center {
            *val = rng.gen_range(-1.0..1.0);
        }
        let norm: f32 = center.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut center {
                *val /= norm;
            }
        }
        cluster_centers.push(center);
    }

    let mut corpus_raw: Vec<Vec<f32>> = Vec::with_capacity(n);
    let mut folded_corpus: Vec<VectorEmbedding> = Vec::with_capacity(n);

    for i in 0..n {
        let cluster = &cluster_centers[i % num_clusters];
        let mut vec = cluster.clone();
        for val in &mut vec {
            *val += rng.gen_range(-0.15..0.15);
        }
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut vec {
                *val /= norm;
            }
        }
        let folded = ComplexWeaver::fold_llm_embedding(&vec);
        corpus_raw.push(vec);
        folded_corpus.push(folded);
    }

    let mut queries_raw: Vec<Vec<f32>> = Vec::with_capacity(num_queries);
    let mut folded_queries: Vec<VectorEmbedding> = Vec::with_capacity(num_queries);
    let mut hard_negatives: Vec<VectorEmbedding> = Vec::with_capacity(num_queries);
    let mut ood_queries: Vec<VectorEmbedding> = Vec::with_capacity(num_queries);
    let mut isotropic_queries: Vec<VectorEmbedding> = Vec::with_capacity(num_queries);

    for i in 0..num_queries {
        let cluster_idx = i % num_clusters;
        let cluster = &cluster_centers[cluster_idx];
        let mut q_vec = cluster.clone();
        for val in &mut q_vec {
            *val += rng.gen_range(-0.08..0.08);
        }
        let norm: f32 = q_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut q_vec {
                *val /= norm;
            }
        }
        queries_raw.push(q_vec.clone());
        folded_queries.push(ComplexWeaver::fold_llm_embedding(&q_vec));

        // Hard Negative: Boundary interpolation between two clusters
        let other_cluster = &cluster_centers[(cluster_idx + 1) % num_clusters];
        let mut hn_vec = vec![0.0f32; real_dim];
        for d in 0..real_dim {
            hn_vec[d] = 0.5 * cluster[d] + 0.5 * other_cluster[d] + rng.gen_range(-0.05..0.05);
        }
        let norm_hn: f32 = hn_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_hn > 0.0 {
            for val in &mut hn_vec {
                *val /= norm_hn;
            }
        }
        hard_negatives.push(ComplexWeaver::fold_llm_embedding(&hn_vec));

        // OOD query
        let mut ood_vec = vec![0.0f32; real_dim];
        for d in 0..real_dim {
            ood_vec[d] = (d as f32).sin() + rng.gen_range(-0.2..0.2);
        }
        let norm_ood: f32 = ood_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_ood > 0.0 {
            for val in &mut ood_vec {
                *val /= norm_ood;
            }
        }
        ood_queries.push(ComplexWeaver::fold_llm_embedding(&ood_vec));

        // Random Isotropic
        let mut iso_vec = vec![0.0f32; real_dim];
        for val in &mut iso_vec {
            *val = rng.gen_range(-1.0..1.0);
        }
        let norm_iso: f32 = iso_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_iso > 0.0 {
            for val in &mut iso_vec {
                *val /= norm_iso;
            }
        }
        isotropic_queries.push(ComplexWeaver::fold_llm_embedding(&iso_vec));
    }

    let corpus = TextRetrievalCorpus {
        name: format!("text-cluster-n{n}"),
        real_dim,
        complex_dim,
        corpus_raw,
        folded_corpus,
        queries_raw,
        folded_queries,
        hard_negatives,
        ood_queries,
        isotropic_queries,
    };

    if let Ok(bytes) = bincode::serialize(&corpus)
        && let Ok(mut file) = File::create(&cache_file)
    {
        let _ = file.write_all(&bytes);
    }

    corpus
}

/// Retrieves or builds a canonical cached Snapshot V2 index for lightning-fast test execution.
pub fn get_or_build_snapshot_v2(
    scale: BenchScale,
    profile: RiveroProfile,
    seed: u64,
) -> (PathBuf, TextRetrievalCorpus) {
    let n = scale.corpus_size();
    let q_count = scale.query_count();
    let corpus = generate_realistic_text_corpus(n, q_count, 64, seed);

    let snap_path = bench_cache_dir().join(format!(
        "snapshot-v{CACHE_VERSION}-n{n}-p{:?}-s{seed:x}.hnsqr",
        profile
    ));

    if snap_path.exists() {
        return (snap_path, corpus);
    }

    // Build index and export canonical snapshot
    let builder = RiveroBulkBuilder::with_profile(profile).with_threads(16);
    let built = builder.build(&corpus.folded_corpus).unwrap();

    let mut config = HNSQRConfig::default();
    config.max_elements = n.max(1000);
    config.rivero_enabled = true;
    let index = HNSQRIndex::new(config, corpus.complex_dim);
    for (i, v) in corpus.folded_corpus.iter().enumerate() {
        index.insert(format!("doc-{i}"), v.clone()).unwrap();
    }
    index.install_rivero_state(built).unwrap();

    index.save_snapshot_v2(&snap_path).unwrap();
    (snap_path, corpus)
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

    let num_clusters = 20;
    let mut centers = Vec::with_capacity(num_clusters);
    for _ in 0..num_clusters {
        let mut c = vec![0.0f32; real_dim];
        for v in &mut c {
            *v = rng.gen_range(-1.0..1.0);
        }
        let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt();
        for v in &mut c {
            *v /= norm;
        }
        centers.push(c);
    }

    let mut corpus = Vec::with_capacity(n);
    let mut metadata = Vec::with_capacity(n);

    // 1. Normal semantic cluster nodes (0..1600)
    for i in 0..1600 {
        let c = &centers[i % num_clusters];
        let mut v = c.clone();
        for val in &mut v {
            *val += rng.gen_range(-0.10..0.10);
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
            v[d] = 0.5 * c1[d] + 0.5 * c2[d] + rng.gen_range(-0.02..0.02);
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
            *val = rng.gen_range(-1.0..1.0);
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
            *val += rng.gen_range(-0.05..0.05);
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
            v[d] = 0.5 * c1[d] + 0.5 * c2[d] + rng.gen_range(-0.01..0.01);
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        for val in &mut v {
            *val /= norm;
        }
        hard_negatives.push(ComplexWeaver::fold_llm_embedding(&v));

        let mut ood_v = vec![0.0f32; real_dim];
        for d in 0..real_dim {
            ood_v[d] = (d as f32).sin() + rng.gen_range(-0.2..0.2);
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
