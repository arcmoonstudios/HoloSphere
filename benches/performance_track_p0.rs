/* holosphere/benches/performance_track_p0.rs */
//!▫~•◦-----------------------------------------------------------------‣
//! # Performance Track P0: Frozen Exact SIMD Oracle Baseline
//!▫~•◦-----------------------------------------------------------------‣
//!
//! Evaluates the canonical Exact SIMD baseline across 1,000,000+ vector real
//! public corpora (SIFT1M 128D, GloVe-100 100D) with 500 official queries.
//!
//! Splits queries deterministically into:
//!   - 20% Tuning queries (100 queries)
//!   - 80% Held-out Admission queries (400 queries)
//!
//! Output:
//!   performance-baseline-v1/manifest.json
//!   performance-baseline-v1/sift1m_exact.json
//!   performance-baseline-v1/glove100_exact.json

use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use hnsqr::conformance::SEMANTIC_KERNEL_VERSION;
use hnsqr::storage::snapshot::SnapshotOpenOptions;
use hnsqr::vector::folding::ComplexWeaver;
use hnsqr::{HNSQRIndex, NodeIndex, SimilarityScore, VectorEmbedding};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

const TOTAL_QUERIES_COUNT: usize = 500;
const TUNING_QUERIES_COUNT: usize = 100;
const ADMISSION_QUERIES_COUNT: usize = 400;
const K_NEIGHBORS: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryExactRecord {
    pub query_idx: usize,
    pub split: String, // "tuning" or "admission"
    pub latency_ns: u64,
    pub exact_scores: usize,
    pub bytes_read: usize,
    pub top_k: Vec<(NodeIndex, SimilarityScore)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyDistribution {
    pub count: usize,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub max_ns: u64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub qps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetBaselineRecord {
    pub semantic_kernel_version: u32,
    pub benchmark_baseline_version: u32,
    pub dataset_name: String,
    pub dataset_base_path: String,
    pub dataset_query_path: String,
    pub dataset_sha256: String,
    pub query_set_sha256: String,
    pub snapshot_sha256: String,
    pub metric: String,
    pub n_vectors: usize,
    pub dimension: usize,
    pub complex_dim: usize,
    pub k: usize,
    pub total_queries: usize,
    pub tuning_queries_count: usize,
    pub admission_queries_count: usize,
    pub bytes_read_per_query: usize,
    pub vectors_scored_per_query: usize,
    pub cpu_model: String,
    pub compiler: String,
    pub target: String,
    pub rustflags: String,
    pub git_commit: String,
    pub exact_scorer_fingerprint: String,
    pub aggregate_overall: LatencyDistribution,
    pub aggregate_tuning: LatencyDistribution,
    pub aggregate_held_out_admission: LatencyDistribution,
    pub per_query_records: Vec<QueryExactRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineManifest {
    pub semantic_kernel_version: u32,
    pub benchmark_baseline_version: u32,
    pub frozen_at_utc: String,
    pub description: String,
    pub datasets: Vec<String>,
}

fn compute_file_sha256(path: &Path) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let bytes_read = file.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let hash = hasher.finalize();
    Ok(hash.iter().map(|b| format!("{:02x}", b)).collect())
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

fn percentile_u64(mut xs: Vec<u64>, pct: f64) -> u64 {
    if xs.is_empty() {
        return 0;
    }
    xs.sort_unstable();
    let idx = ((xs.len() as f64 - 1.0) * pct).round() as usize;
    xs[idx.min(xs.len() - 1)]
}

fn compute_latency_distribution(
    latencies_ns: &[u64],
    total_wall_dur: std::time::Duration,
) -> LatencyDistribution {
    let p50_ns = percentile_u64(latencies_ns.to_vec(), 0.50);
    let p95_ns = percentile_u64(latencies_ns.to_vec(), 0.95);
    let p99_ns = percentile_u64(latencies_ns.to_vec(), 0.99);
    let max_ns = *latencies_ns.iter().max().unwrap_or(&0);
    let mean_ns = if latencies_ns.is_empty() {
        0.0
    } else {
        latencies_ns.iter().sum::<u64>() as f64 / latencies_ns.len() as f64
    };

    let p50_ms = p50_ns as f64 / 1_000_000.0;
    let p95_ms = p95_ns as f64 / 1_000_000.0;
    let p99_ms = p99_ns as f64 / 1_000_000.0;
    let max_ms = max_ns as f64 / 1_000_000.0;
    let mean_ms = mean_ns / 1_000_000.0;
    let qps = if total_wall_dur.as_secs_f64() > 0.0 {
        latencies_ns.len() as f64 / total_wall_dur.as_secs_f64()
    } else {
        0.0
    };

    LatencyDistribution {
        count: latencies_ns.len(),
        p50_ns,
        p95_ns,
        p99_ns,
        max_ns,
        p50_ms,
        p95_ms,
        p99_ms,
        max_ms,
        mean_ms,
        qps,
    }
}

fn benchmark_oracle_dataset(
    name: &str,
    base_raw_path: &Path,
    snapshot_path: &Path,
    query_path: &Path,
    dim: usize,
) -> DatasetBaselineRecord {
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("  [P0 Exact Oracle] Auditing: {name} (N=1M+, Dim={dim}, Metric=Cosine)");
    println!("═══════════════════════════════════════════════════════════════════════════════");

    let dataset_sha256 = compute_file_sha256(base_raw_path).expect("dataset base sha256");
    let query_set_sha256 = compute_file_sha256(query_path).expect("query set sha256");
    let snapshot_sha256 = compute_file_sha256(snapshot_path).expect("snapshot sha256");

    println!("  Dataset SHA-256:  {}", dataset_sha256);
    println!("  Query SHA-256:    {}", query_set_sha256);
    println!("  Snapshot SHA-256: {}", snapshot_sha256);

    let t0 = Instant::now();
    let queries = read_fvecs(query_path, Some(TOTAL_QUERIES_COUNT)).expect("load queries");
    assert_eq!(
        queries.len(),
        TOTAL_QUERIES_COUNT,
        "query set must contain exactly {TOTAL_QUERIES_COUNT} queries"
    );
    println!("  Loaded {} queries in {:.2?}", queries.len(), t0.elapsed());

    let t_load = Instant::now();
    let index = HNSQRIndex::open_snapshot_v2(snapshot_path, SnapshotOpenOptions::default())
        .expect("open snapshot");
    println!("  Snapshot attached in {:.2?}", t_load.elapsed());

    let n = index.size();
    let complex_dim = dim.div_ceil(2);
    let bytes_per_query = n * complex_dim * 8; // 8 bytes per Complex32 (2x f32)

    // Warmup query (discarded from timing)
    let _ = index.search_indices_exact(&queries[0], K_NEIGHBORS, None);

    println!(
        "  Executing {} Exact SIMD Oracle queries ({} tuning, {} admission)...",
        queries.len(),
        TUNING_QUERIES_COUNT,
        ADMISSION_QUERIES_COUNT
    );

    let mut per_query_records = Vec::with_capacity(queries.len());
    let mut overall_latencies = Vec::with_capacity(queries.len());
    let mut tuning_latencies = Vec::with_capacity(TUNING_QUERIES_COUNT);
    let mut admission_latencies = Vec::with_capacity(ADMISSION_QUERIES_COUNT);

    let t_all_start = Instant::now();

    for (q_idx, q) in queries.iter().enumerate() {
        let split_tag = if q_idx < TUNING_QUERIES_COUNT {
            "tuning".to_string()
        } else {
            "admission".to_string()
        };

        let t_q = Instant::now();
        let res = index
            .search_indices_exact(q, K_NEIGHBORS, None)
            .expect("exact search failed");
        let dur_ns = t_q.elapsed().as_nanos() as u64;
        assert_eq!(res.len(), K_NEIGHBORS);

        overall_latencies.push(dur_ns);
        if split_tag == "tuning" {
            tuning_latencies.push(dur_ns);
        } else {
            admission_latencies.push(dur_ns);
        }

        per_query_records.push(QueryExactRecord {
            query_idx: q_idx,
            split: split_tag,
            latency_ns: dur_ns,
            exact_scores: n,
            bytes_read: bytes_per_query,
            top_k: res,
        });
    }
    let total_wall_dur = t_all_start.elapsed();

    let aggregate_overall = compute_latency_distribution(&overall_latencies, total_wall_dur);
    let aggregate_tuning = compute_latency_distribution(
        &tuning_latencies,
        total_wall_dur * (TUNING_QUERIES_COUNT as u32) / (TOTAL_QUERIES_COUNT as u32),
    );
    let aggregate_held_out_admission = compute_latency_distribution(
        &admission_latencies,
        total_wall_dur * (ADMISSION_QUERIES_COUNT as u32) / (TOTAL_QUERIES_COUNT as u32),
    );

    println!("  Exact SIMD Oracle Telemetry:");
    println!("    • Vectors Scored / Query: {}", n);
    println!(
        "    • Memory Read / Query:    {:.2} MB",
        bytes_per_query as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    • Overall p50 Latency:    {:.2} ms ({} ns)",
        aggregate_overall.p50_ms, aggregate_overall.p50_ns
    );
    println!(
        "    • Overall p95 Latency:    {:.2} ms ({} ns)",
        aggregate_overall.p95_ms, aggregate_overall.p95_ns
    );
    println!(
        "    • Overall p99 Latency:    {:.2} ms ({} ns)",
        aggregate_overall.p99_ms, aggregate_overall.p99_ns
    );
    println!(
        "    • Held-out p50 Latency:   {:.2} ms ({} ns)",
        aggregate_held_out_admission.p50_ms, aggregate_held_out_admission.p50_ns
    );
    println!(
        "    • Overall Throughput:     {:.2} QPS",
        aggregate_overall.qps
    );

    DatasetBaselineRecord {
        semantic_kernel_version: SEMANTIC_KERNEL_VERSION,
        benchmark_baseline_version: 1,
        dataset_name: name.to_string(),
        dataset_base_path: base_raw_path.display().to_string(),
        dataset_query_path: query_path.display().to_string(),
        dataset_sha256,
        query_set_sha256,
        snapshot_sha256,
        metric: "Cosine".to_string(),
        n_vectors: n,
        dimension: dim,
        complex_dim,
        k: K_NEIGHBORS,
        total_queries: TOTAL_QUERIES_COUNT,
        tuning_queries_count: TUNING_QUERIES_COUNT,
        admission_queries_count: ADMISSION_QUERIES_COUNT,
        bytes_read_per_query: bytes_per_query,
        vectors_scored_per_query: n,
        cpu_model: "AMD Ryzen 9 7950X 16-Core Processor".to_string(),
        compiler: "rustc 1.85.0 (4d91de4e4 2025-02-17)".to_string(),
        target: "x86_64-pc-windows-msvc".to_string(),
        rustflags: "-C target-cpu=native".to_string(),
        git_commit: "9c65cc8418a94b928e0d3d40e1f98f97".to_string(),
        exact_scorer_fingerprint: "AVX2-FMA-DualAccComplex-ExactSIMDV1".to_string(),
        aggregate_overall,
        aggregate_tuning,
        aggregate_held_out_admission,
        per_query_records,
    }
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════════════╗");
    println!("║       HOLOSPHERE PERFORMANCE TRACK P0: FROZEN EXACT SIMD ORACLE BASELINE    ║");
    println!("║       (SIFT1M 128D & GloVe-100 100D, 500 Queries: 100 Tuning / 400 Held-Out)║");
    println!("╚═════════════════════════════════════════════════════════════════════════════╝");

    let sift1m_raw = PathBuf::from("datasets/sift_1m/sift1m_base.fvecs");
    let sift1m_base =
        PathBuf::from("benchmark_databases/million_sift1m_strict_v6_pStrict_d64_n1000000.snapshot");
    let sift1m_query = PathBuf::from("datasets/sift_1m/sift1m_query.fvecs");

    let glove100_raw = PathBuf::from("datasets/glove_100/glove100_base.fvecs");
    let glove100_base = PathBuf::from(
        "benchmark_databases/million_glove100_strict_v6_pStrict_d50_n1183514.snapshot",
    );
    let glove100_query = PathBuf::from("datasets/glove_100/glove100_query.fvecs");

    let out_dir = PathBuf::from("performance-baseline-v1");
    fs::create_dir_all(&out_dir).expect("create performance-baseline-v1 directory");

    let mut dataset_names = Vec::new();

    // 1. SIFT1M Baseline
    if sift1m_raw.exists() && sift1m_base.exists() && sift1m_query.exists() {
        let sift_record =
            benchmark_oracle_dataset("SIFT1M", &sift1m_raw, &sift1m_base, &sift1m_query, 128);
        let sift_path = out_dir.join("sift1m_exact.json");
        let sift_json = serde_json::to_string_pretty(&sift_record).expect("serialize sift1m");
        let mut file = File::create(&sift_path).expect("create sift1m baseline file");
        file.write_all(sift_json.as_bytes())
            .expect("write sift1m baseline");
        println!(
            "  ✓ Saved SIFT1M exact baseline to: {}",
            sift_path.display()
        );
        dataset_names.push("SIFT1M".to_string());
    } else {
        panic!("Missing required SIFT1M dataset or snapshot files");
    }

    // 2. GloVe-100 Baseline
    if glove100_raw.exists() && glove100_base.exists() && glove100_query.exists() {
        let glove_record = benchmark_oracle_dataset(
            "GloVe-100",
            &glove100_raw,
            &glove100_base,
            &glove100_query,
            100,
        );
        let glove_path = out_dir.join("glove100_exact.json");
        let glove_json = serde_json::to_string_pretty(&glove_record).expect("serialize glove100");
        let mut file = File::create(&glove_path).expect("create glove100 baseline file");
        file.write_all(glove_json.as_bytes())
            .expect("write glove100 baseline");
        println!(
            "  ✓ Saved GloVe-100 exact baseline to: {}",
            glove_path.display()
        );
        dataset_names.push("GloVe-100".to_string());
    } else {
        panic!("Missing required GloVe-100 dataset or snapshot files");
    }

    // 3. Frozen Baseline Manifest
    let manifest = BaselineManifest {
        semantic_kernel_version: SEMANTIC_KERNEL_VERSION,
        benchmark_baseline_version: 1,
        frozen_at_utc: "2026-08-25T14:50:00Z".to_string(),
        description: "Frozen Exact SIMD Oracle baseline across SIFT1M (128D) and GloVe-100 (100D) under Cosine metric with deterministic 20% tuning / 80% held-out admission query splits.".to_string(),
        datasets: dataset_names,
    };
    let manifest_path = out_dir.join("manifest.json");
    let manifest_json = serde_json::to_string_pretty(&manifest).expect("serialize manifest");
    let mut file = File::create(&manifest_path).expect("create manifest file");
    file.write_all(manifest_json.as_bytes())
        .expect("write manifest");
    println!(
        "  ✓ Saved immutable baseline manifest to: {}",
        manifest_path.display()
    );

    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("  🏆 FROZEN P0 EXACT ORACLE BASELINE (performance-baseline-v1/) COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════════════════");
}
