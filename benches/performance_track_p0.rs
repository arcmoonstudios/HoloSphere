/* holosphere/benches/performance_track_p0.rs */
//!▫~•◦-----------------------------------------------------------------‣
//! # Performance Track P0: Frozen Exact SIMD Oracle Baseline
//!▫~•◦-----------------------------------------------------------------‣
//!
//! Evaluates the canonical Exact SIMD baseline across 1,000,000+ vector real
//! public corpora (SIFT1M 128D, GloVe-100 100D) with 500 official queries.
//!
//! Output: benches/baseline/performance-baseline-v1.json (Exact Oracle baseline ONLY).

use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use hnsqr::conformance::SEMANTIC_KERNEL_VERSION;
use hnsqr::storage::snapshot::SnapshotOpenOptions;
use hnsqr::vector::folding::ComplexWeaver;
use hnsqr::{HNSQRIndex, VectorEmbedding};
use serde::{Deserialize, Serialize};

const QUERIES_COUNT: usize = 500;
const K_NEIGHBORS: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExactOracleRecord {
    pub semantic_kernel_version: u32,
    pub benchmark_baseline_version: u32,
    pub dataset_name: String,
    pub dataset_fingerprint: String,
    pub metric: String,
    pub n_vectors: usize,
    pub dimension: usize,
    pub complex_dim: usize,
    pub queries_evaluated: usize,
    pub k: usize,
    pub p50_ns: u64,
    pub p95_ns: u64,
    pub p99_ns: u64,
    pub max_ns: u64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
    pub qps: f64,
    pub bytes_read_per_query: usize,
    pub vectors_scored_per_query: usize,
    pub cpu_architecture: String,
    pub compiler: String,
    pub target: String,
    pub simd_features: String,
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
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

fn benchmark_oracle_dataset(
    name: &str,
    snapshot_path: &Path,
    query_path: &Path,
    dim: usize,
) -> ExactOracleRecord {
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("  [P0 Exact Oracle] Auditing: {name} (N=1M+, Dim={dim}, Metric=Cosine)");
    println!("═══════════════════════════════════════════════════════════════════════════════");

    let t0 = Instant::now();
    let queries = read_fvecs(query_path, Some(QUERIES_COUNT)).expect("load queries");
    println!("  Loaded {} queries in {:.2?}", queries.len(), t0.elapsed());

    assert!(
        snapshot_path.is_file(),
        "missing permanent million-scale database: {}",
        snapshot_path.display()
    );

    let t_load = Instant::now();
    let index = HNSQRIndex::open_snapshot_v2(snapshot_path, SnapshotOpenOptions::default())
        .expect("open snapshot");
    println!("  Snapshot attached in {:.2?}", t_load.elapsed());

    let n = index.size();
    let complex_dim = dim.div_ceil(2);
    let bytes_per_query = n * complex_dim * 8; // 8 bytes per Complex32
    let fp = hex_encode(&index.structural_fingerprint()[..8]);

    // Warmup query
    let _ = index.search_indices_exact(&queries[0], K_NEIGHBORS, None);

    // Measure exact scan latencies across all queries
    println!("  Executing {} Exact SIMD Oracle queries...", queries.len());
    let mut latencies_ns = Vec::with_capacity(queries.len());
    let t_all_start = Instant::now();

    for q in &queries {
        let t_q = Instant::now();
        let res = index
            .search_indices_exact(q, K_NEIGHBORS, None)
            .expect("exact search failed");
        let dur_ns = t_q.elapsed().as_nanos() as u64;
        assert_eq!(res.len(), K_NEIGHBORS);
        latencies_ns.push(dur_ns);
    }
    let total_dur = t_all_start.elapsed();

    let p50_ns = percentile_u64(latencies_ns.clone(), 0.50);
    let p95_ns = percentile_u64(latencies_ns.clone(), 0.95);
    let p99_ns = percentile_u64(latencies_ns.clone(), 0.99);
    let max_ns = *latencies_ns.iter().max().unwrap_or(&0);

    let p50_ms = p50_ns as f64 / 1_000_000.0;
    let p95_ms = p95_ns as f64 / 1_000_000.0;
    let p99_ms = p99_ns as f64 / 1_000_000.0;
    let max_ms = max_ns as f64 / 1_000_000.0;
    let qps = queries.len() as f64 / total_dur.as_secs_f64();

    println!("  Exact SIMD Oracle Telemetry:");
    println!("    • Vectors Scored / Query: {}", n);
    println!(
        "    • Memory Read / Query:    {:.2} MB",
        bytes_per_query as f64 / (1024.0 * 1024.0)
    );
    println!(
        "    • Exact Latency (p50):    {:.2} ms ({} ns)",
        p50_ms, p50_ns
    );
    println!(
        "    • Exact Latency (p95):    {:.2} ms ({} ns)",
        p95_ms, p95_ns
    );
    println!(
        "    • Exact Latency (p99):    {:.2} ms ({} ns)",
        p99_ms, p99_ns
    );
    println!(
        "    • Exact Latency (max):    {:.2} ms ({} ns)",
        max_ms, max_ns
    );
    println!("    • Exact Throughput (QPS): {:.2}", qps);

    ExactOracleRecord {
        semantic_kernel_version: SEMANTIC_KERNEL_VERSION,
        benchmark_baseline_version: 1,
        dataset_name: name.to_string(),
        dataset_fingerprint: fp,
        metric: "Cosine".to_string(),
        n_vectors: n,
        dimension: dim,
        complex_dim,
        queries_evaluated: queries.len(),
        k: K_NEIGHBORS,
        p50_ns,
        p95_ns,
        p99_ns,
        max_ns,
        p50_ms,
        p95_ms,
        p99_ms,
        max_ms,
        qps,
        bytes_read_per_query: bytes_per_query,
        vectors_scored_per_query: n,
        cpu_architecture: "x86_64".to_string(),
        compiler: "rustc 1.85+".to_string(),
        target: "x86_64-pc-windows-msvc".to_string(),
        simd_features: "AVX2/FMA/Dual-Acc Complex".to_string(),
    }
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════════════╗");
    println!("║       HOLOSPHERE PERFORMANCE TRACK P0: FROZEN EXACT SIMD ORACLE BASELINE    ║");
    println!("║       (SIFT1M 128D & GloVe-100 100D, 500 Queries, Single SIMD Scorer)       ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════╝");

    let sift1m_base =
        PathBuf::from("benchmark_databases/million_sift1m_strict_v6_pStrict_d64_n1000000.snapshot");
    let sift1m_query = PathBuf::from("datasets/sift_1m/sift1m_query.fvecs");

    let glove100_base = PathBuf::from(
        "benchmark_databases/million_glove100_strict_v6_pStrict_d50_n1183514.snapshot",
    );
    let glove100_query = PathBuf::from("datasets/glove_100/glove100_query.fvecs");

    let mut records = Vec::new();

    if sift1m_base.exists() && sift1m_query.exists() {
        records.push(benchmark_oracle_dataset(
            "Texmex SIFT1M (1,000,000 Vectors, 128-Dim)",
            &sift1m_base,
            &sift1m_query,
            128,
        ));
    } else {
        eprintln!("Skipping SIFT1M: snapshot or query file not found");
    }

    if glove100_base.exists() && glove100_query.exists() {
        records.push(benchmark_oracle_dataset(
            "GloVe-100 (1,183,514 Vectors, 100-Dim)",
            &glove100_base,
            &glove100_query,
            100,
        ));
    } else {
        eprintln!("Skipping GloVe-100: snapshot or query file not found");
    }

    // Write frozen performance-baseline-v1.json
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let out_dir = manifest_dir.join("benches").join("baseline");
    let _ = std::fs::create_dir_all(&out_dir);
    let baseline_path = out_dir.join("performance-baseline-v1.json");
    let json_bytes = serde_json::to_string_pretty(&records).expect("serialize baseline");
    let mut file = File::create(&baseline_path).expect("create baseline file");
    file.write_all(json_bytes.as_bytes())
        .expect("write baseline");

    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!("  🏆 FROZEN P0 EXACT ORACLE BASELINE SUMMARY TABLE");
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!(
        "  {:<32} │ {:>10} │ {:>6} │ {:>10} │ {:>10} │ {:>10} │ {:>8}",
        "Dataset", "N Vectors", "Dim", "p50 Lat", "p95 Lat", "p99 Lat", "QPS"
    );
    println!("  {}", "─".repeat(95));
    for r in &records {
        println!(
            "  {:<32} │ {:>10} │ {:>6} │ {:>8.2} ms │ {:>8.2} ms │ {:>8.2} ms │ {:>8.1}",
            r.dataset_name, r.n_vectors, r.dimension, r.p50_ms, r.p95_ms, r.p99_ms, r.qps
        );
    }
    println!("═══════════════════════════════════════════════════════════════════════════════");
    println!(
        "[Artifact Frozen] Saved clean baseline to: {}\n",
        baseline_path.display()
    );
}
