/* holosphere/src/bin/hnsqr_build_bench_db.rs */
//!▫~•◦-------------------------------‣
//! # Benchmark Database & Snapshot Generator CLI
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Builds immutable, dataset-backed vector and graph snapshots for reproducible
//! performance and certification benchmark suites.
//!
//! ## Key Capabilities
//! - **Bulk Territory Construction:** Pre-indexes vector datasets using RiveroBulkBuilder.
//! - **Multi-Metric Support:** Emits certified index files across Cosine, Dot Product, and Euclidean metrics.
//! - **Artifact Specialization:** Generates .hnsqr snapshots and standalone frozen index files.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

use hnsqr::rivero::{RiveroBulkBuilder, RiveroProfile, VectorGeometry};
use hnsqr::vector::folding::ComplexWeaver;
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, VectorEmbedding};

const CACHE_VERSION: u32 = 6;

#[derive(Clone, Copy)]
enum ArtifactKind {
    Snapshot,
    Index,
}

fn usage() -> ! {
    eprintln!(
        "Usage: hnsqr_build_bench_db [--kind snapshot|index] [--algorithm rivero|hnsw] [--tag NAME] [--vectors N] [--queries N] [--source-dim D] [--index-dim D] [--profile fast|balanced|strict] [--m M] [--m0 M0] [--ef-construction EFC] [--output DIR]"
    );
    eprintln!(
        "\nExamples:\n  cargo run --release --bin hnsqr_build_bench_db -- --vectors 5000 --queries 64 --source-dim 128 --profile balanced\n  cargo run --release --bin hnsqr_build_bench_db -- --kind index --algorithm hnsw --tag hnsw_sweep --vectors 5000 --source-dim 128 --m 16 --m0 32 --ef-construction 200"
    );
    std::process::exit(2);
}

fn read_fvecs(path: &Path, limit: usize) -> std::io::Result<(Vec<VectorEmbedding>, usize)> {
    let mut file = File::open(path)?;
    let mut dim_buf = [0; 4];
    let mut dimension = 0;
    let mut vectors = Vec::with_capacity(limit);
    while vectors.len() < limit && file.read_exact(&mut dim_buf).is_ok() {
        let current_dim = u32::from_le_bytes(dim_buf) as usize;
        dimension = if dimension == 0 {
            current_dim
        } else {
            dimension
        };
        let mut bytes = vec![0; current_dim * 4];
        file.read_exact(&mut bytes)?;
        let raw: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let norm = raw
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt()
            .max(1e-9);
        let normalized: Vec<f32> = raw.iter().map(|value| value / norm).collect();
        vectors.push(ComplexWeaver::fold_llm_embedding(&normalized));
    }
    Ok((vectors, dimension))
}

fn dataset_path(dim: usize) -> PathBuf {
    let root = Path::new("datasets");
    match dim {
        25 => root.join("glove_25/glove25_base.fvecs"),
        50 => root.join("glove_50/glove50_base.fvecs"),
        100 => root.join("glove_100/glove100_base.fvecs"),
        128 if root.join("sift_1m/sift1m_base.fvecs").exists() => {
            root.join("sift_1m/sift1m_base.fvecs")
        }
        128 => root.join("siftsmall/siftsmall_base.fvecs"),
        512 => root.join("clip_512/clip_base.fvecs"),
        768 if root
            .join("cohere_768_large/cohere_100k_base.fvecs")
            .exists() =>
        {
            root.join("cohere_768_large/cohere_100k_base.fvecs")
        }
        768 => root.join("cohere_768/cohere_base.fvecs"),
        1536 if root.join("openai_1536_large/openai_1m_base.fvecs").exists() => {
            root.join("openai_1536_large/openai_1m_base.fvecs")
        }
        1536 => root.join("openai_1536/openai_base.fvecs"),
        4096 => root.join("arxiv_4096/database_vectors.fvecs"),
        _ => panic!("no checked-in dataset mapping for {dim} dimensions"),
    }
}

fn main() {
    let mut kind = ArtifactKind::Snapshot;
    let mut algorithm = "rivero".to_string();
    let mut tag = None;
    let mut vectors = 5_000usize;
    let mut source_dim = 128usize;
    let mut index_dim = None;
    let mut profile = RiveroProfile::Balanced;
    let mut m = 16usize;
    let mut m0 = 32usize;
    let mut ef_construction = 128usize;
    let mut output = std::env::var_os("HNSQR_BENCH_DATABASE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("benchmark_databases"));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).unwrap_or_else(|| usage());
        match args[i].as_str() {
            "--help" | "-h" => usage(),
            "--kind" => {
                kind = match value.as_str() {
                    "snapshot" => ArtifactKind::Snapshot,
                    "index" => ArtifactKind::Index,
                    _ => usage(),
                }
            }
            "--algorithm" => algorithm = value.to_ascii_lowercase(),
            "--tag" => tag = Some(value.clone()),
            "--vectors" => vectors = value.parse().unwrap_or_else(|_| usage()),
            "--queries" => {} // Queries are read by the benchmark; retained for scriptable invocations.
            "--source-dim" => source_dim = value.parse().unwrap_or_else(|_| usage()),
            "--index-dim" => index_dim = Some(value.parse().unwrap_or_else(|_| usage())),
            "--m" => m = value.parse().unwrap_or_else(|_| usage()),
            "--m0" => m0 = value.parse().unwrap_or_else(|_| usage()),
            "--ef-construction" | "--efc" => {
                ef_construction = value.parse().unwrap_or_else(|_| usage())
            }
            "--profile" => {
                profile = match value.to_ascii_lowercase().as_str() {
                    "fast" => RiveroProfile::Fast,
                    "balanced" => RiveroProfile::Balanced,
                    "strict" => RiveroProfile::Strict,
                    _ => usage(),
                }
            }
            "--output" => output = PathBuf::from(value),
            _ => usage(),
        }
        i += 2;
    }

    let source = dataset_path(source_dim);
    let (corpus, loaded_dim) = read_fvecs(&source, vectors).expect("failed to read dataset");
    assert!(
        !corpus.is_empty(),
        "dataset {} contained no vectors",
        source.display()
    );

    // If the dataset file contains fewer vectors than requested, cap the build to what exists.
    if corpus.len() < vectors {
        eprintln!(
            "warning: requested {} vectors, but '{}' only supplies {} — \
             building snapshot with {} vectors",
            vectors,
            source.display(),
            corpus.len(),
            corpus.len()
        );
        vectors = corpus.len();
    }

    fs::create_dir_all(&output).expect("failed to create benchmark database directory");
    let is_hnsw = algorithm == "hnsw";
    let rivero_enabled = !is_hnsw && matches!(kind, ArtifactKind::Snapshot);

    let (filename, dimension) = match kind {
        ArtifactKind::Snapshot => {
            if is_hnsw {
                (
                    format!("hnsw-v{CACHE_VERSION}-n{vectors}-m{m}-efc{ef_construction}-d{loaded_dim}.snapshot"),
                    loaded_dim.div_ceil(2),
                )
            } else {
                (
                    format!("snapshot-v{CACHE_VERSION}-n{vectors}-p{profile:?}-d{loaded_dim}.hnsqr"),
                    loaded_dim.div_ceil(2),
                )
            }
        }
        ArtifactKind::Index => {
            let tag = tag.unwrap_or_else(|| usage());
            let dim = index_dim.unwrap_or_else(|| loaded_dim.div_ceil(2));
            if is_hnsw {
                (
                    format!("{tag}_v{CACHE_VERSION}_m{m}_efc{ef_construction}_d{dim}_n{vectors}.snapshot"),
                    dim,
                )
            } else {
                (
                    format!("{tag}_v{CACHE_VERSION}_p{profile:?}_d{dim}_n{vectors}.snapshot"),
                    dim,
                )
            }
        }
    };
    let destination = output.join(filename);
    if destination.exists() {
        panic!(
            "refusing to overwrite immutable benchmark database: {}",
            destination.display()
        );
    }

    let mut config = if rivero_enabled {
        HNSQRConfig::strict_rivero_for_dim(dimension)
    } else {
        HNSQRConfig::default()
    };
    config.distance_function = DistanceFunction::Cosine;
    config.max_elements = vectors + 10_000;
    config.rivero_enabled = rivero_enabled;
    config.rivero_fallback_on_underfill = false;
    config.rivero_witness_degree = 32;
    config.ef_construction = ef_construction;
    config.m = m;
    config.m0 = m0;
    config.rivero_address_config.geometry = VectorGeometry::Real;

    let index = HNSQRIndex::new(config, dimension);
    for (slot, vector) in corpus.iter().enumerate() {
        index
            .insert(format!("doc_{slot}"), vector.clone())
            .expect("index insertion failed");
    }

    if rivero_enabled {
        let builder = RiveroBulkBuilder::with_profile(profile)
            .with_address_config(index.config().rivero_address_config)
            .with_distance_function(index.config().distance_function)
            .with_witness_params(32, 16, 8);
        index
            .install_rivero_state(builder.build(&corpus).expect("Rivero build failed"))
            .expect("Rivero installation failed");
        index.freeze_rivero_routing();
    }

    index
        .save_snapshot_v2(&destination)
        .expect("snapshot write failed");
    println!(
        "Built immutable benchmark database from {}: {}",
        source.display(),
        destination.display()
    );
}
