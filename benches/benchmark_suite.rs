//! Snapshot-backed retrieval benchmark over the checked-in public SIFT dataset.
//!
//! This target deliberately measures attachment and query execution only. Index
//! construction belongs to `hnsqr_build_bench_db`, never to a benchmark process.

mod common;

use std::time::Instant;

use common::{BenchScale, DEFAULT_BENCH_SEED, open_prebuilt_snapshot_v2};
use hnsqr::rivero::RiveroProfile;
use hnsqr::storage::snapshot::SnapshotOpenOptions;
use hnsqr::vector::quantization::PolarQuantizedVector;
use hnsqr::{HNSQRIndex, NodeIndex, VectorEmbedding};

fn exact_top_k(corpus: &[VectorEmbedding], query: &VectorEmbedding, k: usize) -> Vec<NodeIndex> {
    let mut scored: Vec<(NodeIndex, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(slot, vector)| (slot as NodeIndex, query.dot_product_complex(vector).re))
        .collect();
    scored.sort_unstable_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    scored.into_iter().take(k).map(|(slot, _)| slot).collect()
}

fn main() {
    const K: usize = 10;
    let (snapshot_path, dataset) =
        open_prebuilt_snapshot_v2(BenchScale::Dev, RiveroProfile::Balanced, DEFAULT_BENCH_SEED);
    assert!(
        !dataset.folded_queries.is_empty(),
        "real benchmark query set is empty"
    );

    println!("HoloSphere snapshot-backed real-dataset benchmark");
    println!("  snapshot: {}", snapshot_path.display());
    println!(
        "  corpus: {} SIFT vectors ({} real / {} complex dimensions)",
        dataset.folded_corpus.len(),
        dataset.real_dim,
        dataset.complex_dim
    );

    let attach_started = Instant::now();
    let index = HNSQRIndex::open_snapshot_v2(&snapshot_path, SnapshotOpenOptions::default())
        .expect("prebuilt benchmark snapshot must open");
    let attach_elapsed = attach_started.elapsed();
    assert_eq!(index.size(), dataset.folded_corpus.len());
    assert_eq!(index.dimension(), dataset.complex_dim);
    println!("  snapshot attachment: {:.2?}", attach_elapsed);

    let queries = &dataset.folded_queries;
    let started = Instant::now();
    let mut recall_sum = 0.0f64;
    for query in queries {
        let expected = exact_top_k(&dataset.folded_corpus, query, K);
        let results = index
            .search_indices_strict(query, K, None)
            .expect("snapshot query must succeed");
        let found = results
            .0
            .iter()
            .filter(|(slot, _)| expected.contains(slot))
            .count();
        recall_sum += found as f64 / K as f64;
    }
    let elapsed = started.elapsed();
    let recall = recall_sum / queries.len() as f64;
    println!(
        "  strict recall@{K}: {:.4}; {:.1} us/query; {:.0} QPS",
        recall,
        elapsed.as_secs_f64() * 1_000_000.0 / queries.len() as f64,
        queries.len() as f64 / elapsed.as_secs_f64()
    );

    let quantization_started = Instant::now();
    let mut mean_error = 0.0f64;
    let mut comparisons = 0usize;
    for query in queries.iter().take(32) {
        let quantized = PolarQuantizedVector::quantize(query.complex_data());
        for vector in dataset.folded_corpus.iter().take(128) {
            let exact = query.projective_overlap(vector);
            let approximate = quantized
                .asymmetric_dot_product(vector.complex_data())
                .norm_sqr()
                / (query.norm_squared() * vector.norm_squared()).max(1e-12);
            mean_error += (exact - approximate.clamp(0.0, 1.0)).abs() as f64;
            comparisons += 1;
        }
    }
    println!(
        "  PQ-C mean projective-overlap error: {:.6} ({} real comparisons in {:.2?})",
        mean_error / comparisons as f64,
        comparisons,
        quantization_started.elapsed()
    );
}
