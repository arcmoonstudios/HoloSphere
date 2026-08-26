mod common;

use std::time::Instant;

use hnsqr::proof::lutz::SemanticRerankPlan;
use hnsqr::storage::segment::SegmentedEngine;
use hnsqr::{NodeIndex, SimilarityScore};

fn percentile(mut latencies: Vec<f64>, p: f64) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let idx = ((latencies.len() as f64 - 1.0) * (p / 100.0)).round() as usize;
    latencies[idx]
}

fn evaluate_ann_benchmarks(real_dim: usize, complex_dim: usize, n: usize, num_queries: usize) {
    let k = 10;
    let (base_path, query_path, _) = common::find_best_matching_dataset(real_dim);
    let (folded_corpus, _) = common::read_fvecs(&base_path, Some(n))
        .unwrap_or_else(|_| panic!("failed to load {}", base_path.display()));
    let (folded_queries, _) = common::read_fvecs(&query_path, Some(num_queries))
        .unwrap_or_else(|_| panic!("failed to load {}", query_path.display()));
    assert!(
        !folded_corpus.is_empty(),
        "dataset '{}' is missing or empty",
        base_path.display()
    );

    // Compute ground-truth exhaustive top-k
    let mut ground_truth = Vec::with_capacity(num_queries);
    for query in &folded_queries {
        let mut exhaustive: Vec<(NodeIndex, SimilarityScore)> = folded_corpus
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx as NodeIndex, (query.dot_product_complex(doc)).re))
            .collect();
        exhaustive.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let topk: Vec<NodeIndex> = exhaustive.into_iter().take(k).map(|(id, _)| id).collect();
        ground_truth.push(topk);
    }

    // Benchmark HNSQR Segmented Engine: Rivero candidate generation + ExactSimd candidate rerank
    // (Note: In SegmentedEngine, ExactSimd performs exact candidate reranking over Rivero proposals, not exhaustive scan)
    let engine = SegmentedEngine::new(complex_dim, 2048);
    let t_build = Instant::now();
    for (i, v) in folded_corpus.iter().enumerate() {
        engine.insert(format!("node_{i}"), v.clone()).unwrap();
    }
    let build_time_s = t_build.elapsed().as_secs_f64();
    let build_vec_per_sec = (n as f64) / build_time_s.max(1e-6);

    let mut latencies = Vec::with_capacity(num_queries);
    let mut total_hits = 0usize;

    for (q_idx, query) in folded_queries.iter().enumerate() {
        let t0 = Instant::now();
        let topk = engine.search(query, k, SemanticRerankPlan::ExactSimd);
        latencies.push(t0.elapsed().as_secs_f64() * 1_000_000.0);

        let gt = &ground_truth[q_idx];
        for (res_id, _) in &topk {
            let s: &str = res_id.as_ref();
            if let Ok(id_num) = s.strip_prefix("node_").unwrap_or("").parse::<NodeIndex>()
                && gt.contains(&id_num)
            {
                total_hits += 1;
            }
        }
    }

    let recall_at_10 = (total_hits as f64) / ((num_queries * k) as f64);
    let p50 = percentile(latencies.clone(), 50.0);
    let p95 = percentile(latencies.clone(), 95.0);
    let p99 = percentile(latencies.clone(), 99.0);
    let qps = 1_000_000.0 / p50.max(0.1);

    println!(
        "  │ {:>6} │ {:>6} │ {:>12.0} v/s │ {:>10.4} │ {:>7.1} µs │ {:>7.1} µs │ {:>7.1} µs │ {:>9.0} QPS│",
        real_dim, n, build_vec_per_sec, recall_at_10, p50, p95, p99, qps
    );
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR STANDARDIZED ANN-BENCHMARKS HARNESS EVALUATION                                 ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    println!(
        "  ┌────────┬────────┬────────────────┬────────────┬──────────┬──────────┬──────────┬──────────────┐"
    );
    println!(
        "  │ Real D │ Corpus │ Build Rate     │ Recall@10  │ Lat (p50)│ Lat (p95)│ Lat (p99)│ Throughput   │"
    );
    println!(
        "  ├────────┼────────┼────────────────┼────────────┼──────────┼──────────┼──────────┼──────────────┤"
    );

    if cfg!(debug_assertions) {
        evaluate_ann_benchmarks(128, 64, 100, 4);
    } else {
        evaluate_ann_benchmarks(128, 64, 5000, 32);
        evaluate_ann_benchmarks(768, 384, 5000, 32);
        evaluate_ann_benchmarks(1536, 768, 5000, 32);
        evaluate_ann_benchmarks(4096, 2048, 2000, 16);
    }

    println!(
        "  └────────┴────────┴────────────────┴────────────┴──────────┴──────────┴──────────┴──────────────┘\n"
    );
}
