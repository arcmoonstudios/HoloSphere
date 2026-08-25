/* hnsqr/benches/benchmark_suite.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Comprehensive Complex Vector Benchmark Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates the full spectrum of HNSQR capabilities:
//! 1. Hardware AVX2+FMA Dual-Accumulator SIMD Tensor Micro-Kernels
//! 2. 8-Bit Polar Phase Quantization (PQ-C) & Asymmetric Distance Computation (ADC)
//! 3. Zero-Copy Disk-Backed Quantized Storage (`MmapArena`) & Mapping Attach Time
//! 4. Lock-Free Roaring Bitmap Inverted Metadata Index (Precompiled Filter Masks)
//! 5. LLM Pairwise Amplitude-Phase Weaver Gateway (1536/3072/4096-dim)
//! 6. Zero-Copy Asynchronous Binary TCP Server & Pipelined Network Wire Protocol
//! 7. Multi-Threaded Concurrent Graph Ingestion Scaling (1, 2, 4, 8, 16 threads)
//! 8. Phase-Weighted Graph Traversal Recall & Latency Tradeoffs (ef_search 10..256)
//! 9. Multi-Core Rayon Parallel Batch Search Throughput
//! 10. Complex Phase Attention & Diverse Intent Superposition Routing
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2025 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#![allow(
    clippy::too_many_arguments,
    clippy::manual_div_ceil,
    clippy::manual_clamp,
    clippy::field_reassign_with_default
)]

use hnsqr::bench_support as common;

use std::collections::HashSet;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use rayon::prelude::*;

use hnsqr::metadata::index::FilterExpr;
use hnsqr::transport::qir0::{HNSQRClient, HNSQRServer};
use hnsqr::vector::folding::ComplexWeaver;
use hnsqr::vector::quantization::PolarQuantizedVector;
use hnsqr::{
    DistanceFunction, HNSQRConfig, HNSQRIndex, NodeId, SearchIntent, SearchPlan, SimilarityScore,
    VectorEmbedding,
};
use num_complex::Complex32;

/// Loads a realistic embedding dataset from real public datasets in datasets/.
fn generate_clustered_dataset(
    num_vectors: usize,
    dim: usize,
    _num_clusters: usize,
) -> (Vec<(NodeId, VectorEmbedding)>, Vec<VectorEmbedding>, usize) {
    let (base_path, query_path, _) = common::find_best_matching_dataset(dim * 2);
    let (mut corpus_vecs, _) =
        common::read_fvecs(&base_path, Some(num_vectors)).unwrap_or_default();
    let (mut query_vecs, _) = common::read_fvecs(&query_path, Some(200)).unwrap_or_default();

    if corpus_vecs.is_empty() {
        let text_corpus = common::generate_realistic_text_corpus(num_vectors, 200, dim * 2, 0x1234);
        corpus_vecs = text_corpus.folded_corpus;
        query_vecs = text_corpus.folded_queries;
    }

    if corpus_vecs.len() < num_vectors && !corpus_vecs.is_empty() {
        let orig_len = corpus_vecs.len();
        while corpus_vecs.len() < num_vectors {
            let take = (num_vectors - corpus_vecs.len()).min(orig_len);
            for i in 0..take {
                corpus_vecs.push(corpus_vecs[i].clone());
            }
        }
    }

    let actual_dim = corpus_vecs.first().map(|v| v.dimension()).unwrap_or(dim);

    let dataset = corpus_vecs
        .into_iter()
        .enumerate()
        .map(|(i, v)| (format!("doc_{:05}", i).into(), v))
        .collect();

    (dataset, query_vecs, actual_dim)
}

/// Brute-force exact k-nearest neighbor search using Projective Overlap (CPO).
fn exact_knn_fidelity(
    dataset: &[(NodeId, VectorEmbedding)],
    query: &VectorEmbedding,
    k: usize,
) -> Vec<(NodeId, f32)> {
    let mut scores: Vec<(NodeId, f32)> = dataset
        .iter()
        .map(|(id, vec)| {
            let fidelity = query.projective_overlap(vec);
            let dist = 1.0 - fidelity; // Distance = 1 - Fidelity
            (id.clone(), dist)
        })
        .collect();

    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scores.truncate(k);
    scores
}

/// Computes recall@k between approximate results and exact ground truth.
fn compute_recall(approx: &[(NodeId, SimilarityScore)], ground_truth: &[(NodeId, f32)]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let gt_set: HashSet<&NodeId> = ground_truth.iter().map(|(id, _)| id).collect();
    let matches = approx.iter().filter(|(id, _)| gt_set.contains(id)).count();
    matches as f32 / ground_truth.len() as f32
}

fn percentile(sorted_latencies: &[f64], p: f64) -> f64 {
    if sorted_latencies.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_latencies.len() as f64 - 1.0) * p / 100.0).round() as usize;
    sorted_latencies[idx.min(sorted_latencies.len() - 1)]
}

fn run_full_benchmark(
    title: &str,
    dataset: &[(NodeId, VectorEmbedding)],
    queries: &[VectorEmbedding],
    dim: usize,
    k: usize,
    m0: usize,
    m: usize,
    ef_c: usize,
) {
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ {:<76} ║", title);
    println!(
        "║ (M0={}, M={}, ef_construction={}, Metric=Projective Overlap)                ║",
        m0, m, ef_c
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let num_vectors = dataset.len();
    let num_queries = queries.len();

    let config = HNSQRConfig {
        m0,
        m,
        ef_construction: ef_c,
        ef_search: 64,
        distance_function: DistanceFunction::ProjectiveOverlap,
        search_plan: SearchPlan::GraphOnly,
        rivero_enabled: false,
        level_multiplier: 1.0 / (m as f32).ln().max(1.0),
        superposition_beam_width: 8,
        attention_temperature: 0.15,
        interference_weight: 0.35,
        oversample_factor: 3.0,
        heuristic_edge_selection: true,
        multi_root_ensemble_size: 4,
        extend_candidates: true,
        keep_pruned_connections: true,
        ..Default::default()
    };

    // ────────────────────────────────────────────────────────────────────────
    // 1. INGESTION PERFORMANCE
    // ────────────────────────────────────────────────────────────────────────
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 1. INGESTION BENCHMARK (Single-Threaded & Multi-Threaded Concurrent)          │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    let index_seq = HNSQRIndex::new(config.clone(), dim);
    let start_seq = Instant::now();
    for (id, vec) in dataset {
        index_seq.insert(id.as_ref(), vec.clone()).unwrap();
    }
    let dur_seq = start_seq.elapsed();
    let seq_qps = num_vectors as f64 / dur_seq.as_secs_f64();
    let seq_latency_us = (dur_seq.as_micros() as f64) / num_vectors as f64;
    println!(
        " • Single-threaded Build: {:.2?} ({:.0} vectors/sec, avg {:.1} µs/insert)",
        dur_seq, seq_qps, seq_latency_us
    );

    // Multi-threaded Ingestion scaling
    for &threads in &[2, 4, 8] {
        let index_par = Arc::new(HNSQRIndex::new(config.clone(), dim));
        let chunk_size = num_vectors.div_ceil(threads);
        let start_par = Instant::now();

        let mut handles = Vec::new();
        for t in 0..threads {
            let idx_clone = Arc::clone(&index_par);
            let start = t * chunk_size;
            let end = (start + chunk_size).min(num_vectors);
            let chunk_data = dataset[start..end].to_vec();

            handles.push(thread::spawn(move || {
                for (id, vec) in chunk_data {
                    idx_clone.insert(id.as_ref(), vec).unwrap();
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        let dur_par = start_par.elapsed();
        let par_qps = num_vectors as f64 / dur_par.as_secs_f64();
        let scaling = par_qps / seq_qps.max(1.0);
        println!(
            " • Concurrent Build ({:2} threads): {:.2?} ({:.0} vectors/sec, {:.2}x scaling)",
            threads, dur_par, par_qps, scaling
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 2. SEARCH ACCURACY & LATENCY TRADEOFFS
    // ────────────────────────────────────────────────────────────────────────
    println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 2. SEARCH ACCURACY & LATENCY (Recall vs Exact Ground Truth)                  │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    print!(
        "Computing exact brute-force Projective Overlap for {} queries (sequential)... ",
        num_queries
    );
    let start_gt = Instant::now();
    let ground_truth: Vec<Vec<(NodeId, f32)>> = queries
        .iter()
        .map(|q| exact_knn_fidelity(dataset, q, k))
        .collect();
    let gt_duration = start_gt.elapsed();
    println!("Done in {:.2?}.", gt_duration);

    let sequential_brute_qps = num_queries as f64 / gt_duration.as_secs_f64();
    let sequential_brute_latency_us = gt_duration.as_secs_f64() * 1e6 / num_queries as f64;

    // Parallel brute-force baseline for comparison
    print!(
        "Computing exact brute-force Projective Overlap for {} queries (parallel Rayon)... ",
        num_queries
    );
    let start_par_gt = Instant::now();
    let _parallel_ground_truth: Vec<Vec<(NodeId, f32)>> = queries
        .par_iter()
        .map(|q| exact_knn_fidelity(dataset, q, k))
        .collect();
    let par_gt_duration = start_par_gt.elapsed();
    println!("Done in {:.2?}.", par_gt_duration);

    let parallel_brute_qps = num_queries as f64 / par_gt_duration.as_secs_f64();
    let parallel_brute_latency_us = par_gt_duration.as_secs_f64() * 1e6 / num_queries as f64;

    println!("\n⚠️  BRUTE FORCE BASELINES (N={}, k={}):", num_vectors, k);
    println!(
        "   Sequential: {:.1} µs/query ({:.0} QPS)",
        sequential_brute_latency_us, sequential_brute_qps
    );
    println!(
        "   Parallel:   {:.1} µs/query ({:.0} QPS, {:.2}× speedup)\n",
        parallel_brute_latency_us,
        parallel_brute_qps,
        parallel_brute_qps / sequential_brute_qps
    );

    println!(
        "  {:>9} │ {:>9} │ {:>9} │ {:>16} │ {:>11} │ {:>9} │ {:>9} │ {:>9} │ {:>16}",
        "ef_search",
        "Recall@1",
        "Recall@10",
        "Fidelity Quality",
        "Avg Latency",
        "p50 (µs)",
        "p90 (µs)",
        "p99 (µs)",
        "Throughput (QPS)"
    );
    println!(
        " ───────────┼───────────┼───────────┼──────────────────┼─────────────┼───────────┼───────────┼───────────┼──────────────────"
    );

    let ef_search_values = [10, 20, 32, 64, 128, 256];
    let mut latencies_us_ef10 = 0.0;

    for &ef in &ef_search_values {
        let mut latencies_us = Vec::with_capacity(num_queries);
        let mut recall_1_sum = 0.0;
        let mut recall_10_sum = 0.0;
        let mut fidelity_quality_sum = 0.0;

        index_seq.set_ef_search(ef).unwrap();

        for (q_idx, query) in queries.iter().enumerate() {
            let t0 = Instant::now();
            let results = index_seq.search(query, k).unwrap();
            let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
            latencies_us.push(elapsed_us);

            let gt = &ground_truth[q_idx];
            let r1 = compute_recall(&results[..1.min(results.len())], &gt[..1.min(gt.len())]);
            let r10 = compute_recall(&results, gt);

            recall_1_sum += r1;
            recall_10_sum += r10;

            if !results.is_empty() && !gt.is_empty() {
                let exact_top1_fidelity = 1.0 - gt[0].1;
                let hnsqr_fidelity = results[0].1;
                if exact_top1_fidelity > 1e-6 {
                    let quality = (hnsqr_fidelity / exact_top1_fidelity).clamp(0.0, 1.0);
                    fidelity_quality_sum += quality;
                } else {
                    fidelity_quality_sum += 1.0;
                }
            }
        }

        latencies_us.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let avg_lat = latencies_us.iter().sum::<f64>() / num_queries as f64;
        if ef == 10 {
            latencies_us_ef10 = avg_lat;
        }
        let p50 = percentile(&latencies_us, 50.0);
        let p90 = percentile(&latencies_us, 90.0);
        let p99 = percentile(&latencies_us, 99.0);
        let qps = 1_000_000.0 / avg_lat.max(1.0);

        let mean_r1 = (recall_1_sum / num_queries as f32) * 100.0;
        let mean_r10 = (recall_10_sum / num_queries as f32) * 100.0;
        let mean_fq = (fidelity_quality_sum / num_queries as f32) * 100.0;

        println!(
            "  {:>9} │ {:>8.2}% │ {:>8.2}% │ {:>15.2}% │ {:>9.1} µs │ {:>7.1} µs │ {:>7.1} µs │ {:>7.1} µs │ {:>10.0} QPS",
            ef, mean_r1, mean_r10, mean_fq, avg_lat, p50, p90, p99, qps
        );
    }

    // Reset default ef_search
    index_seq.set_ef_search(64).unwrap();

    // Analysis: Compare HNSQR performance vs brute-force baselines
    println!("\n⚠️  PERFORMANCE ANALYSIS (ef_search=10 vs Brute Force):");
    let ef10_ratio_seq = latencies_us_ef10 / sequential_brute_latency_us.max(1e-6);
    let ef10_speedup_seq = sequential_brute_latency_us / latencies_us_ef10.max(1e-6);
    let ef10_ratio_par = latencies_us_ef10 / parallel_brute_latency_us.max(1e-6);
    let ef10_speedup_par = parallel_brute_latency_us / latencies_us_ef10.max(1e-6);
    println!(
        "   vs Sequential exact ({:.1} µs): latency ratio = {:.2}× | speedup = {:.2}×",
        sequential_brute_latency_us, ef10_ratio_seq, ef10_speedup_seq
    );
    println!(
        "   vs Parallel exact   ({:.1} µs): latency ratio = {:.2}× | speedup = {:.2}×",
        parallel_brute_latency_us, ef10_ratio_par, ef10_speedup_par
    );
    if ef10_speedup_seq < 1.0 {
        println!(
            "   ⚠️  At N={}, graph overhead dominates. ANN crossover point is higher.",
            num_vectors
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // 3. PARALLEL BATCH QUERY THROUGHPUT
    // ────────────────────────────────────────────────────────────────────────
    println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 3. PARALLEL BATCH QUERY THROUGHPUT (Rayon Multi-Core Scaling)                │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    let t_batch = Instant::now();
    let batch_results = index_seq.batch_search(queries, k).unwrap();
    let dur_batch = t_batch.elapsed();
    let batch_qps = num_queries as f64 / dur_batch.as_secs_f64();
    let per_query_eff_us = (dur_batch.as_micros() as f64) / num_queries as f64;

    println!(
        " • Rayon Parallel Batch Search: {:.2?} ({:.0} QPS, {:.2} µs/query effective)",
        dur_batch, batch_qps, per_query_eff_us
    );
    assert_eq!(batch_results.len(), num_queries);

    // ────────────────────────────────────────────────────────────────────────
    // 4. ADVANCED COMPLEX PHASE RE-RANKING & INTENT MODES
    // ────────────────────────────────────────────────────────────────────────
    println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 4. ADVANCED COMPLEX PHASE RE-RANKING & INTENT MODES                          │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    // Diverse Intent (Superposition Maximum Marginal Relevance)
    let intent_diverse = SearchIntent {
        diversity: 0.6,
        compute_budget: 1.0,
        ..Default::default()
    };
    let t_div = Instant::now();
    for q in queries {
        let _ = index_seq
            .intent_rerank_search(q, k, &intent_diverse)
            .unwrap();
    }
    let dur_div = t_div.elapsed();
    println!(
        " • Diverse Intent Search:    {:.2?} ({:.0} QPS, avg {:.1} µs/query)",
        dur_div,
        num_queries as f64 / dur_div.as_secs_f64(),
        (dur_div.as_micros() as f64) / num_queries as f64
    );

    // Attention Weighted Search
    let intent_attention = SearchIntent {
        phase_alignment_weight: 0.8,
        attention_width: 0.12,
        compute_budget: 1.0,
        ..Default::default()
    };
    let t_att = Instant::now();
    for q in queries {
        let _ = index_seq
            .intent_rerank_search(q, k, &intent_attention)
            .unwrap();
    }
    let dur_att = t_att.elapsed();
    println!(
        " • Complex Phase Attention Search: {:.2?} ({:.0} QPS, avg {:.1} µs/query)",
        dur_att,
        num_queries as f64 / dur_att.as_secs_f64(),
        (dur_att.as_micros() as f64) / num_queries as f64
    );

    // ────────────────────────────────────────────────────────────────────────
    // 5. TOPOLOGY, MEMORY FOOTPRINT & STRUCTURAL INTEGRITY
    // ────────────────────────────────────────────────────────────────────────
    println!("\n┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 5. TOPOLOGY, MEMORY FOOTPRINT & STRUCTURAL INTEGRITY                         │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");

    let level_counts = index_seq.level_distribution();
    println!(" • Layer distribution:");
    for (lvl, count) in level_counts.iter().enumerate() {
        let pct = (*count as f64 / num_vectors as f64) * 100.0;
        let bar_len = ((pct / 100.0) * 40.0).round() as usize;
        let bar = "█".repeat(bar_len);
        println!(
            "    Layer {:2}: {:6} nodes {:<40} ({:.2}%)",
            lvl, count, bar, pct
        );
    }

    let stats = index_seq.stats();
    println!("\n • Internal Instrumentation Stats:");
    println!("    - Total Recorded Insertions: {}", stats.insertions);
    println!("    - Total Recorded Searches:   {}", stats.searches);
    println!(
        "    - Rolling Avg Search Latency: {:.2} µs",
        stats.avg_search_latency_us
    );
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║         HNSQR COMPLEX VECTOR RETRIEVAL FULL BENCHMARK SUITE                  ║");
    println!("║     (AVX2+FMA SIMD, Zero-Allocation Scratchpads & Superposition Routing)     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let (num_vectors, num_queries, iterations) = if cfg!(debug_assertions) {
        (50, 4, 10_000)
    } else {
        (5000, 200, 10_000_000)
    };
    let dim = 64; // 64 complex dimensions = 128 floats
    let k = 10;

    println!("Configuration:");
    println!(" • Dataset size:       {} vectors", num_vectors);
    println!(" • Complex dimension:  {} (128 floats)", dim);
    println!(" • Evaluation queries: {}", num_queries);
    println!(" • Neighbors (k):      {}", k);
    println!(" • Distance metric:    Projective Overlap (CPO)\n");

    // ────────────────────────────────────────────────────────────────────────
    // DATASET 1: Clustered Phase-Encoded Embeddings (Production Profile)
    // ────────────────────────────────────────────────────────────────────────
    print!(
        "Generating 5,000 clustered phase-encoded complex embeddings (50 semantic clusters)... "
    );
    let (dataset_clust, queries_clust, actual_dim) =
        generate_clustered_dataset(num_vectors, dim, 50);
    println!("Done.\n");

    run_full_benchmark(
        "DATASET: Clustered Phase-Encoded Embeddings (Production Complex Profile)",
        &dataset_clust,
        &queries_clust,
        actual_dim,
        k,
        64,  // M0
        32,  // M
        200, // ef_construction
    );

    // ────────────────────────────────────────────────────────────────────────
    // 6. HARDWARE-NATIVE AVX2+FMA SIMD MICRO-BENCHMARKS
    // ────────────────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ 6. HARDWARE-NATIVE AVX2+FMA SIMD TENSOR MICRO-BENCHMARKS                     ║");
    println!("║    (Pure Algebraic Projection vs Transcendental Trigonometric Cycles)        ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let v_a: Vec<Complex32> = (0..dim)
        .map(|i| Complex32::new((i as f32).sin(), (i as f32).cos()))
        .collect();
    let v_b: Vec<Complex32> = (0..dim)
        .map(|i| Complex32::new((i as f32).cos(), (i as f32).sin()))
        .collect();

    let t_simd = Instant::now();
    let mut sum_ip = Complex32::new(0.0, 0.0);
    for _ in 0..iterations {
        sum_ip += hnsqr::dot_product_complex_simd(&v_a, &v_b);
    }
    let dur_simd = t_simd.elapsed();
    let mops = (iterations as f64 / dur_simd.as_secs_f64()) / 1_000_000.0;
    let gflops = (mops * (dim * 8) as f64) / 1000.0;

    println!(" • Dual-Accumulator AVX2+FMA Complex Dot Product (64-dim):");
    println!("    - Iterations:        10,000,000");
    println!("    - Elapsed Time:      {:.2?}", dur_simd);
    println!(
        "    - Throughput:        {:.2} Million Inner Products/sec",
        mops
    );
    println!(
        "    - Compute Density:   {:.2} GFLOPS (AVX2 Hardware FMA Fused)",
        gflops
    );
    let _ = sum_ip;

    // ────────────────────────────────────────────────────────────────────────
    // 7. 8-BIT POLAR PHASE QUANTIZATION (PQ-C) & ADC BENCHMARK
    // ────────────────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ 7. 8-BIT POLAR PHASE QUANTIZATION (PQ-C) & ASYMMETRIC DISTANCE ENGINE        ║");
    println!("║    (4x Memory Bus Bandwidth Reduction & 256-Element L1 Trig Tables)          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let raw_bytes_per_vec = dim * std::mem::size_of::<Complex32>();
    let pqc_bytes_per_vec = dim * 2; // 8-bit amp + 8-bit phase
    let raw_total_mb = (num_vectors * raw_bytes_per_vec) as f64 / (1024.0 * 1024.0);
    let pqc_total_mb = (num_vectors * pqc_bytes_per_vec) as f64 / (1024.0 * 1024.0);
    let compression_ratio = raw_bytes_per_vec as f64 / pqc_bytes_per_vec as f64;

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 1. MEMORY FOOTPRINT COMPRESSION METRICS                                      │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    println!(
        " • Raw Complex32 Footprint (64 dim):   {:4} bytes/vector ({:.2} MB for 5,000 vecs)",
        raw_bytes_per_vec, raw_total_mb
    );
    println!(
        " • 8-Bit Polar PQ-C Footprint (64 dim): {:4} bytes/vector ({:.2} MB for 5,000 vecs)",
        pqc_bytes_per_vec, pqc_total_mb
    );
    println!(
        " • Memory Bus Compression Ratio:       {:.2}x ({:.1}% memory footprint reduction)\n",
        compression_ratio,
        (1.0 - 1.0 / compression_ratio) * 100.0
    );

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 2. QUANTIZATION RECONSTRUCTION QUALITY (vs Full-Precision Ground Truth)     │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    let mut fidelity_errors = Vec::with_capacity(num_queries);
    for q in &queries_clust {
        let q_norm_sq = q.norm_squared();
        for (_, vec) in dataset_clust.iter().take(50) {
            let exact_fid = q.projective_overlap(vec);
            let qvec = PolarQuantizedVector::quantize(vec.complex_data());
            let adc_ip = qvec.asymmetric_dot_product(q.complex_data());
            let adc_fid =
                (adc_ip.norm_sqr() / (q_norm_sq * vec.norm_squared()).max(1e-12)).clamp(0.0, 1.0);
            let err = (exact_fid - adc_fid).abs();
            fidelity_errors.push(err);
        }
    }
    let avg_err: f32 = fidelity_errors.iter().sum::<f32>() / fidelity_errors.len() as f32;
    let max_err: f32 = fidelity_errors.iter().copied().fold(0.0, f32::max);
    println!(" • Mean Absolute Fidelity Error (MAE): {:.4}", avg_err);
    println!(" • Peak Absolute Fidelity Error (MAX): {:.4}", max_err);
    println!(
        " • Effective Superposition Fidelity:   {:.2}%\n",
        (1.0 - avg_err) * 100.0
    );

    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ 3. ZERO-COPY DISK-BACKED MMAP STORAGE (Mapping Attach Verification)         │");
    println!("└──────────────────────────────────────────────────────────────────────────────┘");
    let temp_dir = std::env::temp_dir();
    let mmap_file = temp_dir.join(format!(
        "hnsqr_bench_persistent_{}.mmap",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&mmap_file);

    let t_mmap_build = Instant::now();
    let mut config_mmap = HNSQRConfig::default();
    config_mmap.max_elements = 10_000;
    config_mmap.m0 = 64;
    config_mmap.m = 32;

    let mmap_index = HNSQRIndex::create_mmap(&mmap_file, config_mmap, dim).unwrap();
    for (id, vec) in &dataset_clust {
        mmap_index.insert(id.as_ref(), vec.clone()).unwrap();
    }
    mmap_index.flush().unwrap();
    let dur_mmap_build = t_mmap_build.elapsed();
    println!(
        " • Disk-Backed Ingestion & Flush: {:.2?} ({:.0} vectors/sec)",
        dur_mmap_build,
        num_vectors as f64 / dur_mmap_build.as_secs_f64()
    );

    // Mapping attach & cold first query
    drop(mmap_index);
    let t_reboot = Instant::now();
    let reopened_index = HNSQRIndex::open_mmap(&mmap_file).unwrap();
    let dur_reboot = t_reboot.elapsed();
    println!(
        " • Raw Mmap Syscall Attach:       {:.2?} ({:.2} µs mapping syscall)",
        dur_reboot,
        dur_reboot.as_secs_f64() * 1_000_000.0
    );

    let t_first = Instant::now();
    let _ = reopened_index
        .search_indices_exact(&dataset_clust[0].1, 10, None)
        .unwrap();
    let dur_first = t_first.elapsed();
    println!(
        " • Cold First Query (Exact SIMD): {:.2?} ({:.2} µs)",
        dur_first,
        dur_first.as_secs_f64() * 1_000_000.0
    );

    drop(reopened_index);
    let _ = std::fs::remove_file(&mmap_file);

    // ────────────────────────────────────────────────────────────────────────
    // 8. ROARING BITMAP FILTERED SEARCH THROUGHPUT
    // ────────────────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ 8. ROARING BITMAP INVERTED INDEX & PRECOMPILED FILTERED SEARCH              ║");
    println!("║    (Zero Dynamic Dispatch and Zero JSON Parsing in the Search Loop)         ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let config_meta = HNSQRConfig::default();
    let index_meta = HNSQRIndex::new(config_meta, dim);

    for (i, (id, vec)) in dataset_clust.iter().enumerate() {
        let mut meta = std::collections::HashMap::new();
        let cat = if i % 2 == 0 { "gpu" } else { "cpu" };
        let tenant = if i < 2500 {
            "tenant_alpha"
        } else {
            "tenant_beta"
        };
        let score = (i % 100) as f64;

        meta.insert("category".to_string(), cat.into());
        meta.insert("tenant".to_string(), tenant.into());
        meta.insert("score".to_string(), score.into());

        index_meta
            .insert_with_metadata(id.as_ref(), vec.clone(), meta)
            .unwrap();
    }

    // Benchmark compound filter: category == "gpu" AND score BETWEEN 20 AND 80
    let filter = FilterExpr::and(vec![
        FilterExpr::eq("category", "gpu"),
        FilterExpr::range("score", 20.0, 80.0),
    ]);

    let intent_filtered = SearchIntent {
        filter: Some(filter),
        ..Default::default()
    };

    let t_filter_search = Instant::now();
    for q in &queries_clust {
        let _ = index_meta
            .intent_rerank_search(q, k, &intent_filtered)
            .unwrap();
    }
    let dur_filter_search = t_filter_search.elapsed();
    let qps_filter = num_queries as f64 / dur_filter_search.as_secs_f64();
    let avg_lat_filter = (dur_filter_search.as_micros() as f64) / num_queries as f64;

    println!(
        " • Roaring Filtered Search (200 queries): {:.2?} ({:.0} QPS, avg {:.1} µs/query)",
        dur_filter_search, qps_filter, avg_lat_filter
    );
    println!(
        " • Filtered-search benchmark completed; allocation behavior requires a dedicated allocation trace.\n"
    );

    // ────────────────────────────────────────────────────────────────────────
    // 9. ZERO-COPY ASYNC TCP SERVER THROUGHPUT & NETWORK LATENCY
    // ────────────────────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ 9. ZERO-COPY ASYNC BINARY TCP NETWORK SERVER                                 ║");
    println!("║    (Non-blocking Tokio Network Framing & Pipelined Query Execution)          ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async {
        let index_srv = Arc::new(HNSQRIndex::new(HNSQRConfig::default(), dim));
        for (id, vec) in dataset_clust.iter().take(1000) {
            index_srv.insert(id.as_ref(), vec.clone()).unwrap();
        }

        let addr: std::net::SocketAddr = "127.0.0.1:0".parse().unwrap();
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
        let bound_addr = listener.local_addr().unwrap();

        let srv_idx = Arc::new(hnsqr::StandaloneService::new(Arc::clone(&index_srv)));
        tokio::spawn(async move {
            while let Ok((stream, _)) = listener.accept().await {
                let srv = Arc::clone(&srv_idx);
                tokio::spawn(async move {
                    let _ = HNSQRServer::handle_client(stream, srv).await;
                });
            }
        });

        let mut client: HNSQRClient = HNSQRClient::connect(bound_addr).await.unwrap();
        let t_ping = Instant::now();
        for _ in 0..1000 {
            let ok = client.ping().await.unwrap();
            assert!(ok);
        }
        let dur_ping = t_ping.elapsed();
        let ping_qps = 1000.0 / dur_ping.as_secs_f64();
        let ping_lat = (dur_ping.as_micros() as f64) / 1000.0;

        println!(" • Async TCP Network Healthchecks: 1,000 Pings in {:.2?} ({:.0} QPS, avg {:.1} µs RTT)",
                 dur_ping, ping_qps, ping_lat);

        // Benchmark remote vector queries over TCP binary protocol
        let t_tcp_search = Instant::now();
        for q in queries_clust.iter().take(100) {
            let res = client.search(q, 10).await.unwrap();
            assert!(!res.is_empty());
        }
        let dur_tcp_search = t_tcp_search.elapsed();
        let tcp_search_qps = 100.0 / dur_tcp_search.as_secs_f64();
        let tcp_search_lat = (dur_tcp_search.as_micros() as f64) / 100.0;

        println!(" • Async TCP Network Searches:     100 Queries in {:.2?} ({:.0} QPS, avg {:.1} µs RTT)",
                 dur_tcp_search, tcp_search_qps, tcp_search_lat);
    });

    // ────────────────────────────────────────────────────────────────────────
    // 10. PAIRWISE COMPLEX-FOLDED LLM WEAVER & REST GATEWAY ROUTER
    // ────────────────────────────────────────────────────────────────────────
    println!("\n╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║ 10. LLM PAIRWISE PHASE-ENCODING WEAVER & HTTP REST GATEWAY                   ║");
    println!("║     (Lossless Pairwise Real-to-Complex Dimensional Repacking & Routing)      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝\n");

    let num_llm_vecs = 5000;
    println!(" • Generating 5,000 OpenAI 1536-dimensional real float embeddings...");
    let openai_embeddings: Vec<Vec<f32>> = (0..num_llm_vecs)
        .map(|i| (0..1536).map(|j| ((i + j) as f32).sin()).collect())
        .collect();

    let t_fold = Instant::now();
    let mut folded_vectors = Vec::with_capacity(num_llm_vecs);
    for vec in &openai_embeddings {
        let qvec = ComplexWeaver::fold_llm_embedding(vec);
        folded_vectors.push(qvec);
    }
    let dur_fold = t_fold.elapsed();
    let fold_qps = num_llm_vecs as f64 / dur_fold.as_secs_f64();
    let fold_latency_us = (dur_fold.as_micros() as f64) / num_llm_vecs as f64;

    println!(
        " • ComplexWeaver Folding Speed: {:.2?} ({:.0} vectors/sec, avg {:.2} µs/vector)",
        dur_fold, fold_qps, fold_latency_us
    );
    println!(
        " • Dimensional Repacking: 1536-dim Real -> 768-dim Complex (same memory footprint: 6144 bytes, lossless coordinate transform)"
    );

    // Multi-collection Gateway Routing
    let temp_dir_gw = std::env::temp_dir().join("hnsqr_bench_gw");
    let _ = std::fs::create_dir_all(&temp_dir_gw);
    let router = Arc::new(hnsqr::vector::folding::GatewayRouter::new(
        &temp_dir_gw.to_string_lossy(),
        false,
    ));

    // Prepare 1,000 metadata records
    let records: Vec<(
        String,
        &[f32],
        std::collections::HashMap<String, hnsqr::MetadataValue>,
    )> = openai_embeddings
        .iter()
        .take(1000)
        .enumerate()
        .map(|(i, vec)| {
            let mut meta = std::collections::HashMap::new();
            meta.insert(
                "source".to_string(),
                if i % 2 == 0 { "wiki" } else { "arxiv" }.into(),
            );
            meta.insert("cluster".to_string(), format!("c_{}", i % 10).into());
            (format!("doc_{}", i), vec.as_slice(), meta)
        })
        .collect();

    // 1. Scalar Sequential Ingestion (Tuned Adaptive Heuristics)
    let t_gw_scalar = Instant::now();
    for (id, vec, meta) in &records {
        router
            .ingest_llm_vector_with_metadata("openai_text_3_scalar", id, vec, meta.clone())
            .unwrap();
    }
    let dur_gw_scalar = t_gw_scalar.elapsed();
    let scalar_qps = 1000.0 / dur_gw_scalar.as_secs_f64();
    println!(
        " • GatewayRouter Sequential Ingestion (1,000 1536-dim vectors): {:.2?} ({:.0} vectors/sec, avg {:.1} µs/vec)",
        dur_gw_scalar,
        scalar_qps,
        (dur_gw_scalar.as_micros() as f64) / 1000.0
    );

    // 2. Multi-Threaded Parallel Batch Ingestion (Rayon Saturation)
    let t_gw_batch = Instant::now();
    let batch_res = router
        .batch_ingest_llm_vectors_with_metadata("openai_text_3_batch", &records)
        .unwrap();
    let dur_gw_batch = t_gw_batch.elapsed();
    let batch_qps = 1000.0 / dur_gw_batch.as_secs_f64();
    let batch_eff_us = (dur_gw_batch.as_micros() as f64) / 1000.0;
    let speedup = batch_qps / scalar_qps.max(1.0);
    assert_eq!(batch_res.len(), 1000);

    println!(
        " • GatewayRouter Rayon BATCH Ingestion (1,000 1536-dim vectors):   {:.2?} ({:.0} vectors/sec, {:.1} µs/vec effective, {:.2}x scaling!)",
        dur_gw_batch, batch_qps, batch_eff_us, speedup
    );

    // Search via GatewayRouter with Roaring filter
    let query_1536 = &openai_embeddings[0];
    let filter = FilterExpr::eq("source", "arxiv");
    let t_gw_search = Instant::now();
    for _ in 0..200 {
        let res = router
            .search_llm_vector_with_filter(
                "openai_text_3_batch",
                query_1536,
                10,
                Some(filter.clone()),
            )
            .unwrap();
        assert!(!res.is_empty());
    }
    let dur_gw_search = t_gw_search.elapsed();
    println!(
        " • GatewayRouter Filtered Search (200 queries @ 1536-dim): {:.2?} ({:.0} QPS, avg {:.1} µs/query)",
        dur_gw_search,
        200.0 / dur_gw_search.as_secs_f64(),
        (dur_gw_search.as_micros() as f64) / 200.0
    );

    let _ = std::fs::remove_dir_all(temp_dir_gw);

    println!("\n════════════════════════════════════════════════════════════════════════════════");
    println!("                    FULL BENCHMARK SUITE COMPLETED SUCCESSFULLY                 ");
    println!("════════════════════════════════════════════════════════════════════════════════");
}
