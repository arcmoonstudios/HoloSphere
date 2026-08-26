mod common;

use std::time::Instant;

use common::load_real_dataset_corpus;
use hnsqr::proof::lutz::{LutzCertifier, LutzCode, LutzQueryTable};
use hnsqr::{NodeIndex, SimilarityScore};

fn percentile(mut latencies: Vec<f64>, p: f64) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let idx = ((latencies.len() as f64 - 1.0) * (p / 100.0)).round() as usize;
    latencies[idx]
}

struct PipelineResult {
    real_dim: usize,
    complex_dim: usize,
    base_p50_us: f64,
    base_p95_us: f64,
    base_qps: f64,
    pipe_p50_us: f64,
    pipe_p95_us: f64,
    pipe_qps: f64,
    speedup: f64,
    l0_time_us: f64,
    l1_time_us: f64,
    exact_time_us: f64,
    evals_p50: usize,
    evals_p95: usize,
    raw_bytes: usize,
    l0_bytes: usize,
    l1_bytes: usize,
    exact_bytes: usize,
    bw_reduction: f64,
}

fn run_pipeline_benchmark(
    real_dim: usize,
    _complex_dim: usize,
    num_queries: usize,
) -> PipelineResult {
    let k = 10;
    let candidate_pool_size = 512;
    let dataset = load_real_dataset_corpus(
        candidate_pool_size,
        num_queries,
        real_dim,
        common::DEFAULT_BENCH_SEED,
    );

    let real_dim = dataset.real_dim;
    let complex_dim = dataset.complex_dim;
    let candidate_pool_size = dataset.folded_corpus.len();

    // Pre-encode corpus in LUTz-v2 format with L1 enabled
    let codes: Vec<LutzCode> = dataset
        .folded_corpus
        .iter()
        .map(|v| LutzCode::encode(v, true))
        .collect();

    let candidate_slots: Vec<NodeIndex> = (0..candidate_pool_size as NodeIndex).collect();

    let exact_index = {
        let idx = hnsqr::HNSQRIndex::new(hnsqr::HNSQRConfig::default(), complex_dim);
        for (i, v) in dataset.folded_corpus.iter().enumerate() {
            idx.insert(format!("d{i}"), v.clone()).unwrap();
        }
        idx
    };

    // 1. Measure Baseline: Exhaustive Production Exact SIMD Scan
    let mut base_latencies = Vec::with_capacity(num_queries);
    for query in &dataset.folded_queries {
        let t0 = Instant::now();
        let _ = exact_index
            .search_indices_exact(query, k, None)
            .expect("exact scan");
        base_latencies.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    // 2. Measure LUTz v2 Pipeline: Fast Query Setup -> LUTz L0 -> LUTz L1 -> Exact Certifier
    let mut pipe_latencies = Vec::with_capacity(num_queries);
    let mut l0_times = Vec::with_capacity(num_queries);
    let mut l1_times = Vec::with_capacity(num_queries);
    let mut exact_times = Vec::with_capacity(num_queries);
    let mut evals_list = Vec::with_capacity(num_queries);
    let mut l1_refined_counts = Vec::with_capacity(num_queries);
    let mut all_match = true;

    for (q_idx, query) in dataset.folded_queries.iter().enumerate() {
        let t0 = Instant::now();
        let query_lut = LutzQueryTable::build(query);
        let (certified_topk, diag) = LutzCertifier::certify(
            &query_lut,
            &candidate_slots,
            |slot| Some(&codes[slot as usize]),
            |slot| (query.dot_product_complex(&dataset.folded_corpus[slot as usize])).re,
            k,
        );
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;

        pipe_latencies.push(elapsed_us);
        l0_times.push(diag.l0_prescore_us as f64);
        l1_times.push(diag.l1_refine_us as f64);
        exact_times.push(diag.exact_cert_us as f64);
        evals_list.push(diag.exact_evaluations);
        l1_refined_counts.push(diag.candidates_l1_refined);

        // Ground-Truth Match Check
        let mut exhaustive: Vec<(NodeIndex, SimilarityScore)> = dataset
            .folded_corpus
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx as NodeIndex, (query.dot_product_complex(doc)).re))
            .collect();
        exhaustive.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let true_topk: Vec<(NodeIndex, SimilarityScore)> = exhaustive.into_iter().take(k).collect();

        for rank in 0..k {
            if certified_topk[rank].0 != true_topk[rank].0
                || (certified_topk[rank].1 - true_topk[rank].1).abs() > 1e-5
            {
                all_match = false;
                println!("Mismatch Q{q_idx} Rank {rank}!");
            }
        }
    }

    let mut sorted_evals = evals_list.clone();
    sorted_evals.sort_unstable();
    let p50_evals = sorted_evals[((sorted_evals.len() as f64 - 1.0) * 0.50).round() as usize];
    let p95_evals = sorted_evals[((sorted_evals.len() as f64 - 1.0) * 0.95).round() as usize];

    let base_p50 = percentile(base_latencies.clone(), 50.0);
    let base_p95 = percentile(base_latencies.clone(), 95.0);
    let base_qps = 1_000_000.0 / base_p50.max(0.1);

    let pipe_p50 = percentile(pipe_latencies.clone(), 50.0);
    let pipe_p95 = percentile(pipe_latencies.clone(), 95.0);
    let pipe_qps = 1_000_000.0 / pipe_p50.max(0.1);

    let speedup = base_p50 / pipe_p50.max(0.1);

    // Bandwidth Analysis
    let vector_bytes = complex_dim * 8; // Complex32 is 8 bytes
    let raw_total_bytes = candidate_pool_size * vector_bytes;

    let l0_bytes_per_candidate = complex_dim * 2 + (complex_dim.div_ceil(32) * 4) + 12; // 8b phase + 8b amp + block res + max amp + globals
    let l1_bytes_per_candidate = complex_dim.div_ceil(4) + (complex_dim.div_ceil(32) * 4) + 4; // 2b phase + block res l1 + global res

    let mean_l1_refined = l1_refined_counts.iter().sum::<usize>() as f64 / num_queries as f64;

    let l0_bytes = candidate_pool_size * l0_bytes_per_candidate;
    let l1_bytes = (mean_l1_refined.round() as usize) * l1_bytes_per_candidate;
    let exact_bytes = p50_evals * vector_bytes;
    let total_lutz_bytes = l0_bytes + l1_bytes + exact_bytes;
    let bw_reduction = raw_total_bytes as f64 / total_lutz_bytes.max(1) as f64;

    assert!(all_match, "LUTz certification diverged from exact Top-10");

    PipelineResult {
        real_dim,
        complex_dim,
        base_p50_us: base_p50,
        base_p95_us: base_p95,
        base_qps,
        pipe_p50_us: pipe_p50,
        pipe_p95_us: pipe_p95,
        pipe_qps,
        speedup,
        l0_time_us: percentile(l0_times, 50.0),
        l1_time_us: percentile(l1_times, 50.0),
        exact_time_us: percentile(exact_times, 50.0),
        evals_p50: p50_evals,
        evals_p95: p95_evals,
        raw_bytes: raw_total_bytes,
        l0_bytes,
        l1_bytes,
        exact_bytes,
        bw_reduction,
    }
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR INTEGRATED PIPELINE: RIVERO -> LUTz-v2 -> EXACT HERMITIAN CERTIFIER            ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let dims = [(1536, 768), (4096, 2048), (8192, 4096), (16384, 8192)];

    let mut results = Vec::new();
    for &(r_dim, c_dim) in &dims {
        let res = run_pipeline_benchmark(r_dim, c_dim, 32);
        results.push(res);
    }

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 1: END-TO-END QUERY LATENCY & SPEEDUP (512 Candidates)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "  ┌────────┬─────────┬──────────────────────┬──────────────────────┬──────────┬──────────────────┐"
    );
    println!(
        "  │ Real D │ Cmplx D │ 512 Exact Baseline   │ LUTz v2 Certified    │ Speedup  │ Throughput (QPS) │"
    );
    println!(
        "  ├────────┼─────────┼──────────────────────┼──────────────────────┼──────────┼──────────────────┤"
    );

    for res in &results {
        println!(
            "  │ {:>6} │ {:>7} │ {:>7.1} / {:>7.1} µs │ {:>7.1} / {:>7.1} µs │ {:>7.2}x │ {:>7.0} -> {:>5.0} QPS│",
            res.real_dim,
            res.complex_dim,
            res.base_p50_us,
            res.base_p95_us,
            res.pipe_p50_us,
            res.pipe_p95_us,
            res.speedup,
            res.base_qps,
            res.pipe_qps
        );
    }
    println!(
        "  └────────┴─────────┴──────────────────────┴──────────────────────┴──────────┴──────────────────┘\n"
    );

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 2: COMPONENT STAGE LATENCY BREAKDOWN");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "  ┌────────┬─────────┬──────────────────────┬──────────────────────┬──────────────────────┬─────────────┐"
    );
    println!(
        "  │ Real D │ Cmplx D │ LUTz L0 Prescore     │ L1 Refinement        │ Exact Certifier      │ L (p50/p95) │"
    );
    println!(
        "  ├────────┼─────────┼──────────────────────┼──────────────────────┼──────────────────────┼─────────────┤"
    );

    for res in &results {
        println!(
            "  │ {:>6} │ {:>7} │ {:>18.1} µs │ {:>18.1} µs │ {:>18.1} µs │ {:>4} / {:>4}  │",
            res.real_dim,
            res.complex_dim,
            res.l0_time_us,
            res.l1_time_us,
            res.exact_time_us,
            res.evals_p50,
            res.evals_p95
        );
    }
    println!(
        "  └────────┴─────────┴──────────────────────┴──────────────────────┴──────────────────────┴─────────────┘\n"
    );

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 3: MEMORY TRAFFIC & BANDWIDTH SAVINGS (Bytes Touched per Query)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "  ┌────────┬─────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐"
    );
    println!(
        "  │ Real D │ Cmplx D │ Raw Baseline │ L0 Traffic   │ L1 Traffic   │ Exact Vector │ BW Reduction │"
    );
    println!(
        "  ├────────┼─────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤"
    );

    for res in &results {
        println!(
            "  │ {:>6} │ {:>7} │ {:>9.1} KB │ {:>9.1} KB │ {:>9.1} KB │ {:>9.1} KB │ {:>11.2}x │",
            res.real_dim,
            res.complex_dim,
            res.raw_bytes as f64 / 1024.0,
            res.l0_bytes as f64 / 1024.0,
            res.l1_bytes as f64 / 1024.0,
            res.exact_bytes as f64 / 1024.0,
            res.bw_reduction
        );
    }
    println!(
        "  └────────┴─────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n"
    );
}
