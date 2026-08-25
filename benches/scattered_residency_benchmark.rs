use hnsqr::bench_support as common;

use std::time::Instant;

use common::generate_realistic_text_corpus;
use hnsqr::proof::lutz::{LutzCertifier, LutzCode, LutzQueryTable, exact_rerank_locality_sorted};
use hnsqr::rivero::{RiveroCompiler, RiveroTerritoryIndex};
use hnsqr::{NodeIndex, SimilarityScore};

fn percentile(mut latencies: Vec<f64>, p: f64) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let idx = ((latencies.len() as f64 - 1.0) * (p / 100.0)).round() as usize;
    latencies[idx]
}

struct ResidencyResult {
    real_dim: usize,
    complex_dim: usize,
    exact_contiguous_us: f64,
    exact_scattered_warm_us: f64,
    exact_scattered_locality_us: f64,
    exact_scattered_cold_mmap_us: f64,
    lutz_cold_mmap_us: f64,
    cold_speedup_lutz_vs_exact: f64,
    unique_pages_touched: usize,
    lutz_exact_evals: usize,
}

fn run_residency_benchmark(
    real_dim: usize,
    complex_dim: usize,
    num_queries: usize,
) -> ResidencyResult {
    let k = 10;
    let n = 10_000;
    let dataset =
        generate_realistic_text_corpus(n, num_queries, real_dim, common::DEFAULT_BENCH_SEED);

    let compiler = RiveroCompiler::new(complex_dim);
    let territory_index = RiveroTerritoryIndex::new();

    // Fast direct Rivero insertion
    for (i, v) in dataset.folded_corpus.iter().enumerate() {
        let addr = compiler.compile(v.complex_data());
        territory_index.insert(&addr, i as NodeIndex);
    }

    let lutz_codes: Vec<LutzCode> = dataset
        .folded_corpus
        .iter()
        .map(|v| LutzCode::encode(v, true))
        .collect();

    let vector_bytes = complex_dim * 8; // 8 bytes per Complex32
    let page_size = 4096;
    let cold_page_penalty_us = 2.5; // Estimated 2.5 µs NVMe / OS page fault overhead per cold page

    let mut contiguous_times = Vec::with_capacity(num_queries);
    let mut scattered_warm_times = Vec::with_capacity(num_queries);
    let mut scattered_locality_times = Vec::with_capacity(num_queries);
    let mut scattered_cold_times = Vec::with_capacity(num_queries);
    let mut lutz_cold_times = Vec::with_capacity(num_queries);
    let mut unique_pages_list = Vec::with_capacity(num_queries);
    let mut lutz_evals_list = Vec::with_capacity(num_queries);

    for (q_idx, _raw_q) in dataset.queries_raw.iter().enumerate() {
        let folded_q = &dataset.folded_queries[q_idx];
        let q_addr = compiler.compile(folded_q.complex_data());

        // Extract real Rivero routed candidates (512 candidates)
        let mut candidate_slots: Vec<NodeIndex> =
            territory_index.with_candidates(&q_addr, 512, |cands, _| cands.to_vec());
        candidate_slots.truncate(512);
        let pool_size = candidate_slots.len().max(1);

        // 1. Exact Contiguous (Synthetic 512 Consecutive Slots)
        let contiguous_slots: Vec<NodeIndex> = (0..pool_size as NodeIndex).collect();
        let t0 = Instant::now();
        let mut scored_cont: Vec<(NodeIndex, SimilarityScore)> = contiguous_slots
            .iter()
            .map(|&s| {
                (
                    s,
                    (folded_q.dot_product_complex(&dataset.folded_corpus[s as usize])).re,
                )
            })
            .collect();
        scored_cont.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        contiguous_times.push(t0.elapsed().as_secs_f64() * 1_000_000.0);

        // 2. Exact Scattered Warm (Real Rivero candidate IDs)
        let t1 = Instant::now();
        let mut scored_scatt: Vec<(NodeIndex, SimilarityScore)> = candidate_slots
            .iter()
            .map(|&s| {
                (
                    s,
                    (folded_q.dot_product_complex(&dataset.folded_corpus[s as usize])).re,
                )
            })
            .collect();
        scored_scatt.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scattered_warm_times.push(t1.elapsed().as_secs_f64() * 1_000_000.0);

        // 3. Exact Scattered Locality Sorted (Real Rivero IDs reordered by memory slot ID)
        let t2 = Instant::now();
        let _scored_loc = exact_rerank_locality_sorted(
            &candidate_slots,
            |s| (folded_q.dot_product_complex(&dataset.folded_corpus[s as usize])).re,
            k,
        );
        scattered_locality_times.push(t2.elapsed().as_secs_f64() * 1_000_000.0);

        // Count unique pages spanned by the candidate vectors
        let mut sorted_slots = candidate_slots.clone();
        sorted_slots.sort_unstable();
        let mut unique_pages = 0usize;
        let mut last_page = usize::MAX;
        for &slot in &sorted_slots {
            let page_idx = (slot as usize * vector_bytes) / page_size;
            if page_idx != last_page {
                unique_pages += 1;
                last_page = page_idx;
            }
        }
        unique_pages_list.push(unique_pages);

        // 4. Exact Scattered Cold Mmap (Adding page fault penalties for all unique vector pages)
        let cold_exact_us = (t2.elapsed().as_secs_f64() * 1_000_000.0)
            + (unique_pages as f64 * cold_page_penalty_us);
        scattered_cold_times.push(cold_exact_us);

        // 5. LUTz Cold Mmap (Continuous LUTz codes in cache + cold page penalties for only L exact finalists)
        let t3 = Instant::now();
        let query_lut = LutzQueryTable::build(folded_q);
        let (_certified_topk, diag) = LutzCertifier::certify(
            &query_lut,
            &candidate_slots,
            |s| Some(&lutz_codes[s as usize]),
            |s| (folded_q.dot_product_complex(&dataset.folded_corpus[s as usize])).re,
            k,
        );
        let lutz_compute_us = t3.elapsed().as_secs_f64() * 1_000_000.0;
        let lutz_cold_pages = diag.exact_evaluations; // Each exact finalist touches at most 1 page
        let lutz_cold_us = lutz_compute_us + (lutz_cold_pages as f64 * cold_page_penalty_us);

        lutz_cold_times.push(lutz_cold_us);
        lutz_evals_list.push(diag.exact_evaluations);
    }

    let exact_cont_p50 = percentile(contiguous_times, 50.0);
    let exact_scatt_warm_p50 = percentile(scattered_warm_times, 50.0);
    let exact_scatt_loc_p50 = percentile(scattered_locality_times, 50.0);
    let exact_scatt_cold_p50 = percentile(scattered_cold_times, 50.0);
    let lutz_cold_p50 = percentile(lutz_cold_times, 50.0);

    let mean_unique_pages =
        unique_pages_list.iter().sum::<usize>() / unique_pages_list.len().max(1);
    let mean_lutz_evals = lutz_evals_list.iter().sum::<usize>() / lutz_evals_list.len().max(1);

    ResidencyResult {
        real_dim,
        complex_dim,
        exact_contiguous_us: exact_cont_p50,
        exact_scattered_warm_us: exact_scatt_warm_p50,
        exact_scattered_locality_us: exact_scatt_loc_p50,
        exact_scattered_cold_mmap_us: exact_scatt_cold_p50,
        lutz_cold_mmap_us: lutz_cold_p50,
        cold_speedup_lutz_vs_exact: exact_scatt_cold_p50 / lutz_cold_p50.max(0.1),
        unique_pages_touched: mean_unique_pages,
        lutz_exact_evals: mean_lutz_evals,
    }
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR AXIS 2: RESIDENCY & MEMORY LOCALITY BENCHMARK (N=10,000 CORPUS)                ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let dims = [(4096, 2048), (8192, 4096), (16384, 8192)];

    let mut results = Vec::new();
    for &(r_dim, c_dim) in &dims {
        let res = run_residency_benchmark(r_dim, c_dim, 16);
        results.push(res);
    }

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 1: WARM RAM EXECUTION (Contiguous vs Real Scattered vs Locality Reordered)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "  ┌────────┬─────────┬──────────────────────┬──────────────────────┬──────────────────────┬─────────────┐"
    );
    println!(
        "  │ Real D │ Cmplx D │ Contiguous Exact     │ Scattered Warm Exact │ Locality-Sorted Exact│ Locality Win│"
    );
    println!(
        "  ├────────┼─────────┼──────────────────────┼──────────────────────┼──────────────────────┼─────────────┤"
    );

    for res in &results {
        let loc_win = (res.exact_scattered_warm_us - res.exact_scattered_locality_us)
            / res.exact_scattered_warm_us
            * 100.0;
        println!(
            "  │ {:>6} │ {:>7} │ {:>18.1} µs │ {:>18.1} µs │ {:>18.1} µs │ {:>10.1}% │",
            res.real_dim,
            res.complex_dim,
            res.exact_contiguous_us,
            res.exact_scattered_warm_us,
            res.exact_scattered_locality_us,
            loc_win.max(0.0)
        );
    }
    println!(
        "  └────────┴─────────┴──────────────────────┴──────────────────────┴──────────────────────┴─────────────┘\n"
    );

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 2: COLD MMAP / DISK-PAGE REGIME (512 Scattered Vector Pages vs LUTz)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "  ┌────────┬─────────┬──────────────────────┬──────────────────────┬──────────┬──────────────┬──────────────┐"
    );
    println!(
        "  │ Real D │ Cmplx D │ Exact Cold Mmap      │ LUTz Cold Mmap       │ Speedup  │ Pages Touched│ LUTz Evals L │"
    );
    println!(
        "  ├────────┼─────────┼──────────────────────┼──────────────────────┼──────────┼──────────────┼──────────────┤"
    );

    for res in &results {
        println!(
            "  │ {:>6} │ {:>7} │ {:>18.1} µs │ {:>18.1} µs │ {:>7.2}x │ {:>9} pgs │ {:>9} evs │",
            res.real_dim,
            res.complex_dim,
            res.exact_scattered_cold_mmap_us,
            res.lutz_cold_mmap_us,
            res.cold_speedup_lutz_vs_exact,
            res.unique_pages_touched,
            res.lutz_exact_evals
        );
    }
    println!(
        "  └────────┴─────────┴──────────────────────┴──────────────────────┴──────────┴──────────────┴──────────────┘\n"
    );
}
