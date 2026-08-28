mod common;

use std::time::Instant;

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
    unique_pages_touched: usize,
}

fn run_residency_benchmark(
    real_dim: usize,
    complex_dim: usize,
    num_queries: usize,
) -> ResidencyResult {
    let k = 10;
    let n = 10_000;
    let (base_path, query_path, _) = common::find_best_matching_dataset(real_dim);
    let (folded_corpus, _) = common::read_fvecs(&base_path, Some(n))
        .unwrap_or_else(|_| panic!("failed to load {}", base_path.display()));
    let (folded_queries, _) = common::read_fvecs(&query_path, Some(num_queries))
        .unwrap_or_else(|_| panic!("failed to load {}", query_path.display()));

    let compiler = RiveroCompiler::new(complex_dim);
    let territory_index = RiveroTerritoryIndex::new();

    // Fast direct Rivero insertion
    for (i, v) in folded_corpus.iter().enumerate() {
        let addr = compiler.compile(v.complex_data());
        territory_index.insert(&addr, i as NodeIndex);
    }

    let vector_bytes = complex_dim * 8; // 8 bytes per Complex32
    let page_size = 4096;
    let cold_page_penalty_us = 2.5; // Estimated 2.5 µs NVMe / OS page fault overhead per cold page

    let mut contiguous_times = Vec::with_capacity(num_queries);
    let mut scattered_warm_times = Vec::with_capacity(num_queries);
    let mut scattered_locality_times = Vec::with_capacity(num_queries);
    let mut scattered_cold_times = Vec::with_capacity(num_queries);
    let mut unique_pages_list = Vec::with_capacity(num_queries);

    for q_idx in 0..folded_queries.len() {
        let folded_q = &folded_queries[q_idx];
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
                    (folded_q.dot_product_complex(&folded_corpus[s as usize])).re,
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
                    (folded_q.dot_product_complex(&folded_corpus[s as usize])).re,
                )
            })
            .collect();
        scored_scatt.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scattered_warm_times.push(t1.elapsed().as_secs_f64() * 1_000_000.0);

        // 3. Exact Scattered Locality Sorted (Real Rivero IDs reordered by memory slot ID)
        let t2 = Instant::now();
        let mut sorted_slots = candidate_slots.clone();
        sorted_slots.sort_unstable();
        let mut scored_loc: Vec<(NodeIndex, SimilarityScore)> = sorted_slots
            .iter()
            .map(|&s| {
                (
                    s,
                    (folded_q.dot_product_complex(&folded_corpus[s as usize])).re,
                )
            })
            .collect();
        scored_loc.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored_loc.truncate(k);
        scattered_locality_times.push(t2.elapsed().as_secs_f64() * 1_000_000.0);

        // Count unique pages spanned by the candidate vectors
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
    }

    let exact_cont_p50 = percentile(contiguous_times, 50.0);
    let exact_scatt_warm_p50 = percentile(scattered_warm_times, 50.0);
    let exact_scatt_loc_p50 = percentile(scattered_locality_times, 50.0);
    let exact_scatt_cold_p50 = percentile(scattered_cold_times, 50.0);

    let mean_unique_pages =
        unique_pages_list.iter().sum::<usize>() / unique_pages_list.len().max(1);

    ResidencyResult {
        real_dim,
        complex_dim,
        exact_contiguous_us: exact_cont_p50,
        exact_scattered_warm_us: exact_scatt_warm_p50,
        exact_scattered_locality_us: exact_scatt_loc_p50,
        exact_scattered_cold_mmap_us: exact_scatt_cold_p50,
        unique_pages_touched: mean_unique_pages,
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
    println!(" SECTION 2: COLD MMAP / DISK-PAGE REGIME (512 Scattered Vector Pages)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!("  ┌────────┬─────────┬──────────────────────┬──────────────┐");
    println!("  │ Real D │ Cmplx D │ Exact Cold Mmap      │ Pages Touched│");
    println!("  ├────────┼─────────┼──────────────────────┼──────────────┤");

    for res in &results {
        println!(
            "  │ {:>6} │ {:>7} │ {:>18.1} µs │ {:>9} pgs │",
            res.real_dim,
            res.complex_dim,
            res.exact_scattered_cold_mmap_us,
            res.unique_pages_touched
        );
    }
    println!("  └────────┴─────────┴──────────────────────┴──────────────┘\n");
}
