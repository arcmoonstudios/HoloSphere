mod common;

use std::time::Instant;

use common::{BenchScale, DEFAULT_BENCH_SEED, generate_realistic_text_corpus};
use hnsqr::rivero::RiveroProfile;
use hnsqr::rivero_bulk::RiveroBulkBuilder;
use hnsqr::{HNSQRConfig, HNSQRIndex, VectorEmbedding};

fn benchmark_thread_scaling(vectors: &[VectorEmbedding]) {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        " MULTI-CORE THREAD SCALING & BIT-FOR-BIT DETERMINISM (N = {})",
        vectors.len()
    );
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let thread_counts = [1, 2, 4, 8, 16];
    let mut fingerprints: Vec<[u8; 32]> = Vec::new();

    println!(
        "  ┌────────┬─────────────┬──────────────┬──────────────────┬──────────────────────────────────┐"
    );
    println!(
        "  │ Threads│ Total Time  │ Throughput   │ Parallel Speedup │ Structural SHA256 Fingerprint    │"
    );
    println!(
        "  ├────────┼─────────────┼──────────────┼──────────────────┼──────────────────────────────────┤"
    );

    let mut base_time = 1.0;
    for (idx, &t) in thread_counts.iter().enumerate() {
        let builder = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(t);
        let start = Instant::now();
        let built = builder.build(vectors).expect("Bulk build must succeed");
        let dur_ms = start.elapsed().as_secs_f64() * 1000.0;
        let fp = built.territory.structural_fingerprint();

        if idx == 0 {
            base_time = dur_ms;
        }
        let speedup = base_time / dur_ms;
        let vecs_per_sec = (vectors.len() as f64) / (dur_ms / 1000.0);

        let fp_hex = hex::encode(&fp[..8]);
        println!(
            "  │ {:>6} │ {:>9.2} ms│ {:>8.0} v/s │ {:>15.2}x │ {}... │",
            t, dur_ms, vecs_per_sec, speedup, fp_hex
        );

        fingerprints.push(fp);
    }
    println!(
        "  └────────┴─────────────┴──────────────┴──────────────────┴──────────────────────────────────┘\n"
    );

    let reference_fp = fingerprints[0];
    let all_match = fingerprints.iter().all(|fp| *fp == reference_fp);
    if all_match {
        println!(
            "  ✓ Structural Determinism Verified: Identical bit-for-bit SHA-256 fingerprint across all threads!\n"
        );
    } else {
        panic!("Thread count determinism invariant violated!");
    }
}

fn benchmark_profile_scaling(vectors: &[VectorEmbedding]) {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        " PHASE-BY-PHASE TELEMETRY & STAGE A/B PROFILE COMPARISON (N = {})",
        vectors.len()
    );
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let profiles = [
        ("Fast (8F/8P/32C/8B)", RiveroProfile::Fast),
        ("Balanced (12F/16P/48C/12B)", RiveroProfile::Balanced),
        ("Strict (24F/32P/64C/16B)", RiveroProfile::Strict),
    ];

    println!(
        "  ┌───────────────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬─────────────┬──────────────┐"
    );
    println!(
        "  │ Profile                   │ Compile  │ Shard Red│ Merge Red│ Wit Route│ Stage A %│ Stage B %│ Wit Score│ Wit Final│ Total Time  │ Throughput   │"
    );
    println!(
        "  ├───────────────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼──────────┼─────────────┼──────────────┤"
    );

    for (name, profile) in profiles {
        let builder = RiveroBulkBuilder::with_profile(profile).with_threads(16);
        let built = builder.build(vectors).expect("Bulk build must succeed");
        let t = &built.telemetry;

        println!(
            "  │ {:<25} │ {:>6.1} ms│ {:>6.1} ms│ {:>6.1} ms│ {:>6.1} ms│ {:>7.1}% │ {:>7.1}% │ {:>6.1} ms│ {:>6.1} ms│ {:>9.2} ms│ {:>8.0} v/s │",
            name,
            t.time_address_compile_ms,
            t.time_territory_reduction_ms,
            t.time_territory_merge_ms,
            t.time_witness_routing_ms,
            t.stage_a_accepted_pct,
            t.stage_b_expanded_pct,
            t.time_witness_scoring_ms,
            t.time_witness_finalize_ms,
            t.total_build_time_ms,
            t.throughput_vecs_per_sec,
        );
    }
    println!(
        "  └───────────────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴─────────────┴──────────────┘\n"
    );
}

fn benchmark_incremental_vs_bulk(vectors: &[VectorEmbedding]) {
    let n = vectors.len().min(5000);
    let sample = &vectors[..n];

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" INCREMENTAL VS BULK BUILD COMPARISON (N = {})", n);
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    // 1. Incremental 1T build
    let mut cfg_inc = HNSQRConfig::default();
    cfg_inc.rivero_enabled = true;
    cfg_inc.rivero_fallback_on_underfill = false;
    cfg_inc.rivero_witness_degree = 48;
    let idx_inc = HNSQRIndex::new(cfg_inc, 32);

    let t0 = Instant::now();
    for (i, v) in sample.iter().enumerate() {
        idx_inc.insert(format!("node_{i}"), v.clone()).unwrap();
    }
    let inc_time_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // 2. Bulk build
    let builder = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(16);
    let t1 = Instant::now();
    let _built = builder.build(sample).unwrap();
    let bulk_time_ms = t1.elapsed().as_secs_f64() * 1000.0;

    let speedup = inc_time_ms / bulk_time_ms;

    println!(
        "  * Incremental Online Build Time: {:.2} ms ({:.0} vec/s)",
        inc_time_ms,
        (n as f64) / (inc_time_ms / 1000.0)
    );
    println!(
        "  * Parallel Bulk Build Time:       {:.2} ms ({:.0} vec/s)",
        bulk_time_ms,
        (n as f64) / (bulk_time_ms / 1000.0)
    );
    println!(
        "  * Parallel Construction Speedup:  {:.2}x FASTER than incremental\n",
        speedup
    );
}

mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

fn main() {
    let scale = BenchScale::from_env();
    let n = scale.corpus_size();

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR PARALLEL BULK BUILDER BENCHMARK (Scale: {:?}, N = {})                            ║",
        scale, n
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let corpus = generate_realistic_text_corpus(n, 10, 64, DEFAULT_BENCH_SEED);

    benchmark_thread_scaling(&corpus.folded_corpus);
    benchmark_profile_scaling(&corpus.folded_corpus);
    benchmark_incremental_vs_bulk(&corpus.folded_corpus);

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" BULK BUILDER BENCHMARK COMPLETED SUCCESSFULLY");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════\n"
    );
}
