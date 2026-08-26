mod common;

use std::sync::Arc;
use std::time::Instant;

use common::{BenchScale, DEFAULT_BENCH_SEED, open_prebuilt_snapshot_v2};
use hnsqr::HNSQRIndex;
use hnsqr::rivero::{AdaptivePolicy, RiveroProfile};
use hnsqr::storage::snapshot::{SnapshotOpenOptions, VerificationMode};
use rayon::prelude::*;

fn main() {
    let scale = BenchScale::from_env();
    let n = scale.corpus_size();
    let client_counts = scale.concurrency_clients();

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR HIGH-CONCURRENCY SEARCH MATRIX (Scale: {:?}, N = {})                              ║",
        scale, n
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let (snap_path, corpus) =
        open_prebuilt_snapshot_v2(scale, RiveroProfile::Balanced, DEFAULT_BENCH_SEED);
    let index = Arc::new(
        HNSQRIndex::open_snapshot_v2(
            &snap_path,
            SnapshotOpenOptions {
                verification: VerificationMode::HeaderAndBounds,
                ..Default::default()
            },
        )
        .expect("Snapshot must open successfully"),
    );

    println!(
        "  ┌──────────┬──────────────┬──────────────┬──────────────┬──────────────┬────────────────┐"
    );
    println!(
        "  │ Clients  │ Total QPS    │ p50 Latency  │ p95 Latency  │ p99 Latency  │ p99.9 Latency  │"
    );
    println!(
        "  ├──────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤"
    );

    for &clients in client_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(clients)
            .build()
            .unwrap();
        let queries_per_client = 200;
        let total_queries = clients * queries_per_client;

        let t_start = Instant::now();
        let latencies: Vec<f64> = pool.install(|| {
            (0..clients)
                .into_par_iter()
                .flat_map(|c_idx| {
                    let mut client_lats = Vec::with_capacity(queries_per_client);
                    for q_i in 0..queries_per_client {
                        let q = &corpus.folded_queries
                            [(c_idx * 17 + q_i) % corpus.folded_queries.len()];
                        let t0 = Instant::now();
                        let _ = index
                            .search_indices_adaptive(q, 10, None, AdaptivePolicy::RiveroOnly)
                            .unwrap();
                        client_lats.push(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    client_lats
                })
                .collect()
        });

        let duration_sec = t_start.elapsed().as_secs_f64();
        let qps = total_queries as f64 / duration_sec;

        let mut sorted_lats = latencies;
        sorted_lats.sort_unstable_by(|a, b| a.total_cmp(b));
        let p50 = sorted_lats[(sorted_lats.len() as f64 * 0.50) as usize];
        let p95 = sorted_lats[(sorted_lats.len() as f64 * 0.95) as usize];
        let p99 = sorted_lats[(sorted_lats.len() as f64 * 0.99) as usize];
        let p999 = sorted_lats
            [(sorted_lats.len() as f64 * 0.999).min(sorted_lats.len() as f64 - 1.0) as usize];

        println!(
            "  │ {:>8} │ {:>10.1} QPS│ {:>10.2} ms│ {:>10.2} ms│ {:>10.2} ms│ {:>12.2} ms│",
            clients, qps, p50, p95, p99, p999
        );
    }
    println!(
        "  └──────────┴──────────────┴──────────────┴──────────────┴──────────────┴────────────────┘\n"
    );
}
