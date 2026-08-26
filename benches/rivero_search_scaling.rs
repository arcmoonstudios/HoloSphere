mod common;

use std::time::Instant;

use common::{BenchScale, DEFAULT_BENCH_SEED, open_prebuilt_snapshot_v2};
use hnsqr::HNSQRIndex;
use hnsqr::rivero::{AdaptivePolicy, RiveroProfile};
use hnsqr::storage::snapshot::{SnapshotOpenOptions, VerificationMode};

fn main() {
    let scale = BenchScale::from_env();
    let n = scale.corpus_size();

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR SEARCH SCALING & WORK CEILING BENCHMARK (Scale: {:?}, N = {})                     ║",
        scale, n
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let (snap_path, corpus) =
        open_prebuilt_snapshot_v2(scale, RiveroProfile::Balanced, DEFAULT_BENCH_SEED);
    let index = HNSQRIndex::open_snapshot_v2(
        &snap_path,
        SnapshotOpenOptions {
            verification: VerificationMode::HeaderAndBounds,
            ..Default::default()
        },
    )
    .expect("Snapshot must open successfully");

    println!(
        "  ┌──────────┬───────────┬───────────┬───────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐"
    );
    println!(
        "  │ Corpus N │ p50 (ms)  │ p95 (ms)  │ p99 (ms)  │ Scans / Q    │ Exact Evals  │ Route-Cap %  │ Post-Wit %   │ Wit Amplif % │"
    );
    println!(
        "  ├──────────┼───────────┼───────────┼───────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤"
    );

    let mut latencies_ms = Vec::with_capacity(corpus.folded_queries.len());
    let mut scans_sum = 0usize;
    let mut evals_sum = 0usize;
    let mut route_cap_util_sum = 0.0f64;
    let mut post_wit_exp_sum = 0.0f64;
    let mut wit_amplif_sum = 0.0f64;

    for q in &corpus.folded_queries {
        let t0 = Instant::now();
        let (_, diag) = index
            .search_indices_adaptive(q, 10, None, AdaptivePolicy::RiveroOnly)
            .unwrap();
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        scans_sum += diag.cumulative_resident_scans;
        evals_sum += diag.cumulative_exact_scores;

        let cand_bound = diag.rivero.selected_candidate_bound.max(1) as f64;
        let route_cands = diag.rivero.route_candidates_selected as f64;
        let wit_added = diag.rivero.witness_candidates_added as f64;
        let exact_scores = diag.rivero.exact_score_evaluations as f64;

        route_cap_util_sum += (route_cands / cand_bound) * 100.0;
        post_wit_exp_sum += (exact_scores / cand_bound) * 100.0;
        wit_amplif_sum += (wit_added / route_cands.max(1.0)) * 100.0;
    }

    latencies_ms.sort_unstable_by(|a, b| a.total_cmp(b));
    let p50 = latencies_ms[(latencies_ms.len() as f64 * 0.50) as usize];
    let p95 = latencies_ms[(latencies_ms.len() as f64 * 0.95) as usize];
    let p99 = latencies_ms[(latencies_ms.len() as f64 * 0.99) as usize];

    let n_q = corpus.folded_queries.len() as f64;
    let avg_scans = scans_sum as f64 / n_q;
    let avg_evals = evals_sum as f64 / n_q;
    let avg_route_cap = route_cap_util_sum / n_q;
    let avg_post_wit = post_wit_exp_sum / n_q;
    let avg_amplif = wit_amplif_sum / n_q;

    println!(
        "  │ {:>8} │ {:>8.2} ms│ {:>8.2} ms│ {:>8.2} ms│ {:>12.0} │ {:>12.0} │ {:>11.1}% │ {:>11.1}% │ {:>11.1}% │",
        n, p50, p95, p99, avg_scans, avg_evals, avg_route_cap, avg_post_wit, avg_amplif
    );
    println!(
        "  └──────────┴───────────┴───────────┴───────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n"
    );
}
