mod common;

use common::{BenchScale, DEFAULT_BENCH_SEED, open_prebuilt_snapshot_v2};
use hnsqr::rivero::{AdaptivePolicy, RiveroProfile};
use hnsqr::storage::snapshot::{SnapshotOpenOptions, VerificationMode};
use hnsqr::{HNSQRIndex, NodeIndex};

fn main() {
    let scale = BenchScale::from_env();
    let n = scale.corpus_size();

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR ADAPTIVE CONFIDENCE & FALSE-CONFIDENCE VALIDATION (Scale: {:?}, N = {})          ║",
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

    let workloads = [
        ("Real Semantic", &corpus.folded_queries),
        ("Hard Negatives", &corpus.hard_negatives),
        ("OOD Queries", &corpus.ood_queries),
        ("Random Isotropic", &corpus.isotropic_queries),
    ];

    println!(
        "  ┌───────────────────┬───────────────┬───────────────┬───────────────┬────────────────┬─────────────────┐"
    );
    println!(
        "  │ Workload          │ Fast Accepted │ Balanced Acc. │ Strict Acc.   │ Graph Fallback │ False Confident │"
    );
    println!(
        "  ├───────────────────┼───────────────┼───────────────┼───────────────┼────────────────┼─────────────────┤"
    );

    for (name, q_set) in &workloads {
        let mut fast_count = 0usize;
        let mut balanced_count = 0usize;
        let mut strict_count = 0usize;
        let mut fallback_count = 0usize;
        let mut false_confident_count = 0usize;

        for q in *q_set {
            let (strict_res, _) = index.search_indices_strict(q, 10, None).unwrap();
            let exact_top10: Vec<NodeIndex> = strict_res.iter().map(|(idx, _)| *idx).collect();

            let (adapt_res, diag) = index
                .search_indices_adaptive(q, 10, None, AdaptivePolicy::AllowGraphFallback)
                .unwrap();
            let adapt_top10: Vec<NodeIndex> = adapt_res.iter().map(|(idx, _)| *idx).collect();

            let overlap = adapt_top10
                .iter()
                .filter(|id| exact_top10.contains(id))
                .count();
            let recall = overlap as f64 / 10.0;

            if diag.graph_fallback_used {
                fallback_count += 1;
            } else {
                match diag.stages_executed {
                    1 => fast_count += 1,
                    2 => balanced_count += 1,
                    _ => strict_count += 1,
                }

                if recall < 0.90 {
                    false_confident_count += 1;
                }
            }
        }

        let total = q_set.len() as f64;
        let total_accepted = (fast_count + balanced_count + strict_count) as f64;
        let false_conf_rate = if total_accepted > 0.0 {
            (false_confident_count as f64 / total_accepted) * 100.0
        } else {
            0.0
        };

        println!(
            "  │ {:<17} │ {:>12.1}% │ {:>12.1}% │ {:>12.1}% │ {:>13.1}% │ {:>14.2}% │",
            name,
            (fast_count as f64 / total) * 100.0,
            (balanced_count as f64 / total) * 100.0,
            (strict_count as f64 / total) * 100.0,
            (fallback_count as f64 / total) * 100.0,
            false_conf_rate,
        );
    }
    println!(
        "  └───────────────────┴───────────────┴───────────────┴───────────────┴────────────────┴─────────────────┘\n"
    );
}
