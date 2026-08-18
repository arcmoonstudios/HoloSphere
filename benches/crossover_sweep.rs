mod common;

use std::time::Instant;

use common::generate_realistic_text_corpus;
use hnsqr::rivero::{AdaptivePolicy, RiveroProfile};
use hnsqr::rivero::bulk::RiveroBulkBuilder;
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, NodeIndex, SearchPlan};

fn percentile(mut latencies: Vec<f64>, p: f64) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let idx = ((latencies.len() as f64 - 1.0) * (p / 100.0)).round() as usize;
    latencies[idx]
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR EMPIRICAL SEARCH CROSSOVER & ADAPTIVE STAGE SWEEP (N=500..100K)                ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let sizes = if cfg!(debug_assertions) {
        vec![500, 1_000]
    } else {
        vec![500, 2_000, 10_000, 25_000, 50_000, 75_000, 100_000]
    };
    let num_queries = if cfg!(debug_assertions) { 4 } else { 32 };
    let dim = 32; // 64-dim real folded into 32-dim complex

    println!(
        "  ┌────────┬──────────────┬────────────────────────┬───────────────────────────────────────────┬──────────────┐"
    );
    println!(
        "  │ Corpus │ Exact Scan   │ Fast Rivero            │ Adaptive Rivero (Stages Breakdown)        │ GraphOnly    │"
    );
    println!(
        "  │ Size N │ p50 (ms)     │ p50 (ms)  | Recall@10  │ p50 (ms)  | Recall@10 | Fast / Bal / St   │ p50 (ms)     │"
    );
    println!(
        "  ├────────┼──────────────┼────────────────────────┼───────────────────────────────────────────┼──────────────┤"
    );

    for &n in &sizes {
        let dataset =
            generate_realistic_text_corpus(n, num_queries, dim * 2, common::DEFAULT_BENCH_SEED);

        // Ground truth calculation
        let mut ground_truth: Vec<Vec<NodeIndex>> = Vec::with_capacity(num_queries);
        for q in &dataset.folded_queries {
            let mut scored: Vec<(NodeIndex, f32)> = dataset
                .folded_corpus
                .iter()
                .enumerate()
                .map(|(idx, doc)| {
                    let dot = q.dot_product_complex(doc);
                    (idx as NodeIndex, dot.re)
                })
                .collect();
            scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            ground_truth.push(scored.iter().take(10).map(|s| s.0).collect());
        }

        // Build Index
        let mut config = HNSQRConfig::default();
        config.max_elements = n;
        config.distance_function = DistanceFunction::Cosine;
        config.rivero_enabled = true;
        config.search_plan = SearchPlan::Rivero;

        let index = HNSQRIndex::new(config, dim);
        for (i, vec) in dataset.folded_corpus.iter().enumerate() {
            index.insert(format!("doc-{i}"), vec.clone()).unwrap();
        }

        let builder = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(16);
        let rivero_state = builder.build(&dataset.folded_corpus).unwrap();
        index.install_rivero_state(rivero_state).unwrap();

        // 1. Exact Scan
        let mut exact_lats = Vec::with_capacity(num_queries);
        for (q, _gt) in dataset.folded_queries.iter().zip(ground_truth.iter()) {
            let t0 = Instant::now();
            let _ = index.search_indices_exact(q, 10, None).unwrap();
            exact_lats.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let exact_p50 = percentile(exact_lats, 50.0);

        // 2. Fast Rivero
        let mut fast_lats = Vec::with_capacity(num_queries);
        let mut fast_rec = 0.0f64;
        let fast_cfg = RiveroProfile::Fast.config();
        for (q, gt) in dataset.folded_queries.iter().zip(ground_truth.iter()) {
            let t0 = Instant::now();
            let (res, _) = index
                .search_indices_o1_with_config(q, 10, None, &fast_cfg)
                .unwrap();
            fast_lats.push(t0.elapsed().as_secs_f64() * 1000.0);
            let overlap = res.iter().filter(|s| gt.contains(&s.0)).count();
            fast_rec += overlap as f64 / 10.0;
        }
        let fast_p50 = percentile(fast_lats, 50.0);
        let fast_avg_rec = (fast_rec / num_queries as f64) * 100.0;

        // 3. Adaptive Rivero with stage tracking
        let mut adapt_lats = Vec::with_capacity(num_queries);
        let mut adapt_rec = 0.0f64;
        let mut fast_accepted = 0usize;
        let mut bal_accepted = 0usize;
        let mut strict_accepted = 0usize;
        let mut _fallback_used = 0usize;

        for (q, gt) in dataset.folded_queries.iter().zip(ground_truth.iter()) {
            let t0 = Instant::now();
            let (res, diag) = index
                .search_indices_adaptive(q, 10, None, AdaptivePolicy::AllowGraphFallback)
                .unwrap();
            adapt_lats.push(t0.elapsed().as_secs_f64() * 1000.0);
            let overlap = res.iter().filter(|s| gt.contains(&s.0)).count();
            adapt_rec += overlap as f64 / 10.0;

            match diag.final_profile {
                RiveroProfile::Fast => fast_accepted += 1,
                RiveroProfile::Balanced => bal_accepted += 1,
                RiveroProfile::Strict => strict_accepted += 1,
            }
            if diag.graph_fallback_used {
                _fallback_used += 1;
            }
        }
        let adapt_p50 = percentile(adapt_lats, 50.0);
        let adapt_avg_rec = (adapt_rec / num_queries as f64) * 100.0;

        let fast_pct = (fast_accepted as f64 / num_queries as f64) * 100.0;
        let bal_pct = (bal_accepted as f64 / num_queries as f64) * 100.0;
        let strict_pct = (strict_accepted as f64 / num_queries as f64) * 100.0;

        // 4. GraphOnly Traversal
        let mut graph_lats = Vec::with_capacity(num_queries);
        for (q, _gt) in dataset.folded_queries.iter().zip(ground_truth.iter()) {
            let t0 = Instant::now();
            let _ = index.search_indices_graph(q, 10, None).unwrap();
            graph_lats.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        let graph_p50 = percentile(graph_lats, 50.0);

        println!(
            "  │ {:>6} │ {:>10.3} ms │ {:>6.3} ms | {:>5.1}% │ {:>6.3} ms | {:>5.1}% | {:>3.0}% / {:>2.0}% / {:>2.0}% │ {:>10.3} ms │",
            n,
            exact_p50,
            fast_p50,
            fast_avg_rec,
            adapt_p50,
            adapt_avg_rec,
            fast_pct,
            bal_pct,
            strict_pct,
            graph_p50
        );
    }
    println!(
        "  └────────┴──────────────┴────────────────────────┴───────────────────────────────────────────┴──────────────┘\n"
    );
}
