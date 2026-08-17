mod common;

use common::generate_realistic_text_corpus;
use hnsqr::lutz::{LutzCertifier, LutzCode, LutzGlobalCertified, LutzQueryTable};
use hnsqr::rivero::RiveroProfile;
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, NodeIndex, SimilarityScore};
use std::collections::HashSet;
use std::time::Instant;

#[derive(Default, Debug, Clone)]
struct FunnelStageStats {
    total_gt: usize,
    raw_route_hits: usize,
    post_vote_hits: usize,
    post_witness_hits: usize,
    final_hits: usize,
    lutz_hits: usize,
    saturated_queries: usize,
    total_queries: usize,
    avg_raw_candidates: f64,
    avg_post_witness_candidates: f64,
}

#[derive(Default, Debug, Clone)]
struct GlobalSweepStats {
    candidate_cap: usize,
    rivero_recall: f64,
    certified_recall: f64,
    l0_elim_per_q: f64,
    l1_refine_per_q: f64,
    exact_escalate_per_q: f64,
    latency_p50_us: f64,
    qps: f64,
}

fn percentile(mut latencies: Vec<f64>, p: f64) -> f64 {
    if latencies.is_empty() {
        return 0.0;
    }
    latencies.sort_by(|a, b| a.total_cmp(b));
    let idx = ((latencies.len() as f64 - 1.0) * (p / 100.0)).round() as usize;
    latencies[idx]
}

fn evaluate_funnel_for_profile(
    dist_fn: DistanceFunction,
    cap: usize,
    dataset: &common::TextRetrievalCorpus,
    complex_dim: usize,
    k: usize,
) -> (FunnelStageStats, HNSQRIndex) {
    let n = dataset.folded_corpus.len();
    let num_queries = dataset.folded_queries.len();

    let mut config = HNSQRConfig::strict_rivero_for_dim(complex_dim);
    config.max_elements = n + 1000;
    config.distance_function = dist_fn;
    config.rivero_witness_degree = 16;
    config.rivero_witness_seeds = 32;
    config.rivero_witness_second_seeds = 16;
    config.rivero_cell_budget = 32;

    let hnsqr_index = HNSQRIndex::new(config, complex_dim);
    for (i, v) in dataset.folded_corpus.iter().enumerate() {
        hnsqr_index.insert(format!("node_{i}"), v.clone()).unwrap();
    }

    let lutz_codes: Vec<LutzCode> = dataset
        .folded_corpus
        .iter()
        .map(|v| LutzCode::encode(v, true))
        .collect();

    let mut rivero_cfg = RiveroProfile::Strict.config();
    rivero_cfg.query_candidate_cap = cap;

    let mut stats = FunnelStageStats {
        total_queries: num_queries,
        ..Default::default()
    };

    let mut total_raw_cands = 0usize;
    let mut total_post_witness_cands = 0usize;

    for query in &dataset.folded_queries {
        // Compute Ground Truth
        let mut gt_scored: Vec<(NodeIndex, SimilarityScore)> = dataset
            .folded_corpus
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx as NodeIndex, (query.dot_product_complex(doc)).re))
            .collect();
        gt_scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let gt_topk_vec: Vec<NodeIndex> = gt_scored.into_iter().take(k).map(|(id, _)| id).collect();
        let gt_topk_set: HashSet<NodeIndex> = gt_topk_vec.iter().copied().collect();

        // 1. Trace GT coverage through the pipeline
        let trace = hnsqr_index
            .trace_gt_coverage(query, &gt_topk_vec, rivero_cfg)
            .unwrap();

        stats.total_gt += trace.gt_count;
        stats.raw_route_hits += trace.gt_in_raw_route;
        stats.post_vote_hits += trace.gt_after_vote_selection;
        stats.post_witness_hits += trace.gt_after_witness;
        stats.final_hits += trace.gt_in_final_results;

        // 2. Diagnostics for candidate counts
        let (scored, diag) = hnsqr_index
            .search_indices_o1_with_config(query, k, None, &rivero_cfg)
            .unwrap();

        total_raw_cands += diag.raw_unique_candidates;
        total_post_witness_cands += diag.unique_candidates;

        if diag.unique_candidates >= cap {
            stats.saturated_queries += 1;
        }

        // 3. LUTz survival
        let query_lut = LutzQueryTable::build(query);
        let cand_slots: Vec<NodeIndex> = scored.iter().map(|(s, _)| *s).collect();
        let (certified, _) = LutzCertifier::certify(
            &query_lut,
            &cand_slots,
            |slot| Some(&lutz_codes[slot as usize]),
            |slot| (query.dot_product_complex(&dataset.folded_corpus[slot as usize])).re,
            k,
        );

        let lutz_h = certified
            .iter()
            .filter(|(s, _)| gt_topk_set.contains(s))
            .count();
        stats.lutz_hits += lutz_h;
    }

    stats.avg_raw_candidates = total_raw_cands as f64 / num_queries as f64;
    stats.avg_post_witness_candidates = total_post_witness_cands as f64 / num_queries as f64;

    (stats, hnsqr_index)
}

fn evaluate_global_certified_sweep(
    cap: usize,
    dataset: &common::TextRetrievalCorpus,
    _complex_dim: usize,
    k: usize,
    index: &HNSQRIndex,
) -> GlobalSweepStats {
    let n = dataset.folded_corpus.len();
    let num_queries = dataset.folded_queries.len();

    let lutz_codes: Vec<LutzCode> = dataset
        .folded_corpus
        .iter()
        .map(|v| LutzCode::encode(v, true))
        .collect();

    let mut rivero_cfg = RiveroProfile::Strict.config();
    rivero_cfg.query_candidate_cap = cap;

    let mut rivero_hits = 0usize;
    let mut certified_hits = 0usize;
    let mut total_l0_elim = 0usize;
    let mut total_l1_refine = 0usize;
    let mut total_exact_escalations = 0usize;
    let mut latencies = Vec::with_capacity(num_queries);

    for query in &dataset.folded_queries {
        let mut gt_scored: Vec<(NodeIndex, SimilarityScore)> = dataset
            .folded_corpus
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx as NodeIndex, (query.dot_product_complex(doc)).re))
            .collect();
        gt_scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let gt_topk: HashSet<NodeIndex> = gt_scored.into_iter().take(k).map(|(id, _)| id).collect();

        let t0 = Instant::now();
        // 1. Initial Rivero seed
        let (seed_cands, _) = index
            .search_indices_o1_with_config(query, k, None, &rivero_cfg)
            .unwrap();

        for (s, _) in &seed_cands {
            if gt_topk.contains(s) {
                rivero_hits += 1;
            }
        }

        // 2. Global proof
        let query_lut = LutzQueryTable::build(query);
        let (certified, diag) = LutzGlobalCertified::certify_global(
            &query_lut,
            k,
            &seed_cands,
            n,
            None,
            |_| true,
            |slot| Some(&lutz_codes[slot as usize]),
            |slot| (query.dot_product_complex(&dataset.folded_corpus[slot as usize])).re,
        );
        let elapsed_us = t0.elapsed().as_secs_f64() * 1_000_000.0;
        latencies.push(elapsed_us);

        for (s, _) in &certified {
            if gt_topk.contains(s) {
                certified_hits += 1;
            }
        }

        total_l0_elim += diag.l0_eliminations;
        total_l1_refine += diag.l1_refinements;
        total_exact_escalations += diag.exact_escalations;
    }

    let p50 = percentile(latencies, 50.0);
    let total_gt = (num_queries * k) as f64;

    GlobalSweepStats {
        candidate_cap: cap,
        rivero_recall: (rivero_hits as f64 / total_gt) * 100.0,
        certified_recall: (certified_hits as f64 / total_gt) * 100.0,
        l0_elim_per_q: total_l0_elim as f64 / num_queries as f64,
        l1_refine_per_q: total_l1_refine as f64 / num_queries as f64,
        exact_escalate_per_q: total_exact_escalations as f64 / num_queries as f64,
        latency_p50_us: p50,
        qps: 1_000_000.0 / p50.max(0.1),
    }
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR DENSE RECALL FUNNEL & GLOBAL CERTIFIED RETRIEVAL BENCHMARK                     ║"
    );
    println!(
        "║ (N=25,000, D=1536, K=10)                                                             ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let n = 25_000;
    let real_dim = 1536;
    let complex_dim = 768;
    let k = 10;
    let num_queries = 64;

    println!("Generating realistic synthetic corpus (N={n}, D={real_dim})...");
    let dataset =
        generate_realistic_text_corpus(n, num_queries, real_dim, common::DEFAULT_BENCH_SEED);

    // =========================================================================
    // SECTION 1: DETAILED RECALL FUNNEL (DistanceFunction::Cosine, Cap=2048)
    // =========================================================================
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 1: GROUND TRUTH SURVIVAL FUNNEL (Cap=2048, DistanceFunction::Cosine)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let (funnel, index) =
        evaluate_funnel_for_profile(DistanceFunction::Cosine, 2048, &dataset, complex_dim, k);
    let total_gt = funnel.total_gt as f64;

    println!("  ┌────────────────────────────────────────┬─────────────┬─────────────┐");
    println!("  │ Pipeline Stage                         │ GT Hits/Q   │ Recall %    │");
    println!("  ├────────────────────────────────────────┼─────────────┼─────────────┤");
    println!(
        "  │ 0. Exact Ground Truth (Target Top-10)  │ {:>7.3} / 10│     100.00% │",
        10.0
    );
    println!(
        "  │ 1. Raw Rivero Territory Probes         │ {:>7.3} / 10│ {:>10.2}% │",
        funnel.raw_route_hits as f64 / funnel.total_queries as f64,
        (funnel.raw_route_hits as f64 / total_gt) * 100.0
    );
    println!(
        "  │ 2. Candidate Cap Truncation (Cap=2048) │ {:>7.3} / 10│ {:>10.2}% │",
        funnel.post_vote_hits as f64 / funnel.total_queries as f64,
        (funnel.post_vote_hits as f64 / total_gt) * 100.0
    );
    println!(
        "  │ 3. Witness Expansion (2-Hop Graph)     │ {:>7.3} / 10│ {:>10.2}% │",
        funnel.post_witness_hits as f64 / funnel.total_queries as f64,
        (funnel.post_witness_hits as f64 / total_gt) * 100.0
    );
    println!(
        "  │ 4. LUTz L0/L1 Candidate Certification  │ {:>7.3} / 10│ {:>10.2}% │",
        funnel.lutz_hits as f64 / funnel.total_queries as f64,
        (funnel.lutz_hits as f64 / total_gt) * 100.0
    );
    println!(
        "  │ 5. Final Top-K Exact Output            │ {:>7.3} / 10│ {:>10.2}% │",
        funnel.final_hits as f64 / funnel.total_queries as f64,
        (funnel.final_hits as f64 / total_gt) * 100.0
    );
    println!("  └────────────────────────────────────────┴─────────────┴─────────────┘");
    println!(
        "  • Queries hitting candidate cap:   {:>5.1}%",
        (funnel.saturated_queries as f64 / funnel.total_queries as f64) * 100.0
    );
    println!(
        "  • Avg raw candidates from probes:  {:>7.1}",
        funnel.avg_raw_candidates
    );
    println!(
        "  • Avg candidates after witnesses:  {:>7.1}",
        funnel.avg_post_witness_candidates
    );

    // =========================================================================
    // SECTION 2: GLOBAL CERTIFIED RETRIEVAL SWEEP ACROSS CANDIDATE CAPS
    // =========================================================================
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 2: GLOBAL CERTIFIED RETRIEVAL SWEEP ACROSS CANDIDATE CAPS");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    println!(
        "  ┌──────────────┬───────────────┬──────────────────┬──────────────┬──────────────┬──────────────┬────────────┬─────────────┐"
    );
    println!(
        "  │ CandidateCap │ Rivero Recall │ Certified Recall │ L0 Elim/Q    │ L1 Refine/Q  │ Exact Esc/Q  │ Lat (p50)  │ QPS         │"
    );
    println!(
        "  ├──────────────┼───────────────┼──────────────────┼──────────────┼──────────────┼──────────────┼────────────┼─────────────┤"
    );

    let caps = [256, 512, 1024, 2048, 4096, 8192];
    for &cap in &caps {
        let s = evaluate_global_certified_sweep(cap, &dataset, complex_dim, k, &index);
        println!(
            "  │ {:>12} │ {:>12.2}% │ {:>15.3}% │ {:>12.1} │ {:>12.1} │ {:>12.1} │ {:>8.1} µs│ {:>10.0}  │",
            s.candidate_cap,
            s.rivero_recall,
            s.certified_recall,
            s.l0_elim_per_q,
            s.l1_refine_per_q,
            s.exact_escalate_per_q,
            s.latency_p50_us,
            s.qps
        );
    }
    println!(
        "  └──────────────┴───────────────┴──────────────────┴──────────────┴──────────────┴──────────────┴────────────┴─────────────┘\n"
    );
}
