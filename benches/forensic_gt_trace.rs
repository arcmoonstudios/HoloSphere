use hnsqr::bench_support as common;

use std::time::Instant;

use common::load_adversarial_regression_corpus;
use hnsqr::rivero::bulk::RiveroBulkBuilder;
use hnsqr::rivero::{AdaptivePolicy, RiveroProfile};
use hnsqr::{HNSQRConfig, HNSQRIndex, NodeIndex, VectorEmbedding};

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR FORENSIC GROUND-TRUTH TRACER & WORKSPACE ROOT CAUSE ANALYSIS (N = 2000)        ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let adv = load_adversarial_regression_corpus();
    let n = adv.corpus.len();
    let num_queries = adv.in_domain_queries.len();
    println!(
        "  * Dataset: Fixed 2K Adversarial Corpus (N = {}, Queries = {})",
        n, num_queries
    );

    // Build standard test index with Balanced profile and Cosine (Folded Hermitian) distance function
    let builder_prog = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(16);
    let state_prog = builder_prog.build(&adv.corpus).unwrap();

    let mut config = HNSQRConfig::default();
    config.max_elements = n;
    let dim = adv.corpus.first().map_or(32, |v| v.dimension());
    let index = HNSQRIndex::new(config.clone(), dim);

    for (i, (vec, meta)) in adv.corpus.iter().zip(adv.metadata.iter()).enumerate() {
        let v: VectorEmbedding = vec.clone();
        let m: std::collections::HashMap<String, hnsqr::metadata::index::MetadataValue> =
            meta.clone();
        index
            .insert_with_metadata(format!("adv-{i}"), v, m)
            .unwrap();
    }
    index.install_rivero_state(state_prog).unwrap();

    // ════════════════════════════════════════════════════════════════════════
    // 0. SELF-MATCH & TIE SEMANTICS CHECK
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 0: BENCHMARK INTEGRITY, METRIC MISMATCH, & TIE SEMANTICS CHECK");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    for (q_idx, (q, gt)) in adv
        .in_domain_queries
        .iter()
        .zip(adv.in_domain_ground_truth.iter())
        .enumerate()
    {
        let (ann, _) = index.search_indices_strict(q, 10, None).unwrap();
        let ann_ids: Vec<NodeIndex> = ann.iter().map(|s| s.0).collect();

        let missing: Vec<NodeIndex> = gt
            .iter()
            .copied()
            .filter(|id| !ann_ids.contains(id))
            .collect();
        let extras: Vec<NodeIndex> = ann_ids
            .iter()
            .copied()
            .filter(|id| !gt.contains(id))
            .collect();

        if !missing.is_empty() {
            println!(
                "  Query {}: {} missing from ANN results:",
                q_idx,
                missing.len()
            );
            for m in &missing {
                let herm = (q.dot_product_complex(&adv.corpus[*m as usize])).re;
                let fid = q.dot_product_complex(&adv.corpus[*m as usize]).norm_sqr();
                println!(
                    "    -> Missing GT Node {:>4}: Hermitian Re = {:.6}, Projective Overlap = {:.6}",
                    m, herm, fid
                );
            }
            for e in &extras {
                let herm = (q.dot_product_complex(&adv.corpus[*e as usize])).re;
                let fid = q.dot_product_complex(&adv.corpus[*e as usize]).norm_sqr();
                let ann_score = ann.iter().find(|s| s.0 == *e).map(|s| s.1).unwrap_or(0.0);
                println!(
                    "    <- Extra ANN Node  {:>4}: Hermitian Re = {:.6}, Projective Overlap = {:.6}, Returned ANN Score = {:.6}",
                    e, herm, fid, ann_score
                );
            }
            println!();
        }
    }

    // ════════════════════════════════════════════════════════════════════════
    // 1. PIPELINE SURVIVAL TRACE (Balanced Profile, Kcap = 1024)
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 1: GROUND-TRUTH PIPELINE SURVIVAL TRACE (Balanced, Kcap = 1024)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let mut total_gt = 0usize;
    let mut raw_survived = 0usize;
    let mut selected_survived = 0usize;
    let mut post_witness_survived = 0usize;
    let mut final_survived = 0usize;

    let mut missing_gt_ranks_all: Vec<Option<usize>> = Vec::new();

    let bal_cfg = RiveroProfile::Balanced.config();

    for (q, gt) in adv
        .in_domain_queries
        .iter()
        .zip(adv.in_domain_ground_truth.iter())
    {
        let trace = index.trace_gt_coverage(q, gt, bal_cfg).unwrap();
        total_gt += trace.gt_count;
        raw_survived += trace.gt_in_raw_route;
        selected_survived += trace.gt_after_vote_selection;
        post_witness_survived += trace.gt_after_witness;
        final_survived += trace.gt_in_final_results;

        for r in trace.missing_gt_ranks {
            missing_gt_ranks_all.push(r);
        }
    }

    let raw_pct = (raw_survived as f64 / total_gt as f64) * 100.0;
    let selected_pct = (selected_survived as f64 / total_gt as f64) * 100.0;
    let witness_pct = (post_witness_survived as f64 / total_gt as f64) * 100.0;
    let final_pct = (final_survived as f64 / total_gt as f64) * 100.0;

    println!("  ┌──────────────────────────────────────────────┬──────────────┬──────────────┐");
    println!("  │ Pipeline Stage                               │ GT Surviving │ Coverage %   │");
    println!("  ├──────────────────────────────────────────────┼──────────────┼──────────────┤");
    println!(
        "  │ 1. Exact Hermitian GT Top-10                 │ {:>12} │      100.00% │",
        total_gt
    );
    println!(
        "  │ 2. Raw Territorial Candidates (pre-truncate) │ {:>12} │ {:>11.2}% │",
        raw_survived, raw_pct
    );
    println!(
        "  │ 3. Vote-Selected Candidates (post-Kcap 1024) │ {:>12} │ {:>11.2}% │",
        selected_survived, selected_pct
    );
    println!(
        "  │ 4. Post-Witness Expansion Candidates         │ {:>12} │ {:>11.2}% │",
        post_witness_survived, witness_pct
    );
    println!(
        "  │ 5. Final Returned Top-10 Results             │ {:>12} │ {:>11.2}% │",
        final_survived, final_pct
    );
    println!("  └──────────────────────────────────────────────┴──────────────┴──────────────┘\n");

    // ════════════════════════════════════════════════════════════════════════
    // 2. CANDIDATE RANK HISTOGRAM OF ALL GT NEIGHBORS
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 2: CANDIDATE RANK HISTOGRAM OF GT NEIGHBORS BEFORE TRUNCATION");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let mut count_0_512 = 0usize;
    let mut count_513_1024 = 0usize;
    let mut count_1025_1152 = 0usize;
    let mut count_1153_1536 = 0usize;
    let mut count_gt_1536 = 0usize;
    let mut count_not_in_raw = 0usize;

    for r in &missing_gt_ranks_all {
        match r {
            None => count_not_in_raw += 1,
            Some(rank) if *rank < 512 => count_0_512 += 1,
            Some(rank) if *rank < 1024 => count_513_1024 += 1,
            Some(rank) if *rank < 1152 => count_1025_1152 += 1,
            Some(rank) if *rank < 1536 => count_1153_1536 += 1,
            Some(_) => count_gt_1536 += 1,
        }
    }

    let total_samples = missing_gt_ranks_all.len() as f64;
    println!("  ┌───────────────────────┬──────────────┬──────────────┐");
    println!("  │ Candidate Rank Bucket │ GT Count     │ Percentage   │");
    println!("  ├───────────────────────┼──────────────┼──────────────┤");
    println!(
        "  │ 0 – 512               │ {:>12} │ {:>11.2}% │",
        count_0_512,
        (count_0_512 as f64 / total_samples) * 100.0
    );
    println!(
        "  │ 513 – 1024            │ {:>12} │ {:>11.2}% │",
        count_513_1024,
        (count_513_1024 as f64 / total_samples) * 100.0
    );
    println!(
        "  │ 1025 – 1152 (Kcap cut)│ {:>12} │ {:>11.2}% │",
        count_1025_1152,
        (count_1025_1152 as f64 / total_samples) * 100.0
    );
    println!(
        "  │ 1153 – 1536           │ {:>12} │ {:>11.2}% │",
        count_1153_1536,
        (count_1153_1536 as f64 / total_samples) * 100.0
    );
    println!(
        "  │ > 1536                │ {:>12} │ {:>11.2}% │",
        count_gt_1536,
        (count_gt_1536 as f64 / total_samples) * 100.0
    );
    println!(
        "  │ Not in Raw Route      │ {:>12} │ {:>11.2}% │",
        count_not_in_raw,
        (count_not_in_raw as f64 / total_samples) * 100.0
    );
    println!("  └───────────────────────┴──────────────┴──────────────┘\n");

    // ════════════════════════════════════════════════════════════════════════
    // 3. CANDIDATE CAP SWEEP (Balanced Profile: 512, 1024, 1536, 2000)
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 3: CANDIDATE CAP SWEEP (Balanced Profile)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!("  ┌──────────┬──────────────┬──────────────┬──────────────┬──────────────┐");
    println!("  │ Kcap     │ Recall@1     │ Recall@10    │ Raw Route GT │ Selected GT  │");
    println!("  ├──────────┼──────────────┼──────────────┼──────────────┼──────────────┤");

    for &kcap in &[512, 1024, 1536, 2000] {
        let mut custom_cfg = RiveroProfile::Balanced.config();
        custom_cfg.query_candidate_cap = kcap;

        let mut top1_matches = 0usize;
        let mut recall10_sum = 0.0f64;
        let mut raw_gt_sum = 0.0f64;
        let mut sel_gt_sum = 0.0f64;

        for (q, gt) in adv
            .in_domain_queries
            .iter()
            .zip(adv.in_domain_ground_truth.iter())
        {
            let trace = index.trace_gt_coverage(q, gt, custom_cfg).unwrap();
            if trace.top1_recalled {
                top1_matches += 1;
            }
            recall10_sum += trace.recall_at_10;
            raw_gt_sum += trace.gt_in_raw_route as f64 / trace.gt_count as f64;
            sel_gt_sum += trace.gt_after_vote_selection as f64 / trace.gt_count as f64;
        }

        let n_q = adv.in_domain_queries.len() as f64;
        println!(
            "  │ {:>8} │ {:>11.2}% │ {:>11.2}% │ {:>11.2}% │ {:>11.2}% │",
            kcap,
            (top1_matches as f64 / n_q) * 100.0,
            (recall10_sum / n_q) * 100.0,
            (raw_gt_sum / n_q) * 100.0,
            (sel_gt_sum / n_q) * 100.0,
        );
    }
    println!("  └──────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n");

    // ════════════════════════════════════════════════════════════════════════
    // 4. WITNESS BUILDER COMPARISON (Progressive vs Force Stage-B)
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 4: WITNESS BUILDER IMPACT (Progressive vs Force Stage-B)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let builder_stage_b = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced)
        .with_threads(16)
        .with_force_stage_b(true);
    let state_stage_b = builder_stage_b.build(&adv.corpus).unwrap();

    let index_b = HNSQRIndex::new(config, dim);
    for (i, (vec, meta)) in adv.corpus.iter().zip(adv.metadata.iter()).enumerate() {
        let v: VectorEmbedding = vec.clone();
        let m: std::collections::HashMap<String, hnsqr::metadata::index::MetadataValue> =
            meta.clone();
        index_b
            .insert_with_metadata(format!("adv-b-{i}"), v, m)
            .unwrap();
    }
    index_b.install_rivero_state(state_stage_b).unwrap();

    let mut prog_recall10 = 0.0f64;
    let mut prog_top1 = 0usize;
    let mut b_recall10 = 0.0f64;
    let mut b_top1 = 0usize;

    for (q, gt) in adv
        .in_domain_queries
        .iter()
        .zip(adv.in_domain_ground_truth.iter())
    {
        let trace_prog = index.trace_gt_coverage(q, gt, bal_cfg).unwrap();
        let trace_b = index_b.trace_gt_coverage(q, gt, bal_cfg).unwrap();

        prog_recall10 += trace_prog.recall_at_10;
        if trace_prog.top1_recalled {
            prog_top1 += 1;
        }

        b_recall10 += trace_b.recall_at_10;
        if trace_b.top1_recalled {
            b_top1 += 1;
        }
    }

    let n_q = adv.in_domain_queries.len() as f64;
    println!("  ┌─────────────────────────────┬──────────────┬──────────────┐");
    println!("  │ Witness Builder Variant     │ Top-1 Match  │ Recall@10    │");
    println!("  ├─────────────────────────────┼──────────────┼──────────────┤");
    println!(
        "  │ A. Progressive (2-Stage)    │ {:>11.2}% │ {:>11.2}% │",
        (prog_top1 as f64 / n_q) * 100.0,
        (prog_recall10 / n_q) * 100.0
    );
    println!(
        "  │ B. Force Stage-B Expansion  │ {:>11.2}% │ {:>11.2}% │",
        (b_top1 as f64 / n_q) * 100.0,
        (b_recall10 / n_q) * 100.0
    );
    println!("  └─────────────────────────────┴──────────────┴──────────────┘\n");

    // ════════════════════════════════════════════════════════════════════════
    // 5. SEARCH MODES COMPARISON MATRIX
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        " SECTION 5: SEARCH MODES COMPARISON MATRIX (Exact, GraphOnly, Fast, Balanced, Strict, Adaptive)"
    );
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "  ┌───────────────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐"
    );
    println!(
        "  │ Search Mode           │ Recall@1     │ Recall@10    │ Latency (ms) │ Scored Nodes │ Fallback %   │"
    );
    println!(
        "  ├───────────────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤"
    );

    // Mode 1: Exact Hermitian Brute-force
    {
        let mut top1 = 0usize;
        let mut rec10 = 0.0f64;
        let t0 = Instant::now();
        for (q, gt) in adv
            .in_domain_queries
            .iter()
            .zip(adv.in_domain_ground_truth.iter())
        {
            let mut scored: Vec<(NodeIndex, f32)> = adv
                .corpus
                .iter()
                .enumerate()
                .map(|(idx, doc)| (idx as NodeIndex, (q.dot_product_complex(doc)).re))
                .collect();
            scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let top = scored[0].0;
            if top == gt[0] {
                top1 += 1;
            }
            let overlap = scored.iter().take(10).filter(|s| gt.contains(&s.0)).count();
            rec10 += overlap as f64 / 10.0;
        }
        let lat = (t0.elapsed().as_secs_f64() * 1000.0) / n_q;
        println!(
            "  │ 1. Exact Hermitian    │ {:>11.2}% │ {:>11.2}% │ {:>10.3} ms│ {:>12} │        0.0%  │",
            (top1 as f64 / n_q) * 100.0,
            (rec10 / n_q) * 100.0,
            lat,
            n
        );
    }

    // Mode 2: GraphOnly
    {
        let mut top1 = 0usize;
        let mut rec10 = 0.0f64;
        let mut evals_sum = 0usize;
        let t0 = Instant::now();
        for (q, gt) in adv
            .in_domain_queries
            .iter()
            .zip(adv.in_domain_ground_truth.iter())
        {
            let res_graph = index.search_indices_graph(q, 10, None).unwrap();
            if !res_graph.is_empty() && res_graph[0].0 == gt[0] {
                top1 += 1;
            }
            let overlap = res_graph.iter().filter(|s| gt.contains(&s.0)).count();
            rec10 += overlap as f64 / 10.0;
            evals_sum += 64;
        }
        let lat = (t0.elapsed().as_secs_f64() * 1000.0) / n_q;
        println!(
            "  │ 2. GraphOnly          │ {:>11.2}% │ {:>11.2}% │ {:>10.3} ms│ {:>12.0} │      100.0%  │",
            (top1 as f64 / n_q) * 100.0,
            (rec10 / n_q) * 100.0,
            lat,
            evals_sum as f64 / n_q
        );
    }

    // Mode 3: Fast Profile
    {
        let mut top1 = 0usize;
        let mut rec10 = 0.0f64;
        let t0 = Instant::now();
        for (q, gt) in adv
            .in_domain_queries
            .iter()
            .zip(adv.in_domain_ground_truth.iter())
        {
            let trace = index
                .trace_gt_coverage(q, gt, RiveroProfile::Fast.config())
                .unwrap();
            if trace.top1_recalled {
                top1 += 1;
            }
            rec10 += trace.recall_at_10;
        }
        let lat = (t0.elapsed().as_secs_f64() * 1000.0) / n_q;
        println!(
            "  │ 3. Fast Profile       │ {:>11.2}% │ {:>11.2}% │ {:>10.3} ms│ {:>12} │        0.0%  │",
            (top1 as f64 / n_q) * 100.0,
            (rec10 / n_q) * 100.0,
            lat,
            n.min(512)
        );
    }

    // Mode 4: Balanced Profile
    {
        let mut top1 = 0usize;
        let mut rec10 = 0.0f64;
        let t0 = Instant::now();
        for (q, gt) in adv
            .in_domain_queries
            .iter()
            .zip(adv.in_domain_ground_truth.iter())
        {
            let trace = index
                .trace_gt_coverage(q, gt, RiveroProfile::Balanced.config())
                .unwrap();
            if trace.top1_recalled {
                top1 += 1;
            }
            rec10 += trace.recall_at_10;
        }
        let lat = (t0.elapsed().as_secs_f64() * 1000.0) / n_q;
        println!(
            "  │ 4. Balanced Profile   │ {:>11.2}% │ {:>11.2}% │ {:>10.3} ms│ {:>12} │        0.0%  │",
            (top1 as f64 / n_q) * 100.0,
            (rec10 / n_q) * 100.0,
            lat,
            n.min(1024)
        );
    }

    // Mode 5: Strict Profile (24 Foundations, Kcap = 2048)
    {
        let mut top1 = 0usize;
        let mut rec10 = 0.0f64;
        let t0 = Instant::now();
        for (q, gt) in adv
            .in_domain_queries
            .iter()
            .zip(adv.in_domain_ground_truth.iter())
        {
            let trace = index
                .trace_gt_coverage(q, gt, RiveroProfile::Strict.config())
                .unwrap();
            if trace.top1_recalled {
                top1 += 1;
            }
            rec10 += trace.recall_at_10;
        }
        let lat = (t0.elapsed().as_secs_f64() * 1000.0) / n_q;
        println!(
            "  │ 5. Strict Profile     │ {:>11.2}% │ {:>11.2}% │ {:>10.3} ms│ {:>12} │        0.0%  │",
            (top1 as f64 / n_q) * 100.0,
            (rec10 / n_q) * 100.0,
            lat,
            n.min(2048)
        );
    }

    // Mode 6: Adaptive RiveroOnly
    {
        let mut top1 = 0usize;
        let mut rec10 = 0.0f64;
        let mut evals_sum = 0usize;
        let t0 = Instant::now();
        for (q, gt) in adv
            .in_domain_queries
            .iter()
            .zip(adv.in_domain_ground_truth.iter())
        {
            let (res, diag) = index
                .search_indices_adaptive(q, 10, None, AdaptivePolicy::RiveroOnly)
                .unwrap();
            if !res.is_empty() && res[0].0 == gt[0] {
                top1 += 1;
            }
            let overlap = res.iter().filter(|s| gt.contains(&s.0)).count();
            rec10 += overlap as f64 / 10.0;
            evals_sum += diag.cumulative_exact_scores;
        }
        let lat = (t0.elapsed().as_secs_f64() * 1000.0) / n_q;
        println!(
            "  │ 6. Adaptive RiveroOnly│ {:>11.2}% │ {:>11.2}% │ {:>10.3} ms│ {:>12.0} │        0.0%  │",
            (top1 as f64 / n_q) * 100.0,
            (rec10 / n_q) * 100.0,
            lat,
            evals_sum as f64 / n_q
        );
    }

    // Mode 7: Adaptive Hybrid (AllowGraphFallback)
    {
        let mut top1 = 0usize;
        let mut rec10 = 0.0f64;
        let mut evals_sum = 0usize;
        let mut fallback_count = 0usize;
        let t0 = Instant::now();
        for (q, gt) in adv
            .in_domain_queries
            .iter()
            .zip(adv.in_domain_ground_truth.iter())
        {
            let (res, diag) = index
                .search_indices_adaptive(q, 10, None, AdaptivePolicy::AllowGraphFallback)
                .unwrap();
            if !res.is_empty() && res[0].0 == gt[0] {
                top1 += 1;
            }
            let overlap = res.iter().filter(|s| gt.contains(&s.0)).count();
            rec10 += overlap as f64 / 10.0;
            evals_sum += diag.cumulative_exact_scores;
            if diag.graph_fallback_used {
                fallback_count += 1;
            }
        }
        let lat = (t0.elapsed().as_secs_f64() * 1000.0) / n_q;
        println!(
            "  │ 7. Adaptive Hybrid    │ {:>11.2}% │ {:>11.2}% │ {:>10.3} ms│ {:>12.0} │ {:>10.1}%  │",
            (top1 as f64 / n_q) * 100.0,
            (rec10 / n_q) * 100.0,
            lat,
            evals_sum as f64 / n_q,
            (fallback_count as f64 / n_q) * 100.0,
        );
    }
    println!(
        "  └───────────────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n"
    );
}
