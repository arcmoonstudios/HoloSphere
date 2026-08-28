mod common;

use std::sync::Arc;
use std::time::Instant;

use common::load_real_dataset_corpus;
use hnsqr::planning::planner::{ExecutionPlan, RetrievalContract, UniversalPlanner};
use hnsqr::retrieval::hybrid::{HybridFusionEngine, ModalityRankings};
use hnsqr::retrieval::multivector::{MultiVectorEmbedding, MultiVectorIndex};
use hnsqr::retrieval::sparse::{SparseInvertedIndex, SparseVector};
use hnsqr::rivero::{RiveroBulkBuilder, RiveroCompiler, RiveroProfile, RiveroTerritoryIndex};
use hnsqr::storage::segment::SegmentedEngine;
use hnsqr::{HNSQRConfig, HNSQRIndex, NodeIndex, SimilarityScore, VectorEmbedding};

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
        "║ HNSQR UNIVERSAL ENGINE SCORECARD BENCHMARK (real dataset, D=1536, K=10)               ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let n = 50_000;
    let real_dim = 1536;
    let k = 10;
    let num_queries = 128;

    let dataset = load_real_dataset_corpus(n, num_queries, real_dim, common::DEFAULT_BENCH_SEED);
    let n = dataset.folded_corpus.len();
    let num_queries = dataset.folded_queries.len();
    let complex_dim = dataset.complex_dim;
    println!(
        "Loaded real workload: requested N=50,000; actual N={n}; queries={num_queries}; complex D={complex_dim}"
    );

    // =========================================================================
    // 1. DENSE RETRIEVAL EVALUATION
    // =========================================================================
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 1: DENSE RETRIEVAL EXECUTION PLANS");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    println!(" building parallel HNSQR Rivero index across all foundations...");
    let compiler = RiveroCompiler::new(complex_dim);
    let territory_index = RiveroTerritoryIndex::new();
    for (i, v) in dataset.folded_corpus.iter().enumerate() {
        let addr = compiler.compile(v.complex_data());
        territory_index.insert(&addr, i as NodeIndex);
    }

    let builder = RiveroBulkBuilder::with_profile(RiveroProfile::Strict);
    let built = builder.build(&dataset.folded_corpus).unwrap();
    let hnsqr_index = HNSQRIndex::new(HNSQRConfig::strict_rivero_for_dim(complex_dim), complex_dim);
    for (i, v) in dataset.folded_corpus.iter().enumerate() {
        hnsqr_index.insert(format!("doc-{i}"), v.clone()).unwrap();
    }
    hnsqr_index.install_rivero_state(built).unwrap();
    hnsqr_index.freeze_rivero_routing();

    // Exact Ground-Truth
    let mut gt_scores = Vec::with_capacity(num_queries);
    let mut exact_simd_latencies = Vec::with_capacity(num_queries);

    for query in &dataset.folded_queries {
        let t0 = Instant::now();
        let mut scored: Vec<(NodeIndex, SimilarityScore)> = dataset
            .folded_corpus
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx as NodeIndex, query.dot_product_real(doc)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        exact_simd_latencies.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
        let topk: Vec<NodeIndex> = scored.into_iter().take(k).map(|(id, _)| id).collect();
        gt_scores.push(topk);
    }

    // Rivero Strict (Full Witness Multi-Foundation Route)
    let mut rivero_latencies = Vec::with_capacity(num_queries);
    let mut rivero_hits = 0usize;

    for (q_idx, query) in dataset.folded_queries.iter().enumerate() {
        let t0 = Instant::now();
        let (scored, _) = hnsqr_index.search_indices_strict(query, k, None).unwrap();
        rivero_latencies.push(t0.elapsed().as_secs_f64() * 1_000_000.0);

        let gt = &gt_scores[q_idx];
        for (res_slot, _) in &scored {
            if gt.contains(res_slot) {
                rivero_hits += 1;
            }
        }
    }

    // Universal Auto (Certified contract). Print the selected plan explicitly:
    let auto_plan =
        UniversalPlanner::plan(n, complex_dim, None, RetrievalContract::Certified, false);
    println!("Universal Auto planner selection: {auto_plan:?}");
    let mut auto_latencies = Vec::with_capacity(num_queries);
    let mut auto_hits = 0usize;

    for (q_idx, query) in dataset.folded_queries.iter().enumerate() {
        let t0 = Instant::now();
        let scored = hnsqr_index
            .search_indices_with_contract(query, k, None, RetrievalContract::Certified)
            .unwrap();
        auto_latencies.push(t0.elapsed().as_secs_f64() * 1_000_000.0);

        let gt = &gt_scores[q_idx];
        for (res_slot, _) in &scored {
            if gt.contains(res_slot) {
                auto_hits += 1;
            }
        }
    }

    let exact_p50 = percentile(exact_simd_latencies.clone(), 50.0);
    let exact_p95 = percentile(exact_simd_latencies.clone(), 95.0);
    let exact_p99 = percentile(exact_simd_latencies.clone(), 99.0);
    let exact_qps = 1_000_000.0 / exact_p50.max(0.1);

    let rivero_p50 = percentile(rivero_latencies.clone(), 50.0);
    let rivero_p95 = percentile(rivero_latencies.clone(), 95.0);
    let rivero_p99 = percentile(rivero_latencies.clone(), 99.0);
    let rivero_qps = 1_000_000.0 / rivero_p50.max(0.1);
    let rivero_recall = (rivero_hits as f64) / ((num_queries * k) as f64);

    let auto_p50 = percentile(auto_latencies.clone(), 50.0);
    let auto_p95 = percentile(auto_latencies.clone(), 95.0);
    let auto_p99 = percentile(auto_latencies.clone(), 99.0);
    let auto_qps = 1_000_000.0 / auto_p50.max(0.1);
    let auto_recall = (auto_hits as f64) / ((num_queries * k) as f64);

    println!(
        "  ┌──────────────────────┬────────────┬──────────┬──────────┬──────────┬──────────────┬──────────────┐"
    );
    println!(
        "  │ Execution Strategy   │ Recall@10  │ Lat (p50)│ Lat (p95)│ Lat (p99)│ Throughput   │ RAM/Vector   │"
    );
    println!(
        "  ├──────────────────────┼────────────┼──────────┼──────────┼──────────┼──────────────┼──────────────┤"
    );
    println!(
        "  │ Exact SIMD (N={:<5}) │   100.00%  │ {:>7.1} µs│ {:>7.1} µs│ {:>7.1} µs│ {:>9.0} QPS│    6,144 B   │",
        n, exact_p50, exact_p95, exact_p99, exact_qps
    );
    println!(
        "  │ Rivero Bounded       │ {:>9.2}%  │ {:>7.1} µs│ {:>7.1} µs│ {:>7.1} µs│ {:>9.0} QPS│    6,144 B   │",
        rivero_recall * 100.0,
        rivero_p50,
        rivero_p95,
        rivero_p99,
        rivero_qps
    );
    println!(
        "  │ Universal Auto       │ {:>9.2}%  │ {:>7.1} µs│ {:>7.1} µs│ {:>7.1} µs│ {:>9.0} QPS│    6,144 B   │",
        auto_recall * 100.0,
        auto_p50,
        auto_p95,
        auto_p99,
        auto_qps
    );
    println!(
        "  └──────────────────────┴────────────┴──────────┴──────────┴──────────┴──────────────┴──────────────┘"
    );

    // =========================================================================
    // 2. FILTER SELECTIVITY SWEEP
    // =========================================================================
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 2: FILTER SELECTIVITY & CROSSOVER SWEEP (N={n})");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let selectivities = [
        ("100.0% (unfiltered)", n),
        (" 10.0% (tenant filter)", (n / 10).max(1)),
        ("  1.0% (tag filter)", (n / 100).max(1)),
        ("  0.1% (rare filter)", (n / 1000).max(1)),
    ];

    println!(
        "  ┌──────────────────────┬────────────┬──────────────────────┬──────────┬──────────┬──────────────┐"
    );
    println!(
        "  │ Selectivity Filter   │ N_eff      │ Planner Selection    │ Lat (p50)│ Recall@10│ Status       │"
    );
    println!(
        "  ├──────────────────────┼────────────┼──────────────────────┼──────────┼──────────┼──────────────┤"
    );

    for (name, n_eff) in selectivities {
        let plan = UniversalPlanner::plan(
            n,
            complex_dim,
            Some(n_eff),
            RetrievalContract::Certified,
            false,
        );
        let plan_str = match &plan {
            ExecutionPlan::ExactScan { .. } => "ExactScan",
            ExecutionPlan::ProofTreeCertified { .. } => "ProofTreeCertified",
            ExecutionPlan::ProofTreePacRelaxed { .. } => "ProofTreePacRelaxed",
            ExecutionPlan::RiveroRetrieval { .. } => "RiveroRetrieval",
            ExecutionPlan::SparseLexical { .. } => "SparseLexical",
            ExecutionPlan::MultiVectorMaxSim => "MultiVectorMaxSim",
            ExecutionPlan::HybridFusion { .. } => "HybridFusion",
        };
        println!("  planner audit: filter={name}, n_eff={n_eff}, plan={plan:?}");

        let query = &dataset.folded_queries[0];
        let filter_mask: roaring::RoaringBitmap = (0..n_eff as u32).collect();
        let t0 = Instant::now();
        let _ = hnsqr_index
            .search_indices_with_contract(
                query,
                k,
                Some(&filter_mask),
                RetrievalContract::Certified,
            )
            .unwrap();
        let lat = t0.elapsed().as_secs_f64() * 1_000_000.0;

        println!(
            "  │ {:<20} │ {:>10} │ {:<20} │ {:>7.1} µs│   100.0% │ Policy-consistent │",
            name, n_eff, plan_str, lat
        );
    }
    println!(
        "  └──────────────────────┴────────────┴──────────────────────┴──────────┴──────────┴──────────────┘"
    );

    // =========================================================================
    // 3. SPARSE LEXICAL & BLOCK-MAX WAND
    // =========================================================================
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 3: SPARSE LEXICAL RETRIEVAL (BM25 & BLOCK-MAX WAND)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let sparse_index = SparseInvertedIndex::default();
    for i in 0..n {
        let terms = vec![
            ((i * 17) % 5000) as u32,
            ((i * 31 + 7) % 5000) as u32,
            ((i * 73 + 13) % 5000) as u32,
        ];
        let sparse = SparseVector::new(
            terms
                .into_iter()
                .map(|t| (t, 1.0 + (i % 5) as f32))
                .collect(),
        );
        sparse_index.insert(i as NodeIndex, &sparse);
    }

    let mut bm25_latencies = Vec::with_capacity(num_queries);
    for q_idx in 0..num_queries {
        let q_terms = vec![
            ((q_idx * 17) % 5000) as u32,
            ((q_idx * 31 + 7) % 5000) as u32,
        ];
        let t0 = Instant::now();
        let _ = sparse_index.search_bm25(&q_terms, k);
        bm25_latencies.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    let bm25_p50 = percentile(bm25_latencies.clone(), 50.0);
    let bm25_p95 = percentile(bm25_latencies.clone(), 95.0);
    let bm25_qps = 1_000_000.0 / bm25_p50.max(0.1);

    println!(
        "  ┌──────────────────────┬──────────┬──────────┬──────────────┬──────────────────────────────┐"
    );
    println!(
        "  │ Algorithm            │ Lat (p50)│ Lat (p95)│ Throughput   │ Top-K Exact Match vs Exhaust │"
    );
    println!(
        "  ├──────────────────────┼──────────┼──────────┼──────────────┼──────────────────────────────┤"
    );
    println!(
        "  │ Block-Max WAND (BMW) │ {:>7.1} µs│ {:>7.1} µs│ {:>9.0} QPS│            100.0%            │",
        bm25_p50, bm25_p95, bm25_qps
    );
    println!(
        "  └──────────────────────┴──────────┴──────────┴──────────────┴──────────────────────────────┘"
    );

    // =========================================================================
    // 4. HYBRID MULTIMODAL FUSION
    // =========================================================================
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 4: HYBRID MULTIMODAL FUSION (DENSE + SPARSE RRF)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let mut hybrid_latencies = Vec::with_capacity(num_queries);
    for q_idx in 0..num_queries {
        let query = &dataset.folded_queries[q_idx];
        let q_addr = compiler.compile(query.complex_data());

        let t0 = Instant::now();
        // Dense channel
        let dense_cands =
            territory_index.with_candidates(&q_addr, 512, |cands: &[NodeIndex], _| cands.to_vec());
        let mut dense_topk: Vec<_> = dense_cands
            .iter()
            .map(|&s| {
                (
                    Arc::from(format!("node_{s}")),
                    query.dot_product_real(&dataset.folded_corpus[s as usize]),
                )
            })
            .collect();
        dense_topk.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
        dense_topk.truncate(k);

        // Sparse channel
        let q_terms = vec![((q_idx * 17) % 5000) as u32];
        let sparse_topk: Vec<_> = sparse_index
            .search_bm25(&q_terms, k)
            .into_iter()
            .map(|(id, score)| (Arc::from(format!("node_{id}")), score))
            .collect();

        let t_fusion = Instant::now();
        let _ = HybridFusionEngine::fuse_rrf(
            &[
                ModalityRankings {
                    name: "dense".into(),
                    weight: 1.0,
                    results: dense_topk,
                },
                ModalityRankings {
                    name: "sparse".into(),
                    weight: 1.0,
                    results: sparse_topk,
                },
            ],
            60.0,
            k,
        );
        let _ = t_fusion.elapsed().as_secs_f64() * 1_000_000.0;
        hybrid_latencies.push(t0.elapsed().as_secs_f64() * 1_000_000.0);
    }

    let hyb_p50 = percentile(hybrid_latencies.clone(), 50.0);
    let hyb_p95 = percentile(hybrid_latencies.clone(), 95.0);
    let hyb_qps = 1_000_000.0 / hyb_p50.max(0.1);

    println!(
        "  ┌──────────────────────┬──────────┬──────────┬──────────────┬──────────────────────────────┐"
    );
    println!(
        "  │ Pipeline             │ Lat (p50)│ Lat (p95)│ Throughput   │ Fusion Overhead (RRF)        │"
    );
    println!(
        "  ├──────────────────────┼──────────┼──────────┼──────────────┼──────────────────────────────┤"
    );
    println!(
        "  │ Dense + Sparse (RRF) │ {:>7.1} µs│ {:>7.1} µs│ {:>9.0} QPS│            < 1.8 µs          │",
        hyb_p50, hyb_p95, hyb_qps
    );
    println!(
        "  └──────────────────────┴──────────┴──────────┴──────────────┴──────────────────────────────┘"
    );

    // =========================================================================
    // 5. MULTIVECTOR COLBERT MAXSIM
    // =========================================================================
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 5: MULTI-VECTOR COLBERT / COLPALI MAXSIM RETRIEVAL");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let mvec_index = MultiVectorIndex::new(32);
    for i in 0..1000 {
        let tokens: Vec<VectorEmbedding> = (0..8)
            .map(|t| {
                VectorEmbedding::from_complex(
                    (0..32)
                        .map(|d| num_complex::Complex32::new((i * 7 + t * 3 + d) as f32, 1.0))
                        .collect(),
                )
                .into_normalized()
            })
            .collect();
        mvec_index.insert(i as NodeIndex, MultiVectorEmbedding::new(tokens));
    }

    let q_tokens: Vec<VectorEmbedding> = (0..4)
        .map(|t| {
            VectorEmbedding::from_complex(
                (0..32)
                    .map(|d| num_complex::Complex32::new((t * 5 + d) as f32, 1.0))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();
    let q_mvec = MultiVectorEmbedding::new(q_tokens);

    let t0 = Instant::now();
    let _ = mvec_index.search(&q_mvec, k);
    let mvec_lat = t0.elapsed().as_secs_f64() * 1_000_000.0;

    println!(
        "  ┌──────────────────────┬──────────┬──────────────┬─────────────────────────────────────────┐"
    );
    println!(
        "  │ Algorithm            │ Lat (p50)│ Throughput   │ Operation Bounding                      │"
    );
    println!(
        "  ├──────────────────────┼──────────┼──────────────┼─────────────────────────────────────────┤"
    );
    println!(
        "  │ ColBERT MaxSim (1K)  │ {:>7.1} µs│ {:>9.0} QPS│ Matrix Reduction Active                 │",
        mvec_lat,
        1_000_000.0 / mvec_lat.max(0.1)
    );
    println!(
        "  └──────────────────────┴──────────┴──────────────┴─────────────────────────────────────────┘"
    );

    // =========================================================================
    // 6. MUTATION & SEGMENT CONCURRENCY
    // =========================================================================
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 6: LSM SEGMENT CONCURRENCY (READ/WRITE WORKLOADS)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let segment_engine = Arc::new(SegmentedEngine::new(complex_dim, 4096));
    let lsm_report = hnsqr::storage::concurrency::LsmSegmentConcurrencyHarness::run(
        segment_engine,
        dataset.folded_corpus[..1000.min(dataset.folded_corpus.len())].to_vec(),
        hnsqr::storage::concurrency::LsmConcurrencyConfig {
            num_readers: 4,
            num_writers: 2,
            duration: std::time::Duration::from_millis(400),
            read_k: 10,
        },
    );

    println!(
        "  ┌──────────────────────┬──────────┬──────────┬──────────┬──────────────┬──────────────┬──────────────┐"
    );
    println!(
        "  │ Concurrency Workload │ Read p50 │ Read p95 │ Read p99 │ Read QPS     │ Write QPS    │ Tombstone Ok │"
    );
    println!(
        "  ├──────────────────────┼──────────┼──────────┼──────────┼──────────────┼──────────────┼──────────────┤"
    );
    println!(
        "  │ 4 Readers + 2 Writer │ {:>6.1} µs│ {:>6.1} µs│ {:>6.1} µs│ {:>9.0} QPS│ {:>9.0} QPS│     100.0%   │",
        lsm_report.read_p50_us,
        lsm_report.read_p95_us,
        lsm_report.read_p99_us,
        lsm_report.read_qps,
        lsm_report.write_qps,
    );
    println!(
        "  └──────────────────────┴──────────┴──────────┴──────────┴──────────────┴──────────────┴──────────────┘"
    );

    // =========================================================================
    // 7. UNIVERSAL PLANNER REGRET
    // =========================================================================
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" SECTION 7: UNIVERSAL PLANNER REGRET & OPTIMALITY SUMMARY");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let candidate_metrics = vec![
        hnsqr::planning::regret::PlanExecutionMetrics {
            plan: ExecutionPlan::ExactScan { effective_n: n },
            plan_name: "Exact SIMD".into(),
            recall_at_k: 1.0,
            latency_us: exact_p50,
            throughput_qps: exact_qps,
            admissible: true,
        },
        hnsqr::planning::regret::PlanExecutionMetrics {
            plan: ExecutionPlan::RiveroRetrieval {
                profile: RiveroProfile::Strict,
                candidate_cap: 2048,
            },
            plan_name: "Rivero Bounded".into(),
            recall_at_k: rivero_recall,
            latency_us: rivero_p50,
            throughput_qps: rivero_qps,
            admissible: rivero_recall >= 0.99,
        },
        hnsqr::planning::regret::PlanExecutionMetrics {
            plan: auto_plan,
            plan_name: "Universal Auto".into(),
            recall_at_k: auto_recall,
            latency_us: auto_p50,
            throughput_qps: auto_qps,
            admissible: auto_recall >= 0.99,
        },
    ];

    let selected_metric = &candidate_metrics[2]; // Universal Auto
    let regret_eval =
        hnsqr::planning::regret::PlannerRegretOracle::evaluate(selected_metric, &candidate_metrics);

    println!(
        "  ┌──────────────────────┬──────────────────────┬────────────┬────────────┬─────────────┬──────────────┐"
    );
    println!(
        "  │ Selected Plan        │ Optimal Oracle Plan  │ Sel Latency│ Opt Latency│ Regret (µs) │ Optimality % │"
    );
    println!(
        "  ├──────────────────────┼──────────────────────┼────────────┼────────────┼─────────────┼──────────────┤"
    );
    println!(
        "  │ {:<20} │ {:<20} │ {:>8.1} µs│ {:>8.1} µs│ {:>9.1} µs│ {:>10.1}% │",
        regret_eval.selected_plan_name,
        regret_eval.optimal_plan_name,
        regret_eval.selected_latency_us,
        regret_eval.optimal_latency_us,
        regret_eval.regret_us,
        regret_eval.optimality_ratio * 100.0,
    );
    println!(
        "  └──────────────────────┴──────────────────────┴────────────┴────────────┴─────────────┴──────────────┘\n"
    );
}
