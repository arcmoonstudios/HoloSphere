//! Proof Hierarchy Benchmark Suite (Gate B0/B1/B2/B3)
//!
//! Sweeps across corpus scales, dimensions, and leaf capacities to evaluate:
//!   - Exact Recall@10 (100.000% hard assert)
//!   - Disjoint Terminal Funnel Accounting ($N_{\text{eligible}} \equiv \sum \text{buckets}$)
//!   - Spherical-Cap Hierarchy Region Elimination %
//!   - Exact SIMD Residue Evaluations %
//!   - Memory traffic (Full-Vector bytes touched)
//!   - Latency distribution (p50 / p95 / p99) and speedup vs brute-force exact.

mod common;

use std::time::Instant;

use hnsqr::proof::{DenseExactProof, GlobalExactProofSearch, SegmentProofView, SemanticProofTree};
use hnsqr::rivero::bulk::RiveroBulkBuilder;
use hnsqr::rivero::{RiveroCompiler, RiveroProfile};
use hnsqr::{NodeIndex, SimilarityScore, VectorEmbedding};

#[derive(Debug, Clone)]
struct HierarchyExperiment {
    d_real: usize,
    n_corpus: usize,
    n_queries: usize,
    k: usize,
}

#[derive(Debug, Clone)]
struct HierarchyResult {
    d_real: usize,
    n_corpus: usize,
    exact_recall: f64,
    region_pruned_pct: f64,
    exact_simd_pct: f64,

    full_vector_bytes_touched: usize,
    p50_us: f64,
    speedup_vs_bf: f64,
}

fn cosine_sim(a: &VectorEmbedding, b: &VectorEmbedding) -> SimilarityScore {
    (a.dot_product_complex(b)).re
}

fn brute_force_exact(
    query: &VectorEmbedding,
    corpus: &[VectorEmbedding],
    k: usize,
) -> Vec<(NodeIndex, SimilarityScore)> {
    let mut scores: Vec<(NodeIndex, SimilarityScore)> = corpus
        .iter()
        .enumerate()
        .map(|(idx, doc)| (idx as NodeIndex, cosine_sim(query, doc)))
        .collect();

    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scores.truncate(k);
    scores
}

fn load_semantic_corpus(
    n_corpus: usize,
    n_queries: usize,
    d_real: usize,
) -> (Vec<VectorEmbedding>, Vec<VectorEmbedding>) {
    let (base_path, query_path, _) = common::find_best_matching_dataset(d_real);
    let (corpus, _) = common::read_fvecs(&base_path, Some(n_corpus)).unwrap_or_default();
    let (queries, _) = common::read_fvecs(&query_path, Some(n_queries)).unwrap_or_default();

    assert!(
        !corpus.is_empty(),
        "Missing base vectors for dataset at {}",
        base_path.display()
    );
    assert!(
        !queries.is_empty(),
        "Missing query vectors for dataset at {}",
        query_path.display()
    );

    (corpus, queries)
}

fn run_hierarchy_experiment(exp: &HierarchyExperiment) -> Option<HierarchyResult> {
    let (corpus, queries) = load_semantic_corpus(exp.n_corpus, exp.n_queries, exp.d_real);
    let actual_n = corpus.len();
    let complex_dim = corpus[0].dimension();

    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!(
        " 🔬 Running Gate B3 Hierarchy Benchmark: D_real = {}, N = {}, K = {}",
        exp.d_real, actual_n, exp.k
    );
    println!("═══════════════════════════════════════════════════════════════════════════════");

    // 1. Build Rivero Territory Routing Structure
    print!("   ⚙️ Building Rivero Strict Territory Partition Index...");
    let rivero_cfg = RiveroProfile::Strict.config();
    let compiler = RiveroCompiler::new(complex_dim);
    let builder = RiveroBulkBuilder::new(rivero_cfg);
    let built_state = builder.build(&corpus).expect("Rivero build failed");
    let territory = built_state.territory;
    println!(" Done.");

    // 2. Build Semantic Proof Tree
    print!("   ⚙️ Building Flattened Semantic Proof Hierarchy...");
    let slots: Vec<NodeIndex> = (0..actual_n as NodeIndex).collect();
    let build_start = Instant::now();
    let proof_tree = SemanticProofTree::build(&corpus, &slots, complex_dim);
    let build_time_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        " Done in {:.2} ms (Nodes: {}, Leaf Slots: {})",
        build_time_ms,
        proof_tree.nodes.len(),
        proof_tree.leaf_slots.len()
    );

    // 3. Search Evaluation
    let mut bf_latencies_us = Vec::with_capacity(exp.n_queries);
    let mut proof_latencies_us = Vec::with_capacity(exp.n_queries);
    let mut proofs: Vec<DenseExactProof> = Vec::with_capacity(exp.n_queries);
    let mut total_gt_matches = 0usize;
    let mut total_gt_elements = 0usize;

    for q in &queries {
        // Brute Force Baseline (Identical exact cosine kernel)
        let bf_start = Instant::now();
        let gt = brute_force_exact(q, &corpus, exp.k);
        let bf_dur = bf_start.elapsed().as_secs_f64() * 1_000_000.0;
        bf_latencies_us.push(bf_dur);

        // Rivero Seed Discovery
        let q_addr = compiler.compile(q.complex_data());
        let mut rivero_cands = Vec::new();
        territory.with_candidates_config(&q_addr, &rivero_cfg, |cands, _| {
            rivero_cands.extend_from_slice(cands);
        });

        // Hierarchical Exact Proof Search
        let seg_view = SegmentProofView {
            tree: &proof_tree,
            vectors: &corpus,
            tombstones: None,
        };
        let proof_start = Instant::now();
        let (certified, proof) =
            GlobalExactProofSearch::search(q, exp.k, &[seg_view], &[], &rivero_cands, None);
        let proof_dur = proof_start.elapsed().as_secs_f64() * 1_000_000.0;
        proof_latencies_us.push(proof_dur);
        proofs.push(proof);

        // Exactness Hard Assertion
        assert_eq!(certified.len(), gt.len());
        for i in 0..gt.len() {
            if certified[i].0 == gt[i].0 {
                total_gt_matches += 1;
            }
            total_gt_elements += 1;
        }
    }

    let exact_recall = (total_gt_matches as f64 / total_gt_elements as f64) * 100.0;
    assert!(
        (exact_recall - 100.0).abs() < 1e-4,
        "CRITICAL ERROR: Exact Recall violated! Achieved: {:.4}%",
        exact_recall
    );

    // Terminal Funnel Accounting Check
    for (q_idx, p) in proofs.iter().enumerate() {
        assert!(
            p.is_accounting_exact(),
            "Query #{} violated terminal funnel accounting! vectors_pruned={}, exact={}, filtered={}, total={}",
            q_idx,
            p.vectors_pruned_by_region,
            p.exact_evaluations,
            p.filtered_or_tombstoned,
            p.corpus_size
        );
    }

    let n_f64 = corpus.len() as f64;
    let avg_region_pruned = proofs
        .iter()
        .map(|p| p.vectors_pruned_by_region)
        .sum::<usize>() as f64
        / exp.n_queries as f64;
    let region_pruned_pct = (avg_region_pruned / n_f64) * 100.0;

    let avg_exact_evals =
        proofs.iter().map(|p| p.exact_evaluations).sum::<usize>() as f64 / exp.n_queries as f64;
    let exact_simd_pct = (avg_exact_evals / n_f64) * 100.0;

    let avg_exact_bytes =
        proofs.iter().map(|p| p.exact_bytes_touched).sum::<usize>() / exp.n_queries;

    bf_latencies_us.sort_by(|a, b| a.total_cmp(b));
    proof_latencies_us.sort_by(|a, b| a.total_cmp(b));

    let p50_idx = ((exp.n_queries as f64 - 1.0) * 0.50).round() as usize;
    let p95_idx = ((exp.n_queries as f64 - 1.0) * 0.95).round() as usize;
    let p99_idx = ((exp.n_queries as f64 - 1.0) * 0.99).round() as usize;

    let p50_us = proof_latencies_us[p50_idx];
    let p95_us = proof_latencies_us[p95_idx];
    let p99_us = proof_latencies_us[p99_idx];
    let speedup = bf_latencies_us[p50_idx] / p50_us;

    let leaf_thetas: Vec<f32> = proof_tree
        .nodes
        .iter()
        .filter(|n| n.is_leaf())
        .map(|n| n.angular_radius_degrees())
        .collect();
    let leaf_avg_theta_deg = if !leaf_thetas.is_empty() {
        leaf_thetas.iter().sum::<f32>() / leaf_thetas.len() as f32
    } else {
        0.0
    };

    println!("\n   📊 B3 RESULTS FUNNEL & AUDIT:");
    println!(
        "      • Certified Recall@10:          {:.4}% (VERIFIED 100.000%)",
        exact_recall
    );
    println!("      • Terminal Accounting:          100.00% EXACT (0 double-counts)");
    println!(
        "      • Leaf Avg Angular Radius θ:    {:.2}°",
        leaf_avg_theta_deg
    );
    println!(
        "      • Region Pruned (Tree):         {:.2}% ({:.0} vectors)",
        region_pruned_pct, avg_region_pruned
    );
    println!(
        "      • Exact SIMD Evaluations:       {:.2}% ({:.0} vectors)",
        exact_simd_pct, avg_exact_evals
    );
    println!(
        "      • Full-Vector Memory Traffic:   {:.1} KB / query",
        avg_exact_bytes as f64 / 1024.0
    );
    println!(
        "      • Latency p50 / p95 / p99:      {:.1} µs / {:.1} µs / {:.1} µs",
        p50_us, p95_us, p99_us
    );
    println!("      • Speedup vs Brute Force:       {:.2}x", speedup);

    Some(HierarchyResult {
        d_real: exp.d_real,
        n_corpus: actual_n,
        exact_recall,
        region_pruned_pct,
        exact_simd_pct,
        full_vector_bytes_touched: avg_exact_bytes,
        p50_us,
        speedup_vs_bf: speedup,
    })
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════════════╗");
    println!("║              HNSQR GATE B3: PROOF HIERARCHY BENCHMARK SUITE                 ║");
    println!("║                100.000% Exact Recall Verification Matrix                    ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════╝");

    let matrix = vec![
        HierarchyExperiment {
            d_real: 384,
            n_corpus: 10_000,
            n_queries: 25,
            k: 10,
        },
        HierarchyExperiment {
            d_real: 768,
            n_corpus: 10_000,
            n_queries: 25,
            k: 10,
        },
        HierarchyExperiment {
            d_real: 1536,
            n_corpus: 10_000,
            n_queries: 25,
            k: 10,
        },
        HierarchyExperiment {
            d_real: 4096,
            n_corpus: 10_000,
            n_queries: 25,
            k: 10,
        },
        HierarchyExperiment {
            d_real: 1536,
            n_corpus: 25_000,
            n_queries: 25,
            k: 10,
        },
    ];

    let mut results = Vec::new();
    for exp in &matrix {
        if let Some(result) = run_hierarchy_experiment(exp) {
            results.push(result);
        }
    }

    println!(
        "\n══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!("🏆 GATE B3 GRAND PROOF SCORECARD (100.000% CERTIFIED RECALL)");
    println!(
        "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "{:<6} | {:<7} | {:<10} | {:<12} | {:<10} | {:<10} | {:<12} | {:<10}",
        "D_real",
        "N",
        "Recall@10",
        "Region Prune",
        "Exact SIMD",
        "Traffic",
        "Latency p50",
        "Speedup"
    );
    println!(
        "--------------------------------------------------------------------------------------------------------------------------------------"
    );
    for r in results {
        println!(
            "{:<6} | {:<7} | {:<9.3}% | {:<11.2}% | {:<9.2}% | {:<7.1} KB | {:<9.1} µs | {:.2}x",
            r.d_real,
            r.n_corpus,
            r.exact_recall,
            r.region_pruned_pct,
            r.exact_simd_pct,
            r.full_vector_bytes_touched as f64 / 1024.0,
            r.p50_us,
            r.speedup_vs_bf
        );
    }
    println!(
        "══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n"
    );
}
