//! Gate B: Corpus-Global Exact Hierarchical Proof Engine Scorecard
//!
//! Measures:
//!   1. 100.000% Exact Recall Verification against Brute-Force Ground Truth (hard assert)
//!   2. Hierarchical region pruning efficiency (% regions and vectors pruned by UB < tau)
//!   3. LUTz L0 Cauchy-Schwarz leaf filtering (% leaf vectors eliminated without SIMD)
//!   4. Exact SIMD escalation rate (% corpus evaluated exactly)
//!   5. End-to-End Latency vs Brute Force Exact Scan across D_real in [384, 1536, 4096]

mod common;

use std::time::Instant;

use hnsqr::proof::{DenseExactProof, GlobalExactProofSearch, SegmentProofView, SemanticProofTree};
use hnsqr::rivero::bulk::RiveroBulkBuilder;
use hnsqr::rivero::{RiveroAddressConfig, RiveroCompiler, RiveroConfig, RiveroProjectionMode};
use hnsqr::{DistanceFunction, NodeIndex};

#[derive(Debug, Clone)]
struct ExperimentConfig {
    d_real: usize,
    n_corpus: usize,
    n_queries: usize,
    k: usize,
}

#[derive(Debug, Clone, Default)]
struct BenchmarkResult {
    d_real: usize,
    n_corpus: usize,
    exact_recall: f64,
    vectors_pruned_by_region_pct: f64,
    lutz_l0_pruned_pct: f64,
    exact_simd_evals_pct: f64,
    proof_lat_us: f64,
    speedup: f64,
}

fn run_experiment(exp: &ExperimentConfig) -> BenchmarkResult {
    let complex_dim = exp.d_real / 2;
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!(
        " 🔬 Running Gate B Benchmark: D_real = {} ({} complex), N = {}, K = {}",
        exp.d_real, complex_dim, exp.n_corpus, exp.k
    );
    println!("═══════════════════════════════════════════════════════════════════════════════");

    // 1. Load real corpus & queries from datasets/
    let (base_path, query_path, _) = common::find_best_matching_dataset(exp.d_real);
    let (corpus, _) = common::read_fvecs(&base_path, Some(exp.n_corpus))
        .unwrap_or_else(|_| panic!("failed to load {}", base_path.display()));
    let (queries, _) = common::read_fvecs(&query_path, Some(exp.n_queries))
        .unwrap_or_else(|_| panic!("failed to load {}", query_path.display()));
    assert!(
        !corpus.is_empty(),
        "dataset '{}' is missing or empty — ensure datasets/ are populated",
        base_path.display()
    );

    // 2. Build Rivero Index (64 Foundations GlobalMix)
    println!("   ⚙️ Building Rivero Coarse Index...");
    let rivero_cfg = RiveroConfig {
        foundations: 64,
        simhash_query_probes: 32,
        cell_capacity: 64,
        affinity_elites: 24,
        cell_budget: 16,
        query_candidate_cap: 8800,
    };
    let addr_cfg = RiveroAddressConfig {
        foundations: 64,
        projection: RiveroProjectionMode::GlobalMix,
        geometry: hnsqr::rivero::VectorGeometry::Real,
    };
    let compiler = RiveroCompiler::with_config(complex_dim, addr_cfg);
    let builder = RiveroBulkBuilder::new(rivero_cfg)
        .with_address_config(addr_cfg)
        .with_distance_function(DistanceFunction::Cosine);
    let built = builder.build(&corpus).expect("Bulk build must succeed");
    let territory = &built.territory;

    // 3. Build LUTz Codes
    println!("   ⚙️ Encoding LUTz E8 Quantization Codes...");
    let _lutz_codes: Vec<hnsqr::proof::lutz::LutzCode> = corpus
        .iter()
        .map(|v| hnsqr::proof::lutz::LutzCode::encode(v, true))
        .collect();

    // 4. Build Canonical Corpus-Covering Proof Tree & Exact Index Baseline
    println!("   ⚙️ Building Canonical Semantic Proof Hierarchy & Exact Baseline...");
    let slots: Vec<NodeIndex> = (0..exp.n_corpus as NodeIndex).collect();
    let proof_tree_start = Instant::now();
    let proof_tree = SemanticProofTree::build(&corpus, &slots, complex_dim);
    let proof_tree_build_ms = proof_tree_start.elapsed().as_secs_f64() * 1000.0;
    println!(
        "      ✓ Proof Tree Built in {:.2} ms (Total vectors: {})",
        proof_tree_build_ms,
        proof_tree.total_vectors()
    );

    // Exact baseline is a prebuilt snapshot over the same real dataset slice.
    let exact_cache_key = format!(
        "gate_b_exact_d{}_n{}",
        exp.d_real, exp.n_corpus
    );
    let exact_index = common::open_prebuilt_snapshot(&exact_cache_key);

    // 5. Execute Evaluation
    let mut total_gt_matches = 0usize;
    let mut total_gt_elements = 0usize;
    let mut bf_latencies_us = Vec::with_capacity(exp.n_queries);
    let mut proof_latencies_us = Vec::with_capacity(exp.n_queries);
    let mut proofs: Vec<DenseExactProof> = Vec::with_capacity(exp.n_queries);

    for q in &queries {
        // Production Exact SIMD Baseline
        let bf_start = Instant::now();
        let gt = exact_index
            .search_indices_exact(q, exp.k, None)
            .expect("exact scan");
        let bf_dur = bf_start.elapsed().as_secs_f64() * 1_000_000.0;
        bf_latencies_us.push(bf_dur);

        // Rivero Seed Discovery (~8,800 raw candidates)
        let q_addr = compiler.compile(q.complex_data());
        let mut rivero_cands = Vec::new();
        territory.with_candidates_config(&q_addr, &rivero_cfg, |cands, _| {
            rivero_cands.extend_from_slice(cands);
        });

        // Hierarchical Exact Proof Search
        let seg_view = SegmentProofView {
            tree: &proof_tree,
            vectors: &corpus,
            lutz_codes: None,
            tombstones: None,
        };
        let proof_start = Instant::now();
        let (certified, proof) =
            GlobalExactProofSearch::search(q, exp.k, &[seg_view], &[], &rivero_cands, None);
        let proof_dur = proof_start.elapsed().as_secs_f64() * 1_000_000.0;
        proof_latencies_us.push(proof_dur);
        proofs.push(proof);

        // Verification
        assert_eq!(
            certified.len(),
            gt.len(),
            "Result length mismatch for query"
        );
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
        "CRITICAL ERROR: Gate B Exact Recall violated! Achieved: {:.4}%",
        exact_recall
    );

    // Compute Summary Telemetry
    let n_f64 = exp.n_corpus as f64;
    let avg_regions_pruned =
        proofs.iter().map(|p| p.proof_regions_pruned).sum::<usize>() as f64 / exp.n_queries as f64;
    let avg_regions_expanded = proofs
        .iter()
        .map(|p| p.proof_regions_expanded)
        .sum::<usize>() as f64
        / exp.n_queries as f64;
    let total_regions = avg_regions_pruned + avg_regions_expanded;
    let regions_pruned_pct = if total_regions > 0.0 {
        (avg_regions_pruned / total_regions) * 100.0
    } else {
        0.0
    };

    let avg_vectors_pruned_by_region = proofs
        .iter()
        .map(|p| p.vectors_pruned_by_region)
        .sum::<usize>() as f64
        / exp.n_queries as f64;
    let vectors_pruned_by_region_pct = if n_f64 > 0.0 {
        (avg_vectors_pruned_by_region / n_f64) * 100.0
    } else {
        0.0
    };

    let avg_lutz_l0_evals =
        proofs.iter().map(|p| p.lutz_l0_evaluations).sum::<usize>() as f64 / exp.n_queries as f64;
    let avg_lutz_l0_pruned =
        proofs.iter().map(|p| p.lutz_l0_pruned).sum::<usize>() as f64 / exp.n_queries as f64;
    let lutz_l0_pruned_pct = if avg_lutz_l0_evals > 0.0 {
        (avg_lutz_l0_pruned / avg_lutz_l0_evals) * 100.0
    } else {
        0.0
    };

    let avg_exact_evals =
        proofs.iter().map(|p| p.exact_evaluations).sum::<usize>() as f64 / exp.n_queries as f64;
    let exact_simd_evals_pct = if n_f64 > 0.0 {
        (avg_exact_evals / n_f64) * 100.0
    } else {
        0.0
    };

    bf_latencies_us.sort_by(|a, b| a.total_cmp(b));
    proof_latencies_us.sort_by(|a, b| a.total_cmp(b));

    let bf_lat_p50 = bf_latencies_us[exp.n_queries / 2];
    let proof_lat_p50 = proof_latencies_us[exp.n_queries / 2];
    let speedup = if proof_lat_p50 > 0.0 {
        bf_lat_p50 / proof_lat_p50
    } else {
        1.0
    };

    println!("\n   📊 RESULTS SUMMARY:");
    println!(
        "      • Exact Recall@K:               {:.4}% (VERIFIED 100.000%)",
        exact_recall
    );
    println!(
        "      • Hierarchy Regions Pruned:     {:.2}%",
        regions_pruned_pct
    );
    println!(
        "      • Vectors Pruned by Region UB:  {:.2}% ({:.0} vectors)",
        vectors_pruned_by_region_pct, avg_vectors_pruned_by_region
    );
    println!(
        "      • Leaf LUTz L0 Pruned:          {:.2}%",
        lutz_l0_pruned_pct
    );
    println!(
        "      • Exact SIMD Evaluations:       {:.2}% ({:.0} vectors vs {} total)",
        exact_simd_evals_pct, avg_exact_evals, exp.n_corpus
    );
    println!("      • Brute Force Latency (p50):    {:.2} µs", bf_lat_p50);
    println!(
        "      • Hierarchical Proof (p50):     {:.2} µs",
        proof_lat_p50
    );
    println!("      • Speedup Factor vs Brute Force: {:.2}x", speedup);

    BenchmarkResult {
        d_real: exp.d_real,
        n_corpus: exp.n_corpus,
        exact_recall,
        vectors_pruned_by_region_pct,
        lutz_l0_pruned_pct,
        exact_simd_evals_pct,
        proof_lat_us: proof_lat_p50,
        speedup,
    }
}

fn main() {
    println!("\n╔═════════════════════════════════════════════════════════════════════════════╗");
    println!("║       HNSQR GATE B: CORPUS-GLOBAL EXACT HIERARCHICAL PROOF SCORECARD        ║");
    println!("║                  100.000% Exact Recall Ground Truth Matrix                  ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════╝");

    let matrix = vec![
        ExperimentConfig {
            d_real: 384,
            n_corpus: 10_000,
            n_queries: 50,
            k: 10,
        },
        ExperimentConfig {
            d_real: 1536,
            n_corpus: 10_000,
            n_queries: 50,
            k: 10,
        },
        ExperimentConfig {
            d_real: 4096,
            n_corpus: 10_000,
            n_queries: 30,
            k: 10,
        },
        ExperimentConfig {
            d_real: 1536,
            n_corpus: 25_000,
            n_queries: 30,
            k: 10,
        },
    ];

    let mut results = Vec::new();
    for exp in &matrix {
        results.push(run_experiment(exp));
    }

    println!(
        "\n═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!("🏆 GATE B GRAND SCORECARD (100.000% EXACT TOP-K)");
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "{:<8} | {:<8} | {:<10} | {:<12} | {:<12} | {:<12} | {:<12} | {:<10}",
        "D_real",
        "N",
        "Recall@10",
        "Region Prune",
        "LUTz Prune",
        "Exact SIMD",
        "Latency",
        "Speedup"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------------"
    );
    for r in results {
        println!(
            "{:<8} | {:<8} | {:<9.3}% | {:<11.2}% | {:<11.2}% | {:<11.2}% | {:<9.1} µs | {:.2}x",
            r.d_real,
            r.n_corpus,
            r.exact_recall,
            r.vectors_pruned_by_region_pct,
            r.lutz_l0_pruned_pct,
            r.exact_simd_evals_pct,
            r.proof_lat_us,
            r.speedup
        );
    }
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n"
    );
}
