//! Gate B: Corpus-Global Exact Hierarchical Proof Engine Scorecard
//!
//! This is an admission benchmark, not a success banner: exactness is necessary
//! but insufficient. A certified proof path that does not beat the Exact SIMD
//! baseline is explicitly rejected for production use.
//!
//! Measures:
//!   1. 100.000% Exact Recall Verification against Brute-Force Ground Truth (hard assert)
//!   2. Hierarchical region pruning efficiency (% regions and vectors pruned by UB < tau)
//!   3. LUTz L0 Cauchy-Schwarz leaf filtering (% leaf vectors eliminated without SIMD)
//!   4. Exact SIMD escalation rate (% corpus evaluated exactly)
//!   5. End-to-End Latency vs Brute Force Exact Scan across D_real in [384, 1536, 4096]

mod common;

use std::time::Instant;

use hnsqr::proof::{
    DenseExactProof, GlobalExactProofSearch, ProofBenchmarkArtifact, SegmentProofView,
    proof_benchmark_artifact_filename,
};
use hnsqr::rivero::RiveroProfile;

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
    // 1. Load real corpus & queries from datasets/
    let (base_path, query_path, source_real_dim) = common::find_best_matching_dataset(exp.d_real);
    let (corpus, _) = common::read_fvecs(&base_path, Some(exp.n_corpus))
        .unwrap_or_else(|_| panic!("failed to load {}", base_path.display()));
    let (queries, _) = common::read_fvecs(&query_path, Some(exp.n_queries))
        .unwrap_or_else(|_| panic!("failed to load {}", query_path.display()));
    assert!(
        !corpus.is_empty(),
        "dataset '{}' is missing or empty — ensure datasets/ are populated",
        base_path.display()
    );

    let complex_dim = corpus.first().map_or(exp.d_real / 2, |v| v.dimension());
    println!("\n═══════════════════════════════════════════════════════════════════════════════");
    println!(
        " 🔬 Running Gate B Benchmark: D_real = {} ({} complex), N = {}, K = {}",
        complex_dim * 2,
        complex_dim,
        corpus.len(),
        exp.k
    );
    println!("═══════════════════════════════════════════════════════════════════════════════");

    // 2. Attach immutable proof and exact-index snapshots. Benchmark processes
    // deliberately perform no Rivero construction, LUTz encoding, tree building,
    // or index insertion; those are indexing operations performed by
    // `hnsqr_build_bench_db` once per artifact.
    let actual_n = corpus.len();
    let proof_path = common::bench_cache_dir()
        .join(proof_benchmark_artifact_filename(source_real_dim, actual_n));
    assert!(
        proof_path.is_file(),
        "prebuilt Gate B proof artifact is missing: {}\n\
         Build it once from the real dataset with:\n\
           cargo run --release --bin hnsqr_build_bench_db -- --kind proof --vectors {actual_n} --source-dim {source_real_dim}\n\
         Benchmark processes never build proof state.",
        proof_path.display()
    );
    let proof_artifact = ProofBenchmarkArtifact::load(&proof_path, source_real_dim, actual_n)
        .unwrap_or_else(|error| {
            panic!(
                "invalid Gate B proof artifact {}: {error}",
                proof_path.display()
            )
        });
    println!(
        "   ✓ Attached immutable proof artifact ({} vectors, {} nodes)",
        proof_artifact.vector_count,
        proof_artifact.tree.nodes.len()
    );
    let exact_tag = format!("gate_b_exact_d{source_real_dim}");
    let exact_index =
        common::open_prebuilt_index(&exact_tag, &corpus, complex_dim, RiveroProfile::Balanced);

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

        // Hierarchical Exact Proof Search
        let seg_view = SegmentProofView {
            tree: &proof_artifact.tree,
            vectors: &corpus,
            // Gate B must exercise the progressive L0/L1 cascade it reports.
            // Passing `None` silently converted every leaf into an exact scan.
            lutz_codes: Some(&proof_artifact.lutz_codes),
            tombstones: None,
        };
        let proof_start = Instant::now();
        let (certified, proof) =
            GlobalExactProofSearch::search(q, exp.k, &[seg_view], &[], &[], None);
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
    let n_f64 = corpus.len() as f64;
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
        exact_simd_evals_pct, avg_exact_evals, actual_n
    );
    println!("      • Brute Force Latency (p50):    {:.2} µs", bf_lat_p50);
    println!(
        "      • Hierarchical Proof (p50):     {:.2} µs",
        proof_lat_p50
    );
    println!("      • Speedup Factor vs Brute Force: {:.2}x", speedup);

    BenchmarkResult {
        d_real: complex_dim * 2,
        n_corpus: actual_n,
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
    println!("🔬 GATE B ADMISSION SCORECARD (100.000% EXACT TOP-K REQUIRED)");
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "{:<8} | {:<8} | {:<10} | {:<12} | {:<12} | {:<12} | {:<12} | {:<10} | Status",
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
    let mut admitted = 0usize;
    for r in results {
        let admission = r.exact_recall == 100.0 && r.speedup > 1.0;
        admitted += admission as usize;
        println!(
            "{:<8} | {:<8} | {:<9.3}% | {:<11.2}% | {:<11.2}% | {:<11.2}% | {:<9.1} µs | {:.2}x {}",
            r.d_real,
            r.n_corpus,
            r.exact_recall,
            r.vectors_pruned_by_region_pct,
            r.lutz_l0_pruned_pct,
            r.exact_simd_evals_pct,
            r.proof_lat_us,
            r.speedup,
            if admission { "ADMITTED" } else { "REJECTED" }
        );
    }
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n"
    );
    if admitted == 0 {
        eprintln!(
            "Gate B rejected every configuration: exact proof search did not beat Exact SIMD. \\
             Keep Exact SIMD as the production retrieval path."
        );
        std::process::exit(1);
    }
}
