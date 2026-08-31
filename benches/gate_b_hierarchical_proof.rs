//! Gate B: Corpus-Global Exact Hierarchical Proof Engine Scorecard
//!
//! This is an admission benchmark, not a success banner: exactness is necessary
//! but insufficient. A certified proof path that does not beat the Exact SIMD
//! baseline is explicitly rejected for production use.
//!
//! Measures:
//!   1. 100.000% Exact Recall Verification against Brute-Force Ground Truth (hard assert)
//!   2. Hierarchical region pruning efficiency (% regions and vectors pruned by UB < tau)
//!   3. Exact SIMD escalation rate (% corpus evaluated exactly)
//!   4. End-to-End Latency vs Brute Force Exact Scan across D_real in [384, 1536, 4096]

mod common;

use std::time::Instant;

use hnsqr::proof::{
    DenseExactProof, GlobalExactProofSearch, ProofBenchmarkArtifact, SegmentProofView,
    proof_benchmark_artifact_filename,
};
use hnsqr::retrieval::performance_trial::{
    AdmissionGateStatus, CertifiedEvidence, evaluate_admission_gates,
};
use hnsqr::rivero::RiveroProfile;

#[derive(Debug, Clone)]
struct ExperimentConfig {
    d_real: usize,
    n_corpus: usize,
    n_queries: usize,
    k: usize,
}

#[derive(Debug, Clone)]
struct BenchmarkResult {
    d_real: usize,
    n_corpus: usize,
    exact_recall: f64,
    vectors_pruned_by_region_pct: f64,
    exact_simd_evals_pct: f64,
    proof_lat_us: f64,
    speedup: f64,
    admission: AdmissionGateStatus,
}

fn run_experiment(exp: &ExperimentConfig) -> Option<BenchmarkResult> {
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
    // deliberately perform no Rivero construction, tree building,
    // or index insertion; those are indexing operations performed by
    // `hnsqr_build_bench_db` once per artifact.
    // 2. Attach immutable proof and exact-index snapshots.
    // If pre-baked disk artifact is absent, build in-memory so benchmark is fully self-contained.
    let actual_n = corpus.len();
    let proof_path = common::bench_cache_dir()
        .join(proof_benchmark_artifact_filename(source_real_dim, actual_n));
    let proof_artifact = if proof_path.is_file() {
        ProofBenchmarkArtifact::load(&proof_path, source_real_dim, actual_n).unwrap_or_else(|e| {
            panic!(
                "failed to load proof artifact {}: {e}",
                proof_path.display()
            )
        })
    } else {
        println!(
            "⚙️ Generating in-memory proof hierarchy for D_real={source_real_dim}, N={actual_n}..."
        );
        let slots: Vec<u32> = (0..actual_n as u32).collect();
        let tree = hnsqr::proof::SemanticProofTree::build(&corpus, &slots, complex_dim);
        let artifact = ProofBenchmarkArtifact::new(source_real_dim, tree)
            .unwrap_or_else(|e| panic!("failed to construct in-memory proof artifact: {e}"));
        if let Some(parent) = proof_path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = artifact.save(&proof_path);
        artifact
    };

    let mut bf_latencies_us = Vec::with_capacity(exp.n_queries);
    let mut proof_latencies_us = Vec::with_capacity(exp.n_queries);
    let mut proofs: Vec<DenseExactProof> = Vec::with_capacity(exp.n_queries);

    let mut total_gt_matches = 0;
    let mut total_gt_elements = 0;

    for q in &queries {
        // Brute Force Baseline
        let bf_start = Instant::now();
        let mut scored: Vec<(u32, f32)> = corpus
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx as u32, q.dot_product_real(doc)))
            .collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(exp.k);
        let bf_dur = bf_start.elapsed().as_secs_f64() * 1_000_000.0;
        bf_latencies_us.push(bf_dur);

        // Ground truth pairs
        let gt = scored;

        // Hierarchical Exact Proof Search
        let seg_view = SegmentProofView {
            tree: &proof_artifact.tree,
            vectors: &corpus,
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
    // Production admission is deliberately delegated to the shared retrieval
    // contract. Exactness alone is insufficient: a certified path must also
    // beat the exact baseline measured in the same run.
    let proof_complete = proofs
        .iter()
        .all(|proof| !proof.deadline_exceeded && proof.is_accounting_exact());
    let globally_exact = proofs.iter().all(|proof| proof.globally_exact);
    let admission = evaluate_admission_gates(
        Some(CertifiedEvidence {
            all_queries_proof_complete: proof_complete,
            all_queries_globally_exact: globally_exact,
        }),
        exact_recall / 100.0,
        (bf_lat_p50 * 1_000.0).round() as u64,
        (proof_lat_p50 * 1_000.0).round() as u64,
    );

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
        "      • Exact SIMD Evaluations:       {:.2}% ({:.0} vectors vs {} total)",
        exact_simd_evals_pct, avg_exact_evals, actual_n
    );
    println!("      • Brute Force Latency (p50):    {:.2} µs", bf_lat_p50);
    println!(
        "      • Hierarchical Proof (p50):     {:.2} µs",
        proof_lat_p50
    );
    println!("      • Speedup Factor vs Brute Force: {:.2}x", speedup);

    Some(BenchmarkResult {
        d_real: complex_dim * 2,
        n_corpus: actual_n,
        exact_recall,
        vectors_pruned_by_region_pct,
        exact_simd_evals_pct,
        proof_lat_us: proof_lat_p50,
        speedup,
        admission,
    })
}

fn main() {
    println!("╔═════════════════════════════════════════════════════════════════════════════════╗");
    println!("║       GATE B: CORPUS-GLOBAL EXACT HIERARCHICAL PROOF ENGINE BENCHMARK           ║");
    println!("║       Verification of 100.000% Exact Recall and Sub-linear SIMD Escalation      ║");
    println!("╚═════════════════════════════════════════════════════════════════════════════════╝");

    let configs = vec![
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
            n_queries: 50,
            k: 10,
        },
    ];

    let mut results = Vec::new();
    for cfg in &configs {
        if let Some(res) = run_experiment(cfg) {
            results.push(res);
        }
    }

    println!(
        "\n═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!("                             🏆 GATE B HIERARCHICAL PROOF FINAL SCORECARD");
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(
        "{:<8} | {:<8} | {:<10} | {:<12} | {:<12} | {:<12} | {:<10} | Status",
        "D_real", "N", "Recall@10", "Region Prune", "Exact SIMD", "Latency", "Speedup"
    );
    println!(
        "---------------------------------------------------------------------------------------------------------------"
    );
    let mut admitted = 0usize;
    for r in &results {
        let admission = matches!(
            r.admission,
            AdmissionGateStatus::CertifiedProductionApproved
        );
        admitted += admission as usize;
        println!(
            "{:<8} | {:<8} | {:<9.3}% | {:<11.2}% | {:<11.2}% | {:<9.1} µs | {:.2}x {}",
            r.d_real,
            r.n_corpus,
            r.exact_recall,
            r.vectors_pruned_by_region_pct,
            r.exact_simd_evals_pct,
            r.proof_lat_us,
            r.speedup,
            if admission { "ADMITTED" } else { "REJECTED" }
        );
    }
    println!(
        "═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════\n"
    );
    if results.is_empty() {
        println!("Gate B completed with no runnable rows; missing artifacts were skipped.");
    } else if admitted == 0 {
        eprintln!(
            "Gate B rejected every configuration: exact proof search did not beat Exact SIMD. \\
             Keep Exact SIMD as the production retrieval path."
        );
        std::process::exit(1);
    }
}
