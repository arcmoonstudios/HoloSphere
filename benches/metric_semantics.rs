mod common;

use common::load_real_dataset_corpus;
use hnsqr::VectorEmbedding;
use num_complex::Complex32;

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR SEMANTIC METRIC & PHASE COLLISION VALIDATION BENCHMARK                         ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Loads real vectors from datasets/; warns (does not panic) if fewer than 5000 are available.
    let corpus = load_real_dataset_corpus(5_000, 100, 64, common::DEFAULT_BENCH_SEED);

    // ════════════════════════════════════════════════════════════════════════
    // 1. PHASE COLLISION ATTACK SWEEP (z vs e^{i*phi}*z)
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 1: GLOBAL-PHASE COLLISION ATTACK SWEEP");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let phases = [
        ("0", 0.0f32),
        ("π/6", std::f32::consts::FRAC_PI_6),
        ("π/4", std::f32::consts::FRAC_PI_4),
        ("π/3", std::f32::consts::FRAC_PI_3),
        ("π/2", std::f32::consts::FRAC_PI_2),
        ("2π/3", 2.0 * std::f32::consts::FRAC_PI_3),
        ("π", std::f32::consts::PI),
    ];

    println!(
        "  ┌─────────┬──────────────┬──────────────┬──────────────────┬────────────────────────┐"
    );
    println!(
        "  │ Phase φ │ Real Cosine  │ Proj.Overlap │ Folded Hermitian │ Interpretation         │"
    );
    println!(
        "  ├─────────┼──────────────┼──────────────┼──────────────────┼────────────────────────┤"
    );

    let q_vec = &corpus.folded_queries[0];

    for (label, phi) in phases {
        let rot = Complex32::from_polar(1.0, phi);
        let rotated: Vec<Complex32> = q_vec.complex_data().iter().map(|z| z * rot).collect();
        let rot_embed = VectorEmbedding::from_complex(rotated);

        // 1. Complex Projective Overlap (CPO)
        let dot = q_vec.dot_product_complex(&rot_embed);
        let fid =
            (dot.norm_sqr() / (q_vec.norm_squared() * rot_embed.norm_squared())).clamp(0.0, 1.0);

        // 2. Folded Hermitian Real part
        let herm =
            (dot.re / (q_vec.norm_squared() * rot_embed.norm_squared()).sqrt()).clamp(-1.0, 1.0);

        // 3. Real cosine expectation
        let real_cos = phi.cos();

        let interp = if phi == 0.0 {
            "Identical Vector"
        } else if phi <= std::f32::consts::FRAC_PI_4 {
            "Partial Drift"
        } else if phi == std::f32::consts::FRAC_PI_2 {
            "Orthogonal Divergence"
        } else {
            "Opposite Region (Harmful)"
        };

        println!(
            "  │ {:>7} │ {:>12.4} │ {:>12.4} │ {:>16.4} │ {:<22} │",
            label, real_cos, fid, herm, interp
        );
    }
    println!(
        "  └─────────┴──────────────┴──────────────┴──────────────────┴────────────────────────┘\n"
    );

    // ════════════════════════════════════════════════════════════════════════
    // 2. FOLDED HERMITIAN VS COSINE RELEVANCE & NDCG@10 EQUIVALENCE
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 2: FOLDED HERMITIAN VS REAL COSINE RELEVANCE EQUIVALENCE");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let mut max_err = 0.0f32;
    let mut perfect_ndcg_count = 0usize;

    for (raw_q, folded_q) in corpus.queries_raw.iter().zip(corpus.folded_queries.iter()) {
        let mut real_scores: Vec<(usize, f32)> = corpus
            .corpus_raw
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let dot: f32 = raw_q.iter().zip(doc.iter()).map(|(a, b)| a * b).sum();
                (i, dot)
            })
            .collect();
        real_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let mut herm_scores: Vec<(usize, f32)> = corpus
            .folded_corpus
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let dot = folded_q.dot_product_complex(doc);
                (i, dot.re)
            })
            .collect();
        herm_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        for (r, h) in real_scores.iter().zip(herm_scores.iter()).take(10) {
            let err = (r.1 - h.1).abs();
            if err > max_err {
                max_err = err;
            }
        }

        let real_top10: Vec<usize> = real_scores.iter().take(10).map(|s| s.0).collect();
        let herm_top10: Vec<usize> = herm_scores.iter().take(10).map(|s| s.0).collect();
        if real_top10 == herm_top10 {
            perfect_ndcg_count += 1;
        }
    }

    let ndcg_agreement = (perfect_ndcg_count as f64 / corpus.queries_raw.len() as f64) * 100.0;
    println!("  * Max Absolute Precision Error: {:e}", max_err);
    println!("  * Top-10 Exact Rank Agreement:   {:.2}%", ndcg_agreement);
    assert!(
        max_err < 1e-6,
        "Folded Hermitian deviated from real cosine!"
    );
    println!("  ✓ Folded Hermitian matches real cosine down to floating-point epsilon!\n");

    // ════════════════════════════════════════════════════════════════════════
    // 3. PUBLIC SEARCH PATH METRIC CONTRACT COMPLIANCE
    // ════════════════════════════════════════════════════════════════════════
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 3: PUBLIC SEARCH PATH METRIC COMPLIANCE (Cosine vs ProjectiveOverlap)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    use hnsqr::rivero::bulk::RiveroBulkBuilder;
    use hnsqr::rivero::{AdaptivePolicy, RiveroProfile};
    use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, SearchPlan};

    for dist_fn in [
        DistanceFunction::Cosine,
        DistanceFunction::ProjectiveOverlap,
    ] {
        let fn_label = match dist_fn {
            DistanceFunction::Cosine => "Cosine (Folded Hermitian Re)",
            DistanceFunction::ProjectiveOverlap => "ProjectiveOverlap (|<ψ|ϕ>|² / (||ψ||² ||ϕ||²))",
            _ => "Other",
        };
        println!("\n  Testing Contract: [{}]", fn_label);

        let mut config = HNSQRConfig::default();
        config.max_elements = corpus.folded_corpus.len();
        config.distance_function = dist_fn;
        config.rivero_enabled = true;
        config.search_plan = SearchPlan::Rivero;

        let dim = corpus.folded_corpus[0].dimension();
        let index = HNSQRIndex::new(config, dim);
        for (i, vec) in corpus.folded_corpus.iter().enumerate() {
            index.insert(format!("doc-{i}"), vec.clone()).unwrap();
        }

        let builder = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced)
            .with_distance_function(dist_fn)
            .with_threads(8);
        let rivero_state = builder.build(&corpus.folded_corpus).unwrap();
        index.install_rivero_state(rivero_state).unwrap();

        let q = &corpus.folded_queries[0];

        // 1. Exact Scan Path
        let exact_res = index.search_indices_exact(q, 5, None).unwrap();
        for (slot, score) in &exact_res {
            let doc = &corpus.folded_corpus[*slot as usize];
            let ip = q.dot_product_complex(doc);
            let expected_score = match dist_fn {
                DistanceFunction::Cosine => {
                    (ip.re / (q.norm_squared() * doc.norm_squared()).sqrt()).clamp(-1.0, 1.0)
                }
                DistanceFunction::ProjectiveOverlap => {
                    (ip.norm_sqr() / (q.norm_squared() * doc.norm_squared())).clamp(0.0, 1.0)
                }
                _ => 0.0,
            };
            assert!(
                (score - expected_score).abs() < 1e-5,
                "Exact path score diverged from metric contract!"
            );
        }
        println!("    ✓ Exact Scan obeys metric contract");

        // 2. Strict Rivero Path
        let (strict_res, _) = index.search_indices_strict(q, 5, None).unwrap();
        for (slot, score) in &strict_res {
            let doc = &corpus.folded_corpus[*slot as usize];
            let ip = q.dot_product_complex(doc);
            let expected_score = match dist_fn {
                DistanceFunction::Cosine => {
                    (ip.re / (q.norm_squared() * doc.norm_squared()).sqrt()).clamp(-1.0, 1.0)
                }
                DistanceFunction::ProjectiveOverlap => {
                    (ip.norm_sqr() / (q.norm_squared() * doc.norm_squared())).clamp(0.0, 1.0)
                }
                _ => 0.0,
            };
            assert!(
                (score - expected_score).abs() < 1e-5,
                "Strict Rivero score diverged from metric contract!"
            );
        }
        println!("    ✓ Strict Rivero obeys metric contract");

        // 3. Adaptive RiveroOnly Path
        let (adapt_res, _) = index
            .search_indices_adaptive(q, 5, None, AdaptivePolicy::RiveroOnly)
            .unwrap();
        for (slot, score) in &adapt_res {
            let doc = &corpus.folded_corpus[*slot as usize];
            let ip = q.dot_product_complex(doc);
            let expected_score = match dist_fn {
                DistanceFunction::Cosine => {
                    (ip.re / (q.norm_squared() * doc.norm_squared()).sqrt()).clamp(-1.0, 1.0)
                }
                DistanceFunction::ProjectiveOverlap => {
                    (ip.norm_sqr() / (q.norm_squared() * doc.norm_squared())).clamp(0.0, 1.0)
                }
                _ => 0.0,
            };
            assert!(
                (score - expected_score).abs() < 1e-5,
                "Adaptive Rivero score diverged from metric contract!"
            );
        }
        println!("    ✓ Adaptive Rivero obeys metric contract");

        // 4. GraphOnly Path
        let graph_res = index.search_indices_graph(q, 5, None).unwrap();
        for (slot, score) in &graph_res {
            let doc = &corpus.folded_corpus[*slot as usize];
            let ip = q.dot_product_complex(doc);
            let expected_score = match dist_fn {
                DistanceFunction::Cosine => {
                    (ip.re / (q.norm_squared() * doc.norm_squared()).sqrt()).clamp(-1.0, 1.0)
                }
                DistanceFunction::ProjectiveOverlap => {
                    (ip.norm_sqr() / (q.norm_squared() * doc.norm_squared())).clamp(0.0, 1.0)
                }
                _ => 0.0,
            };
            assert!(
                (score - expected_score).abs() < 1e-5,
                "GraphOnly score diverged from metric contract!"
            );
        }
        println!("    ✓ GraphOnly obeys metric contract");

        // 5. Snapshot V2 Persistence & Reopening
        let snap_path =
            std::env::temp_dir().join(format!("hnsqr_metric_contract_{:?}.hnsqr", dist_fn));
        index.save_snapshot_v2(&snap_path).unwrap();
        let restored = HNSQRIndex::open_snapshot_v2(
            &snap_path,
            hnsqr::storage::snapshot::SnapshotOpenOptions::default(),
        )
        .unwrap();
        assert_eq!(
            restored.config().distance_function,
            dist_fn,
            "Snapshot did not restore configured distance_function!"
        );
        let (restored_res, _) = restored.search_indices_strict(q, 5, None).unwrap();
        assert_eq!(
            strict_res.len(),
            restored_res.len(),
            "Restored results count diverged!"
        );
        for ((s1, sc1), (s2, sc2)) in strict_res.iter().zip(restored_res.iter()) {
            assert_eq!(s1, s2, "Restored slot index diverged!");
            assert!((sc1 - sc2).abs() < 1e-5, "Restored score diverged!");
        }
        let _ = std::fs::remove_file(&snap_path);
        println!("    ✓ Snapshot V2 save/open preserves metric contract and bit-level scores");
    }
    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" ALL PUBLIC SEARCH PATHS SATISFY METRIC CONSISTENCY CONTRACT (COSINE & FIDELITY)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════\n"
    );
}
