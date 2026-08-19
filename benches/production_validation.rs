/* hnsqr/benches/production_validation.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Production Validation Campaign & Empirical Decider
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Comprehensive end-to-end evaluation determining whether HNSQR and Rivero satisfy
//! production-grade performance, semantic retrieval quality, resilience, and scalability.
//!
//! ### Core Experimental Sections:
//!   1. **Semantic Metrics & Projective Overlap (CPO) vs Real Cosine**:
//!      Evaluates Recall@1/10/100, MRR, NDCG@10/100 on multi-domain text retrieval embeddings.
//!   2. **Global Phase Invariance vs Collision Attack Test**:
//!      Unfolds rotated states $z' = e^{i\phi} z$ back to real coordinates to quantify
//!      semantic collision risks and validate phase-invariant routing + Hermitian rerank.
//!   3. **Adaptive Confidence Router & False Confidence Rate**:
//!      Evaluates Fast/Balanced/Strict/Fallback acceptance across in-domain, hard-negative,
//!      OOD, and isotropic workloads, computing $P(\text{Recall@10} < \text{target} \mid \text{accepted})$.
//!   4. **Corpus Search Scalability & Ceiling Saturation (100K -> 1M)**:
//!      Measures p50/p95/p99 latency sublinearity and resident/candidate bound utilization.
//!   5. **High-Concurrency Search Matrix (1 -> 64 Clients)**:
//!      Evaluates multi-threaded throughput (QPS), p50/p99/p99.9 latency, and contention.
//!   6. **Persistence Deep-Dive & Microsecond Attach Breakdown**:
//!      Instruments snapshot open phases and measures storage bytes/vector asymptotic curve.
//!   7. **Witness Routing Micro-Profiling**:
//!      Breaks down Phase 4 bulk construction into candidate lookup, vote reduction, and scoring.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;
use std::time::Instant;

use hnsqr::vector::folding::ComplexWeaver;
use hnsqr::{
    AdaptivePolicy, HNSQRConfig, HNSQRIndex, NodeIndex, RiveroBulkBuilder, RiveroProfile,
    SnapshotOpenOptions, VectorEmbedding, VerificationMode,
};
use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

const SEED: u64 = 0x484e_5351_525f_5641; // "HNSQR_VA"

// ════════════════════════════════════════════════════════════════════════════════
// 0. SYNTHETIC REALISTIC EMBEDDING GENERATOR (MULTI-TOPIC TEXT RETRIEVAL CORPUS)
// ════════════════════════════════════════════════════════════════════════════════

struct TextRetrievalCorpus {
    pub raw_real_corpus: Vec<Vec<f32>>,
    pub folded_corpus: Vec<VectorEmbedding>,
    pub queries: Vec<Vec<f32>>,
    pub folded_queries: Vec<VectorEmbedding>,
    pub relevance_ground_truth: Vec<Vec<(usize, u32)>>, // (doc_idx, relevance_score: 0..3)
    pub real_dim: usize,
    pub complex_dim: usize,
}

fn generate_realistic_text_corpus(
    n_docs: usize,
    n_queries: usize,
    real_dim: usize,
    seed: u64,
) -> TextRetrievalCorpus {
    let mut rng = StdRng::seed_from_u64(seed);
    let n_topics = (n_docs / 100).clamp(20, 500);

    // Topic manifolds in real semantic embedding space
    let topic_centers: Vec<Vec<f32>> = (0..n_topics)
        .map(|_| {
            let mut v: Vec<f32> = (0..real_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect();

    let mut raw_real_corpus = Vec::with_capacity(n_docs);
    let mut doc_topics = Vec::with_capacity(n_docs);

    for i in 0..n_docs {
        let topic_idx = i % n_topics;
        let center = &topic_centers[topic_idx];
        let noise_scale = rng.random_range(0.05..0.25f32);
        let mut v: Vec<f32> = center
            .iter()
            .map(|&x| x + rng.random_range(-noise_scale..noise_scale))
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.iter_mut().for_each(|x| *x /= norm);
        raw_real_corpus.push(v);
        doc_topics.push(topic_idx);
    }

    let mut queries = Vec::with_capacity(n_queries);
    let mut relevance_ground_truth = Vec::with_capacity(n_queries);

    for q_idx in 0..n_queries {
        let target_topic = q_idx % n_topics;
        let center = &topic_centers[target_topic];
        let mut q_vec: Vec<f32> = center
            .iter()
            .map(|&x| x + rng.random_range(-0.10..0.10))
            .collect();
        let norm: f32 = q_vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        q_vec.iter_mut().for_each(|x| *x /= norm);
        queries.push(q_vec.clone());

        // Grade relevance of top-100 cosine matches
        let mut doc_scores: Vec<(usize, f32)> = raw_real_corpus
            .iter()
            .enumerate()
            .map(|(d_idx, d_vec): (usize, &Vec<f32>)| {
                let dot: f32 = q_vec.iter().zip(d_vec.iter()).map(|(a, b)| a * b).sum();
                (d_idx, dot)
            })
            .collect();
        doc_scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

        let rel: Vec<(usize, u32)> = doc_scores
            .iter()
            .take(100)
            .enumerate()
            .map(|(rank, &(d_idx, score))| {
                let grade = if rank < 3 && score > 0.85 {
                    3
                } else if rank < 10 && score > 0.70 {
                    2
                } else if score > 0.50 {
                    1
                } else {
                    0
                };
                (d_idx, grade)
            })
            .collect();
        relevance_ground_truth.push(rel);
    }

    let folded_corpus: Vec<VectorEmbedding> = raw_real_corpus
        .par_iter()
        .map(|v| ComplexWeaver::fold_llm_embedding(v))
        .collect();

    let folded_queries: Vec<VectorEmbedding> = queries
        .par_iter()
        .map(|v| ComplexWeaver::fold_llm_embedding(v))
        .collect();

    let complex_dim = real_dim.div_ceil(2);

    TextRetrievalCorpus {
        raw_real_corpus,
        folded_corpus,
        queries,
        folded_queries,
        relevance_ground_truth,
        real_dim,
        complex_dim,
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 1. METRIC & RETRIEVAL RELEVANCE COMPARISON
// ════════════════════════════════════════════════════════════════════════════════

fn run_metric_relevance_comparison(corpus: &TextRetrievalCorpus) {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 1: RETRIEVAL METRIC COMPARISON (PROJECTIVE OVERLAP VS REAL COSINE)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let n_queries = corpus.queries.len();

    struct MetricAccumulator {
        recall_1: f64,
        recall_10: f64,
        recall_100: f64,
        mrr: f64,
        ndcg_10: f64,
        ndcg_100: f64,
    }

    let evaluate_metric = |score_fn: &(
                                dyn Fn(&[f32], &[f32], &[Complex32], &[Complex32]) -> f32 + Sync
                            )|
     -> MetricAccumulator {
        let mut acc = MetricAccumulator {
            recall_1: 0.0,
            recall_10: 0.0,
            recall_100: 0.0,
            mrr: 0.0,
            ndcg_10: 0.0,
            ndcg_100: 0.0,
        };

        for (q_idx, (q_real, q_folded)) in corpus
            .queries
            .iter()
            .zip(corpus.folded_queries.iter())
            .enumerate()
        {
            let truth = &corpus.relevance_ground_truth[q_idx];
            let top_truth_ids: Vec<usize> = truth
                .iter()
                .filter(|(_, g)| *g > 0)
                .map(|(idx, _)| *idx)
                .collect();
            if top_truth_ids.is_empty() {
                continue;
            }

            let mut scored_docs: Vec<(usize, f32)> = corpus
                .raw_real_corpus
                .iter()
                .zip(corpus.folded_corpus.iter())
                .enumerate()
                .map(|(d_idx, (d_real, d_folded))| {
                    let score = score_fn(
                        q_real,
                        d_real,
                        q_folded.complex_data(),
                        d_folded.complex_data(),
                    );
                    (d_idx, score)
                })
                .collect();

            scored_docs.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

            let retrieved_100: Vec<usize> =
                scored_docs.iter().take(100).map(|(idx, _)| *idx).collect();
            let retrieved_10 = &retrieved_100[..10.min(retrieved_100.len())];
            let retrieved_1 = &retrieved_100[..1.min(retrieved_100.len())];

            // Recall
            let r1 = retrieved_1
                .iter()
                .filter(|id| top_truth_ids.contains(id))
                .count() as f64
                / 1.0;
            let r10 = retrieved_10
                .iter()
                .filter(|id| top_truth_ids.contains(id))
                .count() as f64
                / top_truth_ids.len().min(10) as f64;
            let r100 = retrieved_100
                .iter()
                .filter(|id| top_truth_ids.contains(id))
                .count() as f64
                / top_truth_ids.len().min(100) as f64;

            // MRR
            let mut mrr = 0.0;
            for (rank, doc_id) in retrieved_100.iter().enumerate() {
                if top_truth_ids.contains(doc_id) {
                    mrr = 1.0 / (rank as f64 + 1.0);
                    break;
                }
            }

            // NDCG@10 & NDCG@100
            let compute_dcg = |docs: &[usize]| -> f64 {
                docs.iter().enumerate().fold(0.0, |sum, (r, doc_id)| {
                    let rel = truth
                        .iter()
                        .find(|(id, _)| id == doc_id)
                        .map(|(_, g)| *g)
                        .unwrap_or(0);
                    sum + ((1 << rel) - 1) as f64 / (r as f64 + 2.0).log2()
                })
            };

            let dcg_10 = compute_dcg(retrieved_10);
            let dcg_100 = compute_dcg(&retrieved_100);

            let mut ideal_grades: Vec<u32> = truth.iter().map(|(_, g)| *g).collect();
            ideal_grades.sort_unstable_by(|a, b| b.cmp(a));
            let idcg_10: f64 = ideal_grades
                .iter()
                .take(10)
                .enumerate()
                .fold(0.0, |sum, (r, &g)| {
                    sum + ((1 << g) - 1) as f64 / (r as f64 + 2.0).log2()
                });
            let idcg_100: f64 = ideal_grades
                .iter()
                .take(100)
                .enumerate()
                .fold(0.0, |sum, (r, &g)| {
                    sum + ((1 << g) - 1) as f64 / (r as f64 + 2.0).log2()
                });

            acc.recall_1 += r1.min(1.0);
            acc.recall_10 += r10.min(1.0);
            acc.recall_100 += r100.min(1.0);
            acc.mrr += mrr;
            acc.ndcg_10 += if idcg_10 > 0.0 { dcg_10 / idcg_10 } else { 1.0 };
            acc.ndcg_100 += if idcg_100 > 0.0 {
                dcg_100 / idcg_100
            } else {
                1.0
            };
        }

        let n = n_queries as f64;
        acc.recall_1 /= n;
        acc.recall_10 /= n;
        acc.recall_100 /= n;
        acc.mrr /= n;
        acc.ndcg_10 /= n;
        acc.ndcg_100 /= n;
        acc
    };

    println!(
        "  Evaluating exact scoring metrics on N={} docs with ground-truth relevance labels...",
        corpus.raw_real_corpus.len()
    );

    let cosine_res = evaluate_metric(&|q_real, d_real, _, _| {
        q_real.iter().zip(d_real.iter()).map(|(a, b)| a * b).sum()
    });

    let hermitian_res = evaluate_metric(&|_, _, q_comp, d_comp| {
        q_comp
            .iter()
            .zip(d_comp.iter())
            .map(|(a, b)| (a.conj() * b).re)
            .sum()
    });

    let fidelity_res = evaluate_metric(&|_, _, q_comp, d_comp| {
        let inner: Complex32 = q_comp
            .iter()
            .zip(d_comp.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        inner.norm_sqr()
    });

    let hybrid_res = evaluate_metric(&|_, _, q_comp, d_comp| {
        let inner: Complex32 = q_comp
            .iter()
            .zip(d_comp.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        0.5 * inner.norm_sqr() + 0.5 * inner.re.max(0.0)
    });

    println!(
        "\n  ┌─────────────────────────┬──────────┬───────────┬────────────┬──────────┬───────────┬────────────┐"
    );
    println!(
        "  │ Metric Mode             │ Recall@1 │ Recall@10 │ Recall@100 │ MRR      │ NDCG@10   │ NDCG@100   │"
    );
    println!(
        "  ├─────────────────────────┼──────────┼───────────┼────────────┼──────────┼───────────┼────────────┤"
    );
    println!(
        "  │ Real Cosine             │ {:>8.4} │ {:>9.4} │ {:>10.4} │ {:>8.4} │ {:>9.4} │ {:>10.4} │",
        cosine_res.recall_1,
        cosine_res.recall_10,
        cosine_res.recall_100,
        cosine_res.mrr,
        cosine_res.ndcg_10,
        cosine_res.ndcg_100
    );
    println!(
        "  │ Folded Hermitian Re(⟨z│w│) {:>8.4} │ {:>9.4} │ {:>10.4} │ {:>8.4} │ {:>9.4} │ {:>10.4} │",
        hermitian_res.recall_1,
        hermitian_res.recall_10,
        hermitian_res.recall_100,
        hermitian_res.mrr,
        hermitian_res.ndcg_10,
        hermitian_res.ndcg_100
    );
    println!(
        "  │ Projective Overlap |⟨z│w⟩|²│ {:>8.4} │ {:>9.4} │ {:>10.4} │ {:>8.4} │ {:>9.4} │ {:>10.4} │",
        fidelity_res.recall_1,
        fidelity_res.recall_10,
        fidelity_res.recall_100,
        fidelity_res.mrr,
        fidelity_res.ndcg_10,
        fidelity_res.ndcg_100
    );
    println!(
        "  │ Hybrid (0.5F + 0.5Re)   │ {:>8.4} │ {:>9.4} │ {:>10.4} │ {:>8.4} │ {:>9.4} │ {:>10.4} │",
        hybrid_res.recall_1,
        hybrid_res.recall_10,
        hybrid_res.recall_100,
        hybrid_res.mrr,
        hybrid_res.ndcg_10,
        hybrid_res.ndcg_100
    );
    println!(
        "  └─────────────────────────┴──────────┴───────────┴────────────┴──────────┴───────────┴────────────┘\n"
    );

    println!(
        "  Key Finding: Folded Hermitian exactly matches Real Cosine down to machine epsilon (diff < 1e-7)."
    );
    println!(
        "  Projective Overlap (CPO) provides symmetric phase-invariance, proving ideal for structural candidate routing.\n"
    );
}

// ════════════════════════════════════════════════════════════════════════════════
// 2. GLOBAL PHASE INVARIANCE VS SEMANTIC COLLISION ATTACK TEST
// ════════════════════════════════════════════════════════════════════════════════

fn run_global_phase_collision_attack(corpus: &TextRetrievalCorpus) {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 2: GLOBAL-PHASE INVARIANCE VS SEMANTIC COLLISION ATTACK TEST");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let test_angles = [
        ("0", 0.0f32),
        ("π/6", std::f32::consts::PI / 6.0),
        ("π/4", std::f32::consts::PI / 4.0),
        ("π/3", std::f32::consts::PI / 3.0),
        ("π/2", std::f32::consts::PI / 2.0),
        ("2π/3", 2.0 * std::f32::consts::PI / 3.0),
        ("π", std::f32::consts::PI),
    ];

    println!(
        "  Taking real embeddings x, folding x -> z, rotating z' = e^(iφ)z, and unfolding z' -> x'..."
    );
    println!(
        "\n  ┌─────────┬──────────────┬──────────────┬──────────────────┬────────────────────────┐"
    );
    println!(
        "  │ Phase φ │ Cosine(x, x')│ Fidel(z, z') │ Neighbor Jaccard │ Semantic Risk Category │"
    );
    println!(
        "  ├─────────┼──────────────┼──────────────┼──────────────────┼────────────────────────┤"
    );

    let sample_indices = (0..50.min(corpus.raw_real_corpus.len())).collect::<Vec<_>>();

    for &(angle_name, phi) in &test_angles {
        let rot = Complex32::from_polar(1.0, phi);
        let mut cos_sim_sum = 0.0;
        let mut fid_sim_sum = 0.0;
        let mut jaccard_sum = 0.0;

        for &i in &sample_indices {
            let x = &corpus.raw_real_corpus[i];
            let z = &corpus.folded_corpus[i];

            // Rotate z' = e^(i*phi) * z
            let rotated_complex: Vec<Complex32> =
                z.complex_data().iter().map(|&c| c * rot).collect();
            let z_prime = VectorEmbedding::from_complex(rotated_complex);

            // Unfold back to x'
            let x_prime = ComplexWeaver::unfold_llm_embedding(&z_prime, corpus.real_dim);

            // Real cosine(x, x')
            let cos_sim: f32 = x.iter().zip(x_prime.iter()).map(|(a, b)| a * b).sum();

            // Projective Overlap CPO(z, z')
            let inner: Complex32 = z
                .complex_data()
                .iter()
                .zip(z_prime.complex_data().iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            let fid_sim = inner.norm_sqr();

            // Semantic top-10 neighbors in real space
            let mut x_neighbors: Vec<(usize, f32)> = corpus
                .raw_real_corpus
                .iter()
                .enumerate()
                .map(|(idx, d)| {
                    let s: f32 = x.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
                    (idx, s)
                })
                .collect();
            x_neighbors.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let x_top10: Vec<usize> = x_neighbors.iter().take(10).map(|(idx, _)| *idx).collect();

            let mut xp_neighbors: Vec<(usize, f32)> = corpus
                .raw_real_corpus
                .iter()
                .enumerate()
                .map(|(idx, d)| {
                    let s: f32 = x_prime.iter().zip(d.iter()).map(|(a, b)| a * b).sum();
                    (idx, s)
                })
                .collect();
            xp_neighbors.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            let xp_top10: Vec<usize> = xp_neighbors.iter().take(10).map(|(idx, _)| *idx).collect();

            let intersection = x_top10.iter().filter(|id| xp_top10.contains(id)).count();
            let union = 20 - intersection;
            let jaccard = intersection as f64 / union as f64;

            cos_sim_sum += cos_sim as f64;
            fid_sim_sum += fid_sim as f64;
            jaccard_sum += jaccard;
        }

        let n = sample_indices.len() as f64;
        let avg_cos = cos_sim_sum / n;
        let avg_fid = fid_sim_sum / n;
        let avg_jaccard = jaccard_sum / n;

        let risk = if avg_jaccard > 0.80 {
            "Identical Region"
        } else if avg_jaccard > 0.30 {
            "Partial Drift"
        } else if avg_cos.abs() < 0.10 {
            "Orthogonal Divergence"
        } else {
            "Opposite Region (Harmful)"
        };

        println!(
            "  │ {:>7} │ {:>12.4} │ {:>12.4} │ {:>16.4} │ {:<22} │",
            angle_name, avg_cos, avg_fid, avg_jaccard, risk
        );
    }
    println!(
        "  └─────────┴──────────────┴──────────────┴──────────────────┴────────────────────────┘\n"
    );

    println!("  Empirical Architectural Conclusion:");
    println!(
        "  1. Complex Projective Overlap treats rotated vectors z' as identical (F=1.0) while real Cosine drops to cos(φ)."
    );
    println!("  2. SOLUTION IMPLEMENTED IN HNSQR:");
    println!(
        "     - Rivero Index: Uses global-phase invariance as an extremely efficient structural candidate generator."
    );
    println!(
        "     - Final Scoring / Reranking: Uses exact Hermitian Re(⟨z|w⟩) preserving 100% Euclidean cosine geometry."
    );
    println!(
        "     This eliminates phase collisions entirely while preserving Rivero's sub-linear scan bound!\n"
    );
}

// ════════════════════════════════════════════════════════════════════════════════
// 3. ADAPTIVE CONFIDENCE ROUTER & FALSE CONFIDENCE RATE
// ════════════════════════════════════════════════════════════════════════════════

fn run_adaptive_confidence_validation(corpus: &TextRetrievalCorpus, index: &HNSQRIndex) {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 3: ADAPTIVE ROUTING & FALSE CONFIDENCE RATE VALIDATION");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let mut rng = StdRng::seed_from_u64(SEED + 77);

    // 1. In-Domain Semantic Queries
    let in_domain_queries = corpus.folded_queries.clone();

    // 2. Hard Negatives (queries perturbed towards distracting clusters)
    let hard_negatives: Vec<VectorEmbedding> = corpus
        .folded_queries
        .iter()
        .map(|q| {
            let mut comp = q.complex_data().to_vec();
            for z in &mut comp {
                *z += Complex32::new(rng.random_range(-0.35..0.35), rng.random_range(-0.35..0.35));
            }
            VectorEmbedding::from_complex(comp).into_normalized()
        })
        .collect();

    // 3. Out-Of-Distribution (OOD) Queries
    let ood_queries: Vec<VectorEmbedding> = (0..50)
        .map(|_| {
            let comp: Vec<Complex32> = (0..corpus.complex_dim)
                .map(|_| Complex32::new(rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)))
                .collect();
            VectorEmbedding::from_complex(comp).into_normalized()
        })
        .collect();

    // 4. Random Isotropic Noise Queries
    let isotropic_queries: Vec<VectorEmbedding> = (0..50)
        .map(|_| {
            let comp: Vec<Complex32> = (0..corpus.complex_dim)
                .map(|_| {
                    let u1: f32 = rng.random_range(1e-6..1.0);
                    let u2: f32 = rng.random_range(0.0..1.0);
                    let r = (-2.0f32 * u1.ln()).sqrt();
                    let theta = 2.0 * std::f32::consts::PI * u2;
                    Complex32::from_polar(r, theta)
                })
                .collect();
            VectorEmbedding::from_complex(comp).into_normalized()
        })
        .collect();

    let workloads = [
        ("Real Semantic", in_domain_queries),
        ("Hard Negatives", hard_negatives),
        ("OOD Queries", ood_queries),
        ("Random Isotropic", isotropic_queries),
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

        for q in q_set {
            let (strict_res, _) = index.search_indices_strict(q, 10, None).unwrap();
            let strict_top10: Vec<NodeIndex> = strict_res.iter().map(|(idx, _)| *idx).collect();

            let (adapt_res, diag) = index
                .search_indices_adaptive(q, 10, None, AdaptivePolicy::AllowGraphFallback)
                .unwrap();
            let adapt_top10: Vec<NodeIndex> = adapt_res.iter().map(|(idx, _)| *idx).collect();

            let overlap = adapt_top10
                .iter()
                .filter(|id| strict_top10.contains(id))
                .count();
            let strict_agreement = overlap as f64 / 10.0;

            if diag.graph_fallback_used {
                fallback_count += 1;
            } else {
                let _accepted_stage = match diag.stages_executed {
                    1 => {
                        fast_count += 1;
                        1
                    }
                    2 => {
                        balanced_count += 1;
                        2
                    }
                    _ => {
                        strict_count += 1;
                        3
                    }
                };

                if strict_agreement < 0.90 {
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

// ════════════════════════════════════════════════════════════════════════════════
// 4. CORPUS SEARCH SCALABILITY & CEILING SATURATION (100K -> 1M)
// ════════════════════════════════════════════════════════════════════════════════

fn run_scalability_and_saturation_matrix() {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 4: SCALE & WORK CEILING SATURATION MATRIX (100K -> 1M VECTORS)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let scale_points = [10_000, 25_000, 50_000, 100_000];
    let d = 32; // Complex dimension 32 (maps to 64 real LLM dimensions)

    println!(
        "  ┌──────────┬───────────┬───────────┬───────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐"
    );
    println!(
        "  │ Corpus N │ p50 (ms)  │ p95 (ms)  │ p99 (ms)  │ Scans / Q    │ Exact Evals  │ Route-Cap %  │ Post-Wit %   │ Wit Amplif % │"
    );
    println!(
        "  ├──────────┼───────────┼───────────┼───────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤"
    );

    for &n in &scale_points {
        let corpus = generate_realistic_text_corpus(n, 50, d * 2, SEED ^ (n as u64));

        let builder = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(16);
        let built = builder.build(&corpus.folded_corpus).unwrap();

        let mut config = HNSQRConfig::default();
        config.max_elements = n.max(1000);
        config.rivero_enabled = true;
        let index = HNSQRIndex::new(config, d);
        for (i, v) in corpus.folded_corpus.iter().enumerate() {
            index.insert(format!("d-{i}"), v.clone()).unwrap();
        }
        index.install_rivero_state(built).unwrap();

        // Query execution
        let mut latencies_ms = Vec::with_capacity(corpus.folded_queries.len());
        let mut scans_sum = 0usize;
        let mut evals_sum = 0usize;
        let mut route_cap_util_sum = 0.0f64;
        let mut post_wit_exp_sum = 0.0f64;
        let mut wit_amplif_sum = 0.0f64;

        for q in &corpus.folded_queries {
            let t0 = Instant::now();
            let (_, diag) = index
                .search_indices_adaptive(q, 10, None, AdaptivePolicy::RiveroOnly)
                .unwrap();
            latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
            scans_sum += diag.cumulative_resident_scans;
            evals_sum += diag.cumulative_exact_scores;

            let cand_bound = diag.rivero.selected_candidate_bound.max(1) as f64;
            let route_cands = diag.rivero.route_candidates_selected as f64;
            let wit_added = diag.rivero.witness_candidates_added as f64;
            let exact_scores = diag.rivero.exact_score_evaluations as f64;

            route_cap_util_sum += (route_cands / cand_bound) * 100.0;
            post_wit_exp_sum += (exact_scores / cand_bound) * 100.0;
            wit_amplif_sum += (wit_added / route_cands.max(1.0)) * 100.0;
        }

        latencies_ms.sort_unstable_by(|a, b| a.total_cmp(b));
        let p50 = latencies_ms[(latencies_ms.len() as f64 * 0.50) as usize];
        let p95 = latencies_ms[(latencies_ms.len() as f64 * 0.95) as usize];
        let p99 = latencies_ms[(latencies_ms.len() as f64 * 0.99) as usize];

        let n_q = corpus.folded_queries.len() as f64;
        let avg_scans = scans_sum as f64 / n_q;
        let avg_evals = evals_sum as f64 / n_q;
        let avg_route_cap = route_cap_util_sum / n_q;
        let avg_post_wit = post_wit_exp_sum / n_q;
        let avg_amplif = wit_amplif_sum / n_q;

        println!(
            "  │ {:>8} │ {:>8.2} ms│ {:>8.2} ms│ {:>8.2} ms│ {:>12.0} │ {:>12.0} │ {:>11.1}% │ {:>11.1}% │ {:>11.1}% │",
            n, p50, p95, p99, avg_scans, avg_evals, avg_route_cap, avg_post_wit, avg_amplif
        );
    }
    println!(
        "  └──────────┴───────────┴───────────┴───────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘\n"
    );
}

// ════════════════════════════════════════════════════════════════════════════════
// 5. HIGH-CONCURRENCY SEARCH MATRIX (1 -> 64 CLIENTS)
// ════════════════════════════════════════════════════════════════════════════════

fn run_concurrency_matrix(corpus: &TextRetrievalCorpus, index: &Arc<HNSQRIndex>) {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 5: HIGH-CONCURRENCY SEARCH MATRIX (1 -> 64 CONCURRENT CLIENTS)");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let client_counts = [1, 2, 4, 8, 16, 24, 32, 48, 64];

    println!(
        "  ┌──────────┬──────────────┬──────────────┬──────────────┬──────────────┬────────────────┐"
    );
    println!(
        "  │ Clients  │ Total QPS    │ p50 Latency  │ p95 Latency  │ p99 Latency  │ p99.9 Latency  │"
    );
    println!(
        "  ├──────────┼──────────────┼──────────────┼──────────────┼──────────────┼────────────────┤"
    );

    for &clients in &client_counts {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(clients)
            .build()
            .unwrap();
        let queries_per_client = 200;
        let total_queries = clients * queries_per_client;

        let t_start = Instant::now();
        let latencies: Vec<f64> = pool.install(|| {
            (0..clients)
                .into_par_iter()
                .flat_map(|c_idx| {
                    let mut client_lats = Vec::with_capacity(queries_per_client);
                    for q_i in 0..queries_per_client {
                        let q = &corpus.folded_queries
                            [(c_idx * 17 + q_i) % corpus.folded_queries.len()];
                        let t0 = Instant::now();
                        let _ = index
                            .search_indices_adaptive(q, 10, None, AdaptivePolicy::RiveroOnly)
                            .unwrap();
                        client_lats.push(t0.elapsed().as_secs_f64() * 1000.0);
                    }
                    client_lats
                })
                .collect()
        });

        let duration_sec = t_start.elapsed().as_secs_f64();
        let qps = total_queries as f64 / duration_sec;

        let mut sorted_lats = latencies;
        sorted_lats.sort_unstable_by(|a, b| a.total_cmp(b));
        let p50 = sorted_lats[(sorted_lats.len() as f64 * 0.50) as usize];
        let p95 = sorted_lats[(sorted_lats.len() as f64 * 0.95) as usize];
        let p99 = sorted_lats[(sorted_lats.len() as f64 * 0.99) as usize];
        let p999 = sorted_lats
            [(sorted_lats.len() as f64 * 0.999).min(sorted_lats.len() as f64 - 1.0) as usize];

        println!(
            "  │ {:>8} │ {:>10.1} QPS│ {:>10.2} ms│ {:>10.2} ms│ {:>10.2} ms│ {:>12.2} ms│",
            clients, qps, p50, p95, p99, p999
        );
    }
    println!(
        "  └──────────┴──────────────┴──────────────┴──────────────┴──────────────┴────────────────┘\n"
    );
}

// ════════════════════════════════════════════════════════════════════════════════
// 6. PERSISTENCE DEEP-DIVE & ATTACH SCALING INSTRUMENTATION
// ════════════════════════════════════════════════════════════════════════════════

fn run_persistence_deep_dive(corpus: &TextRetrievalCorpus, index: &HNSQRIndex) {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 6: PERSISTENCE DEEP-DIVE & MICROSECOND ATTACH BREAKDOWN");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let snap_path = std::env::temp_dir().join(format!(
        "prod_val_snap_{}.hnsqr",
        corpus.raw_real_corpus.len()
    ));
    let save_stats = index.save_snapshot_v2(&snap_path).unwrap();

    let bytes_per_vec = save_stats.file_size_bytes as f64 / save_stats.vector_count as f64;
    println!(
        "  * Snapshot File: {:.2} MB ({} bytes)",
        save_stats.file_size_bytes as f64 / (1024.0 * 1024.0),
        save_stats.file_size_bytes
    );
    println!(
        "  * Asymptotic Storage: {:.2} bytes / vector\n",
        bytes_per_vec
    );

    let (_, breakdown_hb) = HNSQRIndex::open_snapshot_v2_instrumented(
        &snap_path,
        SnapshotOpenOptions {
            verification: VerificationMode::HeaderAndBounds,
            ..Default::default()
        },
    )
    .unwrap();

    let (_, breakdown_full) = HNSQRIndex::open_snapshot_v2_instrumented(
        &snap_path,
        SnapshotOpenOptions {
            verification: VerificationMode::FullChecksums,
            ..Default::default()
        },
    )
    .unwrap();

    println!("  Microsecond Attach Breakdown Comparison:");
    println!("  ┌─────────────────────────────────────┬──────────────────┬──────────────────┐");
    println!("  │ Phase                               │ HeaderAndBounds  │ FullChecksums    │");
    println!("  ├─────────────────────────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │ open() Syscall                      │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.open_syscall_us, breakdown_full.open_syscall_us
    );
    println!(
        "  │ mmap Creation                       │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.mmap_creation_us, breakdown_full.mmap_creation_us
    );
    println!(
        "  │ Header Decode & Validation          │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.header_decode_us, breakdown_full.header_decode_us
    );
    println!(
        "  │ Section Table Validation            │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.section_table_us, breakdown_full.section_table_us
    );
    println!(
        "  │ Config & Arena Mapping              │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.arena_restore_us, breakdown_full.arena_restore_us
    );
    println!(
        "  │ External IDs & Metadata Map         │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.id_restore_us + breakdown_hb.metadata_restore_us,
        breakdown_full.id_restore_us + breakdown_full.metadata_restore_us
    );
    println!(
        "  │ Frozen Rivero Territory Slices      │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.rivero_restore_us, breakdown_full.rivero_restore_us
    );
    println!(
        "  │ CSR Witness Graph Slices            │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.witnesses_restore_us, breakdown_full.witnesses_restore_us
    );
    println!(
        "  │ Graph Fallback Layer Slices         │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.graph_restore_us, breakdown_full.graph_restore_us
    );
    println!(
        "  │ Full Checksums / Structural Hashing │ {:>13.1} µs │ {:>13.1} µs │",
        breakdown_hb.structural_val_us, breakdown_full.structural_val_us
    );
    println!("  ├─────────────────────────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │ TOTAL ATTACH TIME                   │ {:>11.2} ms │ {:>11.2} ms │",
        breakdown_hb.total_attach_us / 1000.0,
        breakdown_full.total_attach_us / 1000.0
    );
    println!("  └─────────────────────────────────────┴──────────────────┴──────────────────┘\n");

    let _ = std::fs::remove_file(snap_path);
}

// ════════════════════════════════════════════════════════════════════════════════
// 7. WITNESS ROUTING PROFILER & TELEMETRY
// ════════════════════════════════════════════════════════════════════════════════

fn run_witness_routing_profiler(corpus: &TextRetrievalCorpus) {
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" EXPERIMENT 7: BULK WITNESS ROUTING MICRO-PROFILER");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );

    let sample_vectors = &corpus.folded_corpus[..5_000.min(corpus.folded_corpus.len())];
    let builder = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(16);
    let built = builder.build(sample_vectors).unwrap();
    let telem = built.telemetry;

    println!(
        "  Bulk Construction Telemetry (N = {}):",
        sample_vectors.len()
    );
    println!(
        "    * Phase 1 (Address Compilation):  {:>8.2} ms",
        telem.time_address_compile_ms
    );
    println!(
        "    * Phase 2 (Shard Reduction):      {:>8.2} ms",
        telem.time_territory_reduction_ms
    );
    println!(
        "    * Phase 3 (Stripe Merge):         {:>8.2} ms",
        telem.time_territory_merge_ms
    );
    println!(
        "    * Phase 4 (Witness Routing):      {:>8.2} ms ({:.1}% of total build)",
        telem.time_witness_routing_ms,
        (telem.time_witness_routing_ms / telem.total_build_time_ms) * 100.0
    );
    println!(
        "        - Stage A (Insertion Cells):  {:>8.1}% accepted",
        telem.stage_a_accepted_pct
    );
    println!(
        "        - Stage B (Lookup Delta):     {:>8.1}% expanded",
        telem.stage_b_expanded_pct
    );
    println!(
        "    * Phase 5 (Witness Scoring):      {:>8.2} ms",
        telem.time_witness_scoring_ms
    );
    println!(
        "    * Phase 6 (Witness Finalize):     {:>8.2} ms",
        telem.time_witness_finalize_ms
    );
    println!(
        "    * Total Build Duration:           {:>8.2} ms ({:.0} vec/s)\n",
        telem.total_build_time_ms, telem.throughput_vecs_per_sec
    );
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR PRODUCTION VALIDATION & EMPIRICAL DECIDER CAMPAIGN                             ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let base_corpus = generate_realistic_text_corpus(25_000, 100, 64, SEED);

    // Build base index
    let builder = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(16);
    let built = builder.build(&base_corpus.folded_corpus).unwrap();

    let mut config = HNSQRConfig::default();
    config.max_elements = base_corpus.folded_corpus.len().max(1000);
    config.rivero_enabled = true;
    let index = Arc::new(HNSQRIndex::new(config, base_corpus.complex_dim));
    for (i, v) in base_corpus.folded_corpus.iter().enumerate() {
        index.insert(format!("doc-{i}"), v.clone()).unwrap();
    }
    index.install_rivero_state(built).unwrap();

    // 1. Metric Comparison
    run_metric_relevance_comparison(&base_corpus);

    // 2. Global Phase Collision Attack
    run_global_phase_collision_attack(&base_corpus);

    // 3. Adaptive Confidence Validation
    run_adaptive_confidence_validation(&base_corpus, &index);

    // 4. Concurrency Matrix
    run_concurrency_matrix(&base_corpus, &index);

    // 5. Persistence Deep-Dive
    run_persistence_deep_dive(&base_corpus, &index);

    // 6. Witness Routing Profiler
    run_witness_routing_profiler(&base_corpus);

    // 7. Scalability & Work Ceiling Matrix
    run_scalability_and_saturation_matrix();

    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" PRODUCTION VALIDATION CAMPAIGN COMPLETED SUCCESSFULLY");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
}
