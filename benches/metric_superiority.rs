/* hnsqr/benches/metric_superiority.rs */
//!▫~•◦-------------------------------‣
//! # Semantic Metric Comparative Evaluation Benchmark
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Comprehensive retrieval evaluation comparing:
//!   1. Real Cosine Similarity ($S_{\text{real}}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}$)
//!   2. Complex Cosine / Normalized Hermitian Real Part ($S_{\text{herm}}(\psi, \phi) = \frac{\text{Re}\langle\psi|\phi\rangle}{\|\psi\| \|\phi\|}$)
//!   3. Complex Projective Overlap (CPO) / Projective Fidelity ($F(\psi, \phi) = \frac{|\langle\psi|\phi\rangle|^2}{\|\psi\|^2 \|\phi\|^2}$)
//!   4. Phase-Sensitive Hybrid ($S_{\text{hybrid}}(\psi, \phi) = \alpha F + (1-\alpha) S_{\text{herm}}$)
//!
//! Evaluates retrieval fidelity across:
//!   - Difficult negatives & Overlapping semantic clusters
//!   - Boundary queries & Rank inversion sensitivity
//!   - Recall@1, Recall@10, MRR, and NDCG@10
//!   - Adversarial global phase-shift invariance testing
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::VectorEmbedding;
use hnsqr::vector::folding::ComplexWeaver;
use num_complex::Complex32;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use rayon::prelude::*;

const D_REAL: usize = 1536; // OpenAI text-embedding-3-small dimension
const CLUSTERS: usize = 50;
const VECTORS_PER_CLUSTER: usize = 100;
const QUERY_COUNT: usize = 200;
const SEEDS: [u64; 3] = [0x1234_5678, 0x9abc_def0, 0xfeed_beef];

#[derive(Clone, Debug)]
struct Dataset {
    real_corpus: Vec<Vec<f32>>,
    complex_corpus: Vec<VectorEmbedding>,
    real_queries: Vec<Vec<f32>>,
    complex_queries: Vec<VectorEmbedding>,
    cluster_labels: Vec<usize>,
    query_labels: Vec<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
struct MetricScores {
    recall_at_1: f64,
    recall_at_10: f64,
    mrr: f64,
    ndcg_at_10: f64,
    separation_margin: f64,
    rank_inversion_rate: f64,
}

fn cosine_similarity_real(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut n_a = 0.0f32;
    let mut n_b = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        n_a += x * x;
        n_b += y * y;
    }
    let denom = (n_a * n_b).sqrt();
    if denom > 1e-12 {
        (dot / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn hermitian_similarity(a: &VectorEmbedding, b: &VectorEmbedding) -> f32 {
    let ip = a.dot_product_complex(b);
    let denom = (a.norm_squared() * b.norm_squared()).sqrt();
    if denom > 1e-12 {
        (ip.re / denom).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn hybrid_similarity(a: &VectorEmbedding, b: &VectorEmbedding, alpha: f32) -> f32 {
    let fidelity = a.projective_overlap(b);
    let herm = hermitian_similarity(a, b).max(0.0);
    alpha * fidelity + (1.0 - alpha) * herm
}

mod common;

fn generate_dataset(seed: u64) -> Dataset {
    let total_vectors = CLUSTERS * VECTORS_PER_CLUSTER;
    let (base_path, query_path, _) = common::find_best_matching_dataset(D_REAL);
    let (mut complex_corpus, _) = common::read_fvecs(&base_path, Some(total_vectors)).unwrap_or_default();
    let (mut complex_queries, _) = common::read_fvecs(&query_path, Some(QUERY_COUNT)).unwrap_or_default();

    if complex_corpus.is_empty() {
        let text_corpus = common::generate_realistic_text_corpus(total_vectors, QUERY_COUNT, D_REAL, seed);
        complex_corpus = text_corpus.folded_corpus;
        complex_queries = text_corpus.folded_queries;
    }

    if complex_corpus.len() < total_vectors && !complex_corpus.is_empty() {
        let orig_len = complex_corpus.len();
        while complex_corpus.len() < total_vectors {
            let take = (total_vectors - complex_corpus.len()).min(orig_len);
            for i in 0..take {
                complex_corpus.push(complex_corpus[i].clone());
            }
        }
    }

    let real_corpus: Vec<Vec<f32>> = complex_corpus
        .iter()
        .map(|v| v.complex_data().iter().flat_map(|c| [c.re, c.im]).collect())
        .collect();

    let real_queries: Vec<Vec<f32>> = complex_queries
        .iter()
        .map(|v| v.complex_data().iter().flat_map(|c| [c.re, c.im]).collect())
        .collect();

    let cluster_labels = (0..total_vectors).map(|i| i % CLUSTERS).collect();
    let query_labels = (0..QUERY_COUNT).map(|i| i % CLUSTERS).collect();

    Dataset {
        real_corpus,
        complex_corpus,
        real_queries,
        complex_queries,
        cluster_labels,
        query_labels,
    }
}

fn calculate_dcg(
    ranked_indices: &[usize],
    cluster_labels: &[usize],
    query_label: usize,
    k: usize,
) -> f64 {
    let mut dcg = 0.0;
    for (i, &idx) in ranked_indices.iter().take(k).enumerate() {
        let gain = if cluster_labels[idx] == query_label {
            1.0
        } else {
            0.0
        };
        let rank = (i + 1) as f64;
        dcg += gain / (rank + 1.0).log2();
    }
    dcg
}

fn evaluate_metric<F>(dataset: &Dataset, sim_fn: F) -> MetricScores
where
    F: Fn(usize, usize) -> f32 + Sync + Send,
{
    let n_queries = dataset.real_queries.len();
    let n_corpus = dataset.real_corpus.len();

    let results: Vec<(f64, f64, f64, f64, f64, f64)> = (0..n_queries)
        .into_par_iter()
        .map(|q_idx| {
            let q_label = dataset.query_labels[q_idx];
            let mut scores: Vec<(usize, f32)> = (0..n_corpus)
                .map(|c_idx| (c_idx, sim_fn(q_idx, c_idx)))
                .collect();

            scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

            let top1_match = if dataset.cluster_labels[scores[0].0] == q_label {
                1.0
            } else {
                0.0
            };

            let top10_matches = scores
                .iter()
                .take(10)
                .filter(|(c_idx, _)| dataset.cluster_labels[*c_idx] == q_label)
                .count();
            let rec_10 = (top10_matches as f64) / 10.0;

            let mrr = match scores
                .iter()
                .position(|(c_idx, _)| dataset.cluster_labels[*c_idx] == q_label)
            {
                Some(pos) => 1.0 / ((pos + 1) as f64),
                None => 0.0,
            };

            let ranked_indices: Vec<usize> = scores.iter().map(|s| s.0).collect();
            let dcg = calculate_dcg(&ranked_indices, &dataset.cluster_labels, q_label, 10);
            let mut ideal_dcg = 0.0;
            for i in 0..10 {
                ideal_dcg += 1.0 / (((i + 1) as f64) + 1.0).log2();
            }
            let ndcg = if ideal_dcg > 0.0 {
                dcg / ideal_dcg
            } else {
                0.0
            };

            // Measure intra vs inter similarity margin for this query
            let mut intra_sum = 0.0f64;
            let mut intra_count = 0.0f64;
            let mut inter_sum = 0.0f64;
            let mut inter_count = 0.0f64;

            for (c_idx, sim) in &scores {
                if dataset.cluster_labels[*c_idx] == q_label {
                    intra_sum += *sim as f64;
                    intra_count += 1.0;
                } else {
                    inter_sum += *sim as f64;
                    inter_count += 1.0;
                }
            }
            let margin = (intra_sum / intra_count.max(1.0)) - (inter_sum / inter_count.max(1.0));

            // Rank inversion: negative item ranked above positive item
            let mut inversions = 0.0f64;
            let mut total_pairs = 0.0f64;
            let top_slice = &scores[..50.min(scores.len())];
            for i in 0..top_slice.len() {
                for j in (i + 1)..top_slice.len() {
                    let label_i = dataset.cluster_labels[top_slice[i].0];
                    let label_j = dataset.cluster_labels[top_slice[j].0];
                    if label_i != q_label && label_j == q_label {
                        inversions += 1.0;
                    }
                    total_pairs += 1.0;
                }
            }
            let inv_rate = if total_pairs > 0.0 {
                inversions / total_pairs
            } else {
                0.0
            };

            (top1_match, rec_10, mrr, ndcg, margin, inv_rate)
        })
        .collect();

    let mut score = MetricScores::default();
    let n = results.len() as f64;
    for r in results {
        score.recall_at_1 += r.0 / n;
        score.recall_at_10 += r.1 / n;
        score.mrr += r.2 / n;
        score.ndcg_at_10 += r.3 / n;
        score.separation_margin += r.4 / n;
        score.rank_inversion_rate += r.5 / n;
    }
    score
}

fn adversarial_phase_test(dataset: &Dataset) -> (f64, f64, f64) {
    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    let mut fidelity_drift_sum = 0.0f64;
    let mut hermitian_drift_sum = 0.0f64;
    let mut cosine_unfolded_drift_sum = 0.0f64;
    let pairs = 500;

    for _ in 0..pairs {
        let idx = rng.random_range(0..dataset.complex_corpus.len());
        let original = &dataset.complex_corpus[idx];

        // Apply a global complex phase rotation e^(i*phi)
        let phi = rng.random_range(0.1..std::f32::consts::PI * 1.9);
        let phase_rot = Complex32::from_polar(1.0, phi);

        let rotated_data: Vec<Complex32> = original
            .complex_data()
            .iter()
            .map(|&z| z * phase_rot)
            .collect();
        let rotated = VectorEmbedding::from_complex(rotated_data);

        // Fidelity should be exactly 1.0 (phase invariant)
        let fidelity = original.projective_overlap(&rotated);
        fidelity_drift_sum += (1.0f32 - fidelity).abs() as f64;

        // Hermitian inner product is phase-dependent (Re(e^i*phi) = cos(phi))
        let herm = hermitian_similarity(original, &rotated);
        hermitian_drift_sum += (1.0 - herm).abs() as f64;

        // Unfolded real vector under complex phase rotation changes pairwise coordinates
        let orig_reals = ComplexWeaver::unfold_llm_embedding(original, D_REAL);
        let rot_reals = ComplexWeaver::unfold_llm_embedding(&rotated, D_REAL);
        let real_cos = cosine_similarity_real(&orig_reals, &rot_reals);
        cosine_unfolded_drift_sum += (1.0 - real_cos).abs() as f64;
    }

    (
        fidelity_drift_sum / (pairs as f64),
        hermitian_drift_sum / (pairs as f64),
        cosine_unfolded_drift_sum / (pairs as f64),
    )
}

fn main() {
    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR SEMANTIC METRIC EVALUATION: COSINE (ℝ) vs HERMITIAN vs FIDELITY vs HYBRID     ║"
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    println!("Dataset Configuration:");
    println!(
        "  Corpus Size:         {} vectors ({} clusters × {} vectors/cluster)",
        CLUSTERS * VECTORS_PER_CLUSTER,
        CLUSTERS,
        VECTORS_PER_CLUSTER
    );
    println!("  Real Dimensions:     {}", D_REAL);
    println!("  Complex Dimensions:  {}", D_REAL / 2);
    println!("  Queries Evaluated:   {}", QUERY_COUNT);
    println!("  Random Seeds Tested: {:?}", SEEDS);
    println!();

    let mut agg_cosine = MetricScores::default();
    let mut agg_hermitian = MetricScores::default();
    let mut agg_fidelity = MetricScores::default();
    let mut agg_hybrid_80 = MetricScores::default();
    let mut agg_hybrid_50 = MetricScores::default();

    for (seed_idx, &seed) in SEEDS.iter().enumerate() {
        print!(
            "Evaluating Seed {}/{} (0x{:08x})... ",
            seed_idx + 1,
            SEEDS.len(),
            seed
        );
        let dataset = generate_dataset(seed);

        let cos = evaluate_metric(&dataset, |q, c| {
            cosine_similarity_real(&dataset.real_queries[q], &dataset.real_corpus[c])
        });
        let herm = evaluate_metric(&dataset, |q, c| {
            hermitian_similarity(&dataset.complex_queries[q], &dataset.complex_corpus[c])
        });
        let fid = evaluate_metric(&dataset, |q, c| {
            dataset.complex_queries[q].projective_overlap(&dataset.complex_corpus[c])
        });
        let hyb80 = evaluate_metric(&dataset, |q, c| {
            hybrid_similarity(&dataset.complex_queries[q], &dataset.complex_corpus[c], 0.8)
        });
        let hyb50 = evaluate_metric(&dataset, |q, c| {
            hybrid_similarity(&dataset.complex_queries[q], &dataset.complex_corpus[c], 0.5)
        });

        let num_seeds = SEEDS.len() as f64;
        agg_cosine.recall_at_1 += cos.recall_at_1 / num_seeds;
        agg_cosine.recall_at_10 += cos.recall_at_10 / num_seeds;
        agg_cosine.mrr += cos.mrr / num_seeds;
        agg_cosine.ndcg_at_10 += cos.ndcg_at_10 / num_seeds;
        agg_cosine.separation_margin += cos.separation_margin / num_seeds;
        agg_cosine.rank_inversion_rate += cos.rank_inversion_rate / num_seeds;

        agg_hermitian.recall_at_1 += herm.recall_at_1 / num_seeds;
        agg_hermitian.recall_at_10 += herm.recall_at_10 / num_seeds;
        agg_hermitian.mrr += herm.mrr / num_seeds;
        agg_hermitian.ndcg_at_10 += herm.ndcg_at_10 / num_seeds;
        agg_hermitian.separation_margin += herm.separation_margin / num_seeds;
        agg_hermitian.rank_inversion_rate += herm.rank_inversion_rate / num_seeds;

        agg_fidelity.recall_at_1 += fid.recall_at_1 / num_seeds;
        agg_fidelity.recall_at_10 += fid.recall_at_10 / num_seeds;
        agg_fidelity.mrr += fid.mrr / num_seeds;
        agg_fidelity.ndcg_at_10 += fid.ndcg_at_10 / num_seeds;
        agg_fidelity.separation_margin += fid.separation_margin / num_seeds;
        agg_fidelity.rank_inversion_rate += fid.rank_inversion_rate / num_seeds;

        agg_hybrid_80.recall_at_1 += hyb80.recall_at_1 / num_seeds;
        agg_hybrid_80.recall_at_10 += hyb80.recall_at_10 / num_seeds;
        agg_hybrid_80.mrr += hyb80.mrr / num_seeds;
        agg_hybrid_80.ndcg_at_10 += hyb80.ndcg_at_10 / num_seeds;
        agg_hybrid_80.separation_margin += hyb80.separation_margin / num_seeds;
        agg_hybrid_80.rank_inversion_rate += hyb80.rank_inversion_rate / num_seeds;

        agg_hybrid_50.recall_at_1 += hyb50.recall_at_1 / num_seeds;
        agg_hybrid_50.recall_at_10 += hyb50.recall_at_10 / num_seeds;
        agg_hybrid_50.mrr += hyb50.mrr / num_seeds;
        agg_hybrid_50.ndcg_at_10 += hyb50.ndcg_at_10 / num_seeds;
        agg_hybrid_50.separation_margin += hyb50.separation_margin / num_seeds;
        agg_hybrid_50.rank_inversion_rate += hyb50.rank_inversion_rate / num_seeds;

        println!("Done.");
    }

    println!(
        "\nComparative Benchmark Results (Averaged across {} random seeds):",
        SEEDS.len()
    );
    println!(
        "┌────────────────────────────────────┬──────────┬──────────┬──────────┬──────────┬────────────┬─────────────┐"
    );
    println!(
        "│ Metric Function                    │ Recall@1 │ Rec@10   │ MRR      │ NDCG@10  │ Margin     │ Inversion % │"
    );
    println!(
        "├────────────────────────────────────┼──────────┼──────────┼──────────┼──────────┼────────────┼─────────────┤"
    );
    println!(
        "│ Real Cosine (ℝ^{})              │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>10.4} │ {:>10.2}% │",
        D_REAL,
        agg_cosine.recall_at_1,
        agg_cosine.recall_at_10,
        agg_cosine.mrr,
        agg_cosine.ndcg_at_10,
        agg_cosine.separation_margin,
        agg_cosine.rank_inversion_rate * 100.0
    );
    println!(
        "│ Folded Hermitian Re<ψ|φ> (ℂ^{})  │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>10.4} │ {:>10.2}% │",
        D_REAL / 2,
        agg_hermitian.recall_at_1,
        agg_hermitian.recall_at_10,
        agg_hermitian.mrr,
        agg_hermitian.ndcg_at_10,
        agg_hermitian.separation_margin,
        agg_hermitian.rank_inversion_rate * 100.0
    );
    println!(
        "│ Projective Overlap |<z|w>|² (ℂ^{})  │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>10.4} │ {:>10.2}% │",
        D_REAL / 2,
        agg_fidelity.recall_at_1,
        agg_fidelity.recall_at_10,
        agg_fidelity.mrr,
        agg_fidelity.ndcg_at_10,
        agg_fidelity.separation_margin,
        agg_fidelity.rank_inversion_rate * 100.0
    );
    println!(
        "│ Hybrid (0.8·Overlap + 0.2·Herm)    │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>10.4} │ {:>10.2}% │",
        agg_hybrid_80.recall_at_1,
        agg_hybrid_80.recall_at_10,
        agg_hybrid_80.mrr,
        agg_hybrid_80.ndcg_at_10,
        agg_hybrid_80.separation_margin,
        agg_hybrid_80.rank_inversion_rate * 100.0
    );
    println!(
        "│ Hybrid (0.5·Overlap + 0.5·Herm)    │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>8.4} │ {:>10.4} │ {:>10.2}% │",
        agg_hybrid_50.recall_at_1,
        agg_hybrid_50.recall_at_10,
        agg_hybrid_50.mrr,
        agg_hybrid_50.ndcg_at_10,
        agg_hybrid_50.separation_margin,
        agg_hybrid_50.rank_inversion_rate * 100.0
    );
    println!(
        "└────────────────────────────────────┴──────────┴──────────┴──────────┴──────────┴────────────┴─────────────┘\n"
    );

    println!("Adversarial Global Phase Invariance Test:");
    let dataset = generate_dataset(SEEDS[0]);
    let (fid_drift, herm_drift, real_drift) = adversarial_phase_test(&dataset);
    println!(
        "  - Projective Overlap Drift under Global Phase Rotations: {:.6} (Mathematically Invariant)",
        fid_drift
    );
    println!(
        "  - Hermitian Drift under Global Phase Rotations:        {:.6} (Phase Sensitive)",
        herm_drift
    );
    println!(
        "  - Unfolded Real Cosine Drift under Complex Rotation:   {:.6}",
        real_drift
    );

    println!(
        "\n════════════════════════════════════════════════════════════════════════════════════════"
    );
    println!(" METRIC EVALUATION BENCHMARK COMPLETE");
    println!(
        "════════════════════════════════════════════════════════════════════════════════════════"
    );
}
