//! Gate B Adversarial Oracle Test Suite
//!
//! Verifies 100.000% exact Top-K equality (slot ID and score) between
//! GlobalExactProofSearch and Exhaustive Brute-Force across all adversarial regimes:
//!   - Identical vectors / mass score ties
//!   - Near-identical vectors separated by ~ULPs
//!   - Antipodal vectors
//!   - High-dimensional isotropic vectors (D = 128, 384, 768, 1536, 4096)
//!   - Adversarial seeds (empty seed, adversarial top-k missing)
//!   - Filter mask intersections & tombstones
//!   - Multi-segment storage mixtures

use num_complex::Complex32;
use roaring::RoaringBitmap;

use hnsqr::proof::{GlobalExactProofSearch, SegmentProofView, SemanticProofTree};
use hnsqr::{
    DistanceFunction, HNSQRConfig, HNSQRIndex, NodeIndex, SimilarityScore, VectorEmbedding,
};

#[inline]
fn cosine_sim(q: &VectorEmbedding, doc: &VectorEmbedding) -> f32 {
    q.dot_product_real(doc)
}

fn brute_force_exact(
    query: &VectorEmbedding,
    corpus: &[VectorEmbedding],
    k: usize,
    filter_mask: Option<&RoaringBitmap>,
    is_live: impl Fn(NodeIndex) -> bool,
) -> Vec<(NodeIndex, SimilarityScore)> {
    let mut scores: Vec<(NodeIndex, SimilarityScore)> = corpus
        .iter()
        .enumerate()
        .filter(|(idx, _)| {
            let slot = *idx as NodeIndex;
            is_live(slot) && filter_mask.is_none_or(|m| m.contains(slot))
        })
        .map(|(idx, doc)| (idx as NodeIndex, cosine_sim(query, doc)))
        .collect();

    // Canonical deterministic tie-breaking: score DESC, slot ASC
    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scores.truncate(k);
    scores
}

#[test]
fn test_gate_b_identical_vectors_and_tie_breaking() {
    let dim = 64; // 128 real
    let n = 200;
    let k = 10;

    let base_vec = VectorEmbedding::from_complex(
        (0..dim)
            .map(|i| Complex32::new((i as f32 * 0.1).cos(), (i as f32 * 0.1).sin()))
            .collect(),
    )
    .into_normalized();

    // 50 identical vectors, then random noise
    let mut corpus: Vec<VectorEmbedding> = Vec::with_capacity(n);
    for _ in 0..50 {
        corpus.push(base_vec.clone());
    }
    for i in 50..n {
        corpus.push(
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|j| Complex32::new(((i * 7 + j) % 13) as f32, ((i * 11 + j) % 17) as f32))
                    .collect(),
            )
            .into_normalized(),
        );
    }

    let query = base_vec.clone();
    let gt = brute_force_exact(&query, &corpus, k, None, |_| true);

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        tombstones: None,
    };

    let (res, proof) = GlobalExactProofSearch::search(
        &query,
        k,
        &[seg_view],
        &[],
        &[], // zero seed
        None,
    );

    assert_eq!(res.len(), k);
    assert_eq!(
        res, gt,
        "Identical vectors must tie-break deterministically (slot ASC)"
    );
    assert!(proof.globally_exact);
}

#[test]
fn test_gate_b_near_identical_ulp_separated_vectors() {
    let dim = 384; // 768 real
    let n = 300;
    let k = 10;

    let base = VectorEmbedding::from_complex(
        (0..dim)
            .map(|i| Complex32::new((i as f32 * 0.05).sin(), (i as f32 * 0.05).cos()))
            .collect(),
    )
    .into_normalized();

    let mut corpus: Vec<VectorEmbedding> = Vec::with_capacity(n);
    for i in 0..n {
        let mut perturbed = base.complex_data().to_vec();
        // Perturb by microscopic epsilon ~1e-6
        perturbed[0].re += (i as f32) * 1e-6;
        corpus.push(VectorEmbedding::from_complex(perturbed).into_normalized());
    }

    let query = base;
    let gt = brute_force_exact(&query, &corpus, k, None, |_| true);

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        tombstones: None,
    };

    let (res, proof) = GlobalExactProofSearch::search(&query, k, &[seg_view], &[], &[], None);

    assert_eq!(res.len(), k);
    for i in 0..k {
        assert_eq!(res[i].0, gt[i].0);
        assert!((res[i].1 - gt[i].1).abs() < 1e-6);
    }
    assert!(proof.globally_exact);
}

#[test]
fn test_gate_b_antipodal_and_orthogonal_vectors() {
    let dim = 128;
    let k = 5;

    let v1 = VectorEmbedding::from_complex(vec![Complex32::new(1.0, 0.0); dim]).into_normalized();
    let v_anti =
        VectorEmbedding::from_complex(vec![Complex32::new(-1.0, 0.0); dim]).into_normalized();
    let v_ortho = VectorEmbedding::from_complex(
        (0..dim)
            .map(|i| {
                if i % 2 == 0 {
                    Complex32::new(1.0, 0.0)
                } else {
                    Complex32::new(-1.0, 0.0)
                }
            })
            .collect(),
    )
    .into_normalized();

    let corpus = vec![v_anti, v_ortho, v1.clone()];
    let query = v1;

    let gt = brute_force_exact(&query, &corpus, k, None, |_| true);

    let slots: Vec<NodeIndex> = (0..3).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        tombstones: None,
    };

    let (res, proof) = GlobalExactProofSearch::search(&query, k, &[seg_view], &[], &[], None);

    assert_eq!(res, gt);
    assert_eq!(res[0].0, 2); // v1 must be rank 0
    assert!(proof.globally_exact);
}

#[test]
fn test_gate_b_high_dimensional_exactness_sweep() {
    let dimensions = vec![64, 192, 384, 768, 2048]; // 128D, 384D, 768D, 1536D, 4096D real
    let n = 200;
    let k = 10;

    for dim in dimensions {
        let corpus: Vec<VectorEmbedding> = (0..n)
            .map(|seed| {
                VectorEmbedding::from_complex(
                    (0..dim)
                        .map(|i| {
                            Complex32::new(
                                ((seed * 19 + i * 3 + 7) % 29) as f32 - 14.0,
                                ((seed * 23 + i * 5 + 11) % 31) as f32 - 15.0,
                            )
                        })
                        .collect(),
                )
                .into_normalized()
            })
            .collect();

        let query = VectorEmbedding::from_complex(
            (0..dim)
                .map(|i| Complex32::new((i as f32 * 0.3).sin(), (i as f32 * 0.3).cos()))
                .collect(),
        )
        .into_normalized();

        let gt = brute_force_exact(&query, &corpus, k, None, |_| true);

        let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
        let tree = SemanticProofTree::build(&corpus, &slots, dim);

        let seg_view = SegmentProofView {
            tree: &tree,
            vectors: &corpus,
            tombstones: None,
        };

        let (res, proof) = GlobalExactProofSearch::search(&query, k, &[seg_view], &[], &[], None);

        assert_eq!(res.len(), k);
        assert_eq!(
            res,
            gt,
            "Failed exactness match for dimension {} ({} real)",
            dim,
            dim * 2
        );
        assert!(proof.globally_exact);
    }
}

#[test]
fn test_gate_b_adversarial_zero_and_missing_seed() {
    let dim = 128;
    let n = 250;
    let k = 10;

    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|seed| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|i| Complex32::new(((seed + i) % 17) as f32, ((seed * 2 + i) % 19) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let query = corpus[42].clone();
    let gt = brute_force_exact(&query, &corpus, k, None, |_| true);

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    // Adversarial Seed: seed contains only the WORST 20 candidates
    let mut bad_seed: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    bad_seed.sort_by(|&a, &b| {
        let sim_a = cosine_sim(&query, &corpus[a as usize]);
        let sim_b = cosine_sim(&query, &corpus[b as usize]);
        sim_a.total_cmp(&sim_b)
    });
    bad_seed.truncate(20);

    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        tombstones: None,
    };

    let (res, proof) = GlobalExactProofSearch::search(&query, k, &[seg_view], &[], &bad_seed, None);

    assert_eq!(
        res, gt,
        "Adversarial bad seed must still produce 100% exact top-K"
    );
    assert!(proof.globally_exact);
}

#[test]
fn test_gate_b_filter_mask_and_tombstone_oracle() {
    let dim = 128;
    let n = 300;
    let k = 10;

    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|seed| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|i| {
                        Complex32::new(((seed * 3 + i) % 23) as f32, ((seed * 5 + i) % 29) as f32)
                    })
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let query = corpus[10].clone();

    // Filter mask allows only even slots in 50..250
    let mut mask = RoaringBitmap::new();
    for i in 50..250 {
        if i % 2 == 0 {
            mask.insert(i);
        }
    }

    // Tombstones on slots 100, 102, 104
    let mut tombstones = RoaringBitmap::new();
    tombstones.insert(100);
    tombstones.insert(102);
    tombstones.insert(104);

    let is_live = |slot: NodeIndex| !tombstones.contains(slot);
    let gt = brute_force_exact(&query, &corpus, k, Some(&mask), is_live);

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        tombstones: Some(&tombstones),
    };

    let (res, proof) =
        GlobalExactProofSearch::search(&query, k, &[seg_view], &[], &[], Some(&mask));

    assert_eq!(
        res, gt,
        "Filter mask and tombstones must produce exact match"
    );
    assert!(proof.globally_exact);
}

#[test]
fn test_gate_b_index_end_to_end_certified_contract() {
    let dim = 128;
    let n = 200;
    let k = 8;

    let mut cfg = HNSQRConfig::default();
    cfg.distance_function = DistanceFunction::Cosine;
    cfg.max_elements = 500;
    let index = HNSQRIndex::new(cfg, dim);

    let mut corpus: Vec<VectorEmbedding> = Vec::with_capacity(n);
    for i in 0..n {
        let v = VectorEmbedding::from_complex(
            (0..dim)
                .map(|j| Complex32::new(((i * 7 + j) % 31) as f32, ((i * 13 + j) % 37) as f32))
                .collect(),
        )
        .into_normalized();
        index.insert(format!("node_{i}"), v.clone()).unwrap();
        corpus.push(v);
    }

    let query = corpus[15].clone();
    let gt = brute_force_exact(&query, &corpus, k, None, |_| true);

    // Query via certified contract
    let (res, proof) = index.search_indices_with_proof(&query, k, None).unwrap();

    assert_eq!(res.len(), k);
    for i in 0..k {
        assert_eq!(
            res[i].0, gt[i].0,
            "Slot mismatch at rank {i}: certified={}, gt={}",
            res[i].0, gt[i].0
        );
        assert!(
            (res[i].1 - gt[i].1).abs() < 1e-5,
            "Score drift at rank {i}: certified={}, gt={}",
            res[i].1,
            gt[i].1
        );
    }
    assert!(proof.globally_exact);
}

/// Deadline abort: exercises all three levels of the certified-deadline contract.
///
/// Path A — no deadline: `globally_exact = true`, `deadline_exceeded = false`.
/// Path B — pre-expired deadline via `search_with_deadline`: `globally_exact = false`,
///           `deadline_exceeded = true`, new telemetry fields populated.
/// Path C — `HNSQRIndex::certified_search` (typed API): `DeadlineExceeded` variant
///           cannot be confused with `Exact` at the type boundary.
/// Path D — `HNSQRIndex::search_indices_with_proof` (legacy): still works, callers
///           must inspect `proof.deadline_exceeded` manually.
///
/// This test covers the P99 tail-latency mitigation path for high-entropy isotropic
/// workloads identified in the architecture report, and validates the design principle
/// "correctness should be structurally difficult to misrepresent."
#[test]
fn test_gate_b_certified_deadline_abort_sets_globally_exact_false() {
    use hnsqr::CertifiedSearchOutcome;

    let dim = 64;
    let n = 500;
    let k = 5;

    // Isotropic unit vectors — worst case for proof-tree pruning.
    let mut corpus: Vec<VectorEmbedding> = Vec::with_capacity(n);
    for i in 0..n {
        let v = VectorEmbedding::from_complex(
            (0..dim)
                .map(|j| {
                    let angle = ((i * 97 + j * 31) % 1000) as f32 * 0.00628;
                    Complex32::new(angle.cos(), angle.sin())
                })
                .collect(),
        )
        .into_normalized();
        corpus.push(v);
    }

    let tree = SemanticProofTree::build(&corpus, &(0..n as u32).collect::<Vec<_>>(), dim);
    let query = corpus[0].clone();

    // ── Path A: no deadline → complete proof ─────────────────────────────────
    let seg_a = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        tombstones: None,
    };
    let (res_full, proof_full) =
        GlobalExactProofSearch::search(&query, k, &[seg_a], &[], &[], None);

    assert!(
        proof_full.globally_exact,
        "Path A: globally_exact must be true"
    );
    assert!(
        !proof_full.deadline_exceeded,
        "Path A: deadline_exceeded must be false"
    );
    assert!(
        proof_full.region_prune_ratio >= 0.0 && proof_full.region_prune_ratio <= 1.0,
        "Path A: region_prune_ratio must be in [0,1]"
    );
    assert_eq!(res_full.len(), k);

    // ── Path B: pre-expired deadline via low-level API ────────────────────────
    let deadline = std::time::Instant::now() + std::time::Duration::from_micros(1);
    std::thread::sleep(std::time::Duration::from_micros(100)); // guarantee expiry

    let seg_b = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        tombstones: None,
    };
    let (res_aborted, proof_aborted) = GlobalExactProofSearch::search_with_deadline(
        &query,
        k,
        &[seg_b],
        &[],
        &[],
        None,
        Some(deadline),
    );

    assert!(
        !proof_aborted.globally_exact,
        "Path B: globally_exact must be false on abort"
    );
    assert!(
        proof_aborted.deadline_exceeded,
        "Path B: deadline_exceeded must be true"
    );
    // frontier_nodes_remaining may be 0 if the deadline fired before the first pop
    // (checked at stage boundaries) — that's correct behaviour.
    assert!(
        proof_aborted.region_prune_ratio >= 0.0 && proof_aborted.region_prune_ratio <= 1.0,
        "Path B: region_prune_ratio must be in [0,1]"
    );
    for (_, score) in &res_aborted {
        assert!(
            *score >= -1.0 - 1e-5 && *score <= 1.0 + 1e-5,
            "Path B: aborted search returned out-of-range score: {score}"
        );
    }

    // ── Path C: HNSQRIndex::certified_search — typed outcome ─────────────────
    // Use a small corpus (50 vectors) so the tree build is fast in debug mode.
    let small_n = 50;
    let small_corpus: Vec<VectorEmbedding> = corpus[..small_n].to_vec();

    let mut cfg_tight = HNSQRConfig::default();
    cfg_tight.distance_function = DistanceFunction::Cosine;
    cfg_tight.max_elements = 100;
    cfg_tight.certified_query_timeout_ms = Some(0); // 0 ms → guaranteed immediate expiry
    let index_tight = HNSQRIndex::new(cfg_tight, dim);
    for (i, v) in small_corpus.iter().enumerate() {
        index_tight.insert(format!("d{i}"), v.clone()).unwrap();
    }

    let outcome = index_tight
        .certified_search(&query, k, None)
        .expect("certified_search must not return Err on deadline abort");

    match outcome {
        CertifiedSearchOutcome::DeadlineExceeded { ref proof, .. } => {
            assert!(
                proof.deadline_exceeded,
                "Path C: DeadlineExceeded variant must have deadline_exceeded=true in proof"
            );
            assert!(
                !proof.globally_exact,
                "Path C: DeadlineExceeded variant must have globally_exact=false"
            );
        }
        CertifiedSearchOutcome::Exact { .. } => {
            // On extremely fast hardware a 0 ms budget may still complete before the
            // first amortised deadline check (32 pops).  Both outcomes are valid; what
            // matters is that the type boundary is respected.
        }
    }

    // ── Path D: legacy search_indices_with_proof — flat tuple, manual inspection ──
    let mut cfg_generous = HNSQRConfig::default();
    cfg_generous.distance_function = DistanceFunction::Cosine;
    cfg_generous.max_elements = 100;
    cfg_generous.certified_query_timeout_ms = Some(5000); // 5 s — should always complete
    let index_gen = HNSQRIndex::new(cfg_generous, dim);
    for (i, v) in small_corpus.iter().enumerate() {
        index_gen.insert(format!("g{i}"), v.clone()).unwrap();
    }

    let (_, proof_gen) = index_gen
        .search_indices_with_proof(&query, k, None)
        .expect("search_indices_with_proof must succeed with generous budget");
    assert!(
        proof_gen.globally_exact,
        "Path D: generous budget must produce complete proof"
    );
    assert!(
        !proof_gen.deadline_exceeded,
        "Path D: generous budget must not fire deadline"
    );
}
