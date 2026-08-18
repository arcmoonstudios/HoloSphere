//! Proof Correctness & Formal Invariant Test Suite (B0/B1)
//!
//! Validates the 9 foundational invariants of the corpus-covering semantic proof hierarchy:
//!   1. Coverage: Every segment slot occurs exactly once in the flattened leaf permutation.
//!   2. Partition: For every internal node, child memberships form an exact disjoint partition of parent members.
//!   3. Envelope Containment: All constituent member vectors strictly reside within recorded block and global radii.
//!   4. Query-Bound Oracle: For all queries q, exact_score <= tree.upper_bound(q, node).
//!   5. Global Exactness: (slot, score, ordering) matches exhaustive brute-force.
//!   6. Rivero Independence: Exactness holds with empty seeds.
//!   7. Adversarial Seeds: Exactness holds when seeds contain the worst vectors.
//!   8. Tombstone Invariance: Deleting Top-1 preserves exactness on remaining corpus without tree rebuilds.
//!   9. Multi-Segment Coordination: Mutable + Immutable A + Immutable B + Tombstones exactness.

use std::collections::HashSet;
use num_complex::Complex32;
use roaring::RoaringBitmap;

use hnsqr::proof::{
    GlobalExactProofSearch, PROOF_BLOCK_COMPLEX_DIM, ProofQuery, SegmentProofView,
    SemanticProofTree,
};
use hnsqr::storage::segment::SegmentedEngine;
use hnsqr::{NodeIndex, SimilarityScore, VectorEmbedding};

#[inline]
fn cosine_sim(q: &VectorEmbedding, doc: &VectorEmbedding) -> f32 {
    (q.dot_product_complex(doc)).re
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

    scores.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scores.truncate(k);
    scores
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. TREE COVERAGE: EVERY SLOT OCCURS EXACTLY ONCE
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_proof_tree_covers_every_slot_exactly_once() {
    let dim = 32;
    let n = 250;
    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 7 + d) as f32, (i * 11 + d) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    assert_eq!(tree.leaf_slots.len(), n);

    let mut seen = HashSet::with_capacity(n);
    for &slot in tree.leaf_slots.iter() {
        assert!(
            seen.insert(slot),
            "Duplicate slot {slot} found in proof tree leaf permutation!"
        );
    }

    for slot in 0..n as NodeIndex {
        assert!(
            seen.contains(&slot),
            "Slot {slot} missing from proof tree leaf permutation!"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. CHILD PARTITION: CHILDREN FORM EXACT DISJOINT PARTITION OF PARENT
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_children_form_exact_parent_partition() {
    let dim = 32;
    let n = 300;
    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 13 + d) as f32, (i * 17 + d) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    for (node_idx, node) in tree.nodes.iter().enumerate() {
        if node.is_internal() {
            let parent_members = tree.members(node);
            let mut child_members_collected = Vec::new();

            for child_idx in tree.children(node) {
                let child = tree.node(child_idx);
                let members = tree.members(child);
                child_members_collected.extend_from_slice(members);
            }

            assert_eq!(
                parent_members.len(),
                child_members_collected.len(),
                "Node {node_idx} child membership count mismatch"
            );
            assert_eq!(
                parent_members,
                &child_members_collected[..],
                "Node {node_idx} child members must match parent member sequence"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. ENVELOPE CONTAINMENT: VECTORS STRICTLY RESIDE WITHIN RESIDUAL RADII
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_envelope_containment_bounds_hold() {
    let dim = 64;
    let n = 200;
    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 3 + d * 5) as f32, (i * 7 + d * 11) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    let num_blocks = dim.div_ceil(PROOF_BLOCK_COMPLEX_DIM);

    for node in tree.nodes.iter() {
        let members = tree.members(node);

        for &slot in members {
            let v = &corpus[slot as usize];
            let v_data = v.complex_data();

            for b in 0..num_blocks {
                let code_idx = node.centroid_offset as usize + b;
                let code = &tree.centroid_codes[code_idx];
                let rho_b = tree.block_radii[code_idx];

                let start = b * PROOF_BLOCK_COMPLEX_DIM;
                let end = (start + PROOF_BLOCK_COMPLEX_DIM).min(dim);

                let mut diff_sum_sq = 0.0f32;
                for i in start..end {
                    let diff = v_data[i] - code.coords[i - start];
                    diff_sum_sq += diff.norm_sqr();
                }
                let actual_dist = diff_sum_sq.sqrt();

                assert!(
                    actual_dist <= rho_b + 1e-5,
                    "Block {b} containment violation: actual={actual_dist}, recorded rho={rho_b}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. QUERY BOUND ORACLE: EXACT SCORE <= TREE UPPER BOUND ACROSS ALL QUERIES
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_query_bound_oracle_soundness() {
    let dim = 64;
    let n = 150;
    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 19 + d) as f32, (i * 23 + d) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    // Test across 50 diverse queries
    for q_idx in 0..50 {
        let query = VectorEmbedding::from_complex(
            (0..dim)
                .map(|d| Complex32::new((q_idx * 31 + d * 3) as f32, (q_idx * 37 + d * 7) as f32))
                .collect(),
        )
        .into_normalized();

        let pq = ProofQuery::new(query.complex_data());

        for (node_idx, node) in tree.nodes.iter().enumerate() {
            let ub = tree.upper_bound(&pq, node_idx as u32);

            for &slot in tree.members(node) {
                let exact_score = (query.dot_product_complex(&corpus[slot as usize])).re as f64;
                assert!(
                    exact_score <= ub + 1e-6,
                    "Upper bound soundess violation: exact={exact_score} > ub={ub} for slot {slot}"
                );
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. GLOBAL EXACTNESS: 100.000% MATCH WITH EXHAUSTIVE BASELINE
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_proof_tree_exact_matches_exhaustive_exact() {
    let dim = 128;
    let n = 300;
    let k = 10;

    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 41 + d) as f32, (i * 43 + d) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let query = corpus[25].clone();
    let gt = brute_force_exact(&query, &corpus, k, None, |_| true);

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        lutz_codes: None,
        tombstones: None,
    };

    let (res, proof) = GlobalExactProofSearch::search(
        &query,
        k,
        &[seg_view],
        &[],
        &[],
        None,
    );

    assert_eq!(res.len(), k);
    assert_eq!(res, gt, "Exact match failure against brute force ground truth");
    assert!(proof.globally_exact);
    assert!(proof.is_accounting_exact(), "Terminal funnel accounting must be 100% exact");
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. RIVERO ZERO SEED TEST: EXACTNESS WITH NO SEEDS
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_rivero_zero_seed_test() {
    let dim = 64;
    let n = 200;
    let k = 8;

    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 47 + d) as f32, (i * 53 + d) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let query = corpus[77].clone();
    let gt = brute_force_exact(&query, &corpus, k, None, |_| true);

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        lutz_codes: None,
        tombstones: None,
    };

    let (res, proof) = GlobalExactProofSearch::search(
        &query,
        k,
        &[seg_view],
        &[],
        &[], // Explicitly empty seeds
        None,
    );

    assert_eq!(res, gt);
    assert!(proof.globally_exact);
    assert!(proof.is_accounting_exact());
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. ADVERSARIAL WORST-SEEDS TEST
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_deliberately_bad_seed_test() {
    let dim = 64;
    let n = 200;
    let k = 8;

    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 59 + d) as f32, (i * 61 + d) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let query = corpus[12].clone();
    let gt = brute_force_exact(&query, &corpus, k, None, |_| true);

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    // Adversarial seeds = the 25 worst vectors
    let mut bad_seeds: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    bad_seeds.sort_by(|&a, &b| {
        let sim_a = cosine_sim(&query, &corpus[a as usize]);
        let sim_b = cosine_sim(&query, &corpus[b as usize]);
        sim_a.total_cmp(&sim_b)
    });
    bad_seeds.truncate(25);

    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        lutz_codes: None,
        tombstones: None,
    };

    let (res, proof) = GlobalExactProofSearch::search(
        &query,
        k,
        &[seg_view],
        &[],
        &bad_seeds,
        None,
    );

    assert_eq!(res, gt, "Adversarial worst seeds must not compromise exactness");
    assert!(proof.globally_exact);
    assert!(proof.is_accounting_exact());
}

// ─────────────────────────────────────────────────────────────────────────────
// 8. TOMBSTONES: DELETE TOP-1 & PRESERVE EXACTNESS WITHOUT TREE REBUILD
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_tombstone_oracle_exactness() {
    let dim = 64;
    let n = 200;
    let k = 5;

    let corpus: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 67 + d) as f32, (i * 71 + d) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let query = corpus[33].clone();

    // Find the original GT Top-1
    let original_gt = brute_force_exact(&query, &corpus, k, None, |_| true);
    let top_1_slot = original_gt[0].0;

    // Tombstone the exact Top-1 and Top-3
    let mut tombstones = RoaringBitmap::new();
    tombstones.insert(top_1_slot);
    tombstones.insert(original_gt[2].0);

    let is_live = |slot: NodeIndex| !tombstones.contains(slot);
    let gt_after_delete = brute_force_exact(&query, &corpus, k, None, is_live);

    // Proof tree built BEFORE deletions — unchanged!
    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);

    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        lutz_codes: None,
        tombstones: Some(&tombstones),
    };

    let (res, proof) = GlobalExactProofSearch::search(
        &query,
        k,
        &[seg_view],
        &[],
        &[],
        None,
    );

    assert_eq!(res.len(), k);
    assert_eq!(res, gt_after_delete, "Tombstone search must match live-only ground truth");
    assert!(proof.globally_exact);
    assert!(proof.is_accounting_exact());
}

// ─────────────────────────────────────────────────────────────────────────────
// 9. MULTI-SEGMENT: MUTABLE + IMMUTABLE A + IMMUTABLE B + TOMBSTONES
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_multi_segment_and_mutable_coordination() {
    let dim = 16;
    let engine = SegmentedEngine::new(dim, 25);

    // Insert 100 vectors across multiple segments
    for i in 0..100 {
        let v = VectorEmbedding::from_complex(
            (0..dim)
                .map(|d| Complex32::new((i * 7 + d) as f32, (i * 11 + d) as f32))
                .collect(),
        )
        .into_normalized();
        engine.insert(format!("multi_doc_{i}"), v).unwrap();
    }

    // Delete doc 10, 20, 30
    engine.delete("multi_doc_10");
    engine.delete("multi_doc_20");
    engine.delete("multi_doc_30");

    let query = VectorEmbedding::from_complex(
        (0..dim)
            .map(|d| Complex32::new((25 * 7 + d) as f32, (25 * 11 + d) as f32))
            .collect(),
    )
    .into_normalized();

    let topk = engine.search_with_contract(&query, 5, hnsqr::planning::planner::RetrievalContract::Certified);

    assert_eq!(topk.len(), 5);
    assert_eq!(topk[0].0.as_ref(), "multi_doc_25");

    for (id, _) in &topk {
        assert_ne!(id.as_ref(), "multi_doc_10");
        assert_ne!(id.as_ref(), "multi_doc_20");
        assert_ne!(id.as_ref(), "multi_doc_30");
    }
}
