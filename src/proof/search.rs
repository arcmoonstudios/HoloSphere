/* hnsqr/src/proof/search.rs */
//!▫~•◦-------------------------------‣
//! # Global Proof Search Engine with LUTz L0/L1 Leaf Cascade (Gate B3)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Executes multi-segment branch-and-bound exact Top-K search with:
//!   1. Spherical Proof Tree region elimination (83–93% corpus elimination)
//!   2. Unresolved leaf LUTz L0 Cauchy-Schwarz bound pruning
//!   3. Unresolved leaf LUTz L1 progressive bound pruning
//!   4. Leaf-local winner-first scheduling to maximize early threshold elevation
//!   5. Exact SIMD resolution on genuine residue
//!   6. Strictly normalized terminal accounting: $N_{\text{eligible}} \equiv N_{\text{pruned}} + N_{\text{exact}} + N_{\text{filtered}}$
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use super::bounds::ProofQuery;
use super::lutz::{LutzCode, LutzQueryTable};
use super::tree::SemanticProofTree;
use crate::{NodeIndex, SimilarityScore, VectorEmbedding};

/// Detailed proof certificate and exactness audit telemetry.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct DenseExactProof {
    /// Total live vectors considered across all segments.
    pub corpus_size: usize,

    /// Number of raw candidates produced by coarse Rivero probe.
    pub rivero_raw_candidates: usize,
    /// Number of candidate seeds prescored for initial threshold estimation.
    pub rivero_seed_candidates: usize,
    /// Number of seeds evaluated with exact SIMD to establish initial $\tau$.
    pub rivero_seed_exact_evaluations: usize,

    /// Number of hierarchy envelope nodes popped from the max-UB frontier.
    pub proof_regions_popped: usize,
    /// Number of internal nodes whose children were expanded.
    pub proof_regions_expanded: usize,
    /// Number of region subtrees pruned because $\text{UB}_T(q) < \tau$.
    pub proof_regions_pruned: usize,
    /// Total vectors skipped entirely due to subtree region pruning.
    pub vectors_pruned_by_region: usize,

    /// Number of leaf candidates evaluated against LUTz L0 Cauchy-Schwarz bound.
    pub lutz_l0_evaluations: usize,
    /// Number of leaf candidates eliminated by LUTz L0 bound.
    pub lutz_l0_pruned: usize,

    /// Number of leaf candidates evaluated against LUTz L1 Cauchy-Schwarz bound.
    pub lutz_l1_evaluations: usize,
    /// Number of leaf candidates eliminated by LUTz L1 bound.
    pub lutz_l1_pruned: usize,

    /// Total candidate leaf vectors considered.
    pub leaf_vectors_considered: usize,
    /// Total vectors evaluated with full exact SIMD dot product.
    pub exact_evaluations: usize,
    /// Total vectors skipped due to tombstone or filter mask exclusion.
    pub filtered_or_tombstoned: usize,

    /// Bytes of LUTz L0 quantized codes read during leaf bounding.
    pub l0_bytes_touched: usize,
    /// Bytes of LUTz L1 residual codes read during leaf bounding.
    pub l1_bytes_touched: usize,
    /// Bytes of full float embeddings loaded into SIMD execution.
    pub exact_bytes_touched: usize,

    /// Final K-th similarity score threshold $\tau$.
    pub kth_score: f32,
    /// Maximum upper bound among all pruned or unvisited branches.
    pub max_remaining_upper_bound: f64,

    /// Formally proven $100.000\%$ globally exact flag.
    /// `false` either means the search was aborted by a deadline or some
    /// bookkeeping invariant was not satisfied.  Always check this alongside
    /// `deadline_exceeded` to distinguish the two cases.
    pub globally_exact: bool,

    // ── Deadline telemetry ───────────────────────────────────────────────
    /// Set to `true` when the query deadline fired before the proof frontier
    /// was fully exhausted.  Distinct from `globally_exact` so callers can
    /// tell the difference between "proof incomplete due to budget" and any
    /// other reason `globally_exact` might be false.
    pub deadline_exceeded: bool,

    /// Wall-clock microseconds consumed from the start of `search_with_deadline`
    /// until the search returned.  Populated for both complete and aborted runs.
    pub elapsed_us: u64,

    /// Estimated number of proof-frontier entries that remained in the heap
    /// at the point of deadline abort.  Zero when the search completed normally.
    /// Useful for diagnosing *why* the deadline fired:
    ///   - Large value  → pathological proof geometry (isotropic distribution)
    ///   - Small value  → CPU overload or very tight budget
    pub frontier_nodes_remaining: usize,

    /// Fraction of total proof regions that were pruned without SIMD evaluation.
    /// `pruned_regions / (pruned_regions + expanded_regions)`.
    /// Low ratio near deadline expiry identifies isotropic / high-entropy corpora
    /// where bounding envelopes collapse and force exhaustive traversal.
    pub region_prune_ratio: f64,
}

impl DenseExactProof {
    /// Validates the terminal funnel accounting invariant:
    /// $$N_{\text{eligible}} \equiv N_{\text{region-pruned}} + N_{\text{L0-pruned}} + N_{\text{L1-pruned}} + N_{\text{exact}} + N_{\text{filtered}}$$
    #[inline]
    pub fn is_accounting_exact(&self) -> bool {
        let total = self.vectors_pruned_by_region
            + self.lutz_l0_pruned
            + self.lutz_l1_pruned
            + self.exact_evaluations
            + self.filtered_or_tombstoned;
        total == self.corpus_size
    }
}

/// An entry in the best-bound max-heap exploration frontier.
#[derive(Clone, Copy, Debug)]
pub struct ProofFrontierEntry {
    pub upper_bound: f64,
    pub segment_idx: usize,
    pub node_idx: u32,
}

impl PartialEq for ProofFrontierEntry {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.upper_bound == other.upper_bound
            && self.segment_idx == other.segment_idx
            && self.node_idx == other.node_idx
    }
}

impl Eq for ProofFrontierEntry {}

impl PartialOrd for ProofFrontierEntry {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ProofFrontierEntry {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap: highest upper bound comes first
        self.upper_bound
            .total_cmp(&other.upper_bound)
            .then_with(|| other.segment_idx.cmp(&self.segment_idx))
            .then_with(|| other.node_idx.cmp(&self.node_idx))
    }
}

/// A candidate finalist inside the Top-K min-heap.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Finalist {
    pub slot: NodeIndex,
    pub score: SimilarityScore,
}

impl Eq for Finalist {}

impl PartialOrd for Finalist {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Finalist {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Inverted min-heap: lowest score (or largest slot on tie) at peek() to be evicted first
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.slot.cmp(&other.slot))
    }
}

/// Top-K accumulator enforcing deterministic canonical tie-breaking `(score DESC, slot ASC)`.
#[derive(Clone, Debug)]
pub struct TopKAccumulator {
    pub k: usize,
    pub heap: BinaryHeap<Finalist>,
}

impl TopKAccumulator {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k),
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.heap.len() >= self.k
    }

    #[inline(always)]
    pub fn kth_score(&self) -> f32 {
        if self.heap.len() >= self.k {
            self.heap.peek().unwrap().score
        } else {
            f32::NEG_INFINITY
        }
    }

    /// Offers a candidate. Returns `true` if accepted into Top-K.
    #[inline(always)]
    pub fn offer(&mut self, slot: NodeIndex, score: SimilarityScore) -> bool {
        if self.k == 0 {
            return false;
        }

        for item in self.heap.iter() {
            if item.slot == slot {
                return false;
            }
        }

        if self.heap.len() < self.k {
            self.heap.push(Finalist { slot, score });
            true
        } else {
            let worst = self.heap.peek().unwrap();
            if score > worst.score || (score == worst.score && slot < worst.slot) {
                self.heap.pop();
                self.heap.push(Finalist { slot, score });
                true
            } else {
                false
            }
        }
    }

    /// Consumes the accumulator and returns sorted finalists `(score DESC, slot ASC)`.
    pub fn into_sorted_vec(self) -> Vec<(NodeIndex, SimilarityScore)> {
        let mut list: Vec<Finalist> = self.heap.into_vec();
        list.sort_unstable_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.slot.cmp(&b.slot))
        });
        list.into_iter().map(|f| (f.slot, f.score)).collect()
    }
}

/// Immutable segment descriptor for multi-segment proof search.
pub struct SegmentProofView<'a> {
    pub tree: &'a SemanticProofTree,
    pub vectors: &'a [VectorEmbedding],
    pub lutz_codes: Option<&'a [LutzCode]>,
    pub tombstones: Option<&'a RoaringBitmap>,
}

/// Multi-segment exact proof search engine.
pub struct GlobalExactProofSearch;

impl GlobalExactProofSearch {
    /// Executes a global branch-and-bound exact Top-K search across multiple segment proof trees.
    ///
    /// `deadline` — optional `Instant` after which the frontier loop is aborted early.
    /// When the deadline is hit, results are returned with `proof.globally_exact = false`.
    /// Without a deadline the engine always produces a complete, verified proof.
    pub fn search(
        query_vector: &VectorEmbedding,
        k: usize,
        segments: &[SegmentProofView],
        mutable_candidates: &[(NodeIndex, SimilarityScore)],
        rivero_seed_candidates: &[NodeIndex],
        filter_mask: Option<&RoaringBitmap>,
    ) -> (Vec<(NodeIndex, SimilarityScore)>, DenseExactProof) {
        Self::search_with_deadline(
            query_vector,
            k,
            segments,
            mutable_candidates,
            rivero_seed_candidates,
            filter_mask,
            None,
        )
    }

    /// Same as [`search`] but with an explicit monotonic-clock deadline.
    ///
    /// The deadline is checked at every stage boundary (mutable scan, Rivero
    /// seeding, frontier init) and then every 32 frontier pops during the
    /// branch-and-bound loop.  Checking every 32 pops keeps clock-query
    /// overhead well below 1% even on Windows where `Instant::now()` is
    /// measured in ~100 ns; the maximum overshoot is bounded to ~32 region
    /// evaluations, not 32 individual vectors.
    ///
    /// Returns `proof.deadline_exceeded = true` and `proof.globally_exact = false`
    /// when aborted.  Best-effort partial results are always returned rather than
    /// an empty set so that `hnsqr_doctor` / metrics can observe what was found.
    pub fn search_with_deadline(
        query_vector: &VectorEmbedding,
        k: usize,
        segments: &[SegmentProofView],
        mutable_candidates: &[(NodeIndex, SimilarityScore)],
        rivero_seed_candidates: &[NodeIndex],
        filter_mask: Option<&RoaringBitmap>,
        deadline: Option<std::time::Instant>,
    ) -> (Vec<(NodeIndex, SimilarityScore)>, DenseExactProof) {
        let t_start = std::time::Instant::now();

        let mut proof = DenseExactProof::default();
        let total_corpus: usize = segments
            .iter()
            .map(|s| s.tree.total_vectors())
            .sum::<usize>()
            + mutable_candidates.len();
        proof.corpus_size = total_corpus;

        // Helper: fill elapsed + prune_ratio and return partial results.
        // `frontier_remaining` is the number of entries still in the heap.
        macro_rules! abort_deadline {
            ($topk:expr, $frontier_remaining:expr) => {{
                proof.deadline_exceeded = true;
                proof.globally_exact = false;
                proof.elapsed_us = t_start.elapsed().as_micros() as u64;
                proof.frontier_nodes_remaining = $frontier_remaining;
                let total_regions = proof.proof_regions_pruned + proof.proof_regions_expanded;
                proof.region_prune_ratio = if total_regions > 0 {
                    proof.proof_regions_pruned as f64 / total_regions as f64
                } else {
                    0.0
                };
                proof.kth_score = $topk.kth_score();
                return ($topk.into_sorted_vec(), proof);
            }};
        }

        // Inline deadline check used at stage boundaries where a single
        // `Instant::now()` is acceptable (called at most a handful of times).
        macro_rules! check_deadline_stage {
            ($topk:expr) => {
                if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                    abort_deadline!($topk, 0);
                }
            };
        }

        if k == 0 || total_corpus == 0 {
            proof.elapsed_us = t_start.elapsed().as_micros() as u64;
            return (Vec::new(), proof);
        }

        let query = ProofQuery::new(query_vector.complex_data());
        let query_lut = LutzQueryTable::build(query_vector);
        let mut topk = TopKAccumulator::new(k);
        let mut evaluated_slots = RoaringBitmap::new();

        // ── Stage 1: Mutable exact candidates ────────────────────────────
        for &(slot, score) in mutable_candidates {
            if filter_mask.is_some_and(|m| !m.contains(slot)) {
                proof.filtered_or_tombstoned += 1;
                evaluated_slots.insert(slot);
                continue;
            }
            evaluated_slots.insert(slot);
            topk.offer(slot, score);
            proof.exact_evaluations += 1;
            proof.exact_bytes_touched += query_vector.dimension() * 8;
        }
        check_deadline_stage!(topk);

        // ── Stage 2: Rivero coarse seeds (initial τ elevation) ────────────
        proof.rivero_raw_candidates = rivero_seed_candidates.len();
        let mut seen_seeds = RoaringBitmap::new();

        for &slot in rivero_seed_candidates.iter().take(128.max(k * 2)) {
            if filter_mask.is_some_and(|m| !m.contains(slot)) || seen_seeds.contains(slot) {
                continue;
            }
            seen_seeds.insert(slot);
            proof.rivero_seed_candidates += 1;

            for seg in segments {
                if (slot as usize) < seg.vectors.len() {
                    if seg.tombstones.is_some_and(|t| t.contains(slot)) {
                        proof.filtered_or_tombstoned += 1;
                        evaluated_slots.insert(slot);
                        break;
                    }
                    evaluated_slots.insert(slot);
                    let score = (query_vector.dot_product_complex(&seg.vectors[slot as usize])).re;
                    proof.rivero_seed_exact_evaluations += 1;
                    proof.exact_evaluations += 1;
                    proof.exact_bytes_touched += seg.tree.dimension * 8;
                    topk.offer(slot, score);
                    break;
                }
            }
        }
        // ── Stage 2.5: Fast Flat LUTz Sieve for Isotropic / Diffuse Segments ───
        for seg in segments {
            if !seg.tree.is_spatially_prunable() {
                if let Some(lutz_codes) = seg.lutz_codes {
                    let n_vecs = seg.vectors.len();
                    proof.leaf_vectors_considered += n_vecs;
                    proof.l0_bytes_touched += n_vecs * std::mem::size_of::<LutzCode>();

                    for slot in 0..n_vecs as NodeIndex {
                        if evaluated_slots.contains(slot) {
                            continue;
                        }
                        if seg.tombstones.is_some_and(|t| t.contains(slot))
                            || filter_mask.is_some_and(|m| !m.contains(slot))
                        {
                            evaluated_slots.insert(slot);
                            proof.filtered_or_tombstoned += 1;
                            continue;
                        }

                        let code = &lutz_codes[slot as usize];
                        proof.lutz_l0_evaluations += 1;
                        let approx0 = query_lut.score_candidate_l0(code);
                        let res0 = query_lut.blockwise_residual_l0(code);
                        let ub0 = approx0 + res0 + 1e-7;

                        let curr_tau = topk.kth_score();
                        if topk.is_full() && ub0 < curr_tau {
                            proof.lutz_l0_pruned += 1;
                            evaluated_slots.insert(slot);
                            continue;
                        }

                        // L1 Progressive Refinement
                        if code.codes_l1.is_some() {
                            let res1 = query_lut.blockwise_residual_l1(code) as f64;
                            let ub1 = (approx0 as f64) + res1 + 1e-7;
                            proof.lutz_l1_evaluations += 1;
                            proof.l1_bytes_touched +=
                                code.codes_l1.as_deref().map_or(0, |c| c.len() * 2);

                            if topk.is_full() && ub1 < curr_tau as f64 {
                                proof.lutz_l1_pruned += 1;
                                evaluated_slots.insert(slot);
                                continue;
                            }
                        }

                        // Escalates to exact SIMD evaluation
                        let v = &seg.vectors[slot as usize];
                        let score = (query_vector.dot_product_complex(v)).re;
                        topk.offer(slot, score);
                        proof.exact_evaluations += 1;
                        proof.exact_bytes_touched += seg.tree.dimension * 8;
                        evaluated_slots.insert(slot);
                    }
                } else {
                    for slot in 0..seg.vectors.len() as NodeIndex {
                        if evaluated_slots.contains(slot) {
                            continue;
                        }
                        if seg.tombstones.is_some_and(|t| t.contains(slot))
                            || filter_mask.is_some_and(|m| !m.contains(slot))
                        {
                            evaluated_slots.insert(slot);
                            proof.filtered_or_tombstoned += 1;
                            continue;
                        }
                        let v = &seg.vectors[slot as usize];
                        let score = (query_vector.dot_product_complex(v)).re;
                        topk.offer(slot, score);
                        proof.exact_evaluations += 1;
                        proof.exact_bytes_touched += seg.tree.dimension * 8;
                        evaluated_slots.insert(slot);
                    }
                }
            }
        }
        check_deadline_stage!(topk);

        // ── Stage 3: Frontier initialisation (Spatially Prunable Segments) ───
        let mut frontier: BinaryHeap<ProofFrontierEntry> = BinaryHeap::new();

        for (seg_idx, seg) in segments.iter().enumerate() {
            if !seg.tree.is_spatially_prunable() || seg.tree.nodes.is_empty() {
                continue;
            }
            let ub = seg.tree.upper_bound(&query, seg.tree.root);
            let tau = topk.kth_score() as f64;

            if topk.is_full() && ub < tau {
                proof.proof_regions_pruned += 1;
                let unvisited = seg
                    .tree
                    .members(seg.tree.node(seg.tree.root))
                    .iter()
                    .filter(|&&s| !evaluated_slots.contains(s))
                    .count();
                proof.vectors_pruned_by_region += unvisited;
                if ub > proof.max_remaining_upper_bound {
                    proof.max_remaining_upper_bound = ub;
                }
            } else {
                frontier.push(ProofFrontierEntry {
                    upper_bound: ub,
                    segment_idx: seg_idx,
                    node_idx: seg.tree.root,
                });
            }
        }
        check_deadline_stage!(topk);

        // ── Stage 4: Best-bound branch-and-bound ─────────────────────────
        // Clock is queried every DEADLINE_CHECK_INTERVAL pops to amortise
        // `Instant::now()` cost on platforms where it is non-trivial (Windows).
        // At 32-region granularity the maximum overshoot is bounded to one
        // leaf-batch, never the whole corpus.
        const DEADLINE_CHECK_INTERVAL: usize = 32;

        while let Some(entry) = frontier.pop() {
            proof.proof_regions_popped += 1;

            // Amortised deadline check: once every 32 frontier pops.
            if proof.proof_regions_popped % DEADLINE_CHECK_INTERVAL == 0 {
                if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                    abort_deadline!(topk, frontier.len());
                }
            }

            let tau = topk.kth_score() as f64;

            let total_regions = proof.proof_regions_pruned + proof.proof_regions_expanded;
            let _region_prune_ratio = if total_regions > 0 {
                proof.proof_regions_pruned as f64 / total_regions as f64
            } else {
                0.0
            };

            // Strict admissible upper bound pruning for exact certification
            let effective_ub = entry.upper_bound;

            if topk.is_full() && effective_ub < tau {
                proof.proof_regions_pruned += 1;
                let seg = &segments[entry.segment_idx];
                let node = seg.tree.node(entry.node_idx);
                let unvisited = seg
                    .tree
                    .members(node)
                    .iter()
                    .filter(|&&s| !evaluated_slots.contains(s))
                    .count();
                proof.vectors_pruned_by_region += unvisited;
                if entry.upper_bound > proof.max_remaining_upper_bound {
                    proof.max_remaining_upper_bound = entry.upper_bound;
                }
                continue;
            }

            let seg = &segments[entry.segment_idx];
            let node = seg.tree.node(entry.node_idx);

            if node.is_internal() {
                proof.proof_regions_expanded += 1;
                for child_idx in seg.tree.children(node) {
                    let child_ub = seg.tree.upper_bound(&query, child_idx);
                    let curr_tau = topk.kth_score() as f64;

                    if topk.is_full() && child_ub < curr_tau {
                        proof.proof_regions_pruned += 1;
                        let child_node = seg.tree.node(child_idx);
                        let unvisited = seg
                            .tree
                            .members(child_node)
                            .iter()
                            .filter(|&&s| !evaluated_slots.contains(s))
                            .count();
                        proof.vectors_pruned_by_region += unvisited;
                        if child_ub > proof.max_remaining_upper_bound {
                            proof.max_remaining_upper_bound = child_ub;
                        }
                    } else {
                        frontier.push(ProofFrontierEntry {
                            upper_bound: child_ub,
                            segment_idx: entry.segment_idx,
                            node_idx: child_idx,
                        });
                    }
                }
            } else {
                // Leaf: B3 Progressive LUTz Filtering Cascade
                let mut candidate_slots = Vec::with_capacity(node.member_len as usize);
                for &slot in seg.tree.members(node) {
                    if evaluated_slots.contains(slot) {
                        continue;
                    }
                    if seg.tombstones.is_some_and(|t| t.contains(slot))
                        || filter_mask.is_some_and(|m| !m.contains(slot))
                    {
                        evaluated_slots.insert(slot);
                        proof.filtered_or_tombstoned += 1;
                        continue;
                    }
                    candidate_slots.push(slot);
                }

                if candidate_slots.is_empty() {
                    continue;
                }

                if let Some(lutz_codes) = seg.lutz_codes {
                    let mut scored_cands: Vec<(NodeIndex, f32)> = candidate_slots
                        .into_iter()
                        .map(|s| {
                            let code = &lutz_codes[s as usize];
                            let approx = query_lut.score_candidate_l0(code);
                            (s, approx)
                        })
                        .collect();
                    scored_cands.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

                    for (slot, approx) in scored_cands {
                        proof.leaf_vectors_considered += 1;
                        let curr_tau = topk.kth_score() as f64;
                        let code = &lutz_codes[slot as usize];

                        // B3.1: LUTz L0
                        let res0 = query_lut.blockwise_residual_l0(code) as f64;
                        let ub0 = (approx as f64) + res0 + 1e-7;
                        proof.lutz_l0_evaluations += 1;
                        proof.l0_bytes_touched += code.codes_l0.len() + code.scales_l0.len();

                        if topk.is_full() && ub0 < curr_tau {
                            proof.lutz_l0_pruned += 1;
                            evaluated_slots.insert(slot);
                            continue;
                        }

                        // B3.2: LUTz L1
                        if code.codes_l1.is_some() {
                            let res1 = query_lut.blockwise_residual_l1(code) as f64;
                            let ub1 = (approx as f64) + res1 + 1e-7;
                            proof.lutz_l1_evaluations += 1;
                            proof.l1_bytes_touched +=
                                code.codes_l1.as_deref().map_or(0, |c| c.len() * 2);

                            if topk.is_full() && ub1 < curr_tau {
                                proof.lutz_l1_pruned += 1;
                                evaluated_slots.insert(slot);
                                continue;
                            }
                        }

                        // B3.3: Exact SIMD
                        let v = &seg.vectors[slot as usize];
                        let score = (query_vector.dot_product_complex(v)).re;
                        topk.offer(slot, score);
                        proof.exact_evaluations += 1;
                        proof.exact_bytes_touched += seg.tree.dimension * 8;
                        evaluated_slots.insert(slot);
                    }
                } else {
                    for slot in candidate_slots {
                        proof.leaf_vectors_considered += 1;
                        let v = &seg.vectors[slot as usize];
                        let score = (query_vector.dot_product_complex(v)).re;
                        topk.offer(slot, score);
                        proof.exact_evaluations += 1;
                        proof.exact_bytes_touched += seg.tree.dimension * 8;
                        evaluated_slots.insert(slot);
                    }
                }
            }
        }

        // Normal completion — proof is complete.
        proof.kth_score = topk.kth_score();
        proof.globally_exact = true;
        proof.elapsed_us = t_start.elapsed().as_micros() as u64;
        let total_regions = proof.proof_regions_pruned + proof.proof_regions_expanded;
        proof.region_prune_ratio = if total_regions > 0 {
            proof.proof_regions_pruned as f64 / total_regions as f64
        } else {
            0.0
        };

        (topk.into_sorted_vec(), proof)
    }
}

/// Multi-segment (ε, δ)-PAC relaxed proof search engine.
/// Prunes proof regions when (1 - ε) * UB < τ.
pub struct GlobalPacProofSearch;

impl GlobalPacProofSearch {
    pub fn search(
        query_vector: &VectorEmbedding,
        k: usize,
        segments: &[SegmentProofView],
        mutable_candidates: &[(NodeIndex, SimilarityScore)],
        rivero_seed_candidates: &[NodeIndex],
        filter_mask: Option<&RoaringBitmap>,
        epsilon: f32,
        _delta: f32,
    ) -> (Vec<(NodeIndex, SimilarityScore)>, DenseExactProof) {
        let t_start = std::time::Instant::now();
        let mut proof = DenseExactProof::default();
        let total_corpus: usize = segments
            .iter()
            .map(|s| s.tree.total_vectors())
            .sum::<usize>()
            + mutable_candidates.len();
        proof.corpus_size = total_corpus;

        if k == 0 || total_corpus == 0 {
            proof.elapsed_us = t_start.elapsed().as_micros() as u64;
            return (Vec::new(), proof);
        }

        let query = ProofQuery::new(query_vector.complex_data());
        let query_lut = LutzQueryTable::build(query_vector);
        let mut topk = TopKAccumulator::new(k);
        let mut evaluated_slots = RoaringBitmap::new();

        // 1. Mutable scan
        for &(slot, score) in mutable_candidates {
            if filter_mask.is_some_and(|m| !m.contains(slot)) {
                proof.filtered_or_tombstoned += 1;
                evaluated_slots.insert(slot);
                continue;
            }
            evaluated_slots.insert(slot);
            topk.offer(slot, score);
            proof.exact_evaluations += 1;
            proof.exact_bytes_touched += query_vector.dimension() * 8;
        }

        // 2. Rivero seed candidates
        proof.rivero_raw_candidates = rivero_seed_candidates.len();
        let mut seen_seeds = RoaringBitmap::new();
        for &slot in rivero_seed_candidates.iter().take(128.max(k * 2)) {
            if filter_mask.is_some_and(|m| !m.contains(slot)) || seen_seeds.contains(slot) {
                continue;
            }
            seen_seeds.insert(slot);
            proof.rivero_seed_candidates += 1;

            for seg in segments {
                if (slot as usize) < seg.vectors.len() {
                    if seg.tombstones.is_some_and(|t| t.contains(slot)) {
                        proof.filtered_or_tombstoned += 1;
                        evaluated_slots.insert(slot);
                        break;
                    }
                    evaluated_slots.insert(slot);
                    let score = (query_vector.dot_product_complex(&seg.vectors[slot as usize])).re;
                    proof.rivero_seed_exact_evaluations += 1;
                    proof.exact_evaluations += 1;
                    proof.exact_bytes_touched += seg.tree.dimension * 8;
                    topk.offer(slot, score);
                    break;
                }
            }
        }

        // 3. Spatially prunable segments with (1 - ε) relaxation
        let mut frontier: BinaryHeap<ProofFrontierEntry> = BinaryHeap::new();
        let eps_factor = (1.0 - epsilon as f64).max(0.0);

        for (seg_idx, seg) in segments.iter().enumerate() {
            if !seg.tree.is_spatially_prunable() || seg.tree.nodes.is_empty() {
                continue;
            }
            let ub = seg.tree.upper_bound(&query, seg.tree.root);
            let tau = topk.kth_score() as f64;

            if topk.is_full() && (ub * eps_factor) < tau {
                proof.proof_regions_pruned += 1;
                let unvisited = seg
                    .tree
                    .members(seg.tree.node(seg.tree.root))
                    .iter()
                    .filter(|&&s| !evaluated_slots.contains(s))
                    .count();
                proof.vectors_pruned_by_region += unvisited;
            } else {
                frontier.push(ProofFrontierEntry {
                    upper_bound: ub,
                    segment_idx: seg_idx,
                    node_idx: seg.tree.root,
                });
            }
        }

        // 4. PAC branch-and-bound loop
        while let Some(entry) = frontier.pop() {
            proof.proof_regions_popped += 1;
            let tau = topk.kth_score() as f64;
            let effective_ub = entry.upper_bound * eps_factor;

            if topk.is_full() && effective_ub < tau {
                proof.proof_regions_pruned += 1;
                let seg = &segments[entry.segment_idx];
                let node = seg.tree.node(entry.node_idx);
                let unvisited = seg
                    .tree
                    .members(node)
                    .iter()
                    .filter(|&&s| !evaluated_slots.contains(s))
                    .count();
                proof.vectors_pruned_by_region += unvisited;
                continue;
            }

            let seg = &segments[entry.segment_idx];
            let node = seg.tree.node(entry.node_idx);

            if node.is_internal() {
                proof.proof_regions_expanded += 1;
                for child_idx in seg.tree.children(node) {
                    let child_ub = seg.tree.upper_bound(&query, child_idx);
                    let curr_tau = topk.kth_score() as f64;

                    if topk.is_full() && (child_ub * eps_factor) < curr_tau {
                        proof.proof_regions_pruned += 1;
                        let child_node = seg.tree.node(child_idx);
                        let unvisited = seg
                            .tree
                            .members(child_node)
                            .iter()
                            .filter(|&&s| !evaluated_slots.contains(s))
                            .count();
                        proof.vectors_pruned_by_region += unvisited;
                    } else {
                        frontier.push(ProofFrontierEntry {
                            upper_bound: child_ub,
                            segment_idx: entry.segment_idx,
                            node_idx: child_idx,
                        });
                    }
                }
            } else {
                let mut candidate_slots = Vec::with_capacity(node.member_len as usize);
                for &slot in seg.tree.members(node) {
                    if evaluated_slots.contains(slot) {
                        continue;
                    }
                    if seg.tombstones.is_some_and(|t| t.contains(slot))
                        || filter_mask.is_some_and(|m| !m.contains(slot))
                    {
                        evaluated_slots.insert(slot);
                        proof.filtered_or_tombstoned += 1;
                        continue;
                    }
                    candidate_slots.push(slot);
                }

                if candidate_slots.is_empty() {
                    continue;
                }

                if let Some(lutz_codes) = seg.lutz_codes {
                    let mut scored_cands: Vec<(NodeIndex, f32)> = candidate_slots
                        .into_iter()
                        .map(|s| {
                            let code = &lutz_codes[s as usize];
                            let approx = query_lut.score_candidate_l0(code);
                            (s, approx)
                        })
                        .collect();
                    scored_cands.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

                    for (slot, approx) in scored_cands {
                        proof.leaf_vectors_considered += 1;
                        let curr_tau = topk.kth_score() as f64;
                        let code = &lutz_codes[slot as usize];

                        let res0 = query_lut.blockwise_residual_l0(code) as f64;
                        let ub0 = (approx as f64) + res0 + 1e-7;
                        proof.lutz_l0_evaluations += 1;
                        proof.l0_bytes_touched += code.codes_l0.len() + code.scales_l0.len();

                        if topk.is_full() && (ub0 * eps_factor) < curr_tau {
                            proof.lutz_l0_pruned += 1;
                            evaluated_slots.insert(slot);
                            continue;
                        }

                        if code.codes_l1.is_some() {
                            let res1 = query_lut.blockwise_residual_l1(code) as f64;
                            let ub1 = (approx as f64) + res1 + 1e-7;
                            proof.lutz_l1_evaluations += 1;
                            proof.l1_bytes_touched +=
                                code.codes_l1.as_deref().map_or(0, |c| c.len() * 2);

                            if topk.is_full() && (ub1 * eps_factor) < curr_tau {
                                proof.lutz_l1_pruned += 1;
                                evaluated_slots.insert(slot);
                                continue;
                            }
                        }

                        let v = &seg.vectors[slot as usize];
                        let score = (query_vector.dot_product_complex(v)).re;
                        topk.offer(slot, score);
                        proof.exact_evaluations += 1;
                        proof.exact_bytes_touched += seg.tree.dimension * 8;
                        evaluated_slots.insert(slot);
                    }
                } else {
                    for slot in candidate_slots {
                        proof.leaf_vectors_considered += 1;
                        let v = &seg.vectors[slot as usize];
                        let score = (query_vector.dot_product_complex(v)).re;
                        topk.offer(slot, score);
                        proof.exact_evaluations += 1;
                        proof.exact_bytes_touched += seg.tree.dimension * 8;
                        evaluated_slots.insert(slot);
                    }
                }
            }
        }

        proof.kth_score = topk.kth_score();
        proof.globally_exact = false;
        proof.elapsed_us = t_start.elapsed().as_micros() as u64;
        let total_regions = proof.proof_regions_pruned + proof.proof_regions_expanded;
        proof.region_prune_ratio = if total_regions > 0 {
            proof.proof_regions_pruned as f64 / total_regions as f64
        } else {
            0.0
        };

        (topk.into_sorted_vec(), proof)
    }
}
