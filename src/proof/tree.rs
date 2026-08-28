/* holosphere/src/proof/tree.rs */
//!▫~•◦-------------------------------‣
//! # Flattened Semantic Proof Hierarchy Substrate (Gate B2)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a contiguous, flattened, mmap-friendly semantic proof tree with
//! angular-radius-aware spherical partitioning and exact spherical-cap envelopes:
//!
//! $$\bigcup_{\ell \in \text{Leaves}(T)} \text{members}(\ell) = V_{\text{segment}} \quad\text{and}\quad \text{members}(\ell_i) \cap \text{members}(\ell_j) = \emptyset, \; \forall i \neq j$$
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use num_complex::Complex32;
use serde::{Deserialize, Serialize};

use super::bounds::{
    PROOF_BLOCK_COMPLEX_DIM, ProofCentroidCode, ProofQuery, evaluate_node_upper_bound_f64,
};
use crate::{NodeIndex, VectorEmbedding};

/// Default target capacity for resident vectors in a proof leaf before splitting.
pub const PROOF_LEAF_TARGET: usize = 32;

/// A contiguous, flat proof node in the flattened hierarchy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProofNode {
    /// Index of the first child node in `nodes` (0 if leaf).
    pub first_child: u32,
    /// Number of child branches (0 if leaf, 2 for binary bisection).
    pub child_count: u16,

    /// Start offset in `leaf_slots` array.
    pub member_start: u32,
    /// Number of resident vectors in this subtree.
    pub member_len: u32,

    /// Offset into the `centroid_codes` and `block_radii` flat arrays.
    pub centroid_offset: u32,

    /// $\cos\theta_T = \min_{x \in T} (\hat{c}_T^\top x)_{\text{re}}$.
    pub cos_radius: f32,
    /// $\sin\theta_T = \sqrt{\max(0, 1 - \cos^2\theta_T)}$.
    pub sin_radius: f32,

    /// True global Euclidean residual radius: $\rho_T = \max_{v \in T} \|\hat{v} - \hat{c}_T\|_2$.
    pub global_radius: f32,
    /// Global centroid reconstruction error: $\epsilon_T \ge \|c_T - \hat{c}_T\|_2$.
    pub centroid_error_norm: f32,

    /// Deterministic minimum slot in this subtree for lexicographic tie-breaking.
    pub min_slot: NodeIndex,
}

impl ProofNode {
    /// Returns `true` if this node is an internal branch node.
    #[inline(always)]
    pub fn is_internal(&self) -> bool {
        self.child_count > 0
    }

    /// Returns `true` if this node is a leaf node containing terminal vectors.
    #[inline(always)]
    pub fn is_leaf(&self) -> bool {
        self.child_count == 0
    }

    /// Returns the angular radius of this envelope in degrees.
    #[inline]
    pub fn angular_radius_degrees(&self) -> f32 {
        self.cos_radius.clamp(-1.0, 1.0).acos().to_degrees()
    }
}

/// Detailed manifold geometry profile characterizing intrinsic dimensional dispersion.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ManifoldGeometryProfile {
    /// Intrinsic dimensionality / dispersion ratio computed via pairwise cosine variance scaling.
    pub participation_ratio: f32,
    /// Mean angular radius across sample pairs in radians.
    pub mean_leaf_theta_rad: f32,
    /// Pairwise cosine variance across sample embeddings.
    pub pairwise_cosine_variance: f32,
    /// Determines whether the geometry admits spatial hierarchical pruning (e.g. Clustered vs Diffuse/Isotropic).
    pub is_spatially_prunable: bool,
}

impl Default for ManifoldGeometryProfile {
    fn default() -> Self {
        Self {
            participation_ratio: 10.0,
            mean_leaf_theta_rad: 0.5,
            pairwise_cosine_variance: 0.05,
            is_spatially_prunable: true,
        }
    }
}

/// The flattened, canonical corpus-covering semantic proof hierarchy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemanticProofTree {
    pub dimension: usize,
    pub block_size: usize,
    pub blocks_per_vector: usize,

    /// Flat node-major array of all proof nodes.
    pub nodes: Box<[ProofNode]>,

    /// Canonical permutation of all resident vector slots. Every segment slot occurs exactly once.
    pub leaf_slots: Box<[NodeIndex]>,

    /// Flat node-major block summaries: `centroid_codes[node.centroid_offset + b]`.
    pub centroid_codes: Box<[ProofCentroidCode]>,
    /// Flat node-major block residual radii: `block_radii[node.centroid_offset + b]`.
    pub block_radii: Box<[f32]>,

    /// Root node index (always 0).
    pub root: u32,

    /// Manifold geometry profile characterizing spatial prunability.
    pub manifold_profile: ManifoldGeometryProfile,
}

impl SemanticProofTree {
    /// Builds a canonical, flattened proof hierarchy over the given vectors and slots.
    ///
    /// Every slot in `slots` is guaranteed to be partitioned into exactly one leaf node.
    pub fn build(vectors: &[VectorEmbedding], slots: &[NodeIndex], dimension: usize) -> Self {
        Self::build_with_leaf_target(vectors, slots, dimension, PROOF_LEAF_TARGET)
    }

    /// Computes the manifold geometry profile and anisotropy ratio from a sample of embeddings.
    pub fn compute_manifold_profile(
        vectors: &[VectorEmbedding],
        slots: &[NodeIndex],
        dimension: usize,
    ) -> ManifoldGeometryProfile {
        let sample_size = 256.min(slots.len());
        if sample_size < 8 || dimension == 0 {
            return ManifoldGeometryProfile::default();
        }

        let mut cos_sum = 0.0f32;
        let mut cos_sq_sum = 0.0f32;
        let mut samples = 0usize;

        for i in 0..sample_size {
            let slot_i = slots[i] as usize;
            if slot_i >= vectors.len() {
                continue;
            }
            let vi = &vectors[slot_i];

            for j in (i + 1)..sample_size.min(i + 32) {
                let slot_j = slots[j] as usize;
                if slot_j >= vectors.len() {
                    continue;
                }
                let vj = &vectors[slot_j];
                let s = vi.dot_product_real(vj);
                cos_sum += s;
                cos_sq_sum += s * s;
                samples += 1;
            }
        }

        if samples == 0 {
            return ManifoldGeometryProfile::default();
        }

        let mean_cos = cos_sum / samples as f32;
        let variance = (cos_sq_sum / samples as f32 - mean_cos * mean_cos).max(0.0);

        // For uniform isotropic vectors in D complex dimensions (2D real dimensions),
        // pairwise inner product variance is ~ 1 / (2D).
        // Clustered manifolds exhibit variance >> 1 / (2D) due to inter-cluster separation.
        let d_real = (dimension * 2) as f32;
        let expected_isotropic_var = 1.0 / d_real.max(1.0);
        let anisotropy_ratio = variance / expected_isotropic_var.max(1e-7);

        let is_spatially_prunable = anisotropy_ratio >= 3.0;

        ManifoldGeometryProfile {
            participation_ratio: anisotropy_ratio,
            mean_leaf_theta_rad: mean_cos.clamp(-1.0, 1.0).acos(),
            pairwise_cosine_variance: variance,
            is_spatially_prunable,
        }
    }

    /// Builds a proof tree with a specified leaf capacity target.
    pub fn build_with_leaf_target(
        vectors: &[VectorEmbedding],
        slots: &[NodeIndex],
        dimension: usize,
        leaf_target: usize,
    ) -> Self {
        if slots.is_empty() {
            return Self::empty(dimension);
        }

        let blocks_per_vector = dimension.div_ceil(PROOF_BLOCK_COMPLEX_DIM);
        let profile = Self::compute_manifold_profile(vectors, slots, dimension);
        let builder = TreeBuilder::new(vectors, slots, dimension, blocks_per_vector);
        builder.build_tree(leaf_target, profile)
    }

    /// Returns an empty proof hierarchy for a segment with 0 live vectors.
    pub fn empty(dimension: usize) -> Self {
        let blocks_per_vector = dimension.div_ceil(PROOF_BLOCK_COMPLEX_DIM);
        Self {
            dimension,
            block_size: PROOF_BLOCK_COMPLEX_DIM,
            blocks_per_vector,
            nodes: Box::new([]),
            leaf_slots: Box::new([]),
            centroid_codes: Box::new([]),
            block_radii: Box::new([]),
            root: 0,
            manifold_profile: ManifoldGeometryProfile::default(),
        }
    }

    /// Returns `true` if this segment has geometry amenable to spatial tree hierarchical pruning.
    #[inline(always)]
    pub fn is_spatially_prunable(&self) -> bool {
        self.manifold_profile.is_spatially_prunable
    }

    /// Returns the node at index `idx`.
    #[inline(always)]
    pub fn node(&self, idx: u32) -> &ProofNode {
        &self.nodes[idx as usize]
    }

    /// Returns an iterator over the child indices of a node.
    #[inline(always)]
    pub fn children(&self, node: &ProofNode) -> impl Iterator<Item = u32> {
        let start = node.first_child;
        let count = node.child_count as u32;
        start..(start + count)
    }

    /// Returns the slice of member slots resident in a node.
    #[inline(always)]
    pub fn members(&self, node: &ProofNode) -> &[NodeIndex] {
        let start = node.member_start as usize;
        let end = start + node.member_len as usize;
        &self.leaf_slots[start..end]
    }

    /// Evaluates the provable combined upper bound for a node in `f64`.
    #[inline(always)]
    pub fn upper_bound(&self, query: &ProofQuery, node_idx: u32) -> f64 {
        let node = self.node(node_idx);
        evaluate_node_upper_bound_f64(
            query,
            &self.centroid_codes,
            &self.block_radii,
            node.centroid_offset as usize,
            self.blocks_per_vector,
            node.cos_radius,
            node.sin_radius,
            node.global_radius,
            node.centroid_error_norm,
        )
    }

    /// Total live vectors contained in this proof hierarchy.
    #[inline(always)]
    pub fn total_vectors(&self) -> usize {
        self.leaf_slots.len()
    }
}

/// Recursive builder creating the flat proof tree layout via spherical-cap minimization.
struct TreeBuilder<'a> {
    vectors: &'a [VectorEmbedding],
    slots: Vec<NodeIndex>,
    dimension: usize,
    blocks_per_vector: usize,

    nodes: Vec<ProofNode>,
    centroid_codes: Vec<ProofCentroidCode>,
    block_radii: Vec<f32>,
}

impl<'a> TreeBuilder<'a> {
    fn new(
        vectors: &'a [VectorEmbedding],
        slots: &[NodeIndex],
        dimension: usize,
        blocks_per_vector: usize,
    ) -> Self {
        Self {
            vectors,
            slots: slots.to_vec(),
            dimension,
            blocks_per_vector,
            nodes: Vec::new(),
            centroid_codes: Vec::new(),
            block_radii: Vec::new(),
        }
    }

    fn build_tree(
        mut self,
        leaf_target: usize,
        profile: ManifoldGeometryProfile,
    ) -> SemanticProofTree {
        let total_slots = self.slots.len();
        self.nodes.push(ProofNode::default());
        self.partition_recursive(0, 0, total_slots, leaf_target);

        SemanticProofTree {
            dimension: self.dimension,
            block_size: PROOF_BLOCK_COMPLEX_DIM,
            blocks_per_vector: self.blocks_per_vector,
            nodes: self.nodes.into_boxed_slice(),
            leaf_slots: self.slots.into_boxed_slice(),
            centroid_codes: self.centroid_codes.into_boxed_slice(),
            block_radii: self.block_radii.into_boxed_slice(),
            root: 0,
            manifold_profile: profile,
        }
    }

    fn compute_normalized_centroid(&self, slots_slice: &[NodeIndex]) -> Vec<Complex32> {
        let mut centroid_sum = vec![Complex32::new(0.0, 0.0); self.dimension];
        for &s in slots_slice {
            let v = &self.vectors[s as usize];
            for (d, &z) in v.complex_data().iter().enumerate().take(self.dimension) {
                centroid_sum[d] += z;
            }
        }

        let mut sum_norm_sq = 0.0f32;
        for z in &centroid_sum {
            sum_norm_sq += z.norm_sqr();
        }
        let norm = sum_norm_sq.sqrt();
        if norm > 1e-12 {
            let inv_norm = 1.0f32 / norm;
            for z in &mut centroid_sum {
                *z *= inv_norm;
            }
        } else if !centroid_sum.is_empty() {
            centroid_sum[0] = Complex32::new(1.0, 0.0);
        }
        centroid_sum
    }

    fn partition_recursive(
        &mut self,
        node_idx: usize,
        start: usize,
        len: usize,
        leaf_target: usize,
    ) {
        let centroid_offset = self.centroid_codes.len() as u32;
        let mut min_slot = NodeIndex::MAX;
        for &s in &self.slots[start..(start + len)] {
            if s < min_slot {
                min_slot = s;
            }
        }

        // 1. Compute unit-normalized global complex centroid
        let centroid = self.compute_normalized_centroid(&self.slots[start..(start + len)]);

        // 2. Compute angular radius cos_radius = min_{v in T} (v^T c_T)
        let mut min_cos_sim = 1.0f32;
        let mut max_global_diff = 0.0f32;

        for &s in &self.slots[start..(start + len)] {
            let v = &self.vectors[s as usize];
            let v_data = v.complex_data();

            // Compute cosine dot product with centroid
            let mut dot_re = 0.0f32;
            for (d, &z) in v_data.iter().enumerate().take(self.dimension) {
                dot_re += z.re * centroid[d].re + z.im * centroid[d].im;
            }
            if dot_re < min_cos_sim {
                min_cos_sim = dot_re;
            }

            // Compute global Euclidean difference
            let mut sum_sq = 0.0f32;
            for d in 0..self.dimension {
                let diff = if d < v_data.len() {
                    v_data[d] - centroid[d]
                } else {
                    -centroid[d]
                };
                sum_sq += diff.norm_sqr();
            }
            let diff_norm = sum_sq.sqrt();
            if diff_norm > max_global_diff {
                max_global_diff = diff_norm;
            }
        }

        let cos_radius = min_cos_sim.clamp(-1.0, 1.0);
        let sin_radius = (1.0f32 - cos_radius * cos_radius).max(0.0).sqrt();

        // 3. Compute block summaries and residual block radii
        for b in 0..self.blocks_per_vector {
            let b_start = b * PROOF_BLOCK_COMPLEX_DIM;
            let b_end = (b_start + PROOF_BLOCK_COMPLEX_DIM).min(self.dimension);

            let block_centroid = &centroid[b_start..b_end];
            let code = ProofCentroidCode::from_raw(block_centroid);
            self.centroid_codes.push(code);

            // Block residual radius rho_T,b = max_{v in T} ||v_b - c_b||_2
            let mut max_block_diff = 0.0f32;
            for &s in &self.slots[start..(start + len)] {
                let v = &self.vectors[s as usize];
                let v_data = v.complex_data();
                let mut sum_sq = 0.0f32;
                for i in b_start..b_end {
                    let diff = if i < v_data.len() {
                        v_data[i] - centroid[i]
                    } else {
                        -centroid[i]
                    };
                    sum_sq += diff.norm_sqr();
                }
                let norm = sum_sq.sqrt();
                if norm > max_block_diff {
                    max_block_diff = norm;
                }
            }
            self.block_radii.push(max_block_diff);
        }

        if len <= leaf_target {
            // Base case: Leaf node
            self.nodes[node_idx] = ProofNode {
                first_child: 0,
                child_count: 0,
                member_start: start as u32,
                member_len: len as u32,
                centroid_offset,
                cos_radius,
                sin_radius,
                global_radius: max_global_diff,
                centroid_error_norm: 0.0,
                min_slot,
            };
            return;
        }

        // 4. Multi-Trial Spherical-Cap Minimizing Bisection
        let mut best_left = Vec::with_capacity(len);
        let mut best_right = Vec::with_capacity(len);
        let mut best_penalty = f32::INFINITY;

        // Try candidate pivot pairs (Pivot A, Pivot B)
        let candidates: [(usize, usize); 3] = [
            (0, len / 2),
            (0, len.saturating_sub(1)),
            (len / 3, (2 * len) / 3),
        ];

        for &(p_a_offset, p_b_offset) in &candidates {
            let slot_a = self.slots[start + p_a_offset];
            let slot_b = self.slots[start + p_b_offset];
            if slot_a == slot_b {
                continue;
            }
            let mut pivot_a = self.vectors[slot_a as usize].complex_data().to_vec();
            let mut pivot_b = self.vectors[slot_b as usize].complex_data().to_vec();

            let mut cur_left = Vec::with_capacity(len);
            let mut cur_right = Vec::with_capacity(len);

            // 2 iterations of Spherical 2-Means
            for _iter in 0..2 {
                cur_left.clear();
                cur_right.clear();

                for &s in &self.slots[start..(start + len)] {
                    let v = &self.vectors[s as usize];
                    let v_data = v.complex_data();
                    let mut sim_a = 0.0f32;
                    let mut sim_b = 0.0f32;
                    for d in 0..self.dimension {
                        sim_a += v_data[d].re * pivot_a[d].re + v_data[d].im * pivot_a[d].im;
                        sim_b += v_data[d].re * pivot_b[d].re + v_data[d].im * pivot_b[d].im;
                    }
                    if sim_a >= sim_b {
                        cur_left.push(s);
                    } else {
                        cur_right.push(s);
                    }
                }

                if cur_left.is_empty() || cur_right.is_empty() {
                    break;
                }

                pivot_a = self.compute_normalized_centroid(&cur_left);
                pivot_b = self.compute_normalized_centroid(&cur_right);
            }

            if cur_left.is_empty() || cur_right.is_empty() {
                continue;
            }

            // Evaluate angular penalty of the candidate partition: J = n_L * sin(theta_L) + n_R * sin(theta_R) + balance
            let mut min_cos_l = 1.0f32;
            for &s in &cur_left {
                let v = &self.vectors[s as usize];
                let v_data = v.complex_data();
                let mut dot = 0.0f32;
                for d in 0..self.dimension {
                    dot += v_data[d].re * pivot_a[d].re + v_data[d].im * pivot_a[d].im;
                }
                if dot < min_cos_l {
                    min_cos_l = dot;
                }
            }
            let sin_l = (1.0f32 - min_cos_l.clamp(-1.0, 1.0).powi(2))
                .max(0.0)
                .sqrt();

            let mut min_cos_r = 1.0f32;
            for &s in &cur_right {
                let v = &self.vectors[s as usize];
                let v_data = v.complex_data();
                let mut dot = 0.0f32;
                for d in 0..self.dimension {
                    dot += v_data[d].re * pivot_b[d].re + v_data[d].im * pivot_b[d].im;
                }
                if dot < min_cos_r {
                    min_cos_r = dot;
                }
            }
            let sin_r = (1.0f32 - min_cos_r.clamp(-1.0, 1.0).powi(2))
                .max(0.0)
                .sqrt();

            let balance_diff = (cur_left.len() as f32 - cur_right.len() as f32).abs();
            let penalty = (cur_left.len() as f32) * sin_l
                + (cur_right.len() as f32) * sin_r
                + 0.05 * balance_diff;

            if penalty < best_penalty {
                best_penalty = penalty;
                best_left = cur_left;
                best_right = cur_right;
            }
        }

        // Fallback to median split if all trials degenerated
        if best_left.is_empty() || best_right.is_empty() {
            let mid = len / 2;
            best_left.clear();
            best_right.clear();
            best_left.extend_from_slice(&self.slots[start..(start + mid)]);
            best_right.extend_from_slice(&self.slots[(start + mid)..(start + len)]);
        }

        let left_len = best_left.len();
        self.slots[start..(start + left_len)].copy_from_slice(&best_left);
        self.slots[(start + left_len)..(start + len)].copy_from_slice(&best_right);

        let first_child = self.nodes.len() as u32;
        self.nodes.push(ProofNode::default());
        self.nodes.push(ProofNode::default());

        self.nodes[node_idx] = ProofNode {
            first_child,
            child_count: 2,
            member_start: start as u32,
            member_len: len as u32,
            centroid_offset,
            cos_radius,
            sin_radius,
            global_radius: max_global_diff,
            centroid_error_norm: 0.0,
            min_slot,
        };

        self.partition_recursive(first_child as usize, start, left_len, leaf_target);
        self.partition_recursive(
            (first_child + 1) as usize,
            start + left_len,
            len - left_len,
            leaf_target,
        );
    }
}
