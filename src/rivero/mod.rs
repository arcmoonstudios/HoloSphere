/* hnsqr/src/rivero/mod.rs */
//!▫~•◦-------------------------------‣
//! # Rivero Resolve — Bounded Complex-Vector Address Routing
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Compiles arbitrary-dimensional complex embeddings into a fixed set of E8
//! foundations, then resolves a corpus-independent bounded candidate set using
//! Rivero's C(7,3) insertion and C(9,3) lookup signatures.
//!
//! Address compilation is O(D) in source dimension. Territory lookup performs a
//! fixed number of hash probes and bounded resident reads independent of corpus
//! size. Exact projective-overlap reranking remains bounded by the resolved candidate
//! budget and source dimension.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::cell::RefCell;
use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering};

use num_complex::Complex32;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{NodeIndex, SimilarityScore};

pub mod bulk;
pub mod witness;

pub use bulk::{BuiltRiveroState, BulkBuildTelemetry, RiveroBulkBuilder};
pub use witness::{
    RIVERO_WITNESS_DEFAULT_DEGREE, RIVERO_WITNESS_DEFAULT_SECOND_SEEDS,
    RIVERO_WITNESS_DEFAULT_SEEDS, RIVERO_WITNESS_INLINE_DEGREE, RIVERO_WITNESS_MAX_DEGREE,
    RIVERO_WITNESS_MAX_SEEDS, ScoredWitness, select_top, witness_edge_scan_bound,
    witness_two_hop_edge_scan_bound,
};

/// Number of roots in the canonical E8 root system.
pub const E8_ROOT_COUNT: usize = 240;

const E8_ROOT_LAST_ID: u8 = 239;

/// Top-root count used at insert time. Each node is registered in C(7,3) = 35 cells.
pub const INSERT_TOP_ROOTS: usize = 7;

/// Top-root count used at lookup time. Each query probes C(9,3) = 84 cells.
pub const LOOKUP_TOP_ROOTS: usize = 9;

static E8_ROOTS: LazyLock<[[f32; 8]; E8_ROOT_COUNT]> = LazyLock::new(generate_e8_roots);
/// Maximum number of independent E8 foundations supported by the universal Rivero address.
pub const RIVERO_MAX_FOUNDATIONS: usize = 64;
/// Default number of independent E8 foundations in standard profiles.
pub const RIVERO_DEFAULT_FOUNDATIONS: usize = 24;
/// Number of independent E8 foundations in legacy default profile.
pub const RIVERO_FOUNDATIONS: usize = RIVERO_DEFAULT_FOUNDATIONS;
/// Maximum residents retained in each territorial cell.
pub const RIVERO_CELL_CAPACITY: usize = 64;
/// Residents reserved for deterministic cell-affinity elites.
pub const RIVERO_CELL_AFFINITY_ELITES: usize = 24;
/// Residents reserved for cell-keyed deterministic diversity sampling.
pub const RIVERO_CELL_DIVERSITY_RESIDENTS: usize =
    RIVERO_CELL_CAPACITY - RIVERO_CELL_AFFINITY_ELITES;
/// Rivero address-compiler schema version.
pub const RIVERO_SCHEMA_VERSION: u16 = 2;

/// Strategy for assigning embedding coordinates to MultiLane projection banks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaneAssignment {
    /// Deterministic pseudo-random hashing avoiding continuous coordinate assumptions.
    #[default]
    Hashed,
    /// Round-robin interleaved dimensional assignment.
    Interleaved,
}

/// Address projection architecture for dimensional capacity scaling.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiveroProjectionMode {
    /// All dimensions project into every foundation accumulator.
    #[default]
    GlobalMix,
    /// Dimensions are partitioned into disjoint lane banks with independent E8 foundations.
    MultiLane {
        lanes: u8,
        assignment: LaneAssignment,
    },
}

/// Address configuration defining foundation geometry and projection mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiveroAddressConfig {
    /// Number of active E8 foundations (1 ..= 64). Default: 24.
    pub foundations: u8,
    /// Projection architecture (GlobalMix or MultiLane). Default: GlobalMix.
    pub projection: RiveroProjectionMode,
}

impl Default for RiveroAddressConfig {
    fn default() -> Self {
        Self {
            foundations: RIVERO_DEFAULT_FOUNDATIONS as u8,
            projection: RiveroProjectionMode::GlobalMix,
        }
    }
}
pub(crate) const RIVERO_STRIPES: usize = 64;
pub const RIVERO_E8_LOOKUP_CELLS: usize = 84;
pub const RIVERO_E8_INSERT_CELLS: usize = 35;
const RIVERO_SIMHASH_BITS: usize = 12;
const RIVERO_SIMHASH_PROBE_POOL: usize = 1
    + RIVERO_SIMHASH_BITS
    + (RIVERO_SIMHASH_BITS * (RIVERO_SIMHASH_BITS - 1) / 2)
    + (RIVERO_SIMHASH_BITS * (RIVERO_SIMHASH_BITS - 1) * (RIVERO_SIMHASH_BITS - 2) / 6)
    + (RIVERO_SIMHASH_BITS
        * (RIVERO_SIMHASH_BITS - 1)
        * (RIVERO_SIMHASH_BITS - 2)
        * (RIVERO_SIMHASH_BITS - 3)
        / 24);
pub const RIVERO_SIMHASH_BUILD_PROBES: usize = 1
    + RIVERO_SIMHASH_BITS
    + (RIVERO_SIMHASH_BITS * (RIVERO_SIMHASH_BITS - 1) / 2)
    + (RIVERO_SIMHASH_BITS * (RIVERO_SIMHASH_BITS - 1) * (RIVERO_SIMHASH_BITS - 2) / 6);
pub const RIVERO_SIMHASH_QUERY_PROBES: usize = 32;
/// Maximum vote-ranked candidates exposed to exact serving rerank.
pub const RIVERO_QUERY_CANDIDATE_CAP: usize = 2_048;
/// Maximum vote-ranked candidates exact-scored while constructing witnesses.
pub const RIVERO_BUILD_CANDIDATE_CAP: usize = 1_024;
const RIVERO_BUILD_LOOKUP_CELLS_PER_FOUNDATION: usize =
    RIVERO_E8_LOOKUP_CELLS + RIVERO_SIMHASH_BUILD_PROBES;
const RIVERO_INSERT_CELLS_PER_FOUNDATION: usize = RIVERO_E8_INSERT_CELLS + 1;
const DEFAULT_SCRATCH_CAPACITY: usize =
    RIVERO_FOUNDATIONS * RIVERO_BUILD_LOOKUP_CELLS_PER_FOUNDATION * 16;

/// Standardized Rivero execution profiles established by Pareto frontier analysis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiveroProfile {
    /// Ultra-fast stage: 8 Foundations, 8 SimHash probes, 32 capacity, 8 budget, 512 candidate cap (~1.1 ms).
    #[default]
    Fast,
    /// Balanced production stage: 12 Foundations, 16 SimHash probes, 48 capacity, 12 budget, 1024 candidate cap (~1.7 ms).
    Balanced,
    /// Fully bounded reference stage: 24 Foundations, 32 SimHash probes, 64 capacity, 16 budget, 2048 candidate cap (~3.7 ms).
    Strict,
}

impl RiveroProfile {
    /// Returns the concrete [`RiveroConfig`] for this profile.
    #[must_use]
    pub const fn config(self) -> RiveroConfig {
        match self {
            Self::Fast => RiveroConfig {
                foundations: 8,
                simhash_query_probes: 8,
                cell_capacity: 32,
                affinity_elites: 12,
                cell_budget: 8,
                query_candidate_cap: 512,
            },
            Self::Balanced => RiveroConfig {
                foundations: 12,
                simhash_query_probes: 16,
                cell_capacity: 48,
                affinity_elites: 18,
                cell_budget: 12,
                query_candidate_cap: 1024,
            },
            Self::Strict => RiveroConfig::strict_default(),
        }
    }

    /// Next progressive escalation profile.
    #[must_use]
    pub const fn next_escalation(self) -> Option<Self> {
        match self {
            Self::Fast => Some(Self::Balanced),
            Self::Balanced => Some(Self::Strict),
            Self::Strict => None,
        }
    }
}

/// Operational search mode for HNSQR index queries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiveroSearchMode {
    /// Fixed-work provably bounded single-profile resolution with zero graph fallback.
    #[default]
    Strict,
    /// Multi-stage confidence-driven progressive resolution with state reuse.
    Adaptive,
    /// Classical HNSW graph traversal only (for baseline benchmarking and comparison).
    GraphOnly,
}

/// Fallback policy when Rivero confidence remains uncertain after the final stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdaptivePolicy {
    /// Keep strict theoretical bounds: return best Rivero candidates with no graph traversal.
    #[default]
    RiveroOnly,
    /// Highest recall guarantee: escalate to HNSW graph search if confidence is low.
    AllowGraphFallback,
}

/// Declared semantic retrieval contract for Rivero execution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum RiveroContract {
    /// Bounded proposal candidate universe with hard real-time latency ceilings (corpus-covering Certified proof search remains data-dependent approaching O(N) in worst-case).
    Bounded,
    /// Progressive confidence-driven expansion targeting high statistical recall.
    HighRecall(f32),
    /// Proof-carrying exact retrieval guaranteeing 100.000% Recall@K via UB <= tau frontier.
    #[default]
    Exact,
}

/// Audit telemetry and proof of mathematical exactness for Rivero queries.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct RiveroExactProof {
    /// Score of the K-th finalist at termination.
    pub kth_score: f32,
    /// Maximum upper bound observed among unvisited territories / vectors.
    pub max_unvisited_upper_bound: f32,
    /// Number of territorial cells visited.
    pub territories_visited: usize,
    /// Number of territorial cells mathematically pruned without reading member vectors.
    pub territories_pruned: usize,
    /// Number of vectors pruned by territory-level envelopes.
    pub vectors_territory_pruned: usize,
    /// Number of vectors pruned by LUTz L0 Cauchy-Schwarz bounds.
    pub vectors_l0_pruned: usize,
    /// Number of vectors pruned by LUTz L1 refinement bounds.
    pub vectors_l1_pruned: usize,
    /// Number of vectors evaluated with exact SIMD inner product.
    pub vectors_exact_scored: usize,
    /// Whether mathematical exactness (max_unvisited_upper_bound <= kth_score) was proven.
    pub globally_exact: bool,
}

/// Provable blockwise semantic envelope for an entire territory cell or partition region.
#[derive(Clone, Debug, PartialEq)]
pub struct TerritoryEnvelope {
    /// Number of resident vectors in this cell.
    pub count: u32,
    /// Blockwise complex centroid vector $c_b$ ($B \times 4$ Complex32).
    pub centroids: Box<[Complex32]>,
    /// Blockwise residual radius $\rho_b = \max_{x \in C} \|x_b - c_b\|_2$.
    pub residual_radii: Box<[f32]>,
    /// Global Euclidean residual radius $\sqrt{\sum \rho_b^2}$.
    pub global_residual_radius: f32,
}

impl TerritoryEnvelope {
    /// Constructs an envelope from a set of complex vector slices.
    #[must_use]
    pub fn from_vectors(vectors: &[&[Complex32]], dimension: usize) -> Self {
        if vectors.is_empty() {
            return Self {
                count: 0,
                centroids: Box::new([]),
                residual_radii: Box::new([]),
                global_residual_radius: 0.0,
            };
        }
        let num_blocks = dimension.div_ceil(4);
        let count = vectors.len() as f32;
        let mut centroids = vec![Complex32::new(0.0, 0.0); dimension];

        // 1. Mean centroid
        for v in vectors {
            for (i, &z) in v.iter().enumerate().take(dimension) {
                centroids[i] += z;
            }
        }
        for z in &mut centroids {
            *z /= count;
        }

        // 2. Blockwise residual radii
        let mut residual_radii = vec![0.0f32; num_blocks];
        for v in vectors {
            for b in 0..num_blocks {
                let start = b * 4;
                let end = (start + 4).min(dimension);
                let mut err_sq = 0.0f32;
                for i in start..end {
                    let diff = v[i] - centroids[i];
                    err_sq += diff.re * diff.re + diff.im * diff.im;
                }
                let err = err_sq.sqrt();
                if err > residual_radii[b] {
                    residual_radii[b] = err;
                }
            }
        }

        let global_residual_radius = residual_radii.iter().map(|&r| r * r).sum::<f32>().sqrt();

        Self {
            count: vectors.len() as u32,
            centroids: centroids.into_boxed_slice(),
            residual_radii: residual_radii.into_boxed_slice(),
            global_residual_radius,
        }
    }

    /// Evaluates the provable upper bound: $\text{UB}_{\text{cell}}(q) = \sum_b (\text{Re}\langle q_b, c_b \rangle + \|q_b\|_2 \rho_b)$.
    #[inline(always)]
    pub fn upper_bound(&self, q: &[Complex32], q_block_norms: &[f32], q_global_norm: f32) -> f32 {
        if self.count == 0 {
            return f32::NEG_INFINITY;
        }
        let num_blocks = q_block_norms.len().min(self.residual_radii.len());
        let mut dot_re = 0.0f32;
        let mut cs_sum = 0.0f32;

        for b in 0..num_blocks {
            let start = b * 4;
            let end = (start + 4).min(q.len().min(self.centroids.len()));
            for i in start..end {
                let q_z = q[i];
                let c_z = self.centroids[i];
                dot_re += q_z.re * c_z.re + q_z.im * c_z.im;
            }
            cs_sum += q_block_norms[b] * self.residual_radii[b];
        }

        let cs_bound = cs_sum.min(q_global_norm * self.global_residual_radius);
        dot_re + cs_bound + 1e-5
    }
}

/// Configurable parameters for Rivero candidate routing and Pareto optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiveroConfig {
    /// Number of independent E8 foundations probed (1..=24). Default: 24.
    pub foundations: usize,
    /// Number of nearest SimHash probe signatures per foundation (0..=128). Default: 32.
    pub simhash_query_probes: usize,
    /// Maximum residents retained in each territorial cell (8..=64). Default: 64.
    pub cell_capacity: usize,
    /// Residents reserved for deterministic cell-affinity elites. Default: 24 (or 3/8 of capacity).
    pub affinity_elites: usize,
    /// Maximum residents admitted per probed cell to exact reranking (1..=64). Default: 16.
    pub cell_budget: usize,
    /// Maximum vote-selected candidates exposed to exact reranking. Default: 2048.
    pub query_candidate_cap: usize,
}

impl RiveroConfig {
    /// Strict default configuration with corpus-independent ceiling guarantees.
    #[must_use]
    pub const fn strict_default() -> Self {
        Self {
            foundations: RIVERO_FOUNDATIONS,
            simhash_query_probes: RIVERO_SIMHASH_QUERY_PROBES,
            cell_capacity: RIVERO_CELL_CAPACITY,
            affinity_elites: RIVERO_CELL_AFFINITY_ELITES,
            cell_budget: 16,
            query_candidate_cap: RIVERO_QUERY_CANDIDATE_CAP,
        }
    }

    /// Fast balanced configuration identified via Pareto sweep.
    #[must_use]
    pub const fn fast_balanced() -> Self {
        RiveroProfile::Balanced.config()
    }

    /// Custom configuration with sanitized bounds.
    #[must_use]
    pub fn custom(
        foundations: usize,
        simhash_query_probes: usize,
        cell_capacity: usize,
        cell_budget: usize,
        query_candidate_cap: usize,
    ) -> Self {
        let foundations = foundations.clamp(1, RIVERO_FOUNDATIONS);
        let cell_capacity = cell_capacity.clamp(4, RIVERO_CELL_CAPACITY);
        let affinity_elites = (cell_capacity * 3) / 8;
        let cell_budget = cell_budget.clamp(1, cell_capacity);
        let query_candidate_cap = query_candidate_cap.max(1);
        Self {
            foundations,
            simhash_query_probes,
            cell_capacity,
            affinity_elites,
            cell_budget,
            query_candidate_cap,
        }
    }

    /// Total number of territory cells probed per query for this configuration.
    #[must_use]
    #[inline]
    pub const fn cell_probe_count(&self) -> usize {
        self.foundations * (RIVERO_E8_LOOKUP_CELLS + self.simhash_query_probes)
    }

    /// Maximum resident records read into admission buffer per query.
    #[must_use]
    #[inline]
    pub const fn candidate_read_bound(&self) -> usize {
        self.cell_probe_count() * self.cell_budget
    }

    /// Maximum compact resident codes scanned across all probed cells per query.
    #[must_use]
    #[inline]
    pub const fn resident_scan_bound(&self) -> usize {
        self.cell_probe_count() * self.cell_capacity
    }
}

impl Default for RiveroConfig {
    fn default() -> Self {
        Self::strict_default()
    }
}

/// Deterministic multi-factor confidence evaluation of Rivero candidate routing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RiveroConfidence {
    /// Composite confidence score in [0.0, 1.0].
    pub score: f32,
    /// Normalized vote distribution concentration / low entropy in [0.0, 1.0].
    pub vote_concentration: f32,
    /// Relative margin between top-1 and top-2 candidate votes in [0.0, 1.0].
    pub vote_margin: f32,
    /// Fraction of resident scan bound consumed by this query in [0.0, 1.0].
    pub scan_saturation: f32,
    /// Fraction of candidate cap utilized in [0.0, 1.0].
    pub candidate_saturation: f32,
    /// Fraction of returned candidates supplied by witness expansion in [0.0, 1.0].
    pub witness_dependency: f32,
    /// Exact fidelity gap between top-1 and top-2 scored candidates.
    pub top1_margin: f32,
    /// Exact fidelity gap between top-k and top-(k+1) score boundary.
    pub topk_tail_margin: f32,
    /// Top-k candidate intersection agreement with the preceding stage in [0.0, 1.0].
    pub cross_stage_stability: f32,
    /// Whether this stage recommends escalation to the next Rivero profile.
    pub escalation_recommended: bool,
}

impl RiveroConfidence {
    /// Evaluates routing confidence from candidate voting, saturation, witness, and score distribution.
    #[must_use]
    pub fn evaluate(
        voted: &[VotedCandidate],
        diag: &RiveroRouteDiagnostics,
        scored: &[(NodeIndex, SimilarityScore)],
        witness_candidates_added: usize,
        k: usize,
        cross_stage_stability: Option<f32>,
        profile: RiveroProfile,
    ) -> Self {
        if voted.is_empty() || scored.is_empty() {
            return Self {
                escalation_recommended: true,
                ..Self::default()
            };
        }

        // 1. Vote concentration (Normalized Shannon Entropy of top candidates)
        let total_votes: u32 = voted.iter().take(32).map(|v| v.votes).sum();
        let vote_concentration = if total_votes > 0 && voted.len() > 1 {
            let mut entropy = 0.0f32;
            let top_len = voted.len().min(32);
            for v in &voted[..top_len] {
                if v.votes > 0 {
                    let p = (v.votes as f32) / (total_votes as f32);
                    entropy -= p * p.ln();
                }
            }
            let max_entropy = (top_len as f32).ln().max(1e-4);
            (1.0 - (entropy / max_entropy)).clamp(0.0, 1.0)
        } else if !voted.is_empty() {
            1.0
        } else {
            0.0
        };

        // 2. Vote Margin: relative gap between rank 0 and rank 1
        let vote_margin = if voted.len() >= 2 && voted[0].votes > 0 {
            ((voted[0].votes.saturating_sub(voted[1].votes)) as f32 / (voted[0].votes as f32))
                .clamp(0.0, 1.0)
        } else if voted.len() == 1 {
            1.0
        } else {
            0.0
        };

        // 3. Scan Saturation: resident_scans / resident_scan_bound
        let scan_saturation = if diag.resident_scan_bound > 0 {
            ((diag.resident_scans as f32) / (diag.resident_scan_bound as f32)).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 4. Candidate Saturation: unique_candidates / selected_candidate_bound
        let candidate_saturation = if diag.selected_candidate_bound > 0 {
            ((diag.unique_candidates as f32) / (diag.selected_candidate_bound as f32))
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        // 5. Witness Dependency: witness_added / total_unique
        let total_unique = diag.unique_candidates.max(1);
        let witness_dependency =
            ((witness_candidates_added as f32) / (total_unique as f32)).clamp(0.0, 1.0);

        // 6. Score Margins: in high dimensions, 0.02 gap is high separation
        let top1_margin = if scored.len() >= 2 {
            (scored[0].1 - scored[1].1).max(0.0)
        } else {
            1.0
        };

        let topk_tail_margin = if scored.len() > k && k > 0 {
            (scored[k - 1].1 - scored[k].1).max(0.0)
        } else if scored.len() >= k {
            0.02
        } else {
            0.0
        };

        let stability = cross_stage_stability.unwrap_or(1.0);

        // 7. Territorial consensus strength (top candidate votes vs expected minimum)
        let _vote_strength = if !voted.is_empty() {
            (voted[0].votes as f32 / 6.0).min(1.0)
        } else {
            0.0
        };

        // 8. Quality and density of top Hermitian match
        let top_fidelity = scored.first().map_or(0.0, |s| s.1);
        let kth_fidelity = if scored.len() >= k && k > 0 {
            scored[k - 1].1
        } else {
            scored.last().map_or(0.0, |s| s.1)
        };

        // Semantic quality: In real embedding space, genuine nearest neighbors typically have top_fidelity in 0.35..0.95
        let semantic_quality = if top_fidelity > 0.30 {
            ((top_fidelity - 0.30) / 0.50).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Top-k boundary separation: Margin between rank 0 and rank (k-1)
        let topk_separation = if top_fidelity > kth_fidelity {
            ((top_fidelity - kth_fidelity) / 0.10).clamp(0.0, 1.0)
        } else {
            0.0
        };

        // Rank correlation between voting and Hermitian scoring:
        // Check if top-voted candidate is in top-10 scored
        let vote_score_concordance = if !voted.is_empty() && !scored.is_empty() {
            let top_voted_slot = voted[0].slot;
            if scored.iter().take(5).any(|s| s.0 == top_voted_slot) {
                1.0f32
            } else if scored.iter().take(20).any(|s| s.0 == top_voted_slot) {
                0.6f32
            } else {
                0.2f32
            }
        } else {
            0.0f32
        };

        // Balanced Composite Confidence
        // 25% Territorial Consensus + 50% Semantic Hermitian Metrics + 25% Margins & Dependencies
        let mut score = 0.10 * vote_concentration
            + 0.10 * vote_margin
            + 0.05 * vote_score_concordance
            + 0.35 * semantic_quality
            + 0.15 * topk_separation
            + 0.10 * (1.0 - witness_dependency * 0.5)
            + 0.15 * (top1_margin.min(0.02) / 0.02);

        if let Some(s) = cross_stage_stability {
            score = score * 0.5 + s * 0.5;
        }

        let score = score.clamp(0.0, 1.0);
        let threshold = match profile {
            RiveroProfile::Fast => 0.42,
            RiveroProfile::Balanced => 0.38,
            RiveroProfile::Strict => 0.32,
        };

        let escalation_recommended = score < threshold
            || scored.len() < k
            || top_fidelity < 0.35
            || (cross_stage_stability.is_some_and(|s| s < 0.50));

        Self {
            score,
            vote_concentration,
            vote_margin,
            scan_saturation,
            candidate_saturation,
            witness_dependency,
            top1_margin,
            topk_tail_margin,
            cross_stage_stability: stability,
            escalation_recommended,
        }
    }
}

#[inline]
const fn cell_probe_count_for(simhash_probes: usize) -> usize {
    RIVERO_FOUNDATIONS * (RIVERO_E8_LOOKUP_CELLS + simhash_probes)
}

#[inline]
const fn candidate_read_bound_for(simhash_probes: usize, per_cell_budget: usize) -> usize {
    cell_probe_count_for(simhash_probes) * per_cell_budget
}

#[inline]
const fn resident_scan_bound_for(simhash_probes: usize) -> usize {
    cell_probe_count_for(simhash_probes) * RIVERO_CELL_CAPACITY
}

const FOUNDATION_SEED: u64 = 0x243f_6a88_85a3_08d3;

thread_local! {
    static RIVERO_CANDIDATES: RefCell<Vec<NodeIndex>> =
        RefCell::new(Vec::with_capacity(RIVERO_QUERY_CANDIDATE_CAP));
    static RIVERO_ADMISSIONS: RefCell<Vec<RankedResident>> =
        RefCell::new(Vec::with_capacity(DEFAULT_SCRATCH_CAPACITY));
    static RIVERO_LOOKUP_CELLS: RefCell<Vec<(usize, u64, u32)>> =
        RefCell::new(Vec::with_capacity(RIVERO_MAX_FOUNDATIONS * RIVERO_BUILD_LOOKUP_CELLS_PER_FOUNDATION));
    static RIVERO_INSERT_CELLS: RefCell<Vec<(usize, u64, u32)>> =
        RefCell::new(Vec::with_capacity(RIVERO_MAX_FOUNDATIONS * RIVERO_INSERT_CELLS_PER_FOUNDATION));
    static RIVERO_VOTED_CANDIDATES: RefCell<Vec<VotedCandidate>> =
        RefCell::new(Vec::with_capacity(DEFAULT_SCRATCH_CAPACITY));
}

/// Precomputed projection matrix for Rivero address compilation.
/// Eliminates O(D × F) dynamic hashing on the ingestion/query hot path.
#[derive(Clone, Debug)]
pub struct RiveroCompiler {
    dimension: usize,
    config: RiveroAddressConfig,
    phase_seeds: Vec<[u64; RIVERO_MAX_FOUNDATIONS]>,
    rot_seeds: Vec<[u64; RIVERO_MAX_FOUNDATIONS]>,
    dim_lanes: Vec<u8>,
}

impl RiveroCompiler {
    /// Precomputes the pseudo-random projection matrices for a given dimension and address config.
    #[must_use]
    pub fn with_config(dimension: usize, config: RiveroAddressConfig) -> Self {
        let foundation_count = (config.foundations as usize).clamp(1, RIVERO_MAX_FOUNDATIONS);
        let mut phase_seeds = vec![[0u64; RIVERO_MAX_FOUNDATIONS]; dimension];
        let mut rot_seeds = vec![[0u64; RIVERO_MAX_FOUNDATIONS]; dimension];
        for (index, (p_row, r_row)) in phase_seeds.iter_mut().zip(rot_seeds.iter_mut()).enumerate()
        {
            for foundation in 0..foundation_count {
                p_row[foundation] = splitmix64((index as u64) ^ foundation_seed(foundation));
                r_row[foundation] = splitmix64(
                    (index as u64) ^ foundation_seed(foundation) ^ 0xd1b5_4a32_d192_ed03,
                );
            }
        }

        let dim_lanes = match config.projection {
            RiveroProjectionMode::GlobalMix => vec![0u8; dimension],
            RiveroProjectionMode::MultiLane { lanes, assignment } => {
                let lanes_count = (lanes as usize).max(1);
                (0..dimension)
                    .map(|i| match assignment {
                        LaneAssignment::Hashed => {
                            (splitmix64((i as u64) ^ 0x9e37_79b9_7f4a_7c15) % (lanes_count as u64))
                                as u8
                        }
                        LaneAssignment::Interleaved => (i % lanes_count) as u8,
                    })
                    .collect()
            }
        };

        Self {
            dimension,
            config,
            phase_seeds,
            rot_seeds,
            dim_lanes,
        }
    }

    /// Precomputes default 24-foundation GlobalMix projection matrices.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        Self::with_config(dimension, RiveroAddressConfig::default())
    }

    /// Returns the address configuration used by this compiler.
    #[must_use]
    pub fn config(&self) -> RiveroAddressConfig {
        self.config
    }

    /// Compiles an embedding into its fixed-size Rivero routing address using precomputed weights.
    #[must_use]
    pub fn compile(&self, data: &[Complex32]) -> RiveroAddress {
        let foundation_count = (self.config.foundations as usize).clamp(1, RIVERO_MAX_FOUNDATIONS);
        let mut foundations = [[0.0f32; 8]; RIVERO_MAX_FOUNDATIONS];
        let mut references = [Complex32::new(0.0, 0.0); RIVERO_MAX_FOUNDATIONS];
        let limit = data.len().min(self.dimension);

        match self.config.projection {
            RiveroProjectionMode::GlobalMix => {
                for (index, value) in data.iter().copied().take(limit).enumerate() {
                    if !value.re.is_finite() || !value.im.is_finite() {
                        continue;
                    }
                    let p_row = &self.phase_seeds[index];
                    for foundation in 0..foundation_count {
                        let mixed = p_row[foundation];
                        let weighted = match (mixed >> 20) & 3 {
                            0 => value,
                            1 => -value,
                            2 => Complex32::new(-value.im, value.re),
                            _ => Complex32::new(value.im, -value.re),
                        };
                        references[foundation] += weighted;
                    }
                }

                let fallback_rotation = canonical_phase_rotation(&data[..limit]);
                let mut rotations = [fallback_rotation; RIVERO_MAX_FOUNDATIONS];
                for foundation in 0..foundation_count {
                    let magnitude = references[foundation].norm();
                    if magnitude.is_finite() && magnitude > 1e-6 {
                        rotations[foundation] = references[foundation].conj() / magnitude;
                    }
                }

                for (index, value) in data.iter().copied().take(limit).enumerate() {
                    if !value.re.is_finite() || !value.im.is_finite() {
                        continue;
                    }
                    let r_row = &self.rot_seeds[index];
                    for foundation in 0..foundation_count {
                        let canonical = value * rotations[foundation];
                        let mixed = r_row[foundation];
                        let re_lane = (mixed & 7) as usize;
                        let im_lane = ((mixed >> 8) & 7) as usize;
                        let re_sign = if mixed & (1 << 16) == 0 { 1.0 } else { -1.0 };
                        let im_sign = if mixed & (1 << 17) == 0 { 1.0 } else { -1.0 };
                        foundations[foundation][re_lane] += canonical.re * re_sign;
                        foundations[foundation][im_lane] += canonical.im * im_sign;
                    }
                }
            }
            RiveroProjectionMode::MultiLane { lanes, .. } => {
                let lanes_count = (lanes as usize).max(1);
                let f_per_lane = (foundation_count / lanes_count).max(1);

                for (index, value) in data.iter().copied().take(limit).enumerate() {
                    if !value.re.is_finite() || !value.im.is_finite() {
                        continue;
                    }
                    let lane = (self.dim_lanes[index] as usize).min(lanes_count - 1);
                    let f_start = lane * f_per_lane;
                    let f_end = (f_start + f_per_lane).min(foundation_count);
                    let p_row = &self.phase_seeds[index];

                    for foundation in f_start..f_end {
                        let mixed = p_row[foundation];
                        let weighted = match (mixed >> 20) & 3 {
                            0 => value,
                            1 => -value,
                            2 => Complex32::new(-value.im, value.re),
                            _ => Complex32::new(value.im, -value.re),
                        };
                        references[foundation] += weighted;
                    }
                }

                let fallback_rotation = canonical_phase_rotation(&data[..limit]);
                let mut rotations = [fallback_rotation; RIVERO_MAX_FOUNDATIONS];
                for foundation in 0..foundation_count {
                    let magnitude = references[foundation].norm();
                    if magnitude.is_finite() && magnitude > 1e-6 {
                        rotations[foundation] = references[foundation].conj() / magnitude;
                    }
                }

                for (index, value) in data.iter().copied().take(limit).enumerate() {
                    if !value.re.is_finite() || !value.im.is_finite() {
                        continue;
                    }
                    let lane = (self.dim_lanes[index] as usize).min(lanes_count - 1);
                    let f_start = lane * f_per_lane;
                    let f_end = (f_start + f_per_lane).min(foundation_count);
                    let r_row = &self.rot_seeds[index];

                    for foundation in f_start..f_end {
                        let canonical = value * rotations[foundation];
                        let mixed = r_row[foundation];
                        let re_lane = (mixed & 7) as usize;
                        let im_lane = ((mixed >> 8) & 7) as usize;
                        let re_sign = if mixed & (1 << 16) == 0 { 1.0 } else { -1.0 };
                        let im_sign = if mixed & (1 << 17) == 0 { 1.0 } else { -1.0 };
                        foundations[foundation][re_lane] += canonical.re * re_sign;
                        foundations[foundation][im_lane] += canonical.im * im_sign;
                    }
                }
            }
        }

        for coords in foundations[..foundation_count].iter_mut() {
            if !normalize8_inplace(coords) {
                *coords = [0.0; 8];
                coords[0] = 1.0;
            }
        }

        RiveroAddress {
            schema_version: RIVERO_SCHEMA_VERSION,
            source_dimension: data.len().min(u32::MAX as usize) as u32,
            foundation_count: foundation_count as u8,
            foundations,
        }
    }
}

/// Fixed-size, global-phase-invariant address compiled from a complex embedding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RiveroAddress {
    /// Address compiler schema. Queries and indexes must use the same version.
    pub schema_version: u16,
    /// Original complex dimensionality, retained for compatibility validation.
    pub source_dimension: u32,
    /// Number of active E8 foundations in this address.
    pub foundation_count: u8,
    /// Independent E8 foundations used for territorial resolution.
    pub foundations: [[f32; 8]; RIVERO_MAX_FOUNDATIONS],
}

impl Serialize for RiveroAddress {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("RiveroAddress", 4)?;
        state.serialize_field("schema_version", &self.schema_version)?;
        state.serialize_field("source_dimension", &self.source_dimension)?;
        state.serialize_field("foundation_count", &self.foundation_count)?;
        let count = (self.foundation_count as usize).min(RIVERO_MAX_FOUNDATIONS);
        state.serialize_field("foundations", &self.foundations[..count])?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for RiveroAddress {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct RiveroAddressHelper {
            schema_version: u16,
            source_dimension: u32,
            #[serde(default = "default_foundation_count")]
            foundation_count: u8,
            foundations: Vec<[f32; 8]>,
        }

        fn default_foundation_count() -> u8 {
            RIVERO_DEFAULT_FOUNDATIONS as u8
        }

        let helper = RiveroAddressHelper::deserialize(deserializer)?;
        let mut foundations = [[0.0f32; 8]; RIVERO_MAX_FOUNDATIONS];
        let count = helper.foundations.len().min(RIVERO_MAX_FOUNDATIONS);
        foundations[..count].copy_from_slice(&helper.foundations[..count]);
        let foundation_count = if helper.foundation_count == 0 {
            count as u8
        } else {
            helper.foundation_count.min(RIVERO_MAX_FOUNDATIONS as u8)
        };

        Ok(RiveroAddress {
            schema_version: helper.schema_version,
            source_dimension: helper.source_dimension,
            foundation_count,
            foundations,
        })
    }
}

impl RiveroAddress {
    /// Compiles an arbitrary-dimensional complex vector into a default 24-foundation Rivero address.
    #[must_use]
    pub fn compile(data: &[Complex32]) -> Self {
        let compiler = RiveroCompiler::new(data.len());
        compiler.compile(data)
    }

    /// Compiles an arbitrary-dimensional complex vector with a customized address configuration.
    #[must_use]
    pub fn compile_with_config(data: &[Complex32], config: RiveroAddressConfig) -> Self {
        let compiler = RiveroCompiler::with_config(data.len(), config);
        compiler.compile(data)
    }

    /// Returns the active E8 foundation coordinates.
    #[must_use]
    #[inline]
    pub fn active_foundations(&self) -> &[[f32; 8]] {
        let count = (self.foundation_count as usize).min(RIVERO_MAX_FOUNDATIONS);
        &self.foundations[..count]
    }

    /// Returns the fixed upper bound on resident reads for a per-cell budget.
    #[must_use]
    pub const fn candidate_read_bound(per_cell_budget: usize) -> usize {
        candidate_read_bound_for(RIVERO_SIMHASH_QUERY_PROBES, per_cell_budget)
    }

    /// Returns the fixed upper bound on compact resident scans for one route.
    ///
    /// Dense cells are scanned in full so their query-specific best residents can
    /// be selected. Only `candidate_read_bound(budget)` residents are admitted to
    /// the exact reranker.
    #[must_use]
    pub const fn resident_scan_bound() -> usize {
        resident_scan_bound_for(RIVERO_SIMHASH_QUERY_PROBES)
    }

    /// Returns the exact number of territory cells probed by one strict route.
    #[must_use]
    pub const fn cell_probe_count() -> usize {
        cell_probe_count_for(RIVERO_SIMHASH_QUERY_PROBES)
    }
}

/// Work counters emitted by a single fixed-budget territory resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiveroRouteDiagnostics {
    /// Exact number of territory cells inspected.
    pub cells_probed: usize,
    /// Total resident entries admitted before deduplication.
    pub resident_reads: usize,
    /// Compact resident codes inspected before query-adaptive admission.
    pub resident_scans: usize,
    /// Distinct arena slots produced by the route.
    pub unique_candidates: usize,
    /// Distinct slots observed before global collision-vote selection.
    pub raw_unique_candidates: usize,
    /// Configured hard ceiling for resident reads.
    pub candidate_read_bound: usize,
    /// Configured hard ceiling for compact resident scans.
    pub resident_scan_bound: usize,
    /// Hard ceiling on raw distinct candidates before vote selection.
    pub raw_unique_candidate_bound: usize,
    /// Hard ceiling on candidates exposed to the visitor.
    pub selected_candidate_bound: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VotedCandidate {
    pub slot: NodeIndex,
    pub votes: u32,
    pub dot_sum: i32,
    pub l1_sum: u32,
}

/// Eight-byte resident record: one arena slot plus eight signed 3-bit projected
/// coordinates and an 8-bit insertion-affinity rank.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub(crate) struct CellResident {
    pub(crate) slot: NodeIndex,
    pub(crate) fine_code: u32,
}

impl CellResident {
    #[inline]
    pub(crate) const fn affinity(self) -> u8 {
        (self.fine_code >> 24) as u8
    }

    #[inline]
    pub(crate) const fn projected_code(self) -> u32 {
        self.fine_code & 0x00ff_ffff
    }
}

#[derive(Clone, Copy)]
struct RankedResident {
    dot: i16,
    l1_distance: u16,
    slot: NodeIndex,
}

impl RankedResident {
    const EMPTY: Self = Self {
        dot: i16::MIN,
        l1_distance: u16::MAX,
        slot: NodeIndex::MAX,
    };
}

#[derive(Debug, Default)]
pub(crate) struct CellSlots {
    /// Affinity elites occupy `[..elite_len]` in descending order; diversity
    /// residents occupy the remainder in ascending minhash-priority order.
    pub(crate) slots: Vec<CellResident>,
    pub(crate) elite_len: usize,
    pub(crate) overflowed: bool,
}

impl CellSlots {
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn insert(&mut self, key: u64, fine_code: u32, slot: NodeIndex) -> bool {
        self.insert_with_limits(
            key,
            fine_code,
            slot,
            RIVERO_CELL_CAPACITY,
            RIVERO_CELL_AFFINITY_ELITES,
        )
    }

    #[inline]
    pub(crate) fn insert_with_limits(
        &mut self,
        key: u64,
        fine_code: u32,
        slot: NodeIndex,
        capacity: usize,
        affinity_elites: usize,
    ) -> bool {
        if self.slots.iter().any(|resident| resident.slot == slot) {
            return false;
        }

        let mut resident = CellResident { slot, fine_code };
        if self.elite_len < affinity_elites {
            self.insert_elite(resident);
            return true;
        }

        let elite_tail = self.slots[self.elite_len - 1];
        if affinity_precedes(resident, elite_tail) {
            resident = self.slots.remove(self.elite_len - 1);
            self.elite_len -= 1;
            self.insert_elite(CellResident { slot, fine_code });
            let _ = self.insert_diversity(key, resident, capacity, affinity_elites);
            return true;
        }

        if self.insert_diversity(key, resident, capacity, affinity_elites) {
            return true;
        }

        if self.slots.len() >= capacity {
            self.overflowed = true;
        }
        false
    }

    #[inline]
    fn evict(&mut self, slot: NodeIndex) {
        let Some(position) = self.slots.iter().position(|resident| resident.slot == slot) else {
            return;
        };
        self.slots.remove(position);
        if position < self.elite_len {
            self.elite_len -= 1;
            if self.elite_len < RIVERO_CELL_AFFINITY_ELITES && self.elite_len < self.slots.len() {
                let promote = (self.elite_len..self.slots.len())
                    .max_by(|&lhs, &rhs| affinity_order(self.slots[lhs], self.slots[rhs]).reverse())
                    .expect("non-empty diversity range");
                let resident = self.slots.remove(promote);
                self.insert_elite(resident);
            }
        }
    }

    #[inline]
    fn insert_elite(&mut self, resident: CellResident) {
        let position = self.slots[..self.elite_len]
            .partition_point(|&existing| affinity_precedes(existing, resident));
        self.slots.insert(position, resident);
        self.elite_len += 1;
    }

    #[inline]
    fn insert_diversity(
        &mut self,
        key: u64,
        resident: CellResident,
        capacity: usize,
        affinity_elites: usize,
    ) -> bool {
        let max_diversity = capacity.saturating_sub(affinity_elites);
        let diversity_len = self.slots.len() - self.elite_len;
        let priority = diversity_priority(key, resident);
        let position = self.slots[self.elite_len..].partition_point(|&existing| {
            let existing_priority = diversity_priority(key, existing);
            existing_priority < priority
                || (existing_priority == priority && existing.slot < resident.slot)
        });
        if diversity_len >= max_diversity && position >= max_diversity {
            self.overflowed = true;
            return false;
        }

        self.slots.insert(self.elite_len + position, resident);
        if self.slots.len() > capacity {
            self.slots.pop();
            self.overflowed = true;
        }
        true
    }

    #[inline]
    pub fn merge_from(
        &mut self,
        key: u64,
        other: &CellSlots,
        capacity: usize,
        affinity_elites: usize,
    ) {
        let mut combined: smallvec::SmallVec<[CellResident; 128]> = smallvec::SmallVec::new();
        combined.extend_from_slice(&self.slots);
        for &r in &other.slots {
            if !combined.iter().any(|existing| existing.slot == r.slot) {
                combined.push(r);
            }
        }
        let had_overflow = self.overflowed || other.overflowed || combined.len() > capacity;

        // Sort all candidates by affinity descending to select top elites
        combined.sort_unstable_by(|&lhs, &rhs| affinity_order(lhs, rhs));

        let elite_count = combined.len().min(affinity_elites);
        let mut final_slots: Vec<CellResident> = Vec::with_capacity(capacity);
        final_slots.extend_from_slice(&combined[..elite_count]);

        // Remaining candidates compete for diversity slots
        let remaining = &mut combined[elite_count..];
        remaining.sort_unstable_by(|&lhs, &rhs| {
            let p_lhs = diversity_priority(key, lhs);
            let p_rhs = diversity_priority(key, rhs);
            p_lhs.cmp(&p_rhs).then_with(|| lhs.slot.cmp(&rhs.slot))
        });

        let diversity_cap = capacity.saturating_sub(affinity_elites);
        let diversity_count = remaining.len().min(diversity_cap);
        final_slots.extend_from_slice(&remaining[..diversity_count]);

        self.slots = final_slots;
        self.elite_len = elite_count;
        self.overflowed = had_overflow;
    }

    #[inline]
    fn append_best(
        &self,
        query_code: u32,
        budget: usize,
        candidates: &mut Vec<RankedResident>,
    ) -> (usize, usize) {
        let scanned = self.slots.len();
        let admitted = scanned.min(budget);
        if scanned <= budget {
            candidates.extend(self.slots.iter().map(|resident| {
                let (dot, l1_distance) =
                    projected_similarity(query_code, resident.projected_code());
                RankedResident {
                    dot,
                    l1_distance,
                    slot: resident.slot,
                }
            }));
            return (scanned, admitted);
        }

        let mut ranked = [RankedResident::EMPTY; RIVERO_CELL_CAPACITY];
        for (rank, resident) in ranked.iter_mut().zip(self.slots.iter().copied()) {
            let (dot, l1_distance) = projected_similarity(query_code, resident.projected_code());
            *rank = RankedResident {
                dot,
                l1_distance,
                slot: resident.slot,
            };
        }
        ranked[..scanned].select_nth_unstable_by(budget - 1, ranked_resident_order);
        ranked[..budget].sort_unstable_by(ranked_resident_order);
        candidates.extend_from_slice(&ranked[..budget]);
        (scanned, admitted)
    }
}

/// Concurrent bounded territory index for [`RiveroAddress`] values.
pub struct RiveroTerritoryIndex {
    pub(crate) stripes: Box<[RwLock<HashMap<u64, CellSlots>>]>,
    inserts: AtomicU64,
    overflows: AtomicU64,
}

impl Default for RiveroTerritoryIndex {
    fn default() -> Self {
        let stripes = (0..RIVERO_STRIPES)
            .map(|_| RwLock::new(HashMap::new()))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            stripes,
            inserts: AtomicU64::new(0),
            overflows: AtomicU64::new(0),
        }
    }
}

impl RiveroTerritoryIndex {
    /// Creates an empty territory index.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a node in the bounded C(7,3) cells of every foundation using strict default config.
    pub fn insert(&self, address: &RiveroAddress, slot: NodeIndex) {
        self.insert_with_config(address, slot, &RiveroConfig::strict_default());
    }

    /// Registers a node in the cells defined by a specific [`RiveroConfig`].
    pub fn insert_with_config(
        &self,
        address: &RiveroAddress,
        slot: NodeIndex,
        config: &RiveroConfig,
    ) {
        RIVERO_INSERT_CELLS.with(|scratch| {
            let mut cells = scratch.borrow_mut();
            cells.clear();
            let foundations = config.foundations.min(address.foundation_count as usize);
            for (foundation, coords) in address.foundations[..foundations].iter().enumerate() {
                let projected_code = pack_projected_code(coords);
                let (signatures, count) = insert_sigs(coords, 0);
                for &signature in &signatures[..count] {
                    let key = cell_key(foundation, signature);
                    let affinity = cell_affinity(signature, coords);
                    cells.push((
                        stripe_for(key),
                        key,
                        pack_fine_code(projected_code, affinity, false),
                    ));
                }
                let (signature, margin, _) = simhash_signature(coords, foundation);
                let key = simhash_cell_key(foundation, signature);
                cells.push((
                    stripe_for(key),
                    key,
                    pack_fine_code(projected_code, margin, true),
                ));
            }
            cells.sort_unstable_by_key(|&(stripe, key, _)| (stripe, key));

            let mut cursor = 0usize;
            while cursor < cells.len() {
                let stripe_index = cells[cursor].0;
                let mut stripe = self.stripes[stripe_index].write();
                while cursor < cells.len() && cells[cursor].0 == stripe_index {
                    let (_, key, fine_code) = cells[cursor];
                    let cell = stripe.entry(key).or_default();
                    let overflow_before = cell.overflowed;
                    if cell.insert_with_limits(
                        key,
                        fine_code,
                        slot,
                        config.cell_capacity,
                        config.affinity_elites,
                    ) {
                        self.inserts.fetch_add(1, Ordering::Relaxed);
                    }
                    if !overflow_before && cell.overflowed {
                        self.overflows.fetch_add(1, Ordering::Relaxed);
                    }
                    cursor += 1;
                }
            }
        });
    }

    /// Removes a node from every cell generated by its address.
    pub fn evict(&self, address: &RiveroAddress, slot: NodeIndex) {
        RIVERO_INSERT_CELLS.with(|scratch| {
            let mut cells = scratch.borrow_mut();
            cells.clear();
            for (foundation, coords) in address.active_foundations().iter().enumerate() {
                let (signatures, count) = insert_sigs(coords, 0);
                for &signature in &signatures[..count] {
                    let key = cell_key(foundation, signature);
                    cells.push((stripe_for(key), key, 0));
                }
                let (signature, _, _) = simhash_signature(coords, foundation);
                let key = simhash_cell_key(foundation, signature);
                cells.push((stripe_for(key), key, 0));
            }
            cells.sort_unstable_by_key(|&(stripe, key, _)| (stripe, key));

            let mut cursor = 0usize;
            while cursor < cells.len() {
                let stripe_index = cells[cursor].0;
                let mut stripe = self.stripes[stripe_index].write();
                while cursor < cells.len() && cells[cursor].0 == stripe_index {
                    let key = cells[cursor].1;
                    if let Some(cell) = stripe.get_mut(&key) {
                        cell.evict(slot);
                    }
                    cursor += 1;
                }
            }
        });
    }

    /// Resolves a sorted, deduplicated candidate slice under a fixed resident budget.
    ///
    /// The visitor must not recursively invoke another Rivero lookup on the same
    /// thread because the candidate slice borrows a thread-local scratchpad.
    pub fn with_candidates<R>(
        &self,
        address: &RiveroAddress,
        per_cell_budget: usize,
        visitor: impl FnOnce(&[NodeIndex], RiveroRouteDiagnostics) -> R,
    ) -> R {
        let mut config = RiveroConfig::strict_default();
        config.cell_budget = per_cell_budget;
        self.with_candidates_config(address, &config, visitor)
    }

    /// Resolves a higher-probe, tightly capped candidate set for bounded witness
    /// construction. This never changes the lower serving-time proof bounds.
    pub(crate) fn with_candidates_for_build<R>(
        &self,
        address: &RiveroAddress,
        per_cell_budget: usize,
        visitor: impl FnOnce(&[NodeIndex], RiveroRouteDiagnostics) -> R,
    ) -> R {
        let mut config = RiveroConfig::strict_default();
        config.cell_budget = per_cell_budget;
        config.simhash_query_probes = RIVERO_SIMHASH_BUILD_PROBES;
        config.query_candidate_cap = RIVERO_BUILD_CANDIDATE_CAP;
        self.with_candidates_config(address, &config, visitor)
    }

    /// Resolves candidate slice under a flexible [`RiveroConfig`] configuration.
    pub fn with_candidates_config<R>(
        &self,
        address: &RiveroAddress,
        config: &RiveroConfig,
        visitor: impl FnOnce(&[NodeIndex], RiveroRouteDiagnostics) -> R,
    ) -> R {
        self.with_voted_candidates_config(address, config, |candidates, _voted, diagnostics| {
            visitor(candidates, diagnostics)
        })
    }

    /// Resolves candidate slice alongside the full pre-cap ranked voted candidates.
    pub fn with_voted_candidates_config<R>(
        &self,
        address: &RiveroAddress,
        config: &RiveroConfig,
        visitor: impl FnOnce(&[NodeIndex], &[VotedCandidate], RiveroRouteDiagnostics) -> R,
    ) -> R {
        let budget = config.cell_budget.clamp(1, config.cell_capacity);
        let simhash_probes = config.simhash_query_probes;
        let selected_cap = config.query_candidate_cap;
        let foundations_count = config.foundations.min(address.foundation_count as usize);

        RIVERO_ADMISSIONS.with(|scratch| {
            let mut admissions = scratch.borrow_mut();
            admissions.clear();

            let required = config.candidate_read_bound();
            let scan_bound = config.resident_scan_bound();
            let additional = required.saturating_sub(admissions.capacity());
            if additional > 0 {
                admissions.reserve_exact(additional);
            }

            RIVERO_LOOKUP_CELLS.with(|cell_scratch| {
                let mut cells = cell_scratch.borrow_mut();
                cells.clear();
                for (foundation, coords) in
                    address.foundations[..foundations_count].iter().enumerate()
                {
                    let query_code = pack_projected_code(coords);
                    let (signatures, count) = lookup_sigs(coords, 0);
                    for &signature in &signatures[..count] {
                        let key = cell_key(foundation, signature);
                        cells.push((stripe_for(key), key, query_code));
                    }
                    let (signature, _, margins) = simhash_signature(coords, foundation);
                    let probes = simhash_probe_signatures(signature, &margins);
                    for &probe in probes.iter().take(simhash_probes) {
                        let key = simhash_cell_key(foundation, probe);
                        cells.push((stripe_for(key), key, query_code));
                    }
                }
                cells.sort_unstable_by_key(|&(stripe, key, _)| (stripe, key));

                let mut resident_reads = 0usize;
                let mut resident_scans = 0usize;
                let mut cursor = 0usize;
                while cursor < cells.len() {
                    let stripe_index = cells[cursor].0;
                    let stripe = self.stripes[stripe_index].read();
                    while cursor < cells.len() && cells[cursor].0 == stripe_index {
                        let key = cells[cursor].1;
                        let query_code = cells[cursor].2;
                        if let Some(cell) = stripe.get(&key) {
                            let (scanned, admitted) =
                                cell.append_best(query_code, budget, &mut admissions);
                            resident_scans += scanned;
                            resident_reads += admitted;
                        }
                        cursor += 1;
                    }
                }

                admissions.sort_unstable_by_key(|admission| admission.slot);
                RIVERO_VOTED_CANDIDATES.with(|vote_scratch| {
                    let mut voted = vote_scratch.borrow_mut();
                    voted.clear();
                    let mut admission_cursor = 0usize;
                    while admission_cursor < admissions.len() {
                        let slot = admissions[admission_cursor].slot;
                        let mut votes = 0u32;
                        let mut dot_sum = 0i32;
                        let mut l1_sum = 0u32;
                        while admission_cursor < admissions.len()
                            && admissions[admission_cursor].slot == slot
                        {
                            let admission = admissions[admission_cursor];
                            votes = votes.saturating_add(1);
                            dot_sum = dot_sum.saturating_add(i32::from(admission.dot));
                            l1_sum = l1_sum.saturating_add(u32::from(admission.l1_distance));
                            admission_cursor += 1;
                        }
                        voted.push(VotedCandidate {
                            slot,
                            votes,
                            dot_sum,
                            l1_sum,
                        });
                    }
                    let raw_unique_candidates = voted.len();
                    voted.sort_unstable_by(voted_candidate_order);

                    RIVERO_CANDIDATES.with(|candidate_scratch| {
                        let mut candidates = candidate_scratch.borrow_mut();
                        candidates.clear();
                        candidates.extend(
                            voted
                                .iter()
                                .take(selected_cap)
                                .map(|candidate| candidate.slot),
                        );
                        candidates.sort_unstable();
                        let diagnostics = RiveroRouteDiagnostics {
                            cells_probed: cells.len(),
                            resident_reads,
                            resident_scans,
                            unique_candidates: candidates.len(),
                            raw_unique_candidates,
                            candidate_read_bound: required,
                            resident_scan_bound: scan_bound,
                            raw_unique_candidate_bound: required,
                            selected_candidate_bound: selected_cap,
                        };
                        visitor(&candidates, &voted, diagnostics)
                    })
                })
            })
        })
    }

    /// Removes all routing cells and resets diagnostics.
    pub fn clear(&self) {
        for stripe in &self.stripes {
            stripe.write().clear();
        }
        self.inserts.store(0, Ordering::Relaxed);
        self.overflows.store(0, Ordering::Relaxed);
    }

    /// Returns the number of populated territory cells.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.stripes.iter().map(|stripe| stripe.read().len()).sum()
    }

    /// Returns the number of cells that reached their bounded capacity.
    #[must_use]
    pub fn overflow_count(&self) -> u64 {
        self.overflows.load(Ordering::Relaxed)
    }

    /// Constructs a territory index from pre-reduced parallel stripes.
    #[must_use]
    pub(crate) fn from_stripes(
        stripes: Box<[RwLock<HashMap<u64, CellSlots>>]>,
        inserts: u64,
        overflows: u64,
    ) -> Self {
        Self {
            stripes,
            inserts: AtomicU64::new(inserts),
            overflows: AtomicU64::new(overflows),
        }
    }

    /// Replaces the contents of this territory index with another in-place.
    pub fn replace_from(&self, other: RiveroTerritoryIndex) {
        for (dest, src) in self.stripes.iter().zip(other.stripes.iter()) {
            *dest.write() = std::mem::take(&mut *src.write());
        }
        self.inserts
            .store(other.inserts.load(Ordering::Relaxed), Ordering::Relaxed);
        self.overflows
            .store(other.overflows.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    /// Computes a canonical cryptographic fingerprint over all populated cells and resident codes.
    #[must_use]
    #[allow(clippy::type_complexity)]
    pub fn structural_fingerprint(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        let mut all_cells: Vec<(u64, Vec<(NodeIndex, u32)>, usize, bool)> = Vec::new();
        for stripe in &self.stripes {
            let guard = stripe.read();
            for (&key, cell) in guard.iter() {
                let residents: Vec<(NodeIndex, u32)> =
                    cell.slots.iter().map(|r| (r.slot, r.fine_code)).collect();
                all_cells.push((key, residents, cell.elite_len, cell.overflowed));
            }
        }
        all_cells.sort_unstable_by_key(|c| c.0);

        for (key, residents, elite_len, overflowed) in all_cells {
            hasher.update(key.to_le_bytes());
            hasher.update((elite_len as u32).to_le_bytes());
            hasher.update([if overflowed { 1u8 } else { 0u8 }]);
            hasher.update((residents.len() as u32).to_le_bytes());
            for (slot, fine_code) in residents {
                hasher.update(slot.to_le_bytes());
                hasher.update(fine_code.to_le_bytes());
            }
        }

        hasher.finalize().into()
    }
}

/// Stateful incremental routing state for progressive, zero-redundancy multi-stage escalation.
#[derive(Clone, Debug, Default)]
pub struct AdaptiveRouteState {
    /// Foundations already inspected.
    pub foundations_probed: usize,
    /// SimHash probe signatures already inspected per foundation.
    pub simhash_probes_per_foundation: usize,
    /// Cumulative compact resident codes scanned across all stages.
    pub cumulative_scans: usize,
    /// Cumulative resident reads / admissions across all stages.
    pub cumulative_reads: usize,
    /// Cumulative unique territory cells inspected across all stages.
    pub cells_visited: usize,
    /// Candidate vote accumulator: slot -> (votes, dot_sum, l1_sum).
    pub voted_candidates: HashMap<NodeIndex, (u32, i32, u32)>,
    /// Voted candidates sorted by vote rank in the most recent stage.
    pub current_voted: Vec<VotedCandidate>,
    /// Current routing diagnostics.
    pub current_diagnostics: RiveroRouteDiagnostics,
}

impl AdaptiveRouteState {
    /// Creates a fresh incremental routing state.
    #[must_use]
    pub fn new() -> Self {
        Self {
            foundations_probed: 0,
            simhash_probes_per_foundation: 0,
            cumulative_scans: 0,
            cumulative_reads: 0,
            cells_visited: 0,
            voted_candidates: HashMap::with_capacity(1024),
            current_voted: Vec::with_capacity(512),
            current_diagnostics: RiveroRouteDiagnostics::default(),
        }
    }

    /// Progressively expands candidate routing to `target_profile` probing only newly required cells.
    pub fn expand_to_profile(
        &mut self,
        territory: &RiveroTerritoryIndex,
        address: &RiveroAddress,
        target_profile: RiveroProfile,
    ) {
        self.expand_to_config(territory, address, target_profile.config());
    }

    /// Progressively expands candidate routing to a custom [`RiveroConfig`] probing newly required cells.
    pub fn expand_to_config(
        &mut self,
        territory: &RiveroTerritoryIndex,
        address: &RiveroAddress,
        config: RiveroConfig,
    ) {
        let target_foundations = config.foundations.min(address.foundation_count as usize);
        let target_probes = config.simhash_query_probes;
        let budget = config.cell_budget.clamp(1, config.cell_capacity);

        // 1. Identify delta lookup cells:
        //    a) New foundations (self.foundations_probed .. target_foundations) -> full E8 lookup + target_probes
        //    b) Previously probed foundations (0 .. self.foundations_probed) -> only delta SimHash probes (self.simhash_probes_per_foundation .. target_probes)
        let mut delta_cells: Vec<(usize, u64, u32)> = Vec::new();

        // (a) Delta probes on already probed foundations
        if target_probes > self.simhash_probes_per_foundation && self.foundations_probed > 0 {
            for (foundation, coords) in address.foundations[..self.foundations_probed]
                .iter()
                .enumerate()
            {
                let query_code = pack_projected_code(coords);
                let (signature, _, margins) = simhash_signature(coords, foundation);
                let probes = simhash_probe_signatures(signature, &margins);
                for &probe in probes
                    .iter()
                    .take(target_probes)
                    .skip(self.simhash_probes_per_foundation)
                {
                    let key = simhash_cell_key(foundation, probe);
                    delta_cells.push((stripe_for(key), key, query_code));
                }
            }
        }

        // (b) Entirely new foundations
        if target_foundations > self.foundations_probed {
            for (foundation_idx, coords) in address.foundations
                [self.foundations_probed..target_foundations]
                .iter()
                .enumerate()
            {
                let foundation = self.foundations_probed + foundation_idx;
                let query_code = pack_projected_code(coords);
                let (signatures, count) = lookup_sigs(coords, 0);
                for &signature in &signatures[..count] {
                    let key = cell_key(foundation, signature);
                    delta_cells.push((stripe_for(key), key, query_code));
                }
                let (signature, _, margins) = simhash_signature(coords, foundation);
                let probes = simhash_probe_signatures(signature, &margins);
                for &probe in probes.iter().take(target_probes) {
                    let key = simhash_cell_key(foundation, probe);
                    delta_cells.push((stripe_for(key), key, query_code));
                }
            }
        }

        delta_cells.sort_unstable_by_key(|&(stripe, key, _)| (stripe, key));

        // 2. Scan delta cells and admit residents
        let mut delta_admissions: Vec<RankedResident> =
            Vec::with_capacity(delta_cells.len() * budget);
        let mut resident_scans = 0usize;
        let mut resident_reads = 0usize;

        let mut cursor = 0usize;
        while cursor < delta_cells.len() {
            let stripe_index = delta_cells[cursor].0;
            let stripe = territory.stripes[stripe_index].read();
            while cursor < delta_cells.len() && delta_cells[cursor].0 == stripe_index {
                let key = delta_cells[cursor].1;
                let query_code = delta_cells[cursor].2;
                if let Some(cell) = stripe.get(&key) {
                    let (scanned, admitted) =
                        cell.append_best(query_code, budget, &mut delta_admissions);
                    resident_scans += scanned;
                    resident_reads += admitted;
                }
                cursor += 1;
            }
        }

        self.cumulative_scans += resident_scans;
        self.cumulative_reads += resident_reads;
        self.cells_visited += delta_cells.len();
        self.foundations_probed = self.foundations_probed.max(target_foundations);
        self.simhash_probes_per_foundation = self.simhash_probes_per_foundation.max(target_probes);

        // 3. Accumulate delta admissions into state votes
        for admission in delta_admissions {
            let entry = self
                .voted_candidates
                .entry(admission.slot)
                .or_insert((0, 0, 0));
            entry.0 = entry.0.saturating_add(1);
            entry.1 = entry.1.saturating_add(i32::from(admission.dot));
            entry.2 = entry.2.saturating_add(u32::from(admission.l1_distance));
        }

        // 4. Materialize sorted VotedCandidates
        self.current_voted.clear();
        for (&slot, &(votes, dot_sum, l1_sum)) in &self.voted_candidates {
            self.current_voted.push(VotedCandidate {
                slot,
                votes,
                dot_sum,
                l1_sum,
            });
        }
        self.current_voted.sort_unstable_by(voted_candidate_order);

        let raw_unique_candidates = self.current_voted.len();
        let selected_cap = config.query_candidate_cap;

        self.current_diagnostics = RiveroRouteDiagnostics {
            cells_probed: self.cells_visited,
            resident_reads: self.cumulative_reads,
            resident_scans: self.cumulative_scans,
            unique_candidates: raw_unique_candidates.min(selected_cap),
            raw_unique_candidates,
            candidate_read_bound: config.candidate_read_bound(),
            resident_scan_bound: config.resident_scan_bound(),
            raw_unique_candidate_bound: config.candidate_read_bound(),
            selected_candidate_bound: selected_cap,
        };
    }
}

#[inline]
fn canonical_phase_rotation(data: &[Complex32]) -> Complex32 {
    let anchor = data
        .iter()
        .copied()
        .filter(|value| value.re.is_finite() && value.im.is_finite())
        .max_by(|lhs, rhs| lhs.norm_sqr().total_cmp(&rhs.norm_sqr()))
        .unwrap_or_else(|| Complex32::new(1.0, 0.0));
    let magnitude = anchor.norm();
    if !magnitude.is_finite() || magnitude <= f32::EPSILON {
        Complex32::new(1.0, 0.0)
    } else {
        anchor.conj() / magnitude
    }
}

#[inline]
fn normalize8_inplace(coords: &mut [f32; 8]) -> bool {
    let norm_sq = coords.iter().map(|value| value * value).sum::<f32>();
    if !norm_sq.is_finite() || norm_sq <= f32::EPSILON {
        return false;
    }
    let inverse = norm_sq.sqrt().recip();
    for value in coords {
        *value *= inverse;
    }
    true
}

#[inline]
pub(crate) fn pack_projected_code(coords: &[f32; 8]) -> u32 {
    let mut code = 0u32;
    for (lane, coordinate) in coords.iter().copied().enumerate() {
        let quantized = (coordinate.clamp(-1.0, 1.0) * 3.0).round().clamp(-3.0, 3.0) as i8;
        let encoded = u32::from((quantized + 3) as u8);
        code |= encoded << (lane * 3);
    }
    code
}

#[inline]
fn pack_fine_code(projected_code: u32, affinity: f32, simhash: bool) -> u32 {
    let normalized = if !affinity.is_finite() {
        0.0
    } else if simhash {
        let maximum = RIVERO_SIMHASH_BITS as f32 * 8.0f32.sqrt();
        (affinity / maximum).clamp(0.0, 1.0)
    } else {
        ((affinity + 3.0) / 6.0).clamp(0.0, 1.0)
    };
    let affinity_rank = (normalized * 255.0).round() as u32;
    (projected_code & 0x00ff_ffff) | (affinity_rank << 24)
}

#[inline]
pub(crate) fn projected_similarity(lhs: u32, rhs: u32) -> (i16, u16) {
    let mut dot = 0i16;
    let mut l1_distance = 0u16;
    for lane in 0..8 {
        let shift = lane * 3;
        let left = ((lhs >> shift) & 7) as i16 - 3;
        let right = ((rhs >> shift) & 7) as i16 - 3;
        dot += left * right;
        l1_distance += left.abs_diff(right);
    }
    (dot, l1_distance)
}

#[inline]
fn affinity_order(lhs: CellResident, rhs: CellResident) -> std::cmp::Ordering {
    rhs.affinity()
        .cmp(&lhs.affinity())
        .then_with(|| lhs.slot.cmp(&rhs.slot))
}

#[inline]
fn affinity_precedes(lhs: CellResident, rhs: CellResident) -> bool {
    affinity_order(lhs, rhs).is_lt()
}

#[inline]
fn diversity_priority(key: u64, resident: CellResident) -> u64 {
    splitmix64(
        key ^ u64::from(resident.slot).wrapping_mul(0xd6e8_feb8_6659_fd93)
            ^ u64::from(resident.projected_code()).rotate_left(23),
    )
}

#[inline]
fn ranked_resident_order(lhs: &RankedResident, rhs: &RankedResident) -> std::cmp::Ordering {
    rhs.dot
        .cmp(&lhs.dot)
        .then_with(|| lhs.l1_distance.cmp(&rhs.l1_distance))
        .then_with(|| lhs.slot.cmp(&rhs.slot))
}

#[inline]
fn voted_candidate_order(lhs: &VotedCandidate, rhs: &VotedCandidate) -> std::cmp::Ordering {
    rhs.votes
        .cmp(&lhs.votes)
        .then_with(|| rhs.dot_sum.cmp(&lhs.dot_sum))
        .then_with(|| lhs.l1_sum.cmp(&rhs.l1_sum))
        .then_with(|| lhs.slot.cmp(&rhs.slot))
}

#[inline(always)]
pub(crate) const fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[inline(always)]
const fn foundation_seed(foundation: usize) -> u64 {
    splitmix64(FOUNDATION_SEED ^ (foundation as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
}

#[inline]
pub(crate) fn simhash_signature(
    coords: &[f32; 8],
    foundation: usize,
) -> (u16, f32, [f32; RIVERO_SIMHASH_BITS]) {
    let mut signature = 0u16;
    let mut margin = 0.0f32;
    let mut margins = [0.0f32; RIVERO_SIMHASH_BITS];
    for (bit, bit_margin) in margins.iter_mut().enumerate() {
        let bit_seed = splitmix64(
            foundation_seed(foundation) ^ (bit as u64).wrapping_mul(0xa076_1d64_78bd_642f),
        );
        let mut projection = 0.0f32;
        for (lane, coordinate) in coords.iter().copied().enumerate() {
            let lane_seed =
                splitmix64(bit_seed ^ (lane as u64).wrapping_mul(0xe703_7ed1_a0b4_28db));
            projection += if lane_seed & 1 == 0 {
                coordinate
            } else {
                -coordinate
            };
        }
        if projection >= 0.0 {
            signature |= 1u16 << bit;
        }
        *bit_margin = projection.abs();
        margin += *bit_margin;
    }
    (signature, margin, margins)
}

#[inline]
pub(crate) fn simhash_probe_signatures(
    signature: u16,
    margins: &[f32; RIVERO_SIMHASH_BITS],
) -> [u16; RIVERO_SIMHASH_BUILD_PROBES] {
    let mut pool = [(0.0f32, 0u16); RIVERO_SIMHASH_PROBE_POOL];
    let mut cursor = 0usize;
    pool[cursor] = (0.0, 0);
    cursor += 1;
    for (first, &first_margin) in margins.iter().enumerate() {
        pool[cursor] = (first_margin, 1u16 << first);
        cursor += 1;
    }
    for (first, &first_margin) in margins.iter().enumerate() {
        for (second, &second_margin) in margins.iter().enumerate().skip(first + 1) {
            pool[cursor] = (
                first_margin + second_margin,
                (1u16 << first) | (1u16 << second),
            );
            cursor += 1;
        }
    }
    for (first, &first_margin) in margins.iter().enumerate() {
        for (second, &second_margin) in margins.iter().enumerate().skip(first + 1) {
            for (third, &third_margin) in margins.iter().enumerate().skip(second + 1) {
                pool[cursor] = (
                    first_margin + second_margin + third_margin,
                    (1u16 << first) | (1u16 << second) | (1u16 << third),
                );
                cursor += 1;
            }
        }
    }
    for (first, &first_margin) in margins.iter().enumerate() {
        for (second, &second_margin) in margins.iter().enumerate().skip(first + 1) {
            for (third, &third_margin) in margins.iter().enumerate().skip(second + 1) {
                for (fourth, &fourth_margin) in margins.iter().enumerate().skip(third + 1) {
                    pool[cursor] = (
                        first_margin + second_margin + third_margin + fourth_margin,
                        (1u16 << first) | (1u16 << second) | (1u16 << third) | (1u16 << fourth),
                    );
                    cursor += 1;
                }
            }
        }
    }
    debug_assert_eq!(cursor, RIVERO_SIMHASH_PROBE_POOL);
    pool.sort_unstable_by(|lhs, rhs| lhs.0.total_cmp(&rhs.0).then_with(|| lhs.1.cmp(&rhs.1)));

    let mut probes = [0u16; RIVERO_SIMHASH_BUILD_PROBES];
    for (probe, &(_, mask)) in probes.iter_mut().zip(pool.iter()) {
        *probe = signature ^ mask;
    }
    probes
}

#[inline(always)]
pub(crate) const fn cell_key(foundation: usize, signature: u32) -> u64 {
    ((foundation as u64) << 32) | signature as u64
}

#[inline(always)]
pub(crate) const fn simhash_cell_key(foundation: usize, signature: u16) -> u64 {
    cell_key(foundation, 0x8000_0000 | signature as u32)
}

#[inline(always)]
pub(crate) const fn stripe_for(key: u64) -> usize {
    (splitmix64(key) as usize) & (RIVERO_STRIPES - 1)
}

/// Pairwise cosine-similarity matrix over all 240 normalized E8 roots.
static E8_GRAM_MATRIX: LazyLock<[[f32; E8_ROOT_COUNT]; E8_ROOT_COUNT]> = LazyLock::new(|| {
    let mut matrix = [[0.0f32; E8_ROOT_COUNT]; E8_ROOT_COUNT];
    for (i, row) in matrix.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = dot8(&E8_ROOTS[i], &E8_ROOTS[j]).clamp(-1.0, 1.0);
        }
    }
    matrix
});

#[inline]
fn normalize8(mut coords: [f32; 8]) -> [f32; 8] {
    let overflow_threshold = f32::MAX.sqrt();
    if coords
        .iter()
        .any(|coordinate| coordinate.abs() > overflow_threshold)
    {
        return [0.0; 8];
    }

    let norm_squared = coords.iter().map(|value| value * value).sum::<f32>();
    if norm_squared == 0.0 || !norm_squared.is_finite() {
        return [0.0; 8];
    }

    let inverse_norm = 1.0 / norm_squared.sqrt();
    for coordinate in &mut coords {
        *coordinate *= inverse_norm;
    }
    coords
}

#[inline]
fn dot8(lhs: &[f32; 8], rhs: &[f32; 8]) -> f32 {
    lhs.iter()
        .zip(rhs.iter())
        .map(|(left, right)| left * right)
        .sum()
}

fn generate_e8_roots() -> [[f32; 8]; E8_ROOT_COUNT] {
    let mut roots = [[0.0f32; 8]; E8_ROOT_COUNT];
    let mut root_index = 0usize;
    let inverse_sqrt_two = 1.0 / 2.0f32.sqrt();

    for first in 0..8 {
        for second in (first + 1)..8 {
            for first_sign in [1.0, -1.0] {
                for second_sign in [1.0, -1.0] {
                    let mut root = [0.0f32; 8];
                    root[first] = first_sign * inverse_sqrt_two;
                    root[second] = second_sign * inverse_sqrt_two;
                    roots[root_index] = normalize8(root);
                    root_index += 1;
                }
            }
        }
    }

    for bits in 0..256u16 {
        if bits.count_ones() % 2 == 0 {
            let mut root = [0.0f32; 8];
            for (bit, coordinate) in root.iter_mut().enumerate() {
                *coordinate = if (bits >> bit) & 1 == 1 { -0.5 } else { 0.5 };
            }
            roots[root_index] = normalize8(root);
            root_index += 1;
        }
    }

    assert_eq!(root_index, E8_ROOT_COUNT);
    roots
}

#[inline]
fn descending_score_then_root_id(lhs: (f32, u8), rhs: (f32, u8)) -> CmpOrdering {
    rhs.0.total_cmp(&lhs.0).then_with(|| lhs.1.cmp(&rhs.1))
}

/// Returns the top-N E8 root IDs ranked by descending dot product with `coords`.
///
/// The input is normalized before scoring. Equal scores are ordered by ascending
/// root ID, making boundary and zero-vector routing reproducible across runs and
/// standard-library sort implementations. The function performs no heap allocation.
///
/// # Panics
///
/// A debug assertion fires when `N` exceeds the 240-root codebook. Release builds
/// return all 240 roots and leave the remaining output positions zeroed.
#[inline]
#[must_use]
pub fn top_orbits_stack<const N: usize>(coords: &[f32; 8]) -> ([u8; N], usize) {
    debug_assert!(
        E8_ROOT_COUNT >= N,
        "top_orbits_stack: codebook has {E8_ROOT_COUNT} roots but N={N} requested; \
         spatial signatures will be geometrically incomplete"
    );

    let query = normalize8(*coords);
    let mut scored = [(f32::NEG_INFINITY, 0u8); E8_ROOT_COUNT];
    for root_id in 0u8..=E8_ROOT_LAST_ID {
        let root_index = usize::from(root_id);
        scored[root_index] = (dot8(&query, &E8_ROOTS[root_index]), root_id);
    }

    let count = N.min(E8_ROOT_COUNT);
    if count == 0 {
        return ([0u8; N], 0);
    }

    scored.select_nth_unstable_by(count - 1, |lhs, rhs| {
        descending_score_then_root_id(*lhs, *rhs)
    });
    scored[..count].sort_unstable_by(|lhs, rhs| descending_score_then_root_id(*lhs, *rhs));

    let mut output = [0u8; N];
    for (slot, scored_root) in output.iter_mut().zip(scored[..count].iter()) {
        *slot = scored_root.1;
    }
    (output, count)
}

/// Packs a canonical three-root territorial signature into a `u32`.
///
/// Format: `(scale_tier << 24) | (r0 << 16) | (r1 << 8) | r2`. Root IDs are
/// sorted ascending before packing.
#[inline]
#[must_use]
pub fn pack_sig(scale_tier: u8, a: u8, b: u8, c: u8) -> u32 {
    let mut roots = [a, b, c];
    if roots[0] > roots[1] {
        roots.swap(0, 1);
    }
    if roots[1] > roots[2] {
        roots.swap(1, 2);
    }
    if roots[0] > roots[1] {
        roots.swap(0, 1);
    }

    (u32::from(scale_tier) << 24)
        | (u32::from(roots[0]) << 16)
        | (u32::from(roots[1]) << 8)
        | u32::from(roots[2])
}

/// Unpacks the three root IDs from a territorial signature.
///
/// The scale tier byte (`sig >> 24`) is not included in the result.
#[inline]
#[must_use]
pub const fn unpack_sig(sig: u32) -> [u8; 3] {
    let bytes = sig.to_be_bytes();
    [bytes[1], bytes[2], bytes[3]]
}

/// Fills `out` with sorted C(`count`, 3) signatures from `roots[..count]`.
///
/// Returns the number of signatures written. `MAX` must cover the requested
/// combination count; otherwise the result is truncated to `MAX` entries.
#[inline]
pub fn fill_combos<const MAX: usize>(
    scale_tier: u8,
    roots: &[u8],
    count: usize,
    out: &mut [u32; MAX],
) -> usize {
    let mut written = 0usize;
    'outer: for first in 0..count {
        for second in (first + 1)..count {
            for third in (second + 1)..count {
                if written >= MAX {
                    break 'outer;
                }
                out[written] = pack_sig(scale_tier, roots[first], roots[second], roots[third]);
                written += 1;
            }
        }
    }
    out[..written].sort_unstable();
    written
}

/// Generates the C(7,3) = 35 signatures used to insert a point.
#[inline]
#[must_use]
pub fn insert_sigs(coords: &[f32; 8], scale_tier: u8) -> ([u32; 35], usize) {
    let (roots, count) = top_orbits_stack::<INSERT_TOP_ROOTS>(coords);
    let mut signatures = [0u32; 35];
    let written = fill_combos(scale_tier, &roots, count, &mut signatures);
    (signatures, written)
}

/// Generates the C(9,3) = 84 signatures used to query a point.
#[inline]
#[must_use]
pub fn lookup_sigs(coords: &[f32; 8], scale_tier: u8) -> ([u32; 84], usize) {
    let (roots, count) = top_orbits_stack::<LOOKUP_TOP_ROOTS>(coords);
    let mut signatures = [0u32; 84];
    let written = fill_combos(scale_tier, &roots, count, &mut signatures);
    (signatures, written)
}

/// Returns the canonical cell formed by the three highest-ranked roots.
#[inline]
#[must_use]
pub fn canonical_cobb_sig(coords: &[f32; 8], scale_tier: u8) -> u32 {
    let (roots, count) = top_orbits_stack::<3>(coords);
    debug_assert_eq!(count, 3);
    pack_sig(scale_tier, roots[0], roots[1], roots[2])
}

/// Returns cosine similarity between two canonical E8 roots.
///
/// # Panics
///
/// Panics if either root ID is outside the canonical `0..240` range.
#[inline]
#[must_use]
pub fn root_similarity(root_a: u8, root_b: u8) -> f32 {
    E8_GRAM_MATRIX[root_a as usize][root_b as usize]
}

/// Returns the mean pairwise root similarity across two territorial signatures.
#[inline]
#[must_use]
pub fn signature_similarity(sig_a: u32, sig_b: u32) -> f32 {
    let lhs = unpack_sig(sig_a);
    let rhs = unpack_sig(sig_b);
    let mut total = 0.0f32;
    for &left in &lhs {
        for &right in &rhs {
            total += root_similarity(left, right);
        }
    }
    total / 9.0
}

/// Computes raw geometric affinity to the three-root cell defined by `sig`.
///
/// The input coordinates are intentionally not normalized, preserving magnitude
/// for callers that use affinity as an insertion-time ordering score.
#[inline]
#[must_use]
pub fn cell_affinity(sig: u32, coords: &[f32; 8]) -> f32 {
    unpack_sig(sig)
        .iter()
        .map(|root_id| dot8(coords, &E8_ROOTS[*root_id as usize]))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_is_invariant_under_global_phase() {
        let vector = [
            Complex32::new(0.25, -0.5),
            Complex32::new(0.75, 0.125),
            Complex32::new(-0.4, 0.2),
        ];
        let phase = Complex32::from_polar(1.0, 1.37);
        let rotated = vector.map(|value| value * phase);

        let lhs = RiveroAddress::compile(&vector);
        let rhs = RiveroAddress::compile(&rotated);
        for foundation in 0..RIVERO_FOUNDATIONS {
            for lane in 0..8 {
                assert!(
                    (lhs.foundations[foundation][lane] - rhs.foundations[foundation][lane]).abs()
                        < 1e-5
                );
            }
        }
    }

    #[test]
    fn territory_resolution_is_fixed_and_recalls_self() {
        let index = RiveroTerritoryIndex::new();
        let vector = [Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)];
        let address = RiveroAddress::compile(&vector);
        index.insert(&address, 42);

        index.with_candidates(&address, 8, |candidates, diagnostics| {
            assert!(candidates.contains(&42));
            assert!(candidates.len() <= RiveroAddress::candidate_read_bound(8));
            assert_eq!(diagnostics.cells_probed, RiveroAddress::cell_probe_count());
            assert!(diagnostics.resident_reads <= diagnostics.candidate_read_bound);
            assert!(diagnostics.resident_reads <= diagnostics.resident_scans);
            assert!(diagnostics.resident_scans <= diagnostics.resident_scan_bound);
            assert_eq!(
                diagnostics.resident_scan_bound,
                RiveroAddress::resident_scan_bound()
            );
        });
    }

    #[test]
    fn dense_cell_admission_is_query_adaptive_and_bounded() {
        let far = pack_fine_code(
            pack_projected_code(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            0.0,
            false,
        );
        let near = pack_fine_code(
            pack_projected_code(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            0.0,
            false,
        );
        let mut cell = CellSlots {
            slots: (0..RIVERO_CELL_CAPACITY as u32)
                .map(|slot| CellResident {
                    slot,
                    fine_code: far,
                })
                .collect(),
            elite_len: RIVERO_CELL_AFFINITY_ELITES,
            overflowed: true,
        };
        cell.slots[RIVERO_CELL_CAPACITY - 1] = CellResident {
            slot: 9_999,
            fine_code: near,
        };

        let mut candidates = Vec::new();
        let (scanned, admitted) = cell.append_best(near & 0x00ff_ffff, 1, &mut candidates);
        assert_eq!(scanned, RIVERO_CELL_CAPACITY);
        assert_eq!(admitted, 1);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].slot, 9_999);
    }

    #[test]
    fn mixed_retention_is_bounded_and_insertion_order_independent() {
        const KEY: u64 = 0x0000_0007_0011_2233;
        let residents = (0..128u32)
            .map(|slot| {
                let coordinates = [
                    (slot as f32 * 0.017).sin(),
                    (slot as f32 * 0.031).cos(),
                    0.25,
                    -0.5,
                    0.125,
                    -0.25,
                    0.375,
                    -0.125,
                ];
                let projected = pack_projected_code(&coordinates);
                let affinity = u8::MAX - slot as u8;
                (slot, projected | (u32::from(affinity) << 24))
            })
            .collect::<Vec<_>>();

        let mut forward = CellSlots::default();
        for &(slot, code) in &residents {
            forward.insert(KEY, code, slot);
        }
        let mut reverse = CellSlots::default();
        for &(slot, code) in residents.iter().rev() {
            reverse.insert(KEY, code, slot);
        }

        assert_eq!(std::mem::size_of::<CellResident>(), 8);
        assert_eq!(forward.slots.len(), RIVERO_CELL_CAPACITY);
        assert_eq!(forward.elite_len, RIVERO_CELL_AFFINITY_ELITES);
        assert!(forward.overflowed);
        assert_eq!(forward.slots, reverse.slots);
        assert_eq!(
            forward.slots[..RIVERO_CELL_AFFINITY_ELITES]
                .iter()
                .map(|resident| resident.slot)
                .collect::<Vec<_>>(),
            (0..RIVERO_CELL_AFFINITY_ELITES as u32).collect::<Vec<_>>()
        );
    }

    #[test]
    fn degenerate_addresses_are_finite_and_stable() {
        let zero = RiveroAddress::compile(&[Complex32::new(0.0, 0.0); 4]);
        let non_finite = RiveroAddress::compile(&[
            Complex32::new(f32::NAN, 0.0),
            Complex32::new(0.0, f32::INFINITY),
            Complex32::new(0.0, 0.0),
            Complex32::new(0.0, 0.0),
        ]);
        assert!(
            zero.foundations
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
        assert_eq!(zero.foundations, non_finite.foundations);
    }
    #[test]
    fn same_point_insert_family_is_contained_by_lookup_family() {
        let point = [0.42, -0.31, 0.18, 0.53, -0.27, 0.11, -0.62, 0.09];
        let (insert, insert_count) = insert_sigs(&point, 17);
        let (lookup, lookup_count) = lookup_sigs(&point, 17);

        assert_eq!(insert_count, 35);
        assert_eq!(lookup_count, 84);
        assert!(
            insert[..insert_count]
                .iter()
                .all(|signature| lookup[..lookup_count].binary_search(signature).is_ok())
        );
    }

    #[test]
    fn equal_scores_resolve_by_ascending_root_id() {
        let (roots, count) = top_orbits_stack::<9>(&[0.0; 8]);

        assert_eq!(count, 9);
        assert_eq!(roots, [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn packed_signature_is_canonical() {
        let signature = pack_sig(9, 31, 2, 17);

        assert_eq!(signature >> 24, 9);
        assert_eq!(unpack_sig(signature), [2, 17, 31]);
    }

    #[test]
    fn canonical_roots_are_unit_normalized() {
        for root in E8_ROOTS.iter() {
            let norm_squared = dot8(root, root);
            assert!((norm_squared - 1.0).abs() <= 2.0 * f32::EPSILON);
        }
    }

    #[test]
    fn test_territory_envelope_soundness() {
        let dim = 16;
        let mut vecs: Vec<Vec<Complex32>> = Vec::new();
        for i in 0..10 {
            let v: Vec<Complex32> = (0..dim)
                .map(|d| {
                    Complex32::new(
                        (i as f32 * 0.1 + d as f32 * 0.05).sin(),
                        (i as f32 * 0.1 + d as f32 * 0.05).cos(),
                    )
                })
                .collect();
            vecs.push(v);
        }
        let slices: Vec<&[Complex32]> = vecs.iter().map(|v| v.as_slice()).collect();
        let envelope = TerritoryEnvelope::from_vectors(&slices, dim);

        let query: Vec<Complex32> = (0..dim)
            .map(|d| Complex32::new((d as f32 * 0.2).cos(), (d as f32 * 0.2).sin()))
            .collect();
        let q_block_norms: Vec<f32> = (0..dim.div_ceil(4))
            .map(|b| {
                let start = b * 4;
                let end = (start + 4).min(dim);
                let mut sum = 0.0f32;
                for z in &query[start..end] {
                    sum += z.re * z.re + z.im * z.im;
                }
                sum.sqrt()
            })
            .collect();
        let q_global_norm = q_block_norms.iter().map(|&n| n * n).sum::<f32>().sqrt();

        let ub = envelope.upper_bound(&query, &q_block_norms, q_global_norm);

        for (idx, v) in vecs.iter().enumerate() {
            let mut true_dot_re = 0.0f32;
            for (q_z, v_z) in query.iter().zip(v.iter()) {
                true_dot_re += q_z.re * v_z.re + q_z.im * v_z.im;
            }
            assert!(
                true_dot_re <= ub,
                "Envelope upper bound violated for member {idx}! true={true_dot_re}, ub={ub}"
            );
        }
    }

    #[test]
    fn multi_lane_and_flexible_foundations_compilation() {
        let dim = 768; // 1536D real
        let data: Vec<Complex32> = (0..dim)
            .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.2).cos()))
            .collect();

        // 1. Test 48 Foundations GlobalMix
        let config_48 = RiveroAddressConfig {
            foundations: 48,
            projection: RiveroProjectionMode::GlobalMix,
        };
        let addr_48 = RiveroAddress::compile_with_config(&data, config_48);
        assert_eq!(addr_48.foundation_count, 48);
        assert_eq!(addr_48.active_foundations().len(), 48);

        // 2. Test 48 Foundations MultiLane (4 lanes × 12F, Hashed)
        let config_ml = RiveroAddressConfig {
            foundations: 48,
            projection: RiveroProjectionMode::MultiLane {
                lanes: 4,
                assignment: LaneAssignment::Hashed,
            },
        };
        let addr_ml = RiveroAddress::compile_with_config(&data, config_ml);
        assert_eq!(addr_ml.foundation_count, 48);
        assert_eq!(addr_ml.active_foundations().len(), 48);

        // 3. Test 64 Foundations MultiLane (8 lanes × 8F, Interleaved)
        let config_64 = RiveroAddressConfig {
            foundations: 64,
            projection: RiveroProjectionMode::MultiLane {
                lanes: 8,
                assignment: LaneAssignment::Interleaved,
            },
        };
        let addr_64 = RiveroAddress::compile_with_config(&data, config_64);
        assert_eq!(addr_64.foundation_count, 64);
        assert_eq!(addr_64.active_foundations().len(), 64);
    }
}
