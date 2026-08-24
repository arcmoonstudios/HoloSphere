/* holosphere/src/learning/inference/rune_evo/mod.rs */
//!▫~•◦-------------------------------‣
//! # Rune-EVO Reference-Equivalent Inference Algorithms
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Subsystem containing reference-equivalent mathematical ports of Rune-EVO's
//! geometric inference algorithms.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod analogy;
pub mod barycentric;
pub mod causal;
pub mod evolution;
pub mod reasoning;

pub use analogy::{
    RUNE_ANALOGY_METHOD_ID, RUNE_ANALOGY_METHOD_VERSION, RotorAlignmentResult, RuneAnalogyConfig,
    RuneStructuralAnalogyV1, align_regions, apply_givens_rotation, apply_rotation,
    euclidean_dist_8, identity_rotation, l2_sq_8, mean_alignment_residual, normalize_vector_8,
    optimal_givens_angle, region_centroid,
};
pub use barycentric::{
    BarycentricWeightSemantics, PARALLEL_BLEND_THRESHOLD, RUNE_BARYCENTRIC_METHOD_ID,
    RUNE_BARYCENTRIC_METHOD_VERSION, RuneBarycentricConfig, RuneBarycentricInsight,
    RuneBarycentricV1, infer_between, normalise_weights, parallel_centroid, resolve_barycentric,
    sequential_centroid,
};
pub use causal::{
    BIVECTOR_DIM, CausalOrientation, DirectedWedgeArtifact, DirectedWedgeRequest,
    RUNE_DIRECTED_WEDGE_METHOD_ID, RUNE_DIRECTED_WEDGE_METHOD_VERSION, bivector_contract,
    bivector_strength, build_directed_wedge_edge, causal_bivector,
    geometric_counterfactual_projection,
};
pub use evolution::{
    EvolutionHistoryView, PhaseShift, RUNE_EVOLUTION_METHOD_ID, RUNE_EVOLUTION_METHOD_VERSION,
    RunePhaseEvolutionV1, apply_phase_shift, dot8, gram_schmidt_tangent, snap_to_e8_lattice,
};
pub use reasoning::{
    Cl24BasisError, Cl24Blade, Cl24CompositionArtifact, Cl24EntityBasis, ClosureCandidate,
    ClosureKind, CompositionRule, CompositionRuleRegistry, CompositionSemantics,
    DEFAULT_MAX_TRUNCATION_LOSS_RATIO, DEFAULT_TRUNCATION_TOPK, DurableEvidenceRef,
    MAX_OPERATOR_CHAIN, MultivectorCl24Sparse, RUNE_CLOSURE_METHOD_ID, RUNE_CLOSURE_METHOD_VERSION,
    ReasoningOperator, ReasoningOperatorId, RuneCl24CompositionConfig, RuneClosureEvidenceV1,
    RuneOperatorClass, blade_product_sign, compile_closure, execute_operator_chain,
    leech_to_e8_f32,
};
