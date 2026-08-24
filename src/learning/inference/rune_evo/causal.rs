/* holosphere/src/learning/inference/rune_evo/causal.rs */
//!▫~•◦-------------------------------‣
//! # Rune-EVO Directed Wedge & Geometric Counterfactual Reference Kernel
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Direct reference-equivalent mathematical port of Rune-EVO's `causal.rs`.
//! Computes the anticommutative Clifford algebra Cl(8) grade-2 wedge product (28 bivector components),
//! bivector contraction, and geometric counterfactual projection.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::entity::id::EntityId;
use crate::learning::inference::contract::{InferenceMethodId, InferenceSeed};
use crate::learning::inference::rune_evo::analogy::normalize_vector_8;
use crate::learning::inference::trace::InferenceTrace;
use crate::relation::id::RelationId;

pub const RUNE_DIRECTED_WEDGE_METHOD_ID: InferenceMethodId = InferenceMethodId(103);
pub const RUNE_DIRECTED_WEDGE_METHOD_VERSION: u32 = 1;

/// Number of independent components in a grade-2 element of Cl(8) (8 * 7 / 2 = 28).
pub const BIVECTOR_DIM: usize = 28;

/// Direction of a directed relationship encoded on a hyperedge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CausalOrientation {
    /// Antecedent -> Consequent (from is cause).
    Forward,
    /// Consequent -> Antecedent (to is cause).
    Backward,
}

/// Request for directed wedge encoding over an existing causal hypothesis.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectedWedgeRequest {
    pub antecedent: EntityId,
    pub consequent: EntityId,
    /// Existing hypothesis relation ID required before performing wedge analysis.
    pub hypothesis_relation: RelationId,
    pub orientation: CausalOrientation,
}

/// Durable geometric sidecar artifact containing the signed bivector and counterfactual projection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DirectedWedgeArtifact {
    pub antecedent_coords: [f32; 8],
    pub consequent_coords: [f32; 8],
    pub orientation: CausalOrientation,
    pub bivector: [f32; BIVECTOR_DIM],
    pub reference_strength: f32,
    pub counterfactual_consequent: [f32; 8],
    pub trace: InferenceTrace,
}

/// Computes the grade-2 outer product (bivector) of two 8D vectors: A ∧ B.
///
/// `bv[k]` where `k = i*(i-1)/2 + j` (i > j) stores `a[i]*b[j] - a[j]*b[i]`.
#[must_use]
pub fn causal_bivector(a: &[f32; 8], b: &[f32; 8]) -> [f32; BIVECTOR_DIM] {
    let mut bv = [0.0f32; BIVECTOR_DIM];
    let mut k = 0usize;
    for i in 1..8usize {
        for j in 0..i {
            bv[k] = a[i].mul_add(b[j], -(a[j] * b[i]));
            k += 1;
        }
    }
    bv
}

/// Bivector magnitude in grade-2 space (L2 norm of the 28 components normalized by 2*sqrt(2)).
#[must_use]
pub fn bivector_strength(bv: &[f32; BIVECTOR_DIM]) -> f32 {
    let sq_sum: f32 = bv.iter().map(|v| v * v).sum();
    (sq_sum / 8.0_f32).sqrt().clamp(0.0, 1.0)
}

/// Contracts a grade-2 bivector with an 8D vector.
#[inline]
pub fn bivector_contract(bv: &[f32; BIVECTOR_DIM], v: &[f32; 8]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    let mut k = 0usize;
    for i in 1..8usize {
        for j in 0..i {
            let b = bv[k];
            out[i] += b * v[j];
            out[j] -= b * v[i];
            k += 1;
        }
    }
    out
}

/// Computes the geometric counterfactual projection given an antecedent, consequent, and edge bivector.
///
/// Labeled explicitly as a geometric heuristic rather than a formal structural causal model intervention.
#[must_use]
pub fn geometric_counterfactual_projection(
    antecedent: &[f32; 8],
    consequent: &[f32; 8],
    edge_bivector: &[f32; BIVECTOR_DIM],
) -> [f32; 8] {
    let neg_a: [f32; 8] = std::array::from_fn(|i| -antecedent[i]);
    let neg_bv = causal_bivector(&neg_a, consequent);

    let divergence: f32 = edge_bivector
        .iter()
        .zip(neg_bv.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / (BIVECTOR_DIM as f32);

    let cf_strength = (divergence * 2.0)
        .max(bivector_strength(edge_bivector))
        .clamp(0.0, 1.0);
    let plane_release = bivector_contract(edge_bivector, consequent);

    let blended: [f32; 8] = std::array::from_fn(|i| {
        consequent[i] * (1.0 - cf_strength) - plane_release[i] * cf_strength
    });

    normalize_vector_8(&blended)
}

/// Builds a directed wedge artifact over existing coordinates.
#[must_use]
pub fn build_directed_wedge_edge(
    from: [f32; 8],
    to: [f32; 8],
    orientation: CausalOrientation,
    hypothesis_relation: RelationId,
    snapshot_lsn: u64,
) -> DirectedWedgeArtifact {
    let (antecedent, consequent) = match orientation {
        CausalOrientation::Forward => (from, to),
        CausalOrientation::Backward => (to, from),
    };

    let bivector = causal_bivector(&antecedent, &consequent);
    let reference_strength = bivector_strength(&bivector);
    let counterfactual_consequent =
        geometric_counterfactual_projection(&antecedent, &consequent, &bivector);

    let mut hasher = Sha256::new();
    hasher.update(&reference_strength.to_le_bytes());
    for &b in &bivector {
        hasher.update(&b.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut param_digest = [0u8; 32];
    param_digest.copy_from_slice(&digest);

    let trace = InferenceTrace {
        method: RUNE_DIRECTED_WEDGE_METHOD_ID,
        method_version: RUNE_DIRECTED_WEDGE_METHOD_VERSION,
        source_entities: Vec::new(),
        source_relations: vec![hypothesis_relation],
        source_attempts: Vec::new(),
        snapshot_lsn,
        seed: InferenceSeed::default(),
        parameter_digest: param_digest,
    };

    DirectedWedgeArtifact {
        antecedent_coords: antecedent,
        consequent_coords: consequent,
        orientation,
        bivector,
        reference_strength,
        counterfactual_consequent,
        trace,
    }
}
