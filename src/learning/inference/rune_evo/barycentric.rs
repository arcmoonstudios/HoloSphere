/* holosphere/src/learning/inference/rune_evo/barycentric.rs */
//!▫~•◦-------------------------------‣
//! # Rune-EVO Barycentric Centroid & Geometric Synthesis Reference Kernel
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Direct reference-equivalent mathematical port of Rune-EVO's `inference.rs`.
//! Derives new concepts via weighted centroid triangulation in E8 space and computes
//! reference geometric dispersion and epistemic uncertainty friction.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::entity::id::EntityId;
use crate::learning::inference::candidate::{
    CandidateEntityId, CandidateEntityRef, CandidateRoleBinding, DerivedEntityProposal,
    InferenceCandidateId, InferenceGeometryArtifact, InferenceProposalBundle, InferenceScore,
    RelationProposal,
};
use crate::learning::inference::contract::{
    InferenceError, InferenceMethod, InferenceMethodId, InferenceRequest, InferenceScope,
};
use crate::learning::inference::rune_evo::analogy::euclidean_dist_8;
use crate::learning::inference::trace::InferenceTrace;

pub const RUNE_BARYCENTRIC_METHOD_ID: InferenceMethodId = InferenceMethodId(102);
pub const RUNE_BARYCENTRIC_METHOD_VERSION: u32 = 1;
pub const PARALLEL_BLEND_THRESHOLD: usize = 256;

/// Explicit weight policy for barycentric triangulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BarycentricWeightSemantics {
    /// Rune-EVO V1 reference: allows any weights where sum is not near zero.
    #[default]
    RuneReferenceV1,
}

/// Configuration for Rune-EVO barycentric inference.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuneBarycentricConfig {
    pub parallel_threshold: usize,
    pub weight_semantics: BarycentricWeightSemantics,
    pub derived_relation_type_id: u32,
    pub derived_role_id: u16,
    pub parent_role_id: u16,
}

impl Default for RuneBarycentricConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: PARALLEL_BLEND_THRESHOLD,
            weight_semantics: BarycentricWeightSemantics::RuneReferenceV1,
            derived_relation_type_id: 101, // DERIVED_FROM
            derived_role_id: 1,            // DerivedConcept
            parent_role_id: 2,             // ParentConcept
        }
    }
}

/// Result of a barycentric triangulation inference operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuneBarycentricInsight {
    /// Raw centroid in 8D E8 space.
    pub centroid: [f32; 8],
    /// The weights used to blend the input coordinates, normalized to sum to 1.0.
    pub normalized_weights: Vec<f32>,
    /// Exact Rune-EVO V1 reference friction: F = D + 2U (dispersion + 2 * raw uncertainty).
    pub reference_friction: f32,
}

/// Computes the sequential weighted centroid using FMA accumulation.
pub fn sequential_centroid(
    coords: &[[f32; 8]],
    weights: &[f32],
) -> Result<[f32; 8], InferenceError> {
    let mut acc = [0.0f32; 8];
    let mut norm = 0.0f32;

    for (c, &w) in coords.iter().zip(weights.iter()) {
        for i in 0..8 {
            acc[i] = w.mul_add(c[i], acc[i]);
        }
        norm += w;
    }

    if norm.abs() < 1e-9 {
        return Err(InferenceError::ComputationFailed(
            "all weights are effectively zero; centroid is undefined".into(),
        ));
    }

    Ok(std::array::from_fn(|i| acc[i] / norm))
}

/// Computes parallel chunked weighted centroid.
pub fn parallel_centroid(coords: &[[f32; 8]], weights: &[f32]) -> Result<[f32; 8], InferenceError> {
    // Deterministic chunked reduction
    const CHUNK_SIZE: usize = 64;
    let mut chunk_accs = Vec::new();
    let mut chunk_norms = Vec::new();

    for (c_chunk, w_chunk) in coords.chunks(CHUNK_SIZE).zip(weights.chunks(CHUNK_SIZE)) {
        let mut acc = [0.0f32; 8];
        let mut norm = 0.0f32;
        for (c, &w) in c_chunk.iter().zip(w_chunk.iter()) {
            for i in 0..8 {
                acc[i] = w.mul_add(c[i], acc[i]);
            }
            norm += w;
        }
        chunk_accs.push(acc);
        chunk_norms.push(norm);
    }

    let mut total_acc = [0.0f32; 8];
    let mut total_norm = 0.0f32;
    for (acc, norm) in chunk_accs.into_iter().zip(chunk_norms.into_iter()) {
        for i in 0..8 {
            total_acc[i] += acc[i];
        }
        total_norm += norm;
    }

    if total_norm.abs() < 1e-9 {
        return Err(InferenceError::ComputationFailed(
            "all weights are effectively zero in parallel blend".into(),
        ));
    }

    Ok(std::array::from_fn(|i| total_acc[i] / total_norm))
}

/// Returns a new weight vector normalized to sum to 1.0.
pub fn normalise_weights(weights: &[f32]) -> Vec<f32> {
    let sum: f32 = weights.iter().sum();
    if sum.abs() < 1e-9 {
        vec![1.0 / (weights.len() as f32); weights.len()]
    } else {
        weights.iter().map(|w| w / sum).collect()
    }
}

/// Core reference triangulation kernel.
pub fn resolve_barycentric(
    coords: &[[f32; 8]],
    weights: &[f32],
    config: &RuneBarycentricConfig,
) -> Result<RuneBarycentricInsight, InferenceError> {
    if coords.is_empty() {
        return Err(InferenceError::InvalidParameters(
            "empty input coordinates".into(),
        ));
    }
    if coords.len() != weights.len() {
        return Err(InferenceError::InvalidParameters(format!(
            "mismatched lengths: coords={} vs weights={}",
            coords.len(),
            weights.len()
        )));
    }

    let centroid = if coords.len() >= config.parallel_threshold {
        parallel_centroid(coords, weights)?
    } else {
        sequential_centroid(coords, weights)?
    };

    // Exact Rune-EVO V1 friction computation:
    // D = sum_i (w_i * dist^2)
    // U = (1.0 - mean_raw_weight).max(0.0)
    // F = D + 2.0 * U
    let mut dispersion = 0.0f32;
    let mut raw_weight_sum = 0.0f32;
    for (c, &w) in coords.iter().zip(weights.iter()) {
        let dist = euclidean_dist_8(c, &centroid);
        dispersion += w * dist.powi(2);
        raw_weight_sum += w;
    }
    let mean_raw_weight = raw_weight_sum / (coords.len() as f32);
    let uncertainty = (1.0 - mean_raw_weight).max(0.0);
    let reference_friction = dispersion + 2.0 * uncertainty;

    let normalized_weights = normalise_weights(weights);

    Ok(RuneBarycentricInsight {
        centroid,
        normalized_weights,
        reference_friction,
    })
}

/// Blends between two 8D coordinates with alpha clamped to [0.0, 1.0].
pub fn infer_between(
    coords_a: &[f32; 8],
    coords_b: &[f32; 8],
    alpha: f32,
    config: &RuneBarycentricConfig,
) -> Result<RuneBarycentricInsight, InferenceError> {
    let alpha = alpha.clamp(0.0, 1.0);
    resolve_barycentric(&[*coords_a, *coords_b], &[alpha, 1.0 - alpha], config)
}

/// Reference-equivalent implementation of Rune-EVO Barycentric Synthesis.
pub struct RuneBarycentricV1 {
    pub config: RuneBarycentricConfig,
}

impl RuneBarycentricV1 {
    pub fn new(config: RuneBarycentricConfig) -> Self {
        Self { config }
    }

    /// Blends input coordinates and produces an InferenceProposalBundle.
    pub fn synthesize_bundle(
        &self,
        parent_entities: &[EntityId],
        parent_coords: &[[f32; 8]],
        weights: &[f32],
        snapshot_lsn: u64,
    ) -> Result<InferenceProposalBundle, InferenceError> {
        let insight = resolve_barycentric(parent_coords, weights, &self.config)?;

        // Parameter digest
        let mut hasher = Sha256::new();
        hasher.update(&insight.reference_friction.to_le_bytes());
        for &c in insight.centroid.iter() {
            hasher.update(&c.to_le_bytes());
        }
        for &w in weights {
            hasher.update(&w.to_le_bytes());
        }
        let digest = hasher.finalize();
        let mut param_digest = [0u8; 32];
        param_digest.copy_from_slice(&digest);

        let trace = InferenceTrace {
            method: self.id(),
            method_version: self.version(),
            source_entities: parent_entities.to_vec(),
            source_relations: Vec::new(),
            source_attempts: Vec::new(),
            snapshot_lsn,
            seed: crate::learning::inference::contract::InferenceSeed::default(),
            parameter_digest: param_digest,
        };

        // 1. Proposed Derived Entity
        let derived_local_id = CandidateEntityId(1);
        let entity_proposal = DerivedEntityProposal::new_provisional(
            derived_local_id,
            InferenceGeometryArtifact::E8Coordinates(insight.centroid),
            trace.clone(),
        );

        // 2. Relations: DERIVED_FROM(Proposed(X), Existing(P_i))
        let mut relations = Vec::new();
        for (i, (&parent_id, &norm_w)) in parent_entities
            .iter()
            .zip(insight.normalized_weights.iter())
            .enumerate()
        {
            let bindings = vec![
                CandidateRoleBinding {
                    entity: CandidateEntityRef::Proposed(derived_local_id),
                    role_id: self.config.derived_role_id,
                },
                CandidateRoleBinding {
                    entity: CandidateEntityRef::Existing(parent_id),
                    role_id: self.config.parent_role_id,
                },
            ];

            let confidence_q32 = (((norm_w).clamp(0.0, 1.0) * (1u64 << 32) as f32) as i64).max(0);

            relations.push(RelationProposal::new_provisional(
                InferenceCandidateId(100 + (i as u64)),
                self.config.derived_relation_type_id,
                bindings,
                InferenceScore {
                    confidence_q32,
                    raw_floating: insight.reference_friction,
                },
                trace.clone(),
            ));
        }

        Ok(InferenceProposalBundle {
            entities: vec![entity_proposal],
            relations,
            evolutions: Vec::new(),
        })
    }
}

impl Default for RuneBarycentricV1 {
    fn default() -> Self {
        Self::new(RuneBarycentricConfig::default())
    }
}

impl InferenceMethod for RuneBarycentricV1 {
    fn id(&self) -> InferenceMethodId {
        RUNE_BARYCENTRIC_METHOD_ID
    }

    fn version(&self) -> u32 {
        RUNE_BARYCENTRIC_METHOD_VERSION
    }

    fn name(&self) -> &'static str {
        "RuneBarycentricV1"
    }

    fn infer(
        &self,
        request: &InferenceRequest<'_>,
    ) -> Result<Vec<crate::learning::inference::candidate::InferenceCandidate>, InferenceError>
    {
        let entities = match &request.scope {
            InferenceScope::Region { entities } => {
                if entities.is_empty() {
                    return Ok(Vec::new());
                }
                entities.clone()
            }
            _ => return Ok(Vec::new()),
        };

        let coords: Vec<[f32; 8]> = entities
            .iter()
            .map(|&e| {
                let mut v = [0.0f32; 8];
                v[(e % 8) as usize] = 1.0;
                v
            })
            .collect();

        let weights = vec![1.0 / (coords.len() as f32); coords.len()];
        let bundle =
            self.synthesize_bundle(&entities, &coords, &weights, request.learning_snapshot.lsn)?;

        Ok(bundle.relations)
    }
}
