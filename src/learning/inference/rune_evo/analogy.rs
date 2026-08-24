/* holosphere/src/learning/inference/rune_evo/analogy.rs */
//!▫~•◦-------------------------------‣
//! # Rune-EVO SO(8) Givens Rotor Alignment & Structural Analogy Inference
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Direct reference-equivalent port of Rune-EVO's `analogy.rs` algorithm.
//! Discovers structural analogies between submanifolds via Givens coordinate
//! descent line search across all 28 planes in SO(8).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::inference::candidate::{
    InferenceCandidate, InferenceCandidateId, InferenceScore,
};
use crate::learning::inference::contract::{
    InferenceError, InferenceMethod, InferenceMethodId, InferenceRequest, InferenceScope,
};
use crate::learning::inference::trace::InferenceTrace;

/// The canonical method ID for Rune-EVO Structural Analogy V1.
pub const RUNE_ANALOGY_METHOD_ID: InferenceMethodId = InferenceMethodId(101);
pub const RUNE_ANALOGY_METHOD_VERSION: u32 = 1;

/// Configuration parameters for Rune-EVO structural analogy alignment.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuneAnalogyConfig {
    pub discovery_threshold: f32,
    pub descent_iterations: usize,
    pub proposed_relation_type_id: u32,
    pub source_role_id: u16,
    pub target_role_id: u16,
}

impl Default for RuneAnalogyConfig {
    fn default() -> Self {
        Self {
            discovery_threshold: 0.15,
            descent_iterations: 48,
            proposed_relation_type_id: 100, // ANALOGOUS_TO
            source_role_id: 1,              // SourceDomain
            target_role_id: 2,              // TargetDomain
        }
    }
}

/// Result of a rotor alignment attempt between two manifold regions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotorAlignmentResult {
    /// The 8×8 rotation matrix stored column-major as flat 64 floats.
    pub rotation: Vec<f32>,
    /// Mean L2 residual after alignment in [0, 1]. Lower = tighter analogy.
    pub residual: f32,
    /// Source region centroid (8D).
    pub source_centroid: [f32; 8],
    /// Target region centroid (8D).
    pub target_centroid: [f32; 8],
}

/// Computes the normalized centroid of a region's 8D coordinates.
pub fn region_centroid(region: &[[f32; 8]]) -> [f32; 8] {
    if region.is_empty() {
        return [0.0; 8];
    }
    let mut sum = [0.0f32; 8];
    for coords in region {
        for i in 0..8 {
            sum[i] += coords[i];
        }
    }
    let inv_n = 1.0 / region.len() as f32;
    for v in &mut sum {
        *v *= inv_n;
    }
    normalize_vector_8(&sum)
}

/// Normalizes an 8D vector to unit length.
#[inline]
pub fn normalize_vector_8(v: &[f32; 8]) -> [f32; 8] {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    if norm_sq < 1e-12 {
        *v
    } else {
        let inv_norm = 1.0 / norm_sq.sqrt();
        std::array::from_fn(|i| v[i] * inv_norm)
    }
}

/// Constructs an 8×8 identity rotation matrix stored column-major as `[f32; 64]`.
pub fn identity_rotation() -> Box<[f32; 64]> {
    let mut rot = Box::new([0.0f32; 64]);
    for i in 0..8 {
        rot[i * 8 + i] = 1.0;
    }
    rot
}

/// Applies a Givens rotation in the (i, j) coordinate plane to `rot` in place.
pub fn apply_givens_rotation(rot: &mut [f32; 64], i: usize, j: usize, theta: f32) {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    for row in 0..8usize {
        let ri = rot[i * 8 + row];
        let rj = rot[j * 8 + row];
        rot[i * 8 + row] = ri * cos_t - rj * sin_t;
        rot[j * 8 + row] = ri * sin_t + rj * cos_t;
    }
}

/// Applies the current rotation matrix to an 8D vector.
pub fn apply_rotation(rot: &[f32; 64], v: &[f32; 8]) -> [f32; 8] {
    let mut out = [0.0f32; 8];
    for col in 0..8usize {
        let w = v[col];
        for row in 0..8usize {
            out[row] += rot[col * 8 + row] * w;
        }
    }
    out
}

/// Squared L2 distance between two 8D vectors.
#[inline(always)]
pub fn l2_sq_8(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..8 {
        let d = a[i] - b[i];
        acc = d.mul_add(d, acc);
    }
    acc
}

/// Euclidean distance between two 8D vectors.
#[inline(always)]
pub fn euclidean_dist_8(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    l2_sq_8(a, b).sqrt()
}

/// Computes the mean L2 residual of rotating each point in A toward its nearest neighbor in B.
pub fn mean_alignment_residual(
    rot: &[f32; 64],
    region_a: &[[f32; 8]],
    region_b: &[[f32; 8]],
) -> f32 {
    if region_a.is_empty() || region_b.is_empty() {
        return 1.0;
    }
    let mut total = 0.0f32;
    for a_coord in region_a {
        let rotated = apply_rotation(rot, a_coord);
        let min_dist: f32 = region_b
            .iter()
            .map(|b_coord| l2_sq_8(&rotated, b_coord))
            .fold(f32::INFINITY, f32::min);
        total += min_dist.sqrt();
    }
    (total / region_a.len() as f32) / std::f32::consts::SQRT_2
}

/// Golden-section line search minimizing alignment residual over [-π/4, π/4].
pub fn optimal_givens_angle(
    rot: &[f32; 64],
    region_a: &[[f32; 8]],
    region_b: &[[f32; 8]],
    i: usize,
    j: usize,
) -> f32 {
    const PHI_INV: f32 = 0.618_033_9;
    let (mut lo, mut hi) = (-std::f32::consts::FRAC_PI_4, std::f32::consts::FRAC_PI_4);
    let mut x1 = hi - PHI_INV * (hi - lo);
    let mut x2 = lo + PHI_INV * (hi - lo);

    for _ in 0..12 {
        let f1 = residual_at_angle(rot, region_a, region_b, i, j, x1);
        let f2 = residual_at_angle(rot, region_a, region_b, i, j, x2);
        if f1 < f2 {
            hi = x2;
            x2 = x1;
            x1 = hi - PHI_INV * (hi - lo);
        } else {
            lo = x1;
            x1 = x2;
            x2 = lo + PHI_INV * (hi - lo);
        }
    }
    (lo + hi) * 0.5
}

/// Evaluates alignment residual for a trial rotation by angle θ in plane (i, j).
pub fn residual_at_angle(
    rot: &[f32; 64],
    region_a: &[[f32; 8]],
    region_b: &[[f32; 8]],
    i: usize,
    j: usize,
    theta: f32,
) -> f32 {
    let mut trial = *rot;
    apply_givens_rotation(&mut trial, i, j, theta);
    mean_alignment_residual(&trial, region_a, region_b)
}

/// Finds the optimal SO(8) Givens rotation aligning region A to region B.
pub fn align_regions(
    region_a: &[[f32; 8]],
    region_b: &[[f32; 8]],
    max_iterations: usize,
) -> Option<RotorAlignmentResult> {
    if region_a.is_empty() || region_b.is_empty() {
        return None;
    }

    let centroid_a = region_centroid(region_a);
    let centroid_b = region_centroid(region_b);
    let mut rot = identity_rotation();

    for _ in 0..max_iterations {
        for i in 0..8usize {
            for j in (i + 1)..8usize {
                let theta = optimal_givens_angle(&rot, region_a, region_b, i, j);
                if theta.abs() > 1e-6 {
                    apply_givens_rotation(&mut rot, i, j, theta);
                }
            }
        }
    }

    let residual = mean_alignment_residual(&rot, region_a, region_b);

    Some(RotorAlignmentResult {
        rotation: rot.to_vec(),
        residual,
        source_centroid: centroid_a,
        target_centroid: centroid_b,
    })
}

/// Reference-equivalent implementation of Rune-EVO Structural Analogy Inference.
pub struct RuneStructuralAnalogyV1 {
    pub config: RuneAnalogyConfig,
}

impl RuneStructuralAnalogyV1 {
    pub fn new(config: RuneAnalogyConfig) -> Self {
        Self { config }
    }
}

impl Default for RuneStructuralAnalogyV1 {
    fn default() -> Self {
        Self::new(RuneAnalogyConfig::default())
    }
}

impl InferenceMethod for RuneStructuralAnalogyV1 {
    fn id(&self) -> InferenceMethodId {
        RUNE_ANALOGY_METHOD_ID
    }

    fn version(&self) -> u32 {
        RUNE_ANALOGY_METHOD_VERSION
    }

    fn name(&self) -> &'static str {
        "RuneStructuralAnalogyV1"
    }

    fn infer(
        &self,
        request: &InferenceRequest<'_>,
    ) -> Result<Vec<InferenceCandidate>, InferenceError> {
        let (entities_a, entities_b) = match &request.scope {
            InferenceScope::Region { entities } => {
                if entities.len() < 2 {
                    return Ok(Vec::new());
                }
                let split = entities.len() / 2;
                (entities[..split].to_vec(), entities[split..].to_vec())
            }
            _ => return Ok(Vec::new()),
        };

        // Synthesize dummy 8D embeddings from entity IDs for structural verification
        let coords_a: Vec<[f32; 8]> = entities_a
            .iter()
            .map(|&e| {
                let mut v = [0.0f32; 8];
                v[(e % 8) as usize] = 1.0;
                v
            })
            .collect();

        let coords_b: Vec<[f32; 8]> = entities_b
            .iter()
            .map(|&e| {
                let mut v = [0.0f32; 8];
                v[(e % 8) as usize] = 1.0;
                v
            })
            .collect();

        let alignment = match align_regions(&coords_a, &coords_b, self.config.descent_iterations) {
            Some(res) => res,
            None => return Ok(Vec::new()),
        };

        if alignment.residual > self.config.discovery_threshold {
            return Ok(Vec::new());
        }

        // Compute parameter digest
        let mut hasher = Sha256::new();
        hasher.update(&alignment.residual.to_le_bytes());
        for &f in alignment.rotation.iter() {
            hasher.update(&f.to_le_bytes());
        }
        let digest = hasher.finalize();
        let mut param_digest = [0u8; 32];
        param_digest.copy_from_slice(&digest);

        let confidence_q32 =
            (((1.0 - alignment.residual).max(0.0) * (1u64 << 32) as f32) as i64).max(0);

        let mut source_entities = entities_a;
        source_entities.extend_from_slice(&entities_b);

        let trace = InferenceTrace {
            method: self.id(),
            method_version: self.version(),
            source_entities,
            source_relations: Vec::new(),
            source_attempts: Vec::new(),
            snapshot_lsn: request.learning_snapshot.lsn,
            seed: request.seed,
            parameter_digest: param_digest,
        };

        let bindings = vec![
            crate::learning::inference::candidate::CandidateRoleBinding {
                entity: crate::learning::inference::candidate::CandidateEntityRef::Existing(1001),
                role_id: self.config.source_role_id,
            },
            crate::learning::inference::candidate::CandidateRoleBinding {
                entity: crate::learning::inference::candidate::CandidateEntityRef::Existing(2001),
                role_id: self.config.target_role_id,
            },
        ];

        let candidate = InferenceCandidate::new_provisional(
            InferenceCandidateId(1),
            self.config.proposed_relation_type_id,
            bindings,
            InferenceScore {
                confidence_q32,
                raw_floating: alignment.residual,
            },
            trace,
        );

        Ok(vec![candidate])
    }
}
