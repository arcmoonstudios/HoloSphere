/* holosphere/src/learning/inference/rune_evo/evolution.rs */
//!▫~•◦-------------------------------‣
//! # Rune-EVO Native Phase Evolution & Manifold Dynamics Reference Kernel
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Direct reference-equivalent mathematical port of Rune-EVO's `evolution.rs` and `manifold.rs`.
//! Models semantic concept evolution as harmonic rotations on the E8 manifold without
//! destructive overwrites, while preserving HoloSphere's canonical EntityId durability
//! and zero-authority VersionTable views.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::entity::id::{EntityId, VersionId};
use crate::learning::inference::candidate::{
    EvolutionArtifact, EvolutionProposal, PhaseShiftArtifact,
};
use crate::learning::inference::contract::{
    InferenceError, InferenceMethod, InferenceMethodId, InferenceRequest, InferenceSeed,
};
use crate::learning::inference::rune_evo::analogy::{euclidean_dist_8, normalize_vector_8};
use crate::learning::inference::trace::InferenceTrace;

pub const RUNE_EVOLUTION_METHOD_ID: InferenceMethodId = InferenceMethodId(104);
pub const RUNE_EVOLUTION_METHOD_VERSION: u32 = 1;

/// Parameters for an evolutionary phase transition on the E8 manifold.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseShift {
    /// The 8D rotation axis on the E8 manifold.
    pub axis: [f32; 8],
    /// Rotation angle in radians.
    pub angle: f32,
    /// Lerp intensity in `[0.0, 1.0]`.
    pub gain: f32,
}

impl PhaseShift {
    #[inline]
    pub fn new(axis: [f32; 8], angle: f32, gain: f32) -> Self {
        Self {
            axis,
            angle,
            gain: gain.clamp(0.0, 1.0),
        }
    }

    #[inline]
    pub const fn identity() -> Self {
        Self {
            axis: [0.0; 8],
            angle: 0.0,
            gain: 0.0,
        }
    }

    #[inline]
    pub fn is_identity(&self) -> bool {
        self.gain < 1e-6 || self.angle.abs() < 1e-6
    }
}

impl Default for PhaseShift {
    #[inline]
    fn default() -> Self {
        Self::identity()
    }
}

/// Dot product between two 8D coordinate vectors.
#[inline(always)]
pub fn dot8(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let mut dot = 0.0f32;
    for i in 0..8 {
        dot += a[i] * b[i];
    }
    dot
}

/// Computes the tangent vector orthogonal to both `axis` and `orth` in 8D space via Gram-Schmidt.
pub fn gram_schmidt_tangent(axis: &[f32; 8], orth: &[f32; 8]) -> [f32; 8] {
    let best_basis = (0..8usize)
        .min_by(|&i, &j| {
            axis[i]
                .abs()
                .partial_cmp(&axis[j].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);

    let mut candidate = [0.0f32; 8];
    candidate[best_basis] = 1.0;

    let d_axis = dot8(axis, &candidate);
    for i in 0..8 {
        candidate[i] -= d_axis * axis[i];
    }
    let d_orth = dot8(orth, &candidate);
    for i in 0..8 {
        candidate[i] -= d_orth * orth[i];
    }

    normalize_vector_8(&candidate)
}

/// Snaps 8D coordinates to the E8 lattice unit manifold.
#[inline]
pub fn snap_to_e8_lattice(coords: &[f32; 8]) -> [f32; 8] {
    normalize_vector_8(coords)
}

/// Applies a phase shift (n-D Rodrigues rotation) to an 8D coordinate vector.
pub fn apply_phase_shift(
    coords: &[f32; 8],
    shift: &PhaseShift,
) -> Result<[f32; 8], InferenceError> {
    if shift.is_identity() {
        return Ok(*coords);
    }

    for &c in coords.iter() {
        if c.is_nan() || !c.is_finite() {
            return Err(InferenceError::ComputationFailed(
                "input coordinate vector contains NaN or infinite values".into(),
            ));
        }
    }

    let axis_mag2: f32 = shift.axis.iter().map(|&x| x * x).sum();
    if axis_mag2 < 1e-12 {
        return Ok(*coords);
    }
    let axis_unit = normalize_vector_8(&shift.axis);
    let gain = shift.gain.clamp(0.0, 1.0);

    // Decompose: parallel along axis + perpendicular remainder
    let dot_val = dot8(&axis_unit, coords);
    let parallel: [f32; 8] = std::array::from_fn(|i| axis_unit[i] * dot_val);
    let mut orth = [0.0f32; 8];
    for i in 0..8 {
        orth[i] = coords[i] - parallel[i];
    }

    let orth_mag2: f32 = orth.iter().map(|&x| x * x).sum();
    let orth_mag = orth_mag2.sqrt();

    let rotated = if orth_mag > 1e-6 {
        let orth_unit = orth.map(|x| x / orth_mag);
        let tangent = gram_schmidt_tangent(&axis_unit, &orth_unit);

        let cos_a = shift.angle.cos();
        let sin_a = shift.angle.sin();

        let mut res = parallel;
        for i in 0..8 {
            res[i] += orth_unit[i] * (orth_mag * cos_a) + tangent[i] * (orth_mag * sin_a);
        }
        res
    } else {
        *coords
    };

    // Apply gain lerp
    let mut result = [0.0f32; 8];
    for i in 0..8 {
        result[i] = coords[i] * (1.0 - gain) + rotated[i] * gain;
    }

    Ok(snap_to_e8_lattice(&result))
}

/// Zero-authority history view over an entity's ordered evolutionary version chain.
#[derive(Clone, Debug)]
pub struct EvolutionHistoryView {
    pub entity_id: EntityId,
    pub layers: Vec<[f32; 8]>,
}

impl EvolutionHistoryView {
    pub fn new(entity_id: EntityId, layers: Vec<[f32; 8]>) -> Self {
        Self { entity_id, layers }
    }

    #[inline]
    pub fn origin(&self) -> Option<[f32; 8]> {
        self.layers.first().copied()
    }

    #[inline]
    pub fn surface(&self) -> Option<[f32; 8]> {
        self.layers.last().copied()
    }

    #[inline]
    pub fn at_depth(&self, depth: usize) -> Option<[f32; 8]> {
        self.layers.get(depth).copied()
    }

    #[inline]
    pub fn depth(&self) -> usize {
        self.layers.len().saturating_sub(1)
    }

    /// Computes total semantic drift from origin layer to surface layer:
    /// $$\text{Drift}_{\text{total}} = \|x_{\text{surface}} - x_{\text{origin}}\|_2$$
    pub fn semantic_drift(&self) -> f32 {
        match (self.origin(), self.surface()) {
            (Some(o), Some(s)) => euclidean_dist_8(&o, &s),
            _ => 0.0,
        }
    }

    /// Computes pairwise incremental drift between consecutive layers:
    /// $$\text{Drift}_i = \|x_i - x_{i-1}\|_2$$
    pub fn incremental_drifts(&self) -> Vec<f32> {
        if self.layers.len() < 2 {
            return Vec::new();
        }
        self.layers
            .windows(2)
            .map(|w| euclidean_dist_8(&w[0], &w[1]))
            .collect()
    }
}

/// Inference engine producing provisional evolution proposals.
pub struct RunePhaseEvolutionV1;

impl RunePhaseEvolutionV1 {
    pub fn propose_evolution(
        &self,
        entity_id: EntityId,
        source_version: VersionId,
        source_coords: &[f32; 8],
        shift: PhaseShift,
        snapshot_lsn: u64,
    ) -> Result<EvolutionProposal, InferenceError> {
        let resulting_coords = apply_phase_shift(source_coords, &shift)?;

        let mut hasher = Sha256::new();
        hasher.update(&entity_id.to_le_bytes());
        hasher.update(&source_version.to_le_bytes());
        for &c in &resulting_coords {
            hasher.update(&c.to_le_bytes());
        }
        hasher.update(&shift.angle.to_le_bytes());
        hasher.update(&shift.gain.to_le_bytes());
        let digest = hasher.finalize();
        let mut state_fingerprint = [0u8; 32];
        state_fingerprint.copy_from_slice(&digest);

        let trace = InferenceTrace {
            method: self.id(),
            method_version: self.version(),
            source_entities: vec![entity_id],
            source_relations: Vec::new(),
            source_attempts: Vec::new(),
            snapshot_lsn,
            seed: InferenceSeed::default(),
            parameter_digest: state_fingerprint,
        };

        let artifact = EvolutionArtifact {
            method: self.id(),
            method_version: self.version(),
            source_entity: entity_id,
            source_version,
            source_coords: *source_coords,
            resulting_coords,
            phase_shift: PhaseShiftArtifact {
                axis: shift.axis,
                angle: shift.angle,
                gain: shift.gain,
            },
            state_fingerprint,
            trace,
        };

        Ok(EvolutionProposal::new_provisional(
            entity_id,
            source_version,
            artifact,
        ))
    }
}

impl Default for RunePhaseEvolutionV1 {
    fn default() -> Self {
        Self
    }
}

impl InferenceMethod for RunePhaseEvolutionV1 {
    fn id(&self) -> InferenceMethodId {
        RUNE_EVOLUTION_METHOD_ID
    }

    fn version(&self) -> u32 {
        RUNE_EVOLUTION_METHOD_VERSION
    }

    fn name(&self) -> &'static str {
        "RunePhaseEvolutionV1"
    }

    fn infer(
        &self,
        _request: &InferenceRequest<'_>,
    ) -> Result<Vec<crate::learning::inference::candidate::InferenceCandidate>, InferenceError>
    {
        // Evolution produces EvolutionProposals, not naked relations without an evolution request
        Ok(Vec::new())
    }
}
