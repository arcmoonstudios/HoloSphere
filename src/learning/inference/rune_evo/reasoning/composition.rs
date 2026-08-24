/* holosphere/src/learning/inference/rune_evo/reasoning/composition.rs */
//!▫~•◦-------------------------------‣
//! # Cl(24) Operator Chain Composition Kernel
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Direct reference-equivalent mathematical port of Rune-EVO's `execute_operator_chain_with_config`.
//! Composes sparse Cl(24) geometric operator transforms, tracks blade truncation energy loss,
//! and outputs derivation sidecar artifacts without overwriting canonical semantic endpoints.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::inference::rune_evo::analogy::euclidean_dist_8;
use crate::learning::inference::rune_evo::reasoning::blade::{
    Cl24Blade, MultivectorCl24Sparse, leech_to_e8_f32,
};
use crate::learning::inference::rune_evo::reasoning::operator::ReasoningOperator;

pub const MAX_OPERATOR_CHAIN: usize = 4;
pub const DEFAULT_TRUNCATION_TOPK: usize = 32;
pub const DEFAULT_MAX_TRUNCATION_LOSS_RATIO: f32 = 0.10;

/// Configuration for Cl(24) operator chain composition.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuneCl24CompositionConfig {
    pub max_operator_chain: usize,
    pub truncation_topk: usize,
    pub max_truncation_loss_ratio: f32,
}

impl Default for RuneCl24CompositionConfig {
    fn default() -> Self {
        Self {
            max_operator_chain: MAX_OPERATOR_CHAIN,
            truncation_topk: DEFAULT_TRUNCATION_TOPK,
            max_truncation_loss_ratio: DEFAULT_MAX_TRUNCATION_LOSS_RATIO,
        }
    }
}

/// Immutable geometric artifact recording the Cl(24) composed transform and truncation quality.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Cl24CompositionArtifact {
    pub retained_blades: Vec<Cl24Blade>,
    pub coords24: [f32; 24],
    /// Canonical Leech-to-E8 shadow projection (retained as an algebraic novelty witness,
    /// NOT the semantic conclusion endpoint).
    pub projected_coords8: [f32; 8],
    pub chain_depth: u8,
    pub composition_delta: f32,
    pub max_truncation_loss_ratio: f32,
    pub semantic_fingerprint: [u8; 32],
}

/// Composes an operator chain via repeated sparse geometric products and energy truncation.
pub fn execute_operator_chain(
    operators: &[ReasoningOperator],
    config: &RuneCl24CompositionConfig,
) -> Cl24CompositionArtifact {
    let mut iter = operators
        .iter()
        .filter(|op| !op.transform.is_empty() && op.is_executable());

    let Some(first) = iter.next() else {
        return Cl24CompositionArtifact {
            retained_blades: Vec::new(),
            coords24: [0.0; 24],
            projected_coords8: [0.0; 8],
            chain_depth: 0,
            composition_delta: 0.0,
            max_truncation_loss_ratio: 0.0,
            semantic_fingerprint: [0u8; 32],
        };
    };

    let mut mv = MultivectorCl24Sparse::from_blades(&first.transform);
    let mut chain_depth = 1u8;
    let mut max_truncation_loss = 0.0f32;

    for op in iter.take(config.max_operator_chain.saturating_sub(1)) {
        let rhs = MultivectorCl24Sparse::from_blades(&op.transform);
        let product = mv.geometric_product(&rhs);
        let before_energy = product.energy();
        let truncated = product.truncate_topk(config.truncation_topk);
        let after_energy = truncated.energy();

        let loss_ratio = if before_energy > f32::EPSILON {
            ((before_energy - after_energy).max(0.0) / before_energy).clamp(0.0, 1.0)
        } else {
            0.0
        };

        max_truncation_loss = max_truncation_loss.max(loss_ratio);
        mv = truncated;
        chain_depth += 1;
    }

    let coords24 = mv.to_grade1_coords();
    let projected_coords8 = leech_to_e8_f32(&coords24);

    let chain_target = operators
        .get(chain_depth.saturating_sub(1) as usize)
        .map_or(first.to_coords, |op| op.to_coords);

    let composition_delta = euclidean_dist_8(&first.from_coords, &projected_coords8)
        .max(euclidean_dist_8(&chain_target, &projected_coords8))
        .clamp(0.0, 1.0);

    let retained_blades: Vec<Cl24Blade> = mv
        .blades
        .into_iter()
        .take(config.truncation_topk.min(32))
        .collect();

    let mut hasher = Sha256::new();
    hasher.update(&chain_depth.to_le_bytes());
    hasher.update(&composition_delta.to_le_bytes());
    hasher.update(&max_truncation_loss.to_le_bytes());
    for &c in &coords24 {
        hasher.update(&c.to_le_bytes());
    }
    for b in &retained_blades {
        hasher.update(&b.bitmap.to_le_bytes());
        hasher.update(&b.coeff.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut semantic_fingerprint = [0u8; 32];
    semantic_fingerprint.copy_from_slice(&digest);

    Cl24CompositionArtifact {
        retained_blades,
        coords24,
        projected_coords8,
        chain_depth,
        composition_delta,
        max_truncation_loss_ratio: max_truncation_loss,
        semantic_fingerprint,
    }
}
