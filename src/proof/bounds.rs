/* hnsqr/src/proof/bounds.rs */
//!▫~•◦-------------------------------‣
//! # Spherical-Cap & Cauchy-Schwarz Admissible Proof Bounds (Gate B2)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates provable upper bounds in 64-bit precision over normalized unit spheres:
//!
//! $$\text{UB}_{\text{cap}}(q, T) = s \cos\theta_T + \sqrt{\max(0, 1 - s^2)} \sin\theta_T \quad (\text{if } s < \cos\theta_T \text{ else } 1.0)$$
//! $$\text{UB}_T(q) = \min(\text{UB}_{\text{cap}}, \text{UB}_{\text{global}}, \text{UB}_{\text{block}}) + \epsilon_{\text{safe}}$$
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use num_complex::Complex32;
use serde::{Deserialize, Serialize};

/// Number of complex coordinates per E8 / 8D-real bounding block (4 complex = 8 floats).
pub const PROOF_BLOCK_COMPLEX_DIM: usize = 4;

/// Compact representation of a quantized block centroid with known reconstruction error.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ProofCentroidCode {
    /// Reconstructed centroid complex coordinates for the block.
    pub coords: [Complex32; PROOF_BLOCK_COMPLEX_DIM],
    /// Outward-safe reconstruction error: $\epsilon_{T,b} \ge \|c_{T,b} - \hat{c}_{T,b}\|_2$.
    pub error_norm: f32,
}

impl ProofCentroidCode {
    /// Builds a centroid code from continuous centroid coordinates.
    #[inline]
    pub fn from_raw(block: &[Complex32]) -> Self {
        let mut coords = [Complex32::new(0.0, 0.0); PROOF_BLOCK_COMPLEX_DIM];
        let len = block.len().min(PROOF_BLOCK_COMPLEX_DIM);
        coords[..len].copy_from_slice(&block[..len]);
        Self {
            coords,
            error_norm: 0.0,
        }
    }
}

/// Pre-computed query norms and block decompositions for fast admissible bounding.
#[derive(Clone, Debug)]
pub struct ProofQuery {
    pub complex_data: Vec<Complex32>,
    pub block_norms: Vec<f64>,
    pub global_norm: f64,
}

impl ProofQuery {
    /// Prepares a query vector for fast hierarchical proof evaluation in `f64`.
    pub fn new(q: &[Complex32]) -> Self {
        let num_blocks = q.len().div_ceil(PROOF_BLOCK_COMPLEX_DIM);
        let mut block_norms = Vec::with_capacity(num_blocks);

        for b in 0..num_blocks {
            let start = b * PROOF_BLOCK_COMPLEX_DIM;
            let end = (start + PROOF_BLOCK_COMPLEX_DIM).min(q.len());
            let mut sum_sq = 0.0f64;
            for i in start..end {
                let z = q[i];
                sum_sq += (z.re as f64) * (z.re as f64) + (z.im as f64) * (z.im as f64);
            }
            block_norms.push(sum_sq.sqrt());
        }

        let mut global_sum_sq = 0.0f64;
        for &z in q {
            global_sum_sq += (z.re as f64) * (z.re as f64) + (z.im as f64) * (z.im as f64);
        }
        let global_norm = global_sum_sq.sqrt();

        Self {
            complex_data: q.to_vec(),
            block_norms,
            global_norm,
        }
    }
}

/// Detailed breakdown of competing admissible upper bounds for diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundBreakdown {
    pub ub_cap: f64,
    pub ub_global: f64,
    pub ub_block: f64,
    pub chosen_ub: f64,
}

/// Evaluates the combined admissible upper bound in `f64`.
///
/// Integrates:
/// 1. Spherical-Cap Bound $\text{UB}_{\text{cap}}$ (cosine-native, exact on $S^{D-1}$).
/// 2. Global Ball Bound $\text{UB}_{\text{global}}$.
/// 3. Additive Block Cauchy-Schwarz Bound $\text{UB}_{\text{block}}$.
#[inline(always)]
pub fn evaluate_node_upper_bound_f64(
    query: &ProofQuery,
    centroid_codes: &[ProofCentroidCode],
    block_radii: &[f32],
    centroid_offset: usize,
    num_blocks: usize,
    cos_radius: f32,
    sin_radius: f32,
    global_radius: f32,
    centroid_error_norm: f32,
) -> f64 {
    if num_blocks == 0 {
        return f64::NEG_INFINITY;
    }

    let q = &query.complex_data;
    let mut dot_block = 0.0f64;
    let mut cs_block = 0.0f64;

    for b in 0..num_blocks {
        let code_idx = centroid_offset + b;
        let code = &centroid_codes[code_idx];
        let rho_b = block_radii[code_idx] as f64;
        let eps_b = code.error_norm as f64;
        let q_norm_b = query.block_norms[b];

        let start = b * PROOF_BLOCK_COMPLEX_DIM;
        let end = (start + PROOF_BLOCK_COMPLEX_DIM).min(q.len());

        for i in start..end {
            let q_z = q[i];
            let c_z = code.coords[i - start];
            dot_block += (q_z.re as f64) * (c_z.re as f64) + (q_z.im as f64) * (c_z.im as f64);
        }

        cs_block += q_norm_b * (rho_b + eps_b);
    }

    // 1. Spherical Cap Upper Bound
    let s = dot_block; // q^T c_T (since both are unit normalized)
    let cos_r = cos_radius as f64;
    let sin_r = sin_radius as f64;

    let ub_cap = if s >= cos_r {
        1.0f64
    } else {
        let sin_alpha = (1.0f64 - s * s).max(0.0).sqrt();
        s * cos_r + sin_alpha * sin_r
    };

    // 2. Global Ball Upper Bound
    let ub_global = dot_block + query.global_norm * (global_radius as f64 + centroid_error_norm as f64);

    // 3. Additive Block Upper Bound
    let ub_block = dot_block + cs_block;

    // Minimum of admissible bounds is admissible, with float outward error term
    let min_ub = ub_cap.min(ub_global).min(ub_block);
    let eps_safe = (num_blocks as f64) * 2.0e-15 + 1.0e-7;
    (min_ub + eps_safe).min(1.0f64)
}
