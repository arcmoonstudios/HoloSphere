/* holosphere/src/vector/rotary.rs */
//!▫~•◦-------------------------------‣
//! # Rotary Phase Transformer & Harmonic Coordinate Rotation
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides unitary 2D pairwise phase rotation (RoPE - Rotary Position Embeddings)
//! and harmonic frequency encoding over complex coordinate pairs:
//!
//! $$z_k(p) = z_k(0) \cdot e^{i\, p\,\theta_k}, \quad \theta_k = \text{base}^{-2k/D}$$
//!
//! ## Invariants
//! 1. **Isometric / Unitary**: $\|R_p(z)\| \equiv \|z\|$ for any position $p$.
//! 2. **Relative Shift Equivariance**: $\langle R_{p_1}(u), R_{p_2}(v) \rangle \equiv \langle R_{p_1 - p_2}(u), v \rangle$.
//! 3. **Additive Group Action**: $R_{p_1}(R_{p_2}(z)) \equiv R_{p_1 + p_2}(z)$ and $R_0(z) \equiv z$.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use num_complex::Complex32;
use serde::{Deserialize, Serialize};

use crate::VectorEmbedding;

/// Rotary Position Embedding (RoPE) and 2D coordinate phase modulator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotaryPhaseTransformer {
    pub complex_dim: usize,
    pub base: f32,
    inv_freqs: Vec<f32>,
}

impl RotaryPhaseTransformer {
    /// Constructs a new RotaryPhaseTransformer for a given complex dimension ($\mathbb{C}^d$) and harmonic base (e.g. 10000.0).
    pub fn new(complex_dim: usize, base: f32) -> Self {
        let mut inv_freqs = Vec::with_capacity(complex_dim);
        for k in 0..complex_dim {
            let exponent = (2 * k) as f32 / (complex_dim * 2).max(2) as f32;
            inv_freqs.push(1.0 / base.powf(exponent));
        }
        Self {
            complex_dim,
            base,
            inv_freqs,
        }
    }

    /// Default transformer configured with standard LLM frequency base (10,000.0).
    pub fn default_for_dim(complex_dim: usize) -> Self {
        Self::new(complex_dim, 10_000.0)
    }

    /// Returns precomputed inverse frequency scaling factors $\theta_k$.
    pub fn inverse_frequencies(&self) -> &[f32] {
        &self.inv_freqs
    }

    /// Computes the complex phasor $e^{i\, p\,\theta_k}$ for coordinate $k$ at token/sequence position $p$.
    #[inline(always)]
    pub fn phasor(&self, coord_k: usize, position: usize) -> Complex32 {
        if coord_k >= self.inv_freqs.len() {
            return Complex32::new(1.0, 0.0);
        }
        let angle = position as f32 * self.inv_freqs[coord_k];
        Complex32::from_polar(1.0, angle)
    }

    /// Rotates a mutable slice of complex coordinates in place by the harmonic angle corresponding to `position`.
    pub fn rotate_in_place(&self, complex_data: &mut [Complex32], position: usize) {
        let n = complex_data.len().min(self.inv_freqs.len());
        for k in 0..n {
            let angle = position as f32 * self.inv_freqs[k];
            let phasor = Complex32::from_polar(1.0, angle);
            complex_data[k] *= phasor;
        }
    }

    /// Applies rotary phase shift to a `VectorEmbedding`, returning a newly allocated rotated embedding.
    pub fn apply_rotation(&self, embedding: &VectorEmbedding, position: usize) -> VectorEmbedding {
        let mut data = embedding.complex_data().to_vec();
        self.rotate_in_place(&mut data, position);
        VectorEmbedding::from_complex(data)
    }

    /// Applies rotary phase shift directly on a contiguous flat real slice `&mut [f32]` (interpreting consecutive pairs as 2D coordinates).
    pub fn rotate_real_slice(&self, real_data: &mut [f32], position: usize) {
        let pairs_len = real_data.len() / 2;
        let n = pairs_len.min(self.inv_freqs.len());
        for k in 0..n {
            let angle = position as f32 * self.inv_freqs[k];
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let x = real_data[2 * k];
            let y = real_data[2 * k + 1];
            real_data[2 * k] = x * cos_a - y * sin_a;
            real_data[2 * k + 1] = x * sin_a + y * cos_a;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary_transformer_isometry_norm_preservation() {
        let dim = 32;
        let rope = RotaryPhaseTransformer::new(dim, 10_000.0);

        let v = VectorEmbedding::from_complex(
            (0..dim)
                .map(|i| Complex32::new((i + 1) as f32, (i * 2 + 1) as f32))
                .collect(),
        )
        .into_normalized();

        let initial_norm_sq = v.norm_squared();
        assert!((initial_norm_sq - 1.0).abs() < 1e-5);

        for pos in [0, 1, 5, 42, 1024] {
            let rotated = rope.apply_rotation(&v, pos);
            let rotated_norm_sq = rotated.norm_squared();
            assert!(
                (rotated_norm_sq - initial_norm_sq).abs() < 1e-5,
                "Rotation at pos {pos} must preserve norm! Expected {initial_norm_sq}, got {rotated_norm_sq}"
            );
        }
    }

    #[test]
    fn test_rotary_relative_shift_equivariance() {
        let dim = 16;
        let rope = RotaryPhaseTransformer::new(dim, 10_000.0);

        let u = VectorEmbedding::from_complex(
            (0..dim)
                .map(|i| Complex32::new((i * 3 + 1) as f32, (i + 2) as f32))
                .collect(),
        )
        .into_normalized();

        let v = VectorEmbedding::from_complex(
            (0..dim)
                .map(|i| Complex32::new((i * 5 + 2) as f32, (i * 3 + 4) as f32))
                .collect(),
        )
        .into_normalized();

        // <R_{p+k}(u), R_p(v)> == <R_k(u), v>
        let p = 17;
        let k = 8;

        let u_p_plus_k = rope.apply_rotation(&u, p + k);
        let v_p = rope.apply_rotation(&v, p);
        let dot_shifted = u_p_plus_k.dot_product_real(&v_p);

        let u_k = rope.apply_rotation(&u, k);
        let dot_relative = u_k.dot_product_real(&v);

        assert!(
            (dot_shifted - dot_relative).abs() < 1e-5,
            "Relative shift equivariance violated: dot_shifted={dot_shifted}, dot_relative={dot_relative}"
        );
    }

    #[test]
    fn test_rotary_real_and_complex_equivalence() {
        let dim = 8;
        let rope = RotaryPhaseTransformer::new(dim, 10_000.0);

        let complex_initial: Vec<Complex32> = (0..dim)
            .map(|i| Complex32::new(i as f32 + 1.0, (i * 2) as f32 + 0.5))
            .collect();

        let mut complex_rot = complex_initial.clone();
        rope.rotate_in_place(&mut complex_rot, 23);

        let mut real_rot = Vec::with_capacity(dim * 2);
        for c in &complex_initial {
            real_rot.push(c.re);
            real_rot.push(c.im);
        }
        rope.rotate_real_slice(&mut real_rot, 23);

        for (k, c) in complex_rot.iter().enumerate() {
            assert!((c.re - real_rot[2 * k]).abs() < 1e-5);
            assert!((c.im - real_rot[2 * k + 1]).abs() < 1e-5);
        }
    }
}
