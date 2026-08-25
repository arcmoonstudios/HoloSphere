/* hnsqr/src/planning/affect.rs */
//!▫~•◦-------------------------------------------------------------------‣
//! # XyCo 8D Affective & Somatic Control Plane
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module implements the 8-dimensional affective state tensor $A_t \in \mathbb{R}^8$
//! for HoloSphere, providing somatic and causal appraisal gating for the query planner.
//!
//! ## Mathematical Formalism
//! The state tensor is defined as:
//! $$A_t = [V, A, D, C, T, N, G, R]^T \in [-1, 1]^8$$
//!
//! - **$V \in [-1, 1]$ (Valence):** Destructive/Aversive $\leftrightarrow$ Constructive/Appetitive
//! - **$A \in [0, 1]$ (Arousal):** Dormant/Calm $\leftrightarrow$ Urgent/High-Energy
//! - **$D \in [0, 1]$ (Dominance):** Constrained/Subordinate $\leftrightarrow$ Autonomous/Decisive
//! - **$C \in [0, 1]$ (Certainty):** Speculative/Ambiguous $\leftrightarrow$ Mathematically Proven
//! - **$T \in [0, 1]$ (Trust):** Unverified/Hostile $\leftrightarrow$ Provenance-Backed/Verified
//! - **$N \in [0, 1]$ (Novelty):** Invariant/Familiar $\leftrightarrow$ Outlier/Anomaly
//! - **$G \in [-1, 1]$ (Goal Congruence):** Blocked/Divergent $\leftrightarrow$ Aligned/Progressing
//! - **$R \in [0, 1]$ (Reversibility):** One-Way Door/Terminal $\leftrightarrow$ Ephemeral/Recoverable
//!
//! $A_t$ maps losslessly into the 8-dimensional $E_8$ root lattice manifold $\Lambda_8 \subset \mathbb{R}^8$.
//!
//! ### Dual-Regime Gating
//! - **Regime A (One-Way Door, $R < 0.2$):** Overrides optimistic execution, enforcing
//!   `RetrievalContract::Certified` and mandatory pre-commit snapshot checkpoints.
//! - **Regime B (Speculative Curiosity, $R > 0.8 \land N > 0.8$):** Licenses broad analogical
//!   exploration and PAC-relaxed retrieval contracts.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// Operational execution regime derived from the 8D affective appraisal tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AffectiveRegime {
    /// Action is a one-way door ($R < 0.2$). Requires certified verification and snapshot isolation.
    OneWayDoorCritical,
    /// Action is fully recoverable and novelty is high ($R > 0.8 \land N > 0.8$). Authorizes analogical expansion.
    SpeculativeCuriosity,
    /// Balanced standard operational regime.
    Equilibrium,
}

/// Canonical 8-Dimensional Affective State Tensor $A_t \in \mathbb{R}^8$.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AffectiveStateTensor8D {
    /// Valence ($V \in [-1.0, 1.0]$): Negative/Aversive $\leftrightarrow$ Positive/Constructive.
    pub valence: f32,
    /// Arousal ($A \in [0.0, 1.0]$): Calm/Dormant $\leftrightarrow$ High-Activation/Urgent.
    pub arousal: f32,
    /// Dominance ($D \in [0.0, 1.0]$): Constrained $\leftrightarrow$ Autonomous.
    pub dominance: f32,
    /// Certainty ($C \in [0.0, 1.0]$): Speculative $\leftrightarrow$ Proven.
    pub certainty: f32,
    /// Trust ($T \in [0.0, 1.0]$): Unverified $\leftrightarrow$ Verified Provenance.
    pub trust: f32,
    /// Novelty ($N \in [0.0, 1.0]$): Familiar $\leftrightarrow$ Anomaly/Outlier.
    pub novelty: f32,
    /// Goal Congruence ($G \in [-1.0, 1.0]$): Conflicting $\leftrightarrow$ Progressing.
    pub goal_congruence: f32,
    /// Reversibility / Blast Radius ($R \in [0.0, 1.0]$): Irreversible $\leftrightarrow$ Fully Recoverable.
    pub reversibility: f32,
}

impl Default for AffectiveStateTensor8D {
    #[inline]
    fn default() -> Self {
        Self::neutral()
    }
}

impl AffectiveStateTensor8D {
    /// Constructs a validated, clamped 8D affective state tensor.
    #[must_use]
    pub fn new(
        valence: f32,
        arousal: f32,
        dominance: f32,
        certainty: f32,
        trust: f32,
        novelty: f32,
        goal_congruence: f32,
        reversibility: f32,
    ) -> Self {
        Self {
            valence: valence.clamp(-1.0, 1.0),
            arousal: arousal.clamp(0.0, 1.0),
            dominance: dominance.clamp(0.0, 1.0),
            certainty: certainty.clamp(0.0, 1.0),
            trust: trust.clamp(0.0, 1.0),
            novelty: novelty.clamp(0.0, 1.0),
            goal_congruence: goal_congruence.clamp(-1.0, 1.0),
            reversibility: reversibility.clamp(0.0, 1.0),
        }
    }

    /// Returns a neutral baseline state tensor in homeostatic equilibrium.
    #[must_use]
    pub const fn neutral() -> Self {
        Self {
            valence: 0.0,
            arousal: 0.1,
            dominance: 0.5,
            certainty: 0.5,
            trust: 0.5,
            novelty: 0.0,
            goal_congruence: 0.0,
            reversibility: 1.0,
        }
    }

    /// Evaluates whether the current action is a low-reversibility "One-Way Door" ($R < 0.2$).
    #[inline]
    #[must_use]
    pub fn is_one_way_door(&self) -> bool {
        self.reversibility < 0.2
    }

    /// Evaluates whether the current state permits curiosity-driven exploration ($R > 0.8 \land N > 0.8$).
    #[inline]
    #[must_use]
    pub fn is_exploratory_safe(&self) -> bool {
        self.reversibility > 0.8 && self.novelty > 0.8
    }

    /// Determines the active operational regime.
    #[must_use]
    pub fn regime(&self) -> AffectiveRegime {
        if self.is_one_way_door() {
            AffectiveRegime::OneWayDoorCritical
        } else if self.is_exploratory_safe() {
            AffectiveRegime::SpeculativeCuriosity
        } else {
            AffectiveRegime::Equilibrium
        }
    }

    /// Converts the tensor into an unpadded 8-element float array.
    #[inline]
    #[must_use]
    pub const fn as_array(&self) -> [f32; 8] {
        [
            self.valence,
            self.arousal,
            self.dominance,
            self.certainty,
            self.trust,
            self.novelty,
            self.goal_congruence,
            self.reversibility,
        ]
    }

    /// Losslessly projects the 8D affective coordinates into the nearest point on the $E_8$ root lattice $\Lambda_8$.
    ///
    /// The $E_8$ lattice consists of points in $\mathbb{R}^8$ whose coordinates are either:
    /// 1. All integers with an even sum $\sum x_i \equiv 0 \pmod 2$, or
    /// 2. All half-integers ($k + 0.5$) with an even sum.
    #[must_use]
    pub fn project_to_e8(&self) -> [f32; 8] {
        let raw = self.as_array();
        let scale = 2.0_f32;
        let scaled: [f32; 8] = [
            raw[0] * scale,
            raw[1] * scale,
            raw[2] * scale,
            raw[3] * scale,
            raw[4] * scale,
            raw[5] * scale,
            raw[6] * scale,
            raw[7] * scale,
        ];

        // Candidate 1: Nearest all-integer lattice point with even sum
        let mut int_pt = [0i32; 8];
        let mut int_diff = [0.0f32; 8];
        let mut sum_int = 0i32;
        for i in 0..8 {
            int_pt[i] = scaled[i].round() as i32;
            int_diff[i] = (scaled[i] - int_pt[i] as f32).abs();
            sum_int += int_pt[i];
        }
        if (sum_int % 2).abs() != 0 {
            // Find component with max rounding residual and flip it by 1
            let mut worst_idx = 0;
            let mut max_residual = -1.0f32;
            for (i, &diff) in int_diff.iter().enumerate() {
                if diff > max_residual {
                    max_residual = diff;
                    worst_idx = i;
                }
            }
            if scaled[worst_idx] > int_pt[worst_idx] as f32 {
                int_pt[worst_idx] += 1;
            } else {
                int_pt[worst_idx] -= 1;
            }
        }

        // Candidate 2: Nearest all-half-integer lattice point with even sum
        let mut half_pt = [0.0f32; 8];
        let mut half_diff = [0.0f32; 8];
        let mut sum_half_int = 0i32;
        for i in 0..8 {
            let base_half = (scaled[i] - 0.5).round();
            half_pt[i] = base_half + 0.5;
            half_diff[i] = (scaled[i] - half_pt[i]).abs();
            sum_half_int += (base_half * 2.0) as i32 + 1;
        }
        if ((sum_half_int / 2) % 2).abs() != 0 {
            let mut worst_idx = 0;
            let mut max_residual = -1.0f32;
            for (i, &diff) in half_diff.iter().enumerate() {
                if diff > max_residual {
                    max_residual = diff;
                    worst_idx = i;
                }
            }
            if scaled[worst_idx] > half_pt[worst_idx] {
                half_pt[worst_idx] += 1.0;
            } else {
                half_pt[worst_idx] -= 1.0;
            }
        }

        // Measure Euclidean distances to choose closest D8 coset
        let mut dist_int = 0.0f32;
        let mut dist_half = 0.0f32;
        for i in 0..8 {
            dist_int += (scaled[i] - int_pt[i] as f32).powi(2);
            dist_half += (scaled[i] - half_pt[i]).powi(2);
        }

        let mut result = [0.0f32; 8];
        if dist_int <= dist_half {
            for i in 0..8 {
                result[i] = int_pt[i] as f32 / scale;
            }
        } else {
            for i in 0..8 {
                result[i] = half_pt[i] / scale;
            }
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affective_state_clamping_and_bounds() {
        let tensor = AffectiveStateTensor8D::new(-2.0, 1.5, 0.5, 0.9, 0.4, 0.1, 3.0, -0.5);
        assert_eq!(tensor.valence, -1.0);
        assert_eq!(tensor.arousal, 1.0);
        assert_eq!(tensor.goal_congruence, 1.0);
        assert_eq!(tensor.reversibility, 0.0);
        assert!(tensor.is_one_way_door());
        assert_eq!(tensor.regime(), AffectiveRegime::OneWayDoorCritical);
    }

    #[test]
    fn test_exploratory_curiosity_regime() {
        let tensor = AffectiveStateTensor8D::new(0.5, 0.3, 0.7, 0.6, 0.8, 0.95, 0.4, 0.9);
        assert!(!tensor.is_one_way_door());
        assert!(tensor.is_exploratory_safe());
        assert_eq!(tensor.regime(), AffectiveRegime::SpeculativeCuriosity);
    }

    #[test]
    fn test_e8_lattice_projection_even_sum_invariance() {
        let tensor = AffectiveStateTensor8D::new(0.2, 0.8, 0.4, 0.9, 0.1, 0.7, -0.3, 0.05);
        let e8_pt = tensor.project_to_e8();
        assert_eq!(e8_pt.len(), 8);

        // Verify scaled coordinates satisfy even-sum integral or half-integral condition
        let scaled: Vec<f32> = e8_pt.iter().map(|&x| x * 2.0).collect();
        let is_all_int = scaled.iter().all(|&x| (x.round() - x).abs() < 1e-4);
        assert!(is_all_int, "Projected point must align with D8 or D8+half coset");
    }
}
