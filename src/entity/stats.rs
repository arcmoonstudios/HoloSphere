/* holosphere/src/entity/stats.rs */
//!▫~•◦-------------------------------‣
//! # Deterministic Adjudication Statistics
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Replicated integer and fixed-point sufficient statistics for deterministic
//! evidence accumulation and state-machine promotion/falsification decisions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// Deterministic integer / fixed-point evidence statistics stored durably.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeterministicEvidenceStats {
    /// Total number of successful outcomes observed under this pattern.
    pub successes: u32,
    /// Total number of failed outcomes observed under this pattern.
    pub failures: u32,
    /// Total distinct observation count.
    pub observation_count: u32,
    /// Sum of empirical utility values in Q32.32 fixed point.
    pub utility_sum_q32: i64,
    /// Sum of squared empirical utility values in Q32.32 fixed point (for variance computation).
    pub utility_sq_sum_q32: i64,
    /// Count of direct contradictions or falsifying counterexamples.
    pub contradiction_count: u32,
}

impl DeterministicEvidenceStats {
    /// Records an observation with its integer fixed-point utility.
    pub fn record_observation(&mut self, success: bool, utility_q32: i64) {
        self.observation_count = self.observation_count.saturating_add(1);
        if success {
            self.successes = self.successes.saturating_add(1);
        } else {
            self.failures = self.failures.saturating_add(1);
        }
        self.utility_sum_q32 = self.utility_sum_q32.saturating_add(utility_q32);
        let sq = (utility_q32 as i128 * utility_q32 as i128 / (1 << 32)) as i64;
        self.utility_sq_sum_q32 = self.utility_sq_sum_q32.saturating_add(sq);
    }

    /// Records a formal contradiction.
    pub fn record_contradiction(&mut self) {
        self.contradiction_count = self.contradiction_count.saturating_add(1);
    }

    /// Evaluates deterministic promotion criteria without non-deterministic floating point ops.
    pub fn meets_promotion_threshold(
        &self,
        min_obs: u32,
        min_success: u32,
        min_utility_sum_q32: i64,
        max_contradictions: u32,
    ) -> bool {
        self.observation_count >= min_obs
            && self.successes >= min_success
            && self.utility_sum_q32 >= min_utility_sum_q32
            && self.contradiction_count <= max_contradictions
    }
}
