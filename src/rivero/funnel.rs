/* holosphere/src/rivero/funnel.rs */
//!▫~•◦-------------------------------‣
//! # Rivero Scale-Adaptive Candidate Funnel Primitive
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Dynamically sizes candidate caps, per-cell admission budgets, and witness
//! expansion parameters as a logarithmic function of corpus scale $N$ and dimension $D$.
//! Ensures high recall at million-scale while preserving strict physical work ceilings.
//!
//! ### Mathematical Invariants:
//! 1. **Monotonicity:** $\forall N_1 \le N_2$, $\text{query\_candidate\_cap}(N_1) \le \text{query\_candidate\_cap}(N_2)$.
//! 2. **Work Ceiling:** $\text{query\_candidate\_cap} \le W_{\max}$ ($W_{\max} = 8192$) and $\text{cell\_budget} \le \text{cell\_capacity}$.
//! 3. **Deterministic Evaluation:** Purely functional arithmetic with zero heap allocation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use super::{
    RIVERO_CELL_CAPACITY, RIVERO_MAX_FOUNDATIONS, RIVERO_WITNESS_MAX_DEGREE,
    RIVERO_WITNESS_MAX_SEEDS, RiveroConfig, RiveroProfile,
};

/// Absolute upper ceiling on candidate cap regardless of scale.
pub const RIVERO_MAX_CANDIDATE_CEILING: usize = 8192;

/// Scale-calibrated parameters for Rivero routing.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiveroBudgetParameters {
    pub foundations: usize,
    pub simhash_probes: usize,
    pub cell_budget: usize,
    pub query_candidate_cap: usize,
    pub witness_seeds: usize,
    pub witness_second_seeds: usize,
}

/// Scale-adaptive candidate funnel calculator.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScaleAdaptiveFunnel;

impl ScaleAdaptiveFunnel {
    /// Computes optimal candidate routing and witness budget parameters given
    /// corpus size $N$, complex dimension $D_{\text{complex}}$, and target profile.
    #[must_use]
    pub fn compute_budget(
        corpus_n: usize,
        complex_dim: usize,
        profile: RiveroProfile,
    ) -> RiveroBudgetParameters {
        let base_config = profile.config();
        let base_cap = base_config.query_candidate_cap;
        let base_budget = base_config.cell_budget;

        // Sublinear logarithmic scaling factor above baseline 10,000 vectors
        let scale_factor = if corpus_n <= 10_000 {
            1.0f64
        } else {
            let n_ratio = (corpus_n as f64) / 10_000.0;
            // f(N) = 1.0 + 0.45 * log2(N / 10000)
            1.0 + 0.45 * n_ratio.log2()
        };

        // Dimensional scale multiplier for high-dimensional metric concentration
        let dim_factor = if complex_dim >= 768 {
            1.5f64 // 1536D+ real
        } else if complex_dim >= 384 {
            1.25f64 // 768D real
        } else {
            1.0f64
        };

        let total_mult = scale_factor * dim_factor;

        let query_candidate_cap = ((base_cap as f64) * total_mult).round() as usize;
        let query_candidate_cap = query_candidate_cap.clamp(base_cap, RIVERO_MAX_CANDIDATE_CEILING);

        let cell_budget =
            ((base_budget as f64) * (1.0 + 0.2 * (total_mult - 1.0))).round() as usize;
        let cell_budget = cell_budget.clamp(
            base_budget,
            base_config.cell_capacity.min(RIVERO_CELL_CAPACITY),
        );

        let witness_seeds = ((profile.witness_seeds() as f64) * (1.0 + 0.3 * (scale_factor - 1.0)))
            .round() as usize;
        let witness_seeds = witness_seeds.clamp(profile.witness_seeds(), RIVERO_WITNESS_MAX_SEEDS);

        let witness_second_seeds = ((profile.witness_second_seeds() as f64)
            * (1.0 + 0.2 * (scale_factor - 1.0)))
            .round() as usize;
        let witness_second_seeds =
            witness_second_seeds.clamp(profile.witness_second_seeds(), RIVERO_WITNESS_MAX_SEEDS);

        RiveroBudgetParameters {
            foundations: base_config.foundations.min(RIVERO_MAX_FOUNDATIONS),
            simhash_probes: base_config.simhash_query_probes,
            cell_budget,
            query_candidate_cap,
            witness_seeds,
            witness_second_seeds,
        }
    }

    /// Creates an adaptive [`RiveroConfig`] for a given corpus size and dimension.
    #[must_use]
    pub fn config_for_corpus(
        corpus_n: usize,
        complex_dim: usize,
        profile: RiveroProfile,
    ) -> RiveroConfig {
        let params = Self::compute_budget(corpus_n, complex_dim, profile);
        let mut config = profile.config();
        config.cell_budget = params.cell_budget;
        config.query_candidate_cap = params.query_candidate_cap;
        config.simhash_query_probes = params.simhash_probes;
        config.foundations = params.foundations;
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_monotonicity_invariant() {
        let sizes = [500, 2_000, 10_000, 50_000, 100_000, 500_000, 1_000_000];
        for profile in [
            RiveroProfile::Fast,
            RiveroProfile::Balanced,
            RiveroProfile::Strict,
        ] {
            let mut prev_cap = 0usize;
            let mut prev_budget = 0usize;
            for &n in &sizes {
                let params = ScaleAdaptiveFunnel::compute_budget(n, 64, profile);
                assert!(
                    params.query_candidate_cap >= prev_cap,
                    "Candidate cap non-monotonic at N={n}: {} < {prev_cap}",
                    params.query_candidate_cap
                );
                assert!(
                    params.cell_budget >= prev_budget,
                    "Cell budget non-monotonic at N={n}: {} < {prev_budget}",
                    params.cell_budget
                );
                prev_cap = params.query_candidate_cap;
                prev_budget = params.cell_budget;
            }
        }
    }

    #[test]
    fn test_work_ceiling_invariant() {
        let huge_n = 100_000_000;
        for profile in [
            RiveroProfile::Fast,
            RiveroProfile::Balanced,
            RiveroProfile::Strict,
        ] {
            let params = ScaleAdaptiveFunnel::compute_budget(huge_n, 2048, profile);
            assert!(
                params.query_candidate_cap <= RIVERO_MAX_CANDIDATE_CEILING,
                "Candidate ceiling violated: {} > {}",
                params.query_candidate_cap,
                RIVERO_MAX_CANDIDATE_CEILING
            );
            assert!(
                params.cell_budget <= RIVERO_CELL_CAPACITY,
                "Cell capacity violated: {} > {}",
                params.cell_budget,
                RIVERO_CELL_CAPACITY
            );
            assert!(
                params.witness_seeds <= RIVERO_WITNESS_MAX_SEEDS,
                "Witness seeds ceiling violated: {} > {}",
                params.witness_seeds,
                RIVERO_WITNESS_MAX_SEEDS
            );
        }
    }

    #[test]
    fn test_baseline_preservation_under_10k() {
        let params_5k = ScaleAdaptiveFunnel::compute_budget(5000, 64, RiveroProfile::Strict);
        let base_strict = RiveroProfile::Strict.config();
        assert_eq!(
            params_5k.query_candidate_cap,
            base_strict.query_candidate_cap
        );
        assert_eq!(params_5k.cell_budget, base_strict.cell_budget);
    }
}
