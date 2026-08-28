/* holosphere/src/planning/regret.rs */
//!▫~•◦-------------------------------‣
//! # Universal Planner Decision Regret & Optimality Evaluator
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Quantifies empirical routing regret against the oracle minimum latency among
//! all admissible execution candidates satisfying the requested retrieval contract:
//!
//! $$\text{Regret}(P, Q) = \max(0,\, \text{Latency}(P) - \min_{P^* \in \mathcal{A}} \text{Latency}(P^*))$$
//!
//! ## Invariants
//! 1. **Non-Negativity**: $\text{Regret}(P) \ge 0.0$ for any selected plan $P$.
//! 2. **Identity of Optimality**: If $P \equiv P^*$ (the optimal admissible plan), then $\text{Regret}(P) \equiv 0.0$ and $\text{OptimalityRatio} \equiv 1.0$.
//! 3. **Admissibility Boundary**: Plans failing target recall or SLA criteria are disqualified from being $P^*$.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::planning::planner::ExecutionPlan;

/// Metrics captured from executing a candidate plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanExecutionMetrics {
    pub plan: ExecutionPlan,
    pub plan_name: String,
    pub recall_at_k: f64,
    pub latency_us: f64,
    pub throughput_qps: f64,
    pub admissible: bool,
}

/// Detailed regret evaluation of the planner's selected plan against the empirical oracle.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanRegretEvaluation {
    pub selected_plan_name: String,
    pub optimal_plan_name: String,
    pub selected_latency_us: f64,
    pub optimal_latency_us: f64,
    pub regret_us: f64,
    pub relative_regret_pct: f64,
    pub optimality_ratio: f64,
    pub policy_consistent: bool,
}

/// Workload summary report across multiple queries or configurations.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkloadRegretReport {
    pub total_evaluations: usize,
    pub mean_regret_us: f64,
    pub p95_regret_us: f64,
    pub mean_optimality_ratio: f64,
    pub oracle_agreement_pct: f64,
}

/// Decision regret and optimality oracle primitive.
pub struct PlannerRegretOracle;

impl PlannerRegretOracle {
    /// Evaluates decision regret for a selected plan against all evaluated candidates.
    #[must_use]
    pub fn evaluate(
        selected: &PlanExecutionMetrics,
        candidates: &[PlanExecutionMetrics],
    ) -> PlanRegretEvaluation {
        let admissible: Vec<&PlanExecutionMetrics> =
            candidates.iter().filter(|c| c.admissible).collect();

        let optimal = admissible
            .iter()
            .min_by(|a, b| a.latency_us.total_cmp(&b.latency_us))
            .copied()
            .unwrap_or(selected);

        let regret_us = (selected.latency_us - optimal.latency_us).max(0.0);
        let relative_regret_pct = if optimal.latency_us > 0.0 {
            (regret_us / optimal.latency_us) * 100.0
        } else {
            0.0
        };

        let optimality_ratio = if selected.latency_us > 0.0 {
            (optimal.latency_us / selected.latency_us).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let policy_consistent = selected.admissible;

        PlanRegretEvaluation {
            selected_plan_name: selected.plan_name.clone(),
            optimal_plan_name: optimal.plan_name.clone(),
            selected_latency_us: selected.latency_us,
            optimal_latency_us: optimal.latency_us,
            regret_us,
            relative_regret_pct,
            optimality_ratio,
            policy_consistent,
        }
    }

    /// Aggregates a series of plan evaluations into a workload summary report.
    pub fn aggregate(evaluations: &[PlanRegretEvaluation]) -> WorkloadRegretReport {
        if evaluations.is_empty() {
            return WorkloadRegretReport::default();
        }

        let mut regrets: Vec<f64> = evaluations.iter().map(|e| e.regret_us).collect();
        regrets.sort_by(|a, b| a.total_cmp(b));

        let mean_regret = regrets.iter().sum::<f64>() / regrets.len() as f64;
        let p95_idx = ((regrets.len() as f64 - 1.0) * 0.95).round() as usize;
        let p95_regret = regrets[p95_idx.min(regrets.len() - 1)];

        let mean_opt = evaluations.iter().map(|e| e.optimality_ratio).sum::<f64>()
            / evaluations.len() as f64;
        let matches = evaluations
            .iter()
            .filter(|e| e.selected_plan_name == e.optimal_plan_name)
            .count();
        let agreement = (matches as f64 / evaluations.len() as f64) * 100.0;

        WorkloadRegretReport {
            total_evaluations: evaluations.len(),
            mean_regret_us: mean_regret,
            p95_regret_us: p95_regret,
            mean_optimality_ratio: mean_opt,
            oracle_agreement_pct: agreement,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regret_oracle_invariants() {
        let plan_a = PlanExecutionMetrics {
            plan: ExecutionPlan::ExactScan { effective_n: 1000 },
            plan_name: "ExactScan".into(),
            recall_at_k: 1.0,
            latency_us: 100.0,
            throughput_qps: 10000.0,
            admissible: true,
        };

        let plan_b = PlanExecutionMetrics {
            plan: ExecutionPlan::RiveroRetrieval {
                profile: crate::rivero::RiveroProfile::Strict,
                candidate_cap: 512,
            },
            plan_name: "RiveroStrict".into(),
            recall_at_k: 1.0,
            latency_us: 40.0,
            throughput_qps: 25000.0,
            admissible: true,
        };

        // When Plan A is selected: regret is 60us, optimality is 0.4
        let eval_a = PlannerRegretOracle::evaluate(&plan_a, &[plan_a.clone(), plan_b.clone()]);
        assert_eq!(eval_a.selected_plan_name, "ExactScan");
        assert_eq!(eval_a.optimal_plan_name, "RiveroStrict");
        assert!((eval_a.regret_us - 60.0).abs() < 1e-5);
        assert!((eval_a.optimality_ratio - 0.4).abs() < 1e-5);

        // When Plan B is selected: regret is 0us, optimality is 1.0
        let eval_b = PlannerRegretOracle::evaluate(&plan_b, &[plan_a.clone(), plan_b.clone()]);
        assert_eq!(eval_b.selected_plan_name, "RiveroStrict");
        assert_eq!(eval_b.optimal_plan_name, "RiveroStrict");
        assert_eq!(eval_b.regret_us, 0.0);
        assert_eq!(eval_b.optimality_ratio, 1.0);
    }
}
