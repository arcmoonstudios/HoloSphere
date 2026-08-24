//! Competitive predictive, explanatory, causal, calibration, and robustness evaluation.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::learning::discovery::dsl::{OperatorProgram, ReasoningContext, execute_program};
use crate::learning::discovery::model::{
    DiscoveryCaseId, DiscoveryOutcome, DomainId, ResolutionId,
};
use crate::learning::discovery::operator::{DeclarativeOperator, OperatorEpistemicRecord};
use crate::learning::integrity::EmpiricalRootId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EvaluationRole {
    HeldOut,
    Counterfactual,
    CausalIntervention,
    Adversarial,
    Shadow,
    Monitoring,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationObservation {
    pub case_id: DiscoveryCaseId,
    pub domain: DomainId,
    pub empirical_root: EmpiricalRootId,
    pub role: EvaluationRole,
    pub context: ReasoningContext,
    pub actual_outcome: DiscoveryOutcome,
    pub actual_resolution: Option<ResolutionId>,
    /// Confidence assigned to the program's prediction, Q32 in [0, 1].
    pub predicted_confidence_q32: i64,
    /// Stable group for semantically equivalent adversarial perturbations.
    pub adversarial_group: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompetitiveEvaluationPolicy {
    pub min_observations: usize,
    pub min_domains: usize,
    pub min_independent_roots: usize,
    pub min_accuracy_q32: i64,
    pub min_incumbent_improvement_q32: i64,
    pub min_counterfactual_accuracy_q32: i64,
    pub min_intervention_accuracy_q32: i64,
    pub min_transfer_accuracy_q32: i64,
    pub max_calibration_error_q32: i64,
    pub min_adversarial_robustness_q32: i64,
    pub max_description_bytes: usize,
}

impl Default for CompetitiveEvaluationPolicy {
    fn default() -> Self {
        Self {
            min_observations: 10,
            min_domains: 2,
            min_independent_roots: 10,
            min_accuracy_q32: q32(4, 5),
            min_incumbent_improvement_q32: q32(1, 20),
            min_counterfactual_accuracy_q32: q32(3, 4),
            min_intervention_accuracy_q32: q32(3, 4),
            min_transfer_accuracy_q32: q32(3, 4),
            max_calibration_error_q32: q32(1, 5),
            min_adversarial_robustness_q32: q32(3, 4),
            max_description_bytes: 16_384,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompetitiveOperatorEvaluation {
    pub observations: usize,
    pub correct_predictions: usize,
    pub independent_roots: BTreeSet<EmpiricalRootId>,
    pub domain_accuracy_q32: BTreeMap<DomainId, i64>,
    pub held_out_accuracy_q32: i64,
    pub incumbent_accuracy_q32: i64,
    pub incumbent_improvement_q32: i64,
    pub counterfactual_accuracy_q32: i64,
    pub intervention_accuracy_q32: i64,
    pub transfer_accuracy_q32: i64,
    pub calibration_error_q32: i64,
    pub adversarial_robustness_q32: i64,
    pub description_bytes: usize,
    pub minimum_description_length_score_q32: i64,
    pub counterexamples: BTreeSet<DiscoveryCaseId>,
    pub passed: bool,
}

/// Candidate laws compete with the current admitted operator set. An absent
/// incumbent is represented by zero accuracy, never by silently skipping the gate.
pub fn evaluate_program_competitively(
    program: &OperatorProgram,
    observations: &[EvaluationObservation],
    incumbent_accuracy_q32: i64,
    policy: CompetitiveEvaluationPolicy,
) -> CompetitiveOperatorEvaluation {
    let mut correct = 0usize;
    let mut roots = BTreeSet::new();
    let mut counterexamples = BTreeSet::new();
    let mut domain_counts = BTreeMap::<DomainId, (usize, usize)>::new();
    let mut role_counts = BTreeMap::<EvaluationRole, (usize, usize)>::new();
    let mut calibration_total = 0i128;
    let mut evaluated = 0usize;
    let mut adversarial_groups = BTreeMap::<u64, Vec<bool>>::new();

    for observation in observations {
        let Ok(result) = execute_program(program, &observation.context) else {
            counterexamples.insert(observation.case_id);
            continue;
        };
        if !result.matched {
            continue;
        }
        evaluated += 1;
        roots.insert(observation.empirical_root);
        let outcome_correct = result.predicted_outcomes.is_empty()
            || result
                .predicted_outcomes
                .contains(&observation.actual_outcome);
        let resolution_correct = observation.actual_resolution.is_none_or(|resolution| {
            result.proposed_resolutions.is_empty()
                || result.proposed_resolutions.contains(&resolution)
        });
        let prediction_correct = outcome_correct && resolution_correct;
        if prediction_correct {
            correct += 1;
        } else {
            counterexamples.insert(observation.case_id);
        }
        let domain = domain_counts.entry(observation.domain).or_default();
        domain.1 += 1;
        domain.0 += usize::from(prediction_correct);
        let role = role_counts.entry(observation.role).or_default();
        role.1 += 1;
        role.0 += usize::from(prediction_correct);
        let target = if prediction_correct { 1i64 << 32 } else { 0 };
        calibration_total += (target - observation.predicted_confidence_q32).unsigned_abs() as i128;
        if let Some(group) = observation.adversarial_group {
            adversarial_groups
                .entry(group)
                .or_default()
                .push(prediction_correct);
        }
    }

    let domain_accuracy_q32: BTreeMap<_, _> = domain_counts
        .iter()
        .map(|(domain, (domain_correct, domain_total))| {
            (*domain, ratio_q32(*domain_correct, *domain_total))
        })
        .collect();
    let held_out_accuracy_q32 = ratio_q32(correct, evaluated);
    let transfer_accuracy_q32 = domain_accuracy_q32.values().copied().min().unwrap_or(0);
    let counterfactual_accuracy_q32 = role_accuracy(&role_counts, EvaluationRole::Counterfactual);
    let intervention_accuracy_q32 = role_accuracy(&role_counts, EvaluationRole::CausalIntervention);
    let adversarial_robustness_q32 = if adversarial_groups.is_empty() {
        0
    } else {
        ratio_q32(
            adversarial_groups
                .values()
                .filter(|group| group.iter().all(|correct| *correct))
                .count(),
            adversarial_groups.len(),
        )
    };
    let calibration_error_q32 = if evaluated == 0 {
        1i64 << 32
    } else {
        (calibration_total / evaluated as i128) as i64
    };
    let description_bytes = bincode::serialize(program).map_or(usize::MAX, |bytes| bytes.len());
    let minimum_description_length_score_q32 = if description_bytes == usize::MAX {
        0
    } else {
        ((policy
            .max_description_bytes
            .saturating_sub(description_bytes) as i128)
            << 32)
            .checked_div(policy.max_description_bytes.max(1) as i128)
            .unwrap_or(0) as i64
    };
    let incumbent_improvement_q32 = held_out_accuracy_q32.saturating_sub(incumbent_accuracy_q32);
    let passed = evaluated >= policy.min_observations
        && domain_accuracy_q32.len() >= policy.min_domains
        && roots.len() >= policy.min_independent_roots
        && held_out_accuracy_q32 >= policy.min_accuracy_q32
        && incumbent_improvement_q32 >= policy.min_incumbent_improvement_q32
        && counterfactual_accuracy_q32 >= policy.min_counterfactual_accuracy_q32
        && intervention_accuracy_q32 >= policy.min_intervention_accuracy_q32
        && transfer_accuracy_q32 >= policy.min_transfer_accuracy_q32
        && calibration_error_q32 <= policy.max_calibration_error_q32
        && adversarial_robustness_q32 >= policy.min_adversarial_robustness_q32
        && description_bytes <= policy.max_description_bytes;

    CompetitiveOperatorEvaluation {
        observations: evaluated,
        correct_predictions: correct,
        independent_roots: roots,
        domain_accuracy_q32,
        held_out_accuracy_q32,
        incumbent_accuracy_q32,
        incumbent_improvement_q32,
        counterfactual_accuracy_q32,
        intervention_accuracy_q32,
        transfer_accuracy_q32,
        calibration_error_q32,
        adversarial_robustness_q32,
        description_bytes,
        minimum_description_length_score_q32,
        counterexamples,
        passed,
    }
}

pub fn apply_competitive_evaluation(
    operator: &mut DeclarativeOperator,
    evaluation: &CompetitiveOperatorEvaluation,
) {
    operator.epistemic.predictive_accuracy_q32 = evaluation.held_out_accuracy_q32;
    operator.epistemic.calibration_error_q32 = evaluation.calibration_error_q32;
    operator.epistemic.uncertainty_q32 =
        (1i64 << 32).saturating_sub(evaluation.held_out_accuracy_q32);
    operator.epistemic.counterexamples = evaluation.counterexamples.clone();
    operator
        .epistemic
        .provenance_roots
        .extend(evaluation.independent_roots.iter().copied());
    operator
        .epistemic
        .applicable_domains
        .extend(evaluation.domain_accuracy_q32.keys().copied());
}

pub fn epistemic_record_from_evaluation(
    evaluation: &CompetitiveOperatorEvaluation,
) -> OperatorEpistemicRecord {
    OperatorEpistemicRecord {
        applicable_domains: evaluation.domain_accuracy_q32.keys().copied().collect(),
        counterexamples: evaluation.counterexamples.clone(),
        provenance_roots: evaluation.independent_roots.clone(),
        predictive_accuracy_q32: evaluation.held_out_accuracy_q32,
        calibration_error_q32: evaluation.calibration_error_q32,
        uncertainty_q32: (1i64 << 32).saturating_sub(evaluation.held_out_accuracy_q32),
        ..OperatorEpistemicRecord::default()
    }
}

fn role_accuracy(counts: &BTreeMap<EvaluationRole, (usize, usize)>, role: EvaluationRole) -> i64 {
    counts
        .get(&role)
        .map_or(0, |(correct, total)| ratio_q32(*correct, *total))
}

fn ratio_q32(numerator: usize, denominator: usize) -> i64 {
    if denominator == 0 {
        return 0;
    }
    (((numerator as i128) << 32) / denominator as i128) as i64
}

const fn q32(numerator: usize, denominator: usize) -> i64 {
    ((numerator as i128 * (1i128 << 32)) / denominator as i128) as i64
}
