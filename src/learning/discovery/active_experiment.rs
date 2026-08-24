//! Information-gain experiment planning with sandbox and external-authorization boundaries.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::learning::discovery::dsl::{OperatorProgram, execute_program};
use crate::learning::discovery::evaluation::{
    CompetitiveOperatorEvaluation, EvaluationObservation,
};
use crate::learning::discovery::experiment::ExperimentProposalId;
use crate::learning::discovery::model::{DomainId, FeatureId, ResolutionId};
use crate::learning::discovery::operator::{DeclarativeOperator, DiscoveredOperatorId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ActiveExperimentKind {
    ShadowReplay,
    Simulation,
    DiagnosticObservation,
    AbTest,
    ControlledConfigurationChange,
    MissingEvidenceRequest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    None,
    Low,
    Medium,
    High,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    Proposed,
    Authorized,
    Running,
    Completed,
    Rejected,
    Expired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentAuthorization {
    pub authority_id: u64,
    pub policy_id: u64,
    pub authorized_at_lsn: u64,
    pub expires_at_lsn: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveExperimentProposal {
    pub id: ExperimentProposalId,
    pub kind: ActiveExperimentKind,
    pub status: ExperimentStatus,
    pub operators: Vec<DiscoveredOperatorId>,
    pub target_domains: BTreeSet<DomainId>,
    pub required_features: BTreeSet<FeatureId>,
    pub candidate_resolutions: BTreeSet<ResolutionId>,
    pub expected_information_gain_q32: i64,
    pub risk: RiskLevel,
    pub requires_external_authorization: bool,
    pub maximum_trials: u32,
    pub authorization: Option<ExperimentAuthorization>,
    pub result: Option<SandboxExperimentResult>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentPlanningPolicy {
    pub maximum_risk: RiskLevel,
    pub allow_live_interventions: bool,
    pub maximum_trials: u32,
    pub max_proposals: usize,
    pub min_information_gain_q32: i64,
}

impl Default for ExperimentPlanningPolicy {
    fn default() -> Self {
        Self {
            maximum_risk: RiskLevel::Low,
            allow_live_interventions: false,
            maximum_trials: 100,
            max_proposals: 64,
            min_information_gain_q32: 1,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SandboxExperimentResult {
    pub proposal_id: Option<ExperimentProposalId>,
    pub trials: u32,
    pub operator_matches: BTreeMap<DiscoveredOperatorId, u32>,
    pub resolution_votes: BTreeMap<ResolutionId, u32>,
    pub disagreements: u32,
    pub empirical_case_ids: BTreeSet<u64>,
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ExperimentExecutionError {
    #[error("experiment kind requires external execution")]
    ExternalExecutionRequired,
    #[error("external authorization is required or expired")]
    AuthorizationRequired,
    #[error("experiment risk {actual:?} exceeds policy maximum {maximum:?}")]
    RiskLimitExceeded {
        actual: RiskLevel,
        maximum: RiskLevel,
    },
    #[error("operator program {0:?} is unavailable")]
    MissingProgram(DiscoveredOperatorId),
    #[error("experiment lifecycle does not permit this operation")]
    InvalidLifecycle,
}

pub fn plan_active_experiments(
    operators: &[DeclarativeOperator],
    evaluations: &BTreeMap<DiscoveredOperatorId, CompetitiveOperatorEvaluation>,
    known_domains: &BTreeSet<DomainId>,
    policy: ExperimentPlanningPolicy,
) -> Vec<ActiveExperimentProposal> {
    let mut by_context = BTreeMap::<Vec<_>, Vec<&DeclarativeOperator>>::new();
    for operator in operators {
        by_context
            .entry(operator.predicates.clone())
            .or_default()
            .push(operator);
    }
    let mut proposals = Vec::new();
    for (predicates, group) in by_context {
        let resolutions: BTreeSet<_> = group
            .iter()
            .map(|operator| operator.proposed_resolution())
            .collect();
        let tested_domains: BTreeSet<_> = group
            .iter()
            .filter_map(|operator| evaluations.get(&operator.id))
            .flat_map(|evaluation| evaluation.domain_accuracy_q32.keys().copied())
            .collect();
        let missing_domains: BTreeSet<_> =
            known_domains.difference(&tested_domains).copied().collect();
        let operator_ids: Vec<_> = group.iter().map(|operator| operator.id).collect();
        let features: BTreeSet<FeatureId> = predicates
            .iter()
            .filter_map(|predicate| match predicate {
                crate::learning::discovery::OperatorPredicate::HasFeature(feature) => {
                    Some(*feature)
                }
                crate::learning::discovery::OperatorPredicate::LacksFeature(_) => None,
            })
            .collect();

        if resolutions.len() > 1 {
            push_proposal(
                &mut proposals,
                ActiveExperimentKind::Simulation,
                &operator_ids,
                missing_domains.clone(),
                features.clone(),
                resolutions.clone(),
                q32(resolutions.len(), resolutions.len() + 1),
                RiskLevel::None,
                false,
                policy.maximum_trials,
            );
            push_proposal(
                &mut proposals,
                ActiveExperimentKind::AbTest,
                &operator_ids,
                missing_domains.clone(),
                features.clone(),
                resolutions.clone(),
                q32(
                    resolutions.len() + missing_domains.len(),
                    resolutions.len() + missing_domains.len() + 1,
                ),
                RiskLevel::Medium,
                true,
                policy.maximum_trials,
            );
        }
        if !missing_domains.is_empty() {
            push_proposal(
                &mut proposals,
                ActiveExperimentKind::ShadowReplay,
                &operator_ids,
                missing_domains.clone(),
                features.clone(),
                resolutions.clone(),
                q32(missing_domains.len(), missing_domains.len() + 1),
                RiskLevel::None,
                false,
                policy.maximum_trials,
            );
        }
        for operator in &group {
            let Some(evaluation) = evaluations.get(&operator.id) else {
                push_proposal(
                    &mut proposals,
                    ActiveExperimentKind::MissingEvidenceRequest,
                    &[operator.id],
                    known_domains.clone(),
                    features.clone(),
                    BTreeSet::from([operator.proposed_resolution()]),
                    q32(1, 2),
                    RiskLevel::None,
                    false,
                    policy.maximum_trials,
                );
                continue;
            };
            if evaluation.counterfactual_accuracy_q32 == 0 {
                push_proposal(
                    &mut proposals,
                    ActiveExperimentKind::DiagnosticObservation,
                    &[operator.id],
                    tested_domains.clone(),
                    features.clone(),
                    BTreeSet::from([operator.proposed_resolution()]),
                    q32(1, 3),
                    RiskLevel::Low,
                    false,
                    policy.maximum_trials,
                );
            }
            if evaluation.intervention_accuracy_q32 == 0 {
                push_proposal(
                    &mut proposals,
                    ActiveExperimentKind::ControlledConfigurationChange,
                    &[operator.id],
                    tested_domains.clone(),
                    features.clone(),
                    BTreeSet::from([operator.proposed_resolution()]),
                    q32(1, 2),
                    RiskLevel::High,
                    true,
                    policy.maximum_trials.min(10),
                );
            }
        }
    }
    proposals.retain(|proposal| {
        proposal.expected_information_gain_q32 >= policy.min_information_gain_q32
            && proposal.risk <= policy.maximum_risk
            && (policy.allow_live_interventions
                || !matches!(
                    proposal.kind,
                    ActiveExperimentKind::AbTest
                        | ActiveExperimentKind::ControlledConfigurationChange
                ))
    });
    proposals.sort_by(|left, right| {
        right
            .expected_information_gain_q32
            .cmp(&left.expected_information_gain_q32)
            .then_with(|| left.risk.cmp(&right.risk))
            .then_with(|| left.id.cmp(&right.id))
    });
    proposals.dedup_by_key(|proposal| proposal.id);
    proposals.truncate(policy.max_proposals);
    proposals
}

pub fn authorize_experiment(
    proposal: &mut ActiveExperimentProposal,
    authorization: ExperimentAuthorization,
    current_lsn: u64,
    policy: ExperimentPlanningPolicy,
) -> Result<(), ExperimentExecutionError> {
    if proposal.risk > policy.maximum_risk {
        return Err(ExperimentExecutionError::RiskLimitExceeded {
            actual: proposal.risk,
            maximum: policy.maximum_risk,
        });
    }
    if authorization.authorized_at_lsn > current_lsn || authorization.expires_at_lsn < current_lsn {
        return Err(ExperimentExecutionError::AuthorizationRequired);
    }
    proposal.authorization = Some(authorization);
    proposal.status = ExperimentStatus::Authorized;
    Ok(())
}

pub fn start_experiment(
    proposal: &mut ActiveExperimentProposal,
    current_lsn: u64,
) -> Result<(), ExperimentExecutionError> {
    let authorization = proposal
        .authorization
        .ok_or(ExperimentExecutionError::AuthorizationRequired)?;
    if proposal.status != ExperimentStatus::Authorized
        || authorization.authorized_at_lsn > current_lsn
        || authorization.expires_at_lsn < current_lsn
    {
        return Err(ExperimentExecutionError::AuthorizationRequired);
    }
    proposal.status = ExperimentStatus::Running;
    Ok(())
}

pub fn complete_experiment(
    proposal: &mut ActiveExperimentProposal,
    result: SandboxExperimentResult,
) -> Result<(), ExperimentExecutionError> {
    if proposal.status != ExperimentStatus::Running || result.proposal_id != Some(proposal.id) {
        return Err(ExperimentExecutionError::InvalidLifecycle);
    }
    proposal.result = Some(result);
    proposal.status = ExperimentStatus::Completed;
    Ok(())
}

/// Executes only read-only shadow/simulation work. Live interventions are always
/// returned to an external controller even when authorization is present.
pub fn execute_sandbox_experiment(
    proposal: &ActiveExperimentProposal,
    programs: &BTreeMap<DiscoveredOperatorId, OperatorProgram>,
    observations: &[EvaluationObservation],
) -> Result<SandboxExperimentResult, ExperimentExecutionError> {
    if proposal.status != ExperimentStatus::Running || proposal.authorization.is_none() {
        return Err(ExperimentExecutionError::AuthorizationRequired);
    }
    if matches!(
        proposal.kind,
        ActiveExperimentKind::AbTest | ActiveExperimentKind::ControlledConfigurationChange
    ) {
        return Err(ExperimentExecutionError::ExternalExecutionRequired);
    }
    let mut result = SandboxExperimentResult {
        proposal_id: Some(proposal.id),
        ..SandboxExperimentResult::default()
    };
    for observation in observations.iter().take(proposal.maximum_trials as usize) {
        let mut trial_resolutions = BTreeSet::new();
        for operator in &proposal.operators {
            let program = programs
                .get(operator)
                .ok_or(ExperimentExecutionError::MissingProgram(*operator))?;
            let execution = execute_program(program, &observation.context)
                .map_err(|_| ExperimentExecutionError::MissingProgram(*operator))?;
            if execution.matched {
                *result.operator_matches.entry(*operator).or_default() += 1;
                for resolution in execution.proposed_resolutions {
                    *result.resolution_votes.entry(resolution).or_default() += 1;
                    trial_resolutions.insert(resolution);
                }
            }
        }
        result.disagreements += u32::from(trial_resolutions.len() > 1);
        result.empirical_case_ids.insert(observation.case_id.0);
        result.trials += 1;
    }
    Ok(result)
}

fn push_proposal(
    proposals: &mut Vec<ActiveExperimentProposal>,
    kind: ActiveExperimentKind,
    operators: &[DiscoveredOperatorId],
    target_domains: BTreeSet<DomainId>,
    required_features: BTreeSet<FeatureId>,
    candidate_resolutions: BTreeSet<ResolutionId>,
    information_gain: i64,
    risk: RiskLevel,
    requires_external_authorization: bool,
    maximum_trials: u32,
) {
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_ACTIVE_EXPERIMENT_V1");
    hasher.update([kind as u8]);
    for operator in operators {
        hasher.update(operator.0);
    }
    for domain in &target_domains {
        hasher.update(domain.0.to_le_bytes());
    }
    proposals.push(ActiveExperimentProposal {
        id: ExperimentProposalId(hasher.finalize().into()),
        kind,
        status: ExperimentStatus::Proposed,
        operators: operators.to_vec(),
        target_domains,
        required_features,
        candidate_resolutions,
        expected_information_gain_q32: information_gain,
        risk,
        requires_external_authorization,
        maximum_trials,
        authorization: None,
        result: None,
    });
}

fn q32(numerator: usize, denominator: usize) -> i64 {
    ((numerator as i128 * (1i128 << 32)) / denominator.max(1) as i128) as i64
}
