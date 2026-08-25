/* holosphere/src/learning/discovery/experiment.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Experiment Design & Hypothesis Testing
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Formulates structured empirical interventions to test causal hypotheses and
//! measure the predictive utility of newly synthesized reasoning operators.
//!
//! ## Key Capabilities
//! - **Hypothesis Generation:** Generates targeted validation queries for unproven predicates.
//! - **Outcome Verification:** Measures real-world task success rates to validate theoretical models.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::discovery::model::{DomainId, FeatureId, ResolutionId};
use crate::learning::discovery::operator::{
    DeclarativeOperator, DiscoveredOperatorId, OperatorPredicate,
};
use crate::learning::discovery::validation::OperatorValidation;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ExperimentProposalId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentKind {
    ShadowReplay,
    Simulation,
    ControlledIntervention,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentProposal {
    pub id: ExperimentProposalId,
    pub kind: ExperimentKind,
    pub operators: Vec<DiscoveredOperatorId>,
    pub required_features: BTreeSet<FeatureId>,
    pub candidate_resolutions: BTreeSet<ResolutionId>,
    pub target_domains: BTreeSet<DomainId>,
    pub expected_information_gain_q32: i64,
    pub requires_external_authorization: bool,
}

/// Selects bounded shadow replays. No experiment is executed by this function.
pub fn plan_experiments(
    operators: &[DeclarativeOperator],
    validations: &BTreeMap<DiscoveredOperatorId, OperatorValidation>,
    known_domains: &BTreeSet<DomainId>,
    max_experiments: usize,
) -> Vec<ExperimentProposal> {
    let mut groups: BTreeMap<Vec<OperatorPredicate>, Vec<&DeclarativeOperator>> = BTreeMap::new();
    for operator in operators {
        groups
            .entry(operator.predicates.clone())
            .or_default()
            .push(operator);
    }

    let mut proposals = Vec::new();
    for (predicates, mut group) in groups {
        group.sort_by_key(|operator| operator.id);
        let resolutions: BTreeSet<_> = group
            .iter()
            .map(|operator| operator.proposed_resolution())
            .collect();
        let observed_domains: BTreeSet<_> = group
            .iter()
            .filter_map(|operator| validations.get(&operator.id))
            .flat_map(|validation| validation.supporting_domains.iter().copied())
            .collect();
        let target_domains: BTreeSet<_> = known_domains
            .difference(&observed_domains)
            .copied()
            .collect();
        let competing = resolutions.len() > 1;
        let under_tested = !target_domains.is_empty();
        if !competing && !under_tested {
            continue;
        }
        let required_features = predicates
            .iter()
            .filter_map(|predicate| match predicate {
                OperatorPredicate::HasFeature(feature) => Some(*feature),
                OperatorPredicate::LacksFeature(_) => None,
            })
            .collect();
        let operator_ids: Vec<_> = group.iter().map(|operator| operator.id).collect();
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_EXPERIMENT_PROPOSAL_V1");
        for operator in &operator_ids {
            hasher.update(operator.0);
        }
        for domain in &target_domains {
            hasher.update(domain.0.to_le_bytes());
        }
        let gain_components = usize::from(competing) + target_domains.len();
        proposals.push(ExperimentProposal {
            id: ExperimentProposalId(hasher.finalize().into()),
            kind: ExperimentKind::ShadowReplay,
            operators: operator_ids,
            required_features,
            candidate_resolutions: resolutions,
            target_domains,
            expected_information_gain_q32: ((gain_components as i64) << 32)
                / (gain_components.max(1) as i64 + 1),
            requires_external_authorization: false,
        });
    }
    proposals.sort_by(|left, right| {
        right
            .expected_information_gain_q32
            .cmp(&left.expected_information_gain_q32)
            .then_with(|| left.id.cmp(&right.id))
    });
    proposals.truncate(max_experiments);
    proposals
}
