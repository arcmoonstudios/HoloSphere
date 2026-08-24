//! Held-out falsification and deterministic operator admission metrics.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::learning::discovery::model::{
    DiscoveryCase, DiscoveryOutcome, DomainId, EvidencePartition, ratio_q32,
};
use crate::learning::discovery::operator::DeclarativeOperator;
use crate::learning::integrity::EmpiricalRootId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorValidationPolicy {
    pub min_evaluated_cases: usize,
    pub min_supporting_domains: usize,
    pub min_independent_roots: usize,
    pub min_precision_q32: i64,
    pub min_lift_q32: i64,
    pub max_contradiction_ratio_q32: i64,
    pub min_held_out_domain_passes: usize,
}

impl Default for OperatorValidationPolicy {
    fn default() -> Self {
        Self {
            min_evaluated_cases: 5,
            min_supporting_domains: 2,
            min_independent_roots: 3,
            min_precision_q32: (0.80 * (1u64 << 32) as f64) as i64,
            min_lift_q32: (0.10 * (1u64 << 32) as f64) as i64,
            max_contradiction_ratio_q32: (0.20 * (1u64 << 32) as f64) as i64,
            min_held_out_domain_passes: 2,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorValidation {
    pub evaluated_cases: usize,
    pub successes: usize,
    pub contradictions: usize,
    pub supporting_domains: BTreeSet<DomainId>,
    pub independent_roots: BTreeSet<EmpiricalRootId>,
    pub precision_q32: i64,
    pub baseline_precision_q32: i64,
    pub predictive_lift_q32: i64,
    pub contradiction_ratio_q32: i64,
    pub held_out_domain_passes: usize,
    pub held_out_domain_failures: usize,
}

impl OperatorValidation {
    pub fn meets(&self, policy: &OperatorValidationPolicy) -> bool {
        self.evaluated_cases >= policy.min_evaluated_cases
            && self.supporting_domains.len() >= policy.min_supporting_domains
            && self.independent_roots.len() >= policy.min_independent_roots
            && self.precision_q32 >= policy.min_precision_q32
            && self.predictive_lift_q32 >= policy.min_lift_q32
            && self.contradiction_ratio_q32 <= policy.max_contradiction_ratio_q32
            && self.held_out_domain_passes >= policy.min_held_out_domain_passes
            && self.held_out_domain_failures == 0
    }
}

/// Evaluates one operator only against certified evidence reserved before mining.
/// Each validation domain must independently clear the precision floor.
pub fn validate_operator(
    operator: &DeclarativeOperator,
    cases: &[DiscoveryCase],
    policy: &OperatorValidationPolicy,
) -> OperatorValidation {
    let resolution = operator.proposed_resolution();
    let discovery_roots: BTreeSet<_> = cases
        .iter()
        .filter(|case| {
            case.certified_evidence && case.evidence_partition == EvidencePartition::Discovery
        })
        .flat_map(|case| case.empirical_roots.iter().copied())
        .collect();
    let mut successes = 0usize;
    let mut contradictions = 0usize;
    let mut domains = BTreeSet::new();
    let mut roots = BTreeSet::new();
    let mut domain_counts = std::collections::BTreeMap::<DomainId, (usize, usize)>::new();

    let mut baseline_successes = 0usize;
    let mut baseline_failures = 0usize;
    for case in cases.iter().filter(|case| {
        case.certified_evidence
            && case.evidence_partition == EvidencePartition::Validation
            && case.empirical_roots.is_disjoint(&discovery_roots)
    }) {
        if case.observed_resolution != Some(resolution) {
            continue;
        }
        match case.outcome {
            DiscoveryOutcome::Successful => baseline_successes += 1,
            DiscoveryOutcome::Failed => baseline_failures += 1,
            DiscoveryOutcome::Unknown => continue,
        }
        if !operator.matches(case) {
            continue;
        }
        domains.insert(case.domain);
        roots.extend(case.empirical_roots.iter().copied());
        let counts = domain_counts.entry(case.domain).or_default();
        match case.outcome {
            DiscoveryOutcome::Successful => {
                successes += 1;
                counts.0 += 1;
            }
            DiscoveryOutcome::Failed => {
                contradictions += 1;
                counts.1 += 1;
            }
            DiscoveryOutcome::Unknown => {}
        }
    }

    let evaluated = successes + contradictions;
    let precision_q32 = ratio_q32(successes, evaluated);
    let baseline_precision_q32 =
        ratio_q32(baseline_successes, baseline_successes + baseline_failures);
    let mut held_out_domain_passes = 0;
    let mut held_out_domain_failures = 0;
    for (domain_successes, domain_failures) in domain_counts.values() {
        let domain_precision = ratio_q32(*domain_successes, domain_successes + domain_failures);
        if domain_precision >= policy.min_precision_q32 {
            held_out_domain_passes += 1;
        } else {
            held_out_domain_failures += 1;
        }
    }

    OperatorValidation {
        evaluated_cases: evaluated,
        successes,
        contradictions,
        supporting_domains: domains,
        independent_roots: roots,
        precision_q32,
        baseline_precision_q32,
        predictive_lift_q32: precision_q32.saturating_sub(baseline_precision_q32),
        contradiction_ratio_q32: ratio_q32(contradictions, evaluated),
        held_out_domain_passes,
        held_out_domain_failures,
    }
}
