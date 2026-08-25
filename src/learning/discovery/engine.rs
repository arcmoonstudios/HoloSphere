/* holosphere/src/learning/discovery/engine.rs */
//!▫~•◦-------------------------------‣
//! # Continuous Governed Discovery Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Orchestrates the end-to-end background discovery lifecycle: motif mining, schema
//! induction, operator synthesis, empirical evaluation, falsification, and catalog revision.
//!
//! ## Key Capabilities
//! - **Autonomous Evolution:** Continuously refines internal reasoning models from empirical experience.
//! - **Epistemic Governance:** Enforces the Immutable Safety Kernel and strict progression gates.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::learning::discovery::dsl::{ReasoningContext, execute_program};
use crate::learning::discovery::experiment::{ExperimentProposal, plan_experiments};
use crate::learning::discovery::mining::{
    DiscoveredMotif, InducedSchemaProposal, MotifMinerConfig, induce_schemas, mine_motifs,
};
use crate::learning::discovery::model::{DiscoveryCase, DiscoveryCorpus, DomainId};
use crate::learning::discovery::operator::{
    DeclarativeOperator, DiscoveredOperatorId, NovelResolution, OperatorLifecycle,
};
use crate::learning::discovery::validation::{
    OperatorValidation, OperatorValidationPolicy, validate_operator,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryGovernance {
    ProposeOnly,
    PolicyAuthorized { policy_id: u64, version: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryPolicy {
    pub mining: MotifMinerConfig,
    pub validation: OperatorValidationPolicy,
    pub schema_min_domains: usize,
    pub schema_min_members: usize,
    pub max_experiments: usize,
    pub governance: DiscoveryGovernance,
}

impl Default for DiscoveryPolicy {
    fn default() -> Self {
        Self {
            mining: MotifMinerConfig::default(),
            validation: OperatorValidationPolicy::default(),
            schema_min_domains: 2,
            schema_min_members: 2,
            max_experiments: 32,
            governance: DiscoveryGovernance::ProposeOnly,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorAssessment {
    pub operator: DeclarativeOperator,
    pub validation: OperatorValidation,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryReport {
    pub schemas: Vec<InducedSchemaProposal>,
    pub motifs: Vec<DiscoveredMotif>,
    pub operators: Vec<OperatorAssessment>,
    pub experiments: Vec<ExperimentProposal>,
}

pub struct GovernedDiscoveryEngine {
    pub policy: DiscoveryPolicy,
}

impl GovernedDiscoveryEngine {
    pub fn new(policy: DiscoveryPolicy) -> Self {
        Self { policy }
    }

    pub fn discover(&self, corpus: &DiscoveryCorpus) -> DiscoveryReport {
        let schemas = induce_schemas(
            &corpus.concept_profiles,
            self.policy.schema_min_domains,
            self.policy.schema_min_members,
        );
        let motifs = mine_motifs(&corpus.cases, self.policy.mining);
        let mut assessments = Vec::with_capacity(motifs.len());
        for motif in &motifs {
            let mut operator = DeclarativeOperator::from_motif(motif);
            operator.lifecycle = OperatorLifecycle::Provisional;
            let validation = validate_operator(&operator, &corpus.cases, &self.policy.validation);
            if validation.evaluated_cases >= self.policy.validation.min_evaluated_cases {
                operator.lifecycle = OperatorLifecycle::FalsificationTesting;
            }
            if validation.meets(&self.policy.validation) {
                operator.lifecycle = OperatorLifecycle::Shadow;
            } else if validation.evaluated_cases >= self.policy.validation.min_evaluated_cases {
                operator.lifecycle = OperatorLifecycle::Rejected;
            }
            assessments.push(OperatorAssessment {
                operator,
                validation,
            });
        }
        assessments.sort_by_key(|assessment| assessment.operator.id);

        let operators: Vec<_> = assessments
            .iter()
            .map(|assessment| assessment.operator.clone())
            .collect();
        let validations: BTreeMap<_, _> = assessments
            .iter()
            .map(|assessment| (assessment.operator.id, assessment.validation.clone()))
            .collect();
        let domains: BTreeSet<DomainId> = corpus.cases.iter().map(|case| case.domain).collect();
        let experiments = plan_experiments(
            &operators,
            &validations,
            &domains,
            self.policy.max_experiments,
        );
        DiscoveryReport {
            schemas,
            motifs,
            operators: assessments,
            experiments,
        }
    }
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryCatalogError {
    #[error("Operator ID does not match its declarative definition")]
    InvalidOperatorIdentity,
    #[error("Admitted operator is missing an external governance authority")]
    MissingAdmissionAuthority,
    #[error("Operator definition changed during a lifecycle transition")]
    DefinitionChanged,
    #[error("Operator commit LSN {incoming} is not newer than {current}")]
    NonMonotonicCommit { current: u64, incoming: u64 },
    #[error("Operator transition from {from:?} to {to:?} is forbidden")]
    ForbiddenTransition {
        from: OperatorLifecycle,
        to: OperatorLifecycle,
    },
}

/// Durable catalog used by future reasoning cycles. Writes are crate-private so
/// callers must apply replicated learning mutations rather than mutate it directly.
#[derive(Default)]
pub struct DiscoveryCatalog {
    operators: RwLock<BTreeMap<DiscoveredOperatorId, Vec<DeclarativeOperator>>>,
}

impl DiscoveryCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn snapshot(&self) -> Vec<DeclarativeOperator> {
        self.operators
            .read()
            .values()
            .filter_map(|versions| versions.last().cloned())
            .collect()
    }

    pub fn snapshot_at(&self, lsn: u64) -> Vec<DeclarativeOperator> {
        self.operators
            .read()
            .values()
            .filter_map(|versions| {
                versions
                    .iter()
                    .rev()
                    .find(|operator| operator.committed_lsn <= lsn)
                    .cloned()
            })
            .collect()
    }

    pub(crate) fn history_snapshot(&self) -> Vec<DeclarativeOperator> {
        self.operators
            .read()
            .values()
            .flat_map(|versions| versions.iter().cloned())
            .collect()
    }

    pub(crate) fn replace_from(&self, operators: Vec<DeclarativeOperator>) {
        let mut rebuilt = BTreeMap::<DiscoveredOperatorId, Vec<DeclarativeOperator>>::new();
        for operator in operators {
            rebuilt.entry(operator.id).or_default().push(operator);
        }
        for versions in rebuilt.values_mut() {
            versions.sort_by_key(|operator| operator.committed_lsn);
        }
        *self.operators.write() = rebuilt;
    }

    pub(crate) fn apply(
        &self,
        mut operator: DeclarativeOperator,
        expected_previous: Option<OperatorLifecycle>,
        commit_lsn: u64,
    ) -> Result<(), DiscoveryCatalogError> {
        let mut catalog = self.operators.write();
        validate_catalog_transition(&catalog, &operator, expected_previous, commit_lsn)?;
        operator.committed_lsn = commit_lsn;
        catalog.entry(operator.id).or_default().push(operator);
        Ok(())
    }

    /// Fail-closed validation used before a replicated state-machine batch is
    /// allowed to publish any of its prepared deltas.
    pub(crate) fn prevalidate(
        &self,
        operator: &DeclarativeOperator,
        expected_previous: Option<OperatorLifecycle>,
        commit_lsn: u64,
    ) -> Result<(), DiscoveryCatalogError> {
        validate_catalog_transition(
            &self.operators.read(),
            operator,
            expected_previous,
            commit_lsn,
        )
    }

    pub fn recommend(&self, case: &DiscoveryCase) -> Vec<NovelResolution> {
        let mut resolutions: Vec<_> = self
            .operators
            .read()
            .values()
            .filter_map(|versions| versions.last())
            .filter(|operator| {
                matches!(
                    operator.lifecycle,
                    OperatorLifecycle::Admitted | OperatorLifecycle::Monitored
                ) && operator.matches(case)
            })
            .map(|operator| NovelResolution {
                operator_id: operator.id,
                resolution: operator.proposed_resolution(),
                matched_predicates: operator.predicates.clone(),
                source_motifs: operator.source_motifs.clone(),
                source_hypergraph_motifs: operator.source_hypergraph_motifs.clone(),
            })
            .collect();
        resolutions.sort_by_key(|resolution| (resolution.resolution, resolution.operator_id));
        resolutions.dedup_by_key(|resolution| resolution.resolution);
        resolutions
    }

    pub fn recommend_in_context(&self, context: &ReasoningContext) -> Vec<NovelResolution> {
        let mut resolutions = Vec::new();
        for operator in self
            .operators
            .read()
            .values()
            .filter_map(|versions| versions.last())
            .filter(|operator| {
                matches!(
                    operator.lifecycle,
                    OperatorLifecycle::Admitted | OperatorLifecycle::Monitored
                )
            })
        {
            let Ok(result) = execute_program(&operator.program, context) else {
                continue;
            };
            if !result.matched || !result.unsatisfied_constraints.is_empty() {
                continue;
            }
            for resolution in result.proposed_resolutions {
                resolutions.push(NovelResolution {
                    operator_id: operator.id,
                    resolution,
                    matched_predicates: operator.predicates.clone(),
                    source_motifs: operator.source_motifs.clone(),
                    source_hypergraph_motifs: operator.source_hypergraph_motifs.clone(),
                });
            }
        }
        resolutions.sort_by_key(|resolution| (resolution.resolution, resolution.operator_id));
        resolutions.dedup_by_key(|resolution| resolution.resolution);
        resolutions
    }
}

fn validate_catalog_transition(
    catalog: &BTreeMap<DiscoveredOperatorId, Vec<DeclarativeOperator>>,
    operator: &DeclarativeOperator,
    expected_previous: Option<OperatorLifecycle>,
    commit_lsn: u64,
) -> Result<(), DiscoveryCatalogError> {
    if !operator.has_valid_identity() {
        return Err(DiscoveryCatalogError::InvalidOperatorIdentity);
    }
    if operator.lifecycle == OperatorLifecycle::Admitted && operator.admission_authority.is_none() {
        return Err(DiscoveryCatalogError::MissingAdmissionAuthority);
    }
    if let Some(existing) = catalog
        .get(&operator.id)
        .and_then(|versions| versions.last())
    {
        if existing.predicates != operator.predicates
            || existing.effect != operator.effect
            || existing.version != operator.version
            || existing.source_motifs != operator.source_motifs
            || existing.source_hypergraph_motifs != operator.source_hypergraph_motifs
            || existing.program != operator.program
        {
            return Err(DiscoveryCatalogError::DefinitionChanged);
        }
        if commit_lsn <= existing.committed_lsn {
            return Err(DiscoveryCatalogError::NonMonotonicCommit {
                current: existing.committed_lsn,
                incoming: commit_lsn,
            });
        }
        if expected_previous != Some(existing.lifecycle)
            || !allowed_transition(existing.lifecycle, operator.lifecycle)
        {
            return Err(DiscoveryCatalogError::ForbiddenTransition {
                from: existing.lifecycle,
                to: operator.lifecycle,
            });
        }
    } else if expected_previous.is_some()
        || !matches!(
            operator.lifecycle,
            OperatorLifecycle::Generated | OperatorLifecycle::Provisional
        )
    {
        return Err(DiscoveryCatalogError::ForbiddenTransition {
            from: OperatorLifecycle::Generated,
            to: operator.lifecycle,
        });
    }
    Ok(())
}

fn allowed_transition(from: OperatorLifecycle, to: OperatorLifecycle) -> bool {
    matches!(
        (from, to),
        (OperatorLifecycle::Generated, OperatorLifecycle::Provisional)
            | (
                OperatorLifecycle::Provisional,
                OperatorLifecycle::FalsificationTesting
            )
            | (OperatorLifecycle::Provisional, OperatorLifecycle::Rejected)
            | (
                OperatorLifecycle::FalsificationTesting,
                OperatorLifecycle::Shadow
            )
            | (
                OperatorLifecycle::FalsificationTesting,
                OperatorLifecycle::Rejected
            )
            | (
                OperatorLifecycle::Shadow,
                OperatorLifecycle::ShadowValidated
            )
            | (OperatorLifecycle::Shadow, OperatorLifecycle::Rejected)
            | (
                OperatorLifecycle::ShadowValidated,
                OperatorLifecycle::Admitted
            )
            | (
                OperatorLifecycle::ShadowValidated,
                OperatorLifecycle::Rejected
            )
            | (OperatorLifecycle::Admitted, OperatorLifecycle::Monitored)
            | (OperatorLifecycle::Admitted, OperatorLifecycle::Deprecated)
            | (OperatorLifecycle::Admitted, OperatorLifecycle::Superseded)
            | (OperatorLifecycle::Monitored, OperatorLifecycle::Deprecated)
            | (OperatorLifecycle::Monitored, OperatorLifecycle::Superseded)
            | (OperatorLifecycle::Deprecated, OperatorLifecycle::Superseded)
    ) || from == to
}
