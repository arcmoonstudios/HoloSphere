//! Continuous governed discovery lifecycle, immutable safety kernel, audit chain, and revision.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::learning::discovery::active_experiment::{
    ActiveExperimentProposal, ExperimentPlanningPolicy, plan_active_experiments,
};
use crate::learning::discovery::dsl::{
    ConditionExpression, OperatorProgram, compose_programs, synthesize_program_from_motif,
    validate_program,
};
use crate::learning::discovery::evaluation::{
    CompetitiveEvaluationPolicy, CompetitiveOperatorEvaluation, EvaluationObservation,
    EvaluationRole, apply_competitive_evaluation, evaluate_program_competitively,
};
use crate::learning::discovery::hyper_motif::{
    HypergraphMotifPolicy, TemporalHypergraphMotif, mine_temporal_hypergraph_motifs,
};
use crate::learning::discovery::knowledge::KnowledgeSnapshot;
use crate::learning::discovery::mapping::{
    ConceptMappingHypothesis, MappingInductionPolicy, MappingLifecycle, MappingValidation,
    MappingValidationPolicy, derive_concept_behaviors, learn_concept_mappings,
    validate_concept_mapping,
};
use crate::learning::discovery::model::{DiscoveryCaseId, FeatureId};
use crate::learning::discovery::operator::{
    DeclarativeOperator, DiscoveredOperatorId, GovernanceAuthority, OperatorLifecycle,
};
use crate::learning::discovery::schema::{
    EvolvedSchemaId, EvolvedSchemaProposal, SchemaInductionPolicy, SchemaProposalState,
    SchemaValidation, SchemaValidationPolicy, induce_evolved_schemas, validate_evolved_schema,
};
use crate::learning::discovery::state::{
    DiscoveryStateMutation, DiscoveryStateSnapshot, GovernedMappingRecord, GovernedSchemaRecord,
};
use crate::learning::integrity::EmpiricalRootId;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImmutableSafetyKernel {
    version: u32,
    certified_retrieval_required: bool,
    provenance_required: bool,
    temporal_isolation_required: bool,
    governed_admission_required: bool,
    evidence_independence_required: bool,
    circular_support_prevention_required: bool,
    resource_limits_required: bool,
    sandbox_required: bool,
    audit_chain_required: bool,
    rollback_is_compensating_only: bool,
    maximum_operator_ast_nodes: u32,
    digest: [u8; 32],
}

impl ImmutableSafetyKernel {
    pub fn v1(maximum_operator_ast_nodes: u32) -> Self {
        let mut kernel = Self {
            version: 1,
            certified_retrieval_required: true,
            provenance_required: true,
            temporal_isolation_required: true,
            governed_admission_required: true,
            evidence_independence_required: true,
            circular_support_prevention_required: true,
            resource_limits_required: true,
            sandbox_required: true,
            audit_chain_required: true,
            rollback_is_compensating_only: true,
            maximum_operator_ast_nodes,
            digest: [0; 32],
        };
        kernel.digest = kernel.compute_digest();
        kernel
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn maximum_operator_ast_nodes(&self) -> u32 {
        self.maximum_operator_ast_nodes
    }

    pub fn verify(&self) -> Result<(), SafetyKernelViolation> {
        if self.version != 1
            || !self.certified_retrieval_required
            || !self.provenance_required
            || !self.temporal_isolation_required
            || !self.governed_admission_required
            || !self.evidence_independence_required
            || !self.circular_support_prevention_required
            || !self.resource_limits_required
            || !self.sandbox_required
            || !self.audit_chain_required
            || !self.rollback_is_compensating_only
            || self.maximum_operator_ast_nodes == 0
            || self.compute_digest() != self.digest
        {
            return Err(SafetyKernelViolation::ModifiedOrInvalid);
        }
        Ok(())
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_IMMUTABLE_DISCOVERY_KERNEL_V1");
        hasher.update(self.version.to_le_bytes());
        hasher.update([
            self.certified_retrieval_required as u8,
            self.provenance_required as u8,
            self.temporal_isolation_required as u8,
            self.governed_admission_required as u8,
            self.evidence_independence_required as u8,
            self.circular_support_prevention_required as u8,
            self.resource_limits_required as u8,
            self.sandbox_required as u8,
            self.audit_chain_required as u8,
            self.rollback_is_compensating_only as u8,
        ]);
        hasher.update(self.maximum_operator_ast_nodes.to_le_bytes());
        hasher.finalize().into()
    }
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum SafetyKernelViolation {
    #[error("immutable discovery safety kernel was modified or is invalid")]
    ModifiedOrInvalid,
    #[error("operator admission lacks external governance authority")]
    MissingGovernanceAuthority,
    #[error("operator violates immutable resource bounds")]
    ResourceBoundViolation,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryAuditAction {
    SchemaProposed([u8; 32]),
    MappingProposed([u8; 32]),
    SchemaTransition {
        schema: EvolvedSchemaId,
        from: Option<SchemaProposalState>,
        to: SchemaProposalState,
    },
    MappingTransition {
        mapping: crate::learning::discovery::mapping::MappingHypothesisId,
        from: Option<MappingLifecycle>,
        to: MappingLifecycle,
    },
    MotifDiscovered([u8; 32]),
    OperatorTransition {
        operator: DiscoveredOperatorId,
        from: Option<OperatorLifecycle>,
        to: OperatorLifecycle,
    },
    ExperimentProposed([u8; 32]),
    MonitoringEvaluated {
        operator: DiscoveredOperatorId,
        accuracy_q32: i64,
    },
    RevisionProposed {
        previous: DiscoveredOperatorId,
        revision: DiscoveredOperatorId,
    },
    CompensatingRollbackProposed {
        current: DiscoveredOperatorId,
        restoration: DiscoveredOperatorId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryAuditEntry {
    pub sequence: u64,
    pub lsn: u64,
    pub action: DiscoveryAuditAction,
    pub previous_hash: [u8; 32],
    pub entry_hash: [u8; 32],
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryAuditLog {
    entries: Vec<DiscoveryAuditEntry>,
}

impl DiscoveryAuditLog {
    pub fn append(&mut self, lsn: u64, action: DiscoveryAuditAction) -> [u8; 32] {
        let previous_hash = self
            .entries
            .last()
            .map_or([0; 32], |entry| entry.entry_hash);
        let sequence = self.entries.len() as u64;
        let entry_hash = audit_hash(sequence, lsn, &action, previous_hash);
        self.entries.push(DiscoveryAuditEntry {
            sequence,
            lsn,
            action,
            previous_hash,
            entry_hash,
        });
        entry_hash
    }

    pub fn entries(&self) -> &[DiscoveryAuditEntry] {
        &self.entries
    }

    pub fn verify(&self) -> bool {
        Self::verify_entries(&self.entries)
    }

    pub fn verify_entries(entries: &[DiscoveryAuditEntry]) -> bool {
        let mut previous = [0; 32];
        for (sequence, entry) in entries.iter().enumerate() {
            if entry.sequence != sequence as u64
                || entry.previous_hash != previous
                || entry.entry_hash
                    != audit_hash(entry.sequence, entry.lsn, &entry.action, previous)
            {
                return false;
            }
            previous = entry.entry_hash;
        }
        true
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MonitoringPolicy {
    pub min_observations: usize,
    pub min_accuracy_q32: i64,
    pub max_calibration_error_q32: i64,
    pub max_failure_ratio_q32: i64,
}

impl Default for MonitoringPolicy {
    fn default() -> Self {
        Self {
            min_observations: 20,
            min_accuracy_q32: q32(3, 4),
            max_calibration_error_q32: q32(1, 4),
            max_failure_ratio_q32: q32(1, 4),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorTransitionPlan {
    pub operator: DeclarativeOperator,
    pub expected_previous: Option<OperatorLifecycle>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaTransitionPlan {
    pub record: GovernedSchemaRecord,
    pub expected_previous: Option<SchemaProposalState>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappingTransitionPlan {
    pub record: GovernedMappingRecord,
    pub expected_previous: Option<MappingLifecycle>,
}

/// One consensus-log action. Actions are deliberately not collapsed into one
/// LSN because each epistemic transition is a separately observable commit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ReplicatedDiscoveryAction {
    Operator(OperatorTransitionPlan),
    State(DiscoveryStateMutation),
    Audit(DiscoveryAuditAction),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorRevisionProposal {
    pub previous: DiscoveredOperatorId,
    pub revision: DeclarativeOperator,
    pub excluded_counterexamples: BTreeSet<DiscoveryCaseId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContinuousDiscoveryPolicy {
    pub safety_kernel: ImmutableSafetyKernel,
    pub schema: SchemaInductionPolicy,
    pub schema_validation: SchemaValidationPolicy,
    pub mappings: MappingInductionPolicy,
    pub mapping_validation: MappingValidationPolicy,
    pub motifs: HypergraphMotifPolicy,
    pub evaluation: CompetitiveEvaluationPolicy,
    pub experiments: ExperimentPlanningPolicy,
    pub monitoring: MonitoringPolicy,
    pub admission_authority: Option<GovernanceAuthority>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContinuousDiscoveryInput {
    pub knowledge: KnowledgeSnapshot,
    /// Must be a later, independently pinned snapshot. If absent or not later,
    /// schema and mapping hypotheses remain Proposed.
    pub validation_knowledge: Option<KnowledgeSnapshot>,
    pub evaluation_observations: Vec<EvaluationObservation>,
    /// Completed, replicated experiments whose returned observations are
    /// automatically folded into this cycle's falsification evidence.
    pub completed_experiments: Vec<ActiveExperimentProposal>,
    pub incumbent_accuracy_q32: i64,
    pub admitted_operators: Vec<DeclarativeOperator>,
    /// Previously replicated Provisional/Falsification/Shadow candidates,
    /// including revisions, to continue through later discovery cycles.
    pub pending_operators: Vec<DeclarativeOperator>,
    pub monitoring_observations: BTreeMap<DiscoveredOperatorId, Vec<EvaluationObservation>>,
    /// Latest replicated governed state at the cycle's pinned read LSN.
    pub prior_discovery_state: Option<DiscoveryStateSnapshot>,
    /// Latest replicated operator versions, including terminal records. This
    /// makes motif re-mining idempotent across continuous cycles.
    pub known_operators: Vec<DeclarativeOperator>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContinuousDiscoveryReport {
    pub schemas: Vec<EvolvedSchemaProposal>,
    pub schema_validations: BTreeMap<EvolvedSchemaId, SchemaValidation>,
    pub schema_transition_plans: Vec<Vec<SchemaTransitionPlan>>,
    pub mappings: Vec<ConceptMappingHypothesis>,
    pub mapping_validations:
        BTreeMap<crate::learning::discovery::mapping::MappingHypothesisId, MappingValidation>,
    pub mapping_transition_plans: Vec<Vec<MappingTransitionPlan>>,
    pub motifs: Vec<TemporalHypergraphMotif>,
    pub evaluations: BTreeMap<DiscoveredOperatorId, CompetitiveOperatorEvaluation>,
    pub transition_plans: Vec<Vec<OperatorTransitionPlan>>,
    pub monitoring_transitions: Vec<OperatorTransitionPlan>,
    pub revisions: Vec<OperatorRevisionProposal>,
    pub experiments: Vec<ActiveExperimentProposal>,
    pub audit: DiscoveryAuditLog,
}

impl ContinuousDiscoveryReport {
    /// Produces the complete deterministic mutation stream needed to make this
    /// report durable. The caller submits each action through Raft in order.
    pub fn replicated_actions(
        &self,
        install_kernel: Option<ImmutableSafetyKernel>,
    ) -> Vec<ReplicatedDiscoveryAction> {
        let mut actions = Vec::new();
        if let Some(kernel) = install_kernel {
            actions.push(ReplicatedDiscoveryAction::State(
                DiscoveryStateMutation::InstallSafetyKernel { kernel },
            ));
        }
        for plans in &self.schema_transition_plans {
            for plan in plans {
                actions.push(ReplicatedDiscoveryAction::State(
                    DiscoveryStateMutation::UpsertSchema {
                        record: plan.record.clone(),
                        expected_previous: plan.expected_previous,
                    },
                ));
            }
        }
        for plans in &self.mapping_transition_plans {
            for plan in plans {
                actions.push(ReplicatedDiscoveryAction::State(
                    DiscoveryStateMutation::UpsertMapping {
                        record: plan.record.clone(),
                        expected_previous: plan.expected_previous,
                    },
                ));
            }
        }
        for plans in &self.transition_plans {
            actions.extend(
                plans
                    .iter()
                    .cloned()
                    .map(ReplicatedDiscoveryAction::Operator),
            );
        }
        for (operator, evaluation) in &self.evaluations {
            actions.push(ReplicatedDiscoveryAction::State(
                DiscoveryStateMutation::RecordEvaluation {
                    operator: *operator,
                    evaluation: evaluation.clone(),
                },
            ));
        }
        actions.extend(
            self.monitoring_transitions
                .iter()
                .cloned()
                .map(ReplicatedDiscoveryAction::Operator),
        );
        for revision in &self.revisions {
            let mut provisional = revision.revision.clone();
            provisional.lifecycle = OperatorLifecycle::Provisional;
            actions.push(ReplicatedDiscoveryAction::Operator(
                OperatorTransitionPlan {
                    operator: provisional,
                    expected_previous: None,
                },
            ));
        }
        for experiment in &self.experiments {
            actions.push(ReplicatedDiscoveryAction::State(
                DiscoveryStateMutation::UpsertExperiment {
                    experiment: experiment.clone(),
                    expected_previous: None,
                },
            ));
        }
        actions.extend(
            self.audit
                .entries()
                .iter()
                .map(|entry| ReplicatedDiscoveryAction::Audit(entry.action.clone())),
        );
        actions
    }
}

pub struct ContinuousDiscoveryEngine {
    pub policy: ContinuousDiscoveryPolicy,
    cycle: u64,
}

impl ContinuousDiscoveryEngine {
    pub fn new(policy: ContinuousDiscoveryPolicy) -> Result<Self, SafetyKernelViolation> {
        policy.safety_kernel.verify()?;
        Ok(Self { policy, cycle: 0 })
    }

    pub fn run_cycle(
        &mut self,
        input: &ContinuousDiscoveryInput,
    ) -> Result<ContinuousDiscoveryReport, SafetyKernelViolation> {
        self.policy.safety_kernel.verify()?;
        self.cycle = self.cycle.saturating_add(1);
        let audit_lsn = input.knowledge.lsn;
        let mut report = ContinuousDiscoveryReport::default();
        let induction_snapshot_roots: BTreeSet<_> = input
            .knowledge
            .concept_profiles
            .iter()
            .flat_map(|profile| profile.empirical_roots.iter().copied())
            .chain(
                input
                    .knowledge
                    .hyperedges
                    .iter()
                    .flat_map(|edge| edge.empirical_roots.iter().copied()),
            )
            .chain(
                input
                    .knowledge
                    .cases
                    .iter()
                    .flat_map(|case| case.empirical_roots.iter().copied()),
            )
            .collect();
        let mut evaluation_observations = input.evaluation_observations.clone();
        for experiment in input.completed_experiments.iter().filter(|experiment| {
            experiment.status
                == crate::learning::discovery::active_experiment::ExperimentStatus::Completed
        }) {
            if let Some(result) = &experiment.result {
                evaluation_observations.extend(result.observations.iter().cloned());
            }
        }
        let mut seen_observations = BTreeSet::new();
        evaluation_observations.retain(|observation| {
            seen_observations.insert(
                bincode::serialize(observation)
                    .expect("evaluation observations are deterministically serializable"),
            )
        });
        report.schemas = induce_evolved_schemas(&input.knowledge, self.policy.schema);
        report.mappings = learn_concept_mappings(
            &derive_concept_behaviors(&input.knowledge),
            self.policy.mappings,
        );
        report.motifs = mine_temporal_hypergraph_motifs(&input.knowledge, self.policy.motifs);
        let known_operator_ids: BTreeSet<_> = input
            .known_operators
            .iter()
            .chain(&input.admitted_operators)
            .chain(&input.pending_operators)
            .map(|operator| operator.id)
            .collect();
        let prior_schemas: BTreeMap<_, _> = input
            .prior_discovery_state
            .as_ref()
            .into_iter()
            .flat_map(|state| state.schemas.iter())
            .map(|record| (record.proposal.id, record))
            .collect();
        let prior_mappings: BTreeMap<_, _> = input
            .prior_discovery_state
            .as_ref()
            .into_iter()
            .flat_map(|state| state.mappings.iter())
            .map(|record| (record.hypothesis.id, record))
            .collect();
        for schema in &report.schemas {
            let mut plans = Vec::new();
            let prior = prior_schemas.get(&schema.id).copied();
            let base = prior.map_or(schema, |record| &record.proposal);
            let mut current = prior.map_or(SchemaProposalState::Proposed, |record| {
                record.proposal.state
            });
            if matches!(
                current,
                SchemaProposalState::Admitted
                    | SchemaProposalState::Rejected
                    | SchemaProposalState::Deprecated
            ) {
                continue;
            }
            if prior.is_none() {
                report
                    .audit
                    .append(audit_lsn, DiscoveryAuditAction::SchemaProposed(schema.id.0));
                push_schema_transition(
                    &mut plans,
                    &mut report.audit,
                    audit_lsn,
                    base,
                    SchemaProposalState::Proposed,
                    None,
                    None,
                    None,
                );
            }
            let validation = input
                .validation_knowledge
                .as_ref()
                .filter(|snapshot| snapshot.lsn > input.knowledge.lsn)
                .map(|validation_snapshot| {
                    validate_evolved_schema(
                        base,
                        validation_snapshot,
                        &report.schemas,
                        self.policy.schema_validation,
                    )
                })
                .or_else(|| prior.and_then(|record| record.validation.clone()));
            if let Some(validation) = validation {
                report
                    .schema_validations
                    .insert(base.id, validation.clone());
                if current == SchemaProposalState::Proposed {
                    push_schema_transition(
                        &mut plans,
                        &mut report.audit,
                        audit_lsn,
                        base,
                        SchemaProposalState::FalsificationTesting,
                        Some(SchemaProposalState::Proposed),
                        Some(validation.clone()),
                        None,
                    );
                    current = SchemaProposalState::FalsificationTesting;
                }
                if validation.passed {
                    if current == SchemaProposalState::FalsificationTesting {
                        push_schema_transition(
                            &mut plans,
                            &mut report.audit,
                            audit_lsn,
                            base,
                            SchemaProposalState::ShadowValidated,
                            Some(SchemaProposalState::FalsificationTesting),
                            Some(validation.clone()),
                            None,
                        );
                        current = SchemaProposalState::ShadowValidated;
                    }
                    if current == SchemaProposalState::ShadowValidated {
                        if let Some(authority) = self.policy.admission_authority {
                            push_schema_transition(
                                &mut plans,
                                &mut report.audit,
                                audit_lsn,
                                base,
                                SchemaProposalState::Admitted,
                                Some(SchemaProposalState::ShadowValidated),
                                Some(validation),
                                Some(authority),
                            );
                        }
                    }
                } else if current == SchemaProposalState::FalsificationTesting {
                    push_schema_transition(
                        &mut plans,
                        &mut report.audit,
                        audit_lsn,
                        base,
                        SchemaProposalState::Rejected,
                        Some(SchemaProposalState::FalsificationTesting),
                        Some(validation),
                        None,
                    );
                }
            }
            report.schema_transition_plans.push(plans);
        }
        for mapping in &report.mappings {
            let mut plans = Vec::new();
            let prior = prior_mappings.get(&mapping.id).copied();
            let base = prior.map_or(mapping, |record| &record.hypothesis);
            let mut current = prior.map_or(MappingLifecycle::Proposed, |record| {
                record.hypothesis.lifecycle
            });
            if matches!(
                current,
                MappingLifecycle::Confirmed
                    | MappingLifecycle::Rejected
                    | MappingLifecycle::Deprecated
            ) {
                continue;
            }
            if prior.is_none() {
                report.audit.append(
                    audit_lsn,
                    DiscoveryAuditAction::MappingProposed(mapping.id.0),
                );
                push_mapping_transition(
                    &mut plans,
                    &mut report.audit,
                    audit_lsn,
                    base,
                    MappingLifecycle::Proposed,
                    None,
                    None,
                    None,
                );
            }
            let validation = input
                .validation_knowledge
                .as_ref()
                .filter(|snapshot| snapshot.lsn > input.knowledge.lsn)
                .map(|validation_snapshot| {
                    validate_concept_mapping(
                        base,
                        validation_snapshot,
                        self.policy.mapping_validation,
                    )
                })
                .or_else(|| prior.and_then(|record| record.validation.clone()));
            if let Some(validation) = validation {
                report
                    .mapping_validations
                    .insert(base.id, validation.clone());
                if current == MappingLifecycle::Proposed {
                    push_mapping_transition(
                        &mut plans,
                        &mut report.audit,
                        audit_lsn,
                        base,
                        MappingLifecycle::FalsificationTesting,
                        Some(MappingLifecycle::Proposed),
                        Some(validation.clone()),
                        None,
                    );
                    current = MappingLifecycle::FalsificationTesting;
                }
                if validation.passed {
                    if current == MappingLifecycle::FalsificationTesting {
                        push_mapping_transition(
                            &mut plans,
                            &mut report.audit,
                            audit_lsn,
                            base,
                            MappingLifecycle::ShadowValidated,
                            Some(MappingLifecycle::FalsificationTesting),
                            Some(validation.clone()),
                            None,
                        );
                        current = MappingLifecycle::ShadowValidated;
                    }
                    if current == MappingLifecycle::ShadowValidated {
                        if let Some(authority) = self.policy.admission_authority {
                            push_mapping_transition(
                                &mut plans,
                                &mut report.audit,
                                audit_lsn,
                                base,
                                MappingLifecycle::Confirmed,
                                Some(MappingLifecycle::ShadowValidated),
                                Some(validation),
                                Some(authority),
                            );
                        }
                    }
                } else if current == MappingLifecycle::FalsificationTesting {
                    push_mapping_transition(
                        &mut plans,
                        &mut report.audit,
                        audit_lsn,
                        base,
                        MappingLifecycle::Rejected,
                        Some(MappingLifecycle::FalsificationTesting),
                        Some(validation),
                        None,
                    );
                }
            }
            report.mapping_transition_plans.push(plans);
        }

        let mut candidate_operators = Vec::new();
        for motif in &report.motifs {
            report
                .audit
                .append(audit_lsn, DiscoveryAuditAction::MotifDiscovered(motif.id.0));
            let mut program = synthesize_program_from_motif(motif);
            program.bounds.max_ast_nodes = program
                .bounds
                .max_ast_nodes
                .min(self.policy.safety_kernel.maximum_operator_ast_nodes());
            validate_program(&program)
                .map_err(|_| SafetyKernelViolation::ResourceBoundViolation)?;
            let Some(mut operator) = DeclarativeOperator::from_program(
                program,
                motif.id,
                motif.empirical_roots.clone(),
                motif.supporting_domains.clone(),
            ) else {
                continue;
            };
            if known_operator_ids.contains(&operator.id) {
                continue;
            }
            operator.epistemic.training_snapshot_roots = induction_snapshot_roots.clone();
            let independent_validation: Vec<_> = evaluation_observations
                .iter()
                .filter(|observation| {
                    !operator
                        .epistemic
                        .training_snapshot_roots
                        .contains(&observation.empirical_root)
                })
                .cloned()
                .collect();
            operator.epistemic.training_evidence = input
                .knowledge
                .cases
                .iter()
                .filter(|case| {
                    case.evidence_partition
                        == crate::learning::discovery::EvidencePartition::Discovery
                })
                .map(|case| case.id)
                .collect();
            operator.epistemic.validation_evidence = independent_validation
                .iter()
                .map(|observation| observation.case_id)
                .collect();
            let evaluation = evaluate_program_competitively(
                &operator.program,
                &independent_validation,
                input.incumbent_accuracy_q32,
                self.policy.evaluation,
            );
            apply_competitive_evaluation(&mut operator, &evaluation);
            report.evaluations.insert(operator.id, evaluation.clone());

            let mut transitions = Vec::new();
            push_transition(
                &mut transitions,
                &mut report.audit,
                audit_lsn,
                &operator,
                OperatorLifecycle::Provisional,
                None,
            );
            push_transition(
                &mut transitions,
                &mut report.audit,
                audit_lsn,
                &operator,
                OperatorLifecycle::FalsificationTesting,
                Some(OperatorLifecycle::Provisional),
            );
            if evaluation.passed {
                push_transition(
                    &mut transitions,
                    &mut report.audit,
                    audit_lsn,
                    &operator,
                    OperatorLifecycle::Shadow,
                    Some(OperatorLifecycle::FalsificationTesting),
                );
                let shadow_count = independent_validation
                    .iter()
                    .filter(|observation| observation.role == EvaluationRole::Shadow)
                    .count();
                if shadow_count >= self.policy.evaluation.min_observations {
                    push_transition(
                        &mut transitions,
                        &mut report.audit,
                        audit_lsn,
                        &operator,
                        OperatorLifecycle::ShadowValidated,
                        Some(OperatorLifecycle::Shadow),
                    );
                    if let Some(authority) = self.policy.admission_authority {
                        let mut admitted = operator.clone();
                        admitted.admission_authority = Some(authority);
                        push_transition(
                            &mut transitions,
                            &mut report.audit,
                            audit_lsn,
                            &admitted,
                            OperatorLifecycle::Admitted,
                            Some(OperatorLifecycle::ShadowValidated),
                        );
                        operator = admitted;
                    }
                }
            } else {
                push_transition(
                    &mut transitions,
                    &mut report.audit,
                    audit_lsn,
                    &operator,
                    OperatorLifecycle::Rejected,
                    Some(OperatorLifecycle::FalsificationTesting),
                );
            }
            if let Some(last) = transitions.last() {
                operator.lifecycle = last.operator.lifecycle;
            }
            report.transition_plans.push(transitions);
            candidate_operators.push(operator);
        }

        for pending in &input.pending_operators {
            if !matches!(
                pending.lifecycle,
                OperatorLifecycle::Provisional
                    | OperatorLifecycle::FalsificationTesting
                    | OperatorLifecycle::Shadow
                    | OperatorLifecycle::ShadowValidated
            ) || !pending.has_valid_identity()
                || validate_program(&pending.program).is_err()
            {
                continue;
            }
            let mut operator = pending.clone();
            let independent_validation: Vec<_> = evaluation_observations
                .iter()
                .filter(|observation| {
                    !operator
                        .epistemic
                        .training_snapshot_roots
                        .contains(&observation.empirical_root)
                })
                .cloned()
                .collect();
            operator.epistemic.validation_evidence = independent_validation
                .iter()
                .map(|observation| observation.case_id)
                .collect();
            let evaluation = evaluate_program_competitively(
                &operator.program,
                &independent_validation,
                input.incumbent_accuracy_q32,
                self.policy.evaluation,
            );
            apply_competitive_evaluation(&mut operator, &evaluation);
            report.evaluations.insert(operator.id, evaluation.clone());
            let mut transitions = Vec::new();
            let mut current = operator.lifecycle;
            if current == OperatorLifecycle::Provisional {
                push_transition(
                    &mut transitions,
                    &mut report.audit,
                    audit_lsn,
                    &operator,
                    OperatorLifecycle::FalsificationTesting,
                    Some(OperatorLifecycle::Provisional),
                );
                current = OperatorLifecycle::FalsificationTesting;
            }
            if evaluation.passed {
                if current == OperatorLifecycle::FalsificationTesting {
                    push_transition(
                        &mut transitions,
                        &mut report.audit,
                        audit_lsn,
                        &operator,
                        OperatorLifecycle::Shadow,
                        Some(OperatorLifecycle::FalsificationTesting),
                    );
                    current = OperatorLifecycle::Shadow;
                }
                let shadow_count = independent_validation
                    .iter()
                    .filter(|observation| observation.role == EvaluationRole::Shadow)
                    .count();
                if current == OperatorLifecycle::Shadow
                    && shadow_count >= self.policy.evaluation.min_observations
                {
                    push_transition(
                        &mut transitions,
                        &mut report.audit,
                        audit_lsn,
                        &operator,
                        OperatorLifecycle::ShadowValidated,
                        Some(OperatorLifecycle::Shadow),
                    );
                    current = OperatorLifecycle::ShadowValidated;
                }
                if current == OperatorLifecycle::ShadowValidated {
                    if let Some(authority) = self.policy.admission_authority {
                        operator.admission_authority = Some(authority);
                        push_transition(
                            &mut transitions,
                            &mut report.audit,
                            audit_lsn,
                            &operator,
                            OperatorLifecycle::Admitted,
                            Some(OperatorLifecycle::ShadowValidated),
                        );
                        current = OperatorLifecycle::Admitted;
                    }
                }
            } else {
                push_transition(
                    &mut transitions,
                    &mut report.audit,
                    audit_lsn,
                    &operator,
                    OperatorLifecycle::Rejected,
                    Some(current),
                );
                current = OperatorLifecycle::Rejected;
            }
            if let Some(last) = transitions.last() {
                operator = last.operator.clone();
            }
            if current == OperatorLifecycle::Admitted {
                if let Some(previous) = operator.epistemic.previous_version.and_then(|id| {
                    input
                        .admitted_operators
                        .iter()
                        .find(|candidate| candidate.id == id)
                }) {
                    let mut superseded = previous.clone();
                    superseded.lifecycle = OperatorLifecycle::Superseded;
                    report.monitoring_transitions.push(OperatorTransitionPlan {
                        operator: superseded,
                        expected_previous: Some(previous.lifecycle),
                    });
                }
            }
            report.transition_plans.push(transitions);
            candidate_operators.push(operator);
        }

        for admitted in &input.admitted_operators {
            let Some(observations) = input.monitoring_observations.get(&admitted.id) else {
                continue;
            };
            let independent_monitoring: Vec<_> = observations
                .iter()
                .filter(|observation| {
                    !admitted
                        .epistemic
                        .training_snapshot_roots
                        .contains(&observation.empirical_root)
                })
                .cloned()
                .collect();
            let monitoring_evaluation = evaluate_program_competitively(
                &admitted.program,
                &independent_monitoring,
                0,
                CompetitiveEvaluationPolicy {
                    min_observations: self.policy.monitoring.min_observations,
                    min_domains: 1,
                    min_independent_roots: self.policy.monitoring.min_observations,
                    min_accuracy_q32: self.policy.monitoring.min_accuracy_q32,
                    min_incumbent_improvement_q32: 0,
                    min_counterfactual_accuracy_q32: 0,
                    min_intervention_accuracy_q32: 0,
                    min_transfer_accuracy_q32: self.policy.monitoring.min_accuracy_q32,
                    max_calibration_error_q32: self.policy.monitoring.max_calibration_error_q32,
                    min_adversarial_robustness_q32: 0,
                    max_description_bytes: self.policy.evaluation.max_description_bytes,
                },
            );
            report.audit.append(
                audit_lsn,
                DiscoveryAuditAction::MonitoringEvaluated {
                    operator: admitted.id,
                    accuracy_q32: monitoring_evaluation.held_out_accuracy_q32,
                },
            );
            let failure_ratio =
                (1i64 << 32).saturating_sub(monitoring_evaluation.held_out_accuracy_q32);
            let target = if monitoring_evaluation.observations
                >= self.policy.monitoring.min_observations
                && monitoring_evaluation.held_out_accuracy_q32
                    >= self.policy.monitoring.min_accuracy_q32
                && monitoring_evaluation.calibration_error_q32
                    <= self.policy.monitoring.max_calibration_error_q32
                && failure_ratio <= self.policy.monitoring.max_failure_ratio_q32
            {
                OperatorLifecycle::Monitored
            } else {
                OperatorLifecycle::Deprecated
            };
            let mut monitored = admitted.clone();
            monitored.lifecycle = target;
            monitored.epistemic.monitoring_observations = monitoring_evaluation.observations as u64;
            monitored.epistemic.monitoring_failures = monitoring_evaluation
                .observations
                .saturating_sub(monitoring_evaluation.correct_predictions)
                as u64;
            apply_competitive_evaluation(&mut monitored, &monitoring_evaluation);
            report.monitoring_transitions.push(OperatorTransitionPlan {
                operator: monitored,
                expected_previous: Some(admitted.lifecycle),
            });
            if target == OperatorLifecycle::Deprecated {
                if let Some(revision) = revise_from_counterexamples(
                    admitted,
                    observations,
                    &monitoring_evaluation.counterexamples,
                    self.policy.safety_kernel.maximum_operator_ast_nodes(),
                ) {
                    report.audit.append(
                        audit_lsn,
                        DiscoveryAuditAction::RevisionProposed {
                            previous: admitted.id,
                            revision: revision.id,
                        },
                    );
                    report.revisions.push(OperatorRevisionProposal {
                        previous: admitted.id,
                        revision,
                        excluded_counterexamples: monitoring_evaluation.counterexamples,
                    });
                }
            }
        }

        report.experiments = plan_active_experiments(
            &candidate_operators,
            &report.evaluations,
            &input
                .knowledge
                .hyperedges
                .iter()
                .map(|edge| edge.domain)
                .collect(),
            self.policy.experiments,
        );
        if let Some(prior) = &input.prior_discovery_state {
            let existing: BTreeSet<_> = prior
                .experiments
                .iter()
                .map(|experiment| experiment.id)
                .collect();
            report
                .experiments
                .retain(|experiment| !existing.contains(&experiment.id));
        }
        for experiment in &report.experiments {
            report.audit.append(
                audit_lsn,
                DiscoveryAuditAction::ExperimentProposed(experiment.id.0),
            );
        }
        debug_assert!(report.audit.verify());
        Ok(report)
    }
}

/// Rollback never rewrites history. It creates a new revision whose program is
/// copied from an earlier admitted version and must traverse all admission gates.
pub fn propose_compensating_rollback(
    current: &DeclarativeOperator,
    historical: &DeclarativeOperator,
    audit: &mut DiscoveryAuditLog,
    lsn: u64,
) -> Option<OperatorRevisionProposal> {
    let restoration = current.revise_with_program(historical.program.clone())?;
    audit.append(
        lsn,
        DiscoveryAuditAction::CompensatingRollbackProposed {
            current: current.id,
            restoration: restoration.id,
        },
    );
    Some(OperatorRevisionProposal {
        previous: current.id,
        revision: restoration,
        excluded_counterexamples: BTreeSet::new(),
    })
}

fn push_transition(
    transitions: &mut Vec<OperatorTransitionPlan>,
    audit: &mut DiscoveryAuditLog,
    lsn: u64,
    base: &DeclarativeOperator,
    lifecycle: OperatorLifecycle,
    expected_previous: Option<OperatorLifecycle>,
) {
    let mut operator = base.clone();
    operator.lifecycle = lifecycle;
    if lifecycle != OperatorLifecycle::Admitted {
        operator.admission_authority = None;
    }
    audit.append(
        lsn,
        DiscoveryAuditAction::OperatorTransition {
            operator: operator.id,
            from: expected_previous,
            to: lifecycle,
        },
    );
    transitions.push(OperatorTransitionPlan {
        operator,
        expected_previous,
    });
}

#[allow(clippy::too_many_arguments)]
fn push_schema_transition(
    transitions: &mut Vec<SchemaTransitionPlan>,
    audit: &mut DiscoveryAuditLog,
    lsn: u64,
    base: &EvolvedSchemaProposal,
    state: SchemaProposalState,
    expected_previous: Option<SchemaProposalState>,
    validation: Option<SchemaValidation>,
    authority: Option<GovernanceAuthority>,
) {
    let mut proposal = base.clone();
    proposal.state = state;
    audit.append(
        lsn,
        DiscoveryAuditAction::SchemaTransition {
            schema: proposal.id,
            from: expected_previous,
            to: state,
        },
    );
    transitions.push(SchemaTransitionPlan {
        record: GovernedSchemaRecord {
            proposal,
            validation,
            authority,
        },
        expected_previous,
    });
}

#[allow(clippy::too_many_arguments)]
fn push_mapping_transition(
    transitions: &mut Vec<MappingTransitionPlan>,
    audit: &mut DiscoveryAuditLog,
    lsn: u64,
    base: &ConceptMappingHypothesis,
    lifecycle: MappingLifecycle,
    expected_previous: Option<MappingLifecycle>,
    validation: Option<MappingValidation>,
    authority: Option<GovernanceAuthority>,
) {
    let mut hypothesis = base.clone();
    hypothesis.lifecycle = lifecycle;
    audit.append(
        lsn,
        DiscoveryAuditAction::MappingTransition {
            mapping: hypothesis.id,
            from: expected_previous,
            to: lifecycle,
        },
    );
    transitions.push(MappingTransitionPlan {
        record: GovernedMappingRecord {
            hypothesis,
            validation,
            authority,
        },
        expected_previous,
    });
}

fn revise_from_counterexamples(
    operator: &DeclarativeOperator,
    observations: &[EvaluationObservation],
    counterexamples: &BTreeSet<DiscoveryCaseId>,
    max_ast_nodes: u32,
) -> Option<DeclarativeOperator> {
    let exclusions: Vec<_> = observations
        .iter()
        .filter(|observation| counterexamples.contains(&observation.case_id))
        .filter_map(|observation| observation.context.case.as_ref())
        .map(|case| {
            ConditionExpression::Not(Box::new(ConditionExpression::All(
                case.features
                    .iter()
                    .copied()
                    .map(ConditionExpression::FeaturePresent)
                    .collect(),
            )))
        })
        .collect();
    if exclusions.is_empty() {
        return None;
    }
    let guard = OperatorProgram {
        condition: ConditionExpression::All(exclusions),
        effects: Vec::new(),
        bounds: operator.program.bounds,
    };
    let mut revised_program =
        compose_programs(&[operator.program.clone(), guard], operator.program.bounds).ok()?;
    revised_program.bounds.max_ast_nodes = revised_program.bounds.max_ast_nodes.min(max_ast_nodes);
    validate_program(&revised_program).ok()?;
    let mut revision = operator.revise_with_program(revised_program)?;
    for observation in observations
        .iter()
        .filter(|observation| counterexamples.contains(&observation.case_id))
    {
        revision
            .epistemic
            .training_evidence
            .insert(observation.case_id);
        revision
            .epistemic
            .training_snapshot_roots
            .insert(observation.empirical_root);
    }
    Some(revision)
}

fn audit_hash(
    sequence: u64,
    lsn: u64,
    action: &DiscoveryAuditAction,
    previous_hash: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_DISCOVERY_AUDIT_CHAIN_V1");
    hasher.update(sequence.to_le_bytes());
    hasher.update(lsn.to_le_bytes());
    hasher.update(previous_hash);
    hasher.update(bincode::serialize(action).expect("audit actions are serializable"));
    hasher.finalize().into()
}

const fn q32(numerator: usize, denominator: usize) -> i64 {
    ((numerator as i128 * (1i128 << 32)) / denominator as i128) as i64
}
