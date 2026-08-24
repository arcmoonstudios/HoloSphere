//! Governed open-ended discovery: schema induction, motif mining, operator
//! synthesis, falsification, admission, and active experiment selection.

pub mod active_experiment;
pub mod checkpoint;
pub mod dsl;
pub mod engine;
pub mod evaluation;
pub mod experiment;
pub mod hyper_motif;
pub mod knowledge;
pub mod lifecycle;
pub mod mapping;
pub mod mining;
pub mod model;
pub mod operator;
pub mod projection;
pub mod schema;
pub mod state;
pub mod validation;

pub use active_experiment::{
    ActiveExperimentKind, ActiveExperimentProposal, ExperimentAuthorization,
    ExperimentExecutionError, ExperimentPlanningPolicy, ExperimentStatus, RiskLevel,
    SandboxExperimentResult, authorize_experiment, complete_experiment, execute_sandbox_experiment,
    plan_active_experiments, start_experiment,
};
pub use checkpoint::{
    DISCOVERY_CHECKPOINT_VERSION, DiscoveryCheckpointError, GovernedDiscoveryCheckpoint,
};

pub use dsl::{
    ComparisonOperator, ConditionExpression, DslEffect, HypergraphTransformation,
    NumericExpression, OperatorProgram, OperatorSandboxError, ProgramCost, ProgramResult,
    ReasoningContext, ResourceCostBounds, compose_programs, execute_program,
    synthesize_program_from_motif, validate_program,
};
pub use evaluation::{
    CompetitiveEvaluationPolicy, CompetitiveOperatorEvaluation, EvaluationObservation,
    EvaluationRole, apply_competitive_evaluation, epistemic_record_from_evaluation,
    evaluate_program_competitively,
};

pub use hyper_motif::{
    HypergraphMotifId, HypergraphMotifKind, HypergraphMotifPolicy, TemporalHypergraphMotif,
    mine_temporal_hypergraph_motifs,
};
pub use knowledge::{
    HyperedgeMember, KnowledgeSnapshot, NumericAttributeId, TemporalHyperedge, TemporalInterval,
};
pub use lifecycle::{
    ContinuousDiscoveryEngine, ContinuousDiscoveryInput, ContinuousDiscoveryPolicy,
    ContinuousDiscoveryReport, DiscoveryAuditAction, DiscoveryAuditEntry, DiscoveryAuditLog,
    ImmutableSafetyKernel, MappingTransitionPlan, MonitoringPolicy, OperatorRevisionProposal,
    OperatorTransitionPlan, ReplicatedDiscoveryAction, SafetyKernelViolation, SchemaTransitionPlan,
    propose_compensating_rollback,
};
pub use mapping::{
    ConceptBehavior, ConceptMappingHypothesis, ConfirmedConceptMappingIndex, MappingHypothesisId,
    MappingInductionPolicy, MappingLifecycle, MappingValidation, MappingValidationPolicy,
    derive_concept_behaviors, learn_concept_mappings, validate_concept_mapping,
};

pub use engine::{
    DiscoveryCatalog, DiscoveryCatalogError, DiscoveryGovernance, DiscoveryPolicy, DiscoveryReport,
    GovernedDiscoveryEngine, OperatorAssessment,
};
pub use experiment::{ExperimentKind, ExperimentProposal, ExperimentProposalId, plan_experiments};
pub use mining::{
    DiscoveredMotif, InducedSchemaProposal, MotifId, MotifMinerConfig, SchemaProposalId,
    induce_schemas, mine_motifs,
};
pub use model::{
    ConceptId, ConceptProfile, DiscoveryCase, DiscoveryCaseId, DiscoveryCorpus, DiscoveryOutcome,
    DomainId, EvidencePartition, FeatureId, ResolutionId, StructuralRole,
};
pub use operator::{
    DeclarativeOperator, DiscoveredOperatorId, GovernanceAuthority, NovelResolution,
    OperatorEffect, OperatorEpistemicRecord, OperatorLifecycle, OperatorPredicate,
};
pub use projection::{
    ExperienceProjectionPolicy, ExperienceProjectionReport, KnowledgeProjectionPolicy,
    KnowledgeProjectionReport, ProjectionSkip, ProjectionSkipReason, project_experience,
    project_knowledge,
};
pub use schema::{
    EvolvedSchemaId, EvolvedSchemaKind, EvolvedSchemaProposal, ProposedRole, SchemaInductionPolicy,
    SchemaProposalState, SchemaValidation, SchemaValidationPolicy, induce_evolved_schemas,
    materialize_proposed_relation_type, materialize_relation_type, validate_evolved_schema,
};
pub use state::{
    DiscoveryStateError, DiscoveryStateMutation, DiscoveryStateSnapshot, GovernedDiscoveryState,
    GovernedMappingRecord, GovernedSchemaRecord,
};
pub use validation::{OperatorValidation, OperatorValidationPolicy, validate_operator};

#[cfg(test)]
mod acceptance;

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::Arc;

    use super::*;
    use crate::learning::integrity::EmpiricalRootId;
    use crate::learning::{LearningMutation, LearningSegment};
    use crate::relation::RelationSegment;

    const BOUNDED_CAPACITY: FeatureId = FeatureId(1);
    const DEMAND_EXCEEDS_RELEASE: FeatureId = FeatureId(2);
    const QUEUE_AMPLIFICATION: FeatureId = FeatureId(3);
    const BACKPRESSURE: ResolutionId = ResolutionId(10);

    fn case(
        id: u64,
        domain: u64,
        features: &[FeatureId],
        resolution: Option<ResolutionId>,
        outcome: DiscoveryOutcome,
        evidence_partition: EvidencePartition,
    ) -> DiscoveryCase {
        DiscoveryCase {
            id: DiscoveryCaseId(id),
            domain: DomainId(domain),
            snapshot_lsn: id,
            features: features.iter().copied().collect(),
            observed_resolution: resolution,
            outcome,
            evidence_partition,
            empirical_roots: BTreeSet::from([EmpiricalRootId(id)]),
            certified_evidence: true,
        }
    }

    fn cross_domain_corpus() -> DiscoveryCorpus {
        let mut cases = Vec::new();
        let mut next_id = 1u64;
        for domain in 1..=3u64 {
            for _ in 0..2 {
                cases.push(case(
                    next_id,
                    domain,
                    &[
                        BOUNDED_CAPACITY,
                        DEMAND_EXCEEDS_RELEASE,
                        QUEUE_AMPLIFICATION,
                    ],
                    Some(BACKPRESSURE),
                    DiscoveryOutcome::Successful,
                    EvidencePartition::Discovery,
                ));
                next_id += 1;
            }
            // Same resolution without the discovered structure fails, lowering
            // the unconditional baseline and giving the motif predictive lift.
            cases.push(case(
                next_id,
                domain,
                &[FeatureId(100 + domain)],
                Some(BACKPRESSURE),
                DiscoveryOutcome::Failed,
                EvidencePartition::Discovery,
            ));
            next_id += 1;

            // These independent cases are reserved before mining and are the
            // only evidence allowed to authorize admission.
            for _ in 0..2 {
                cases.push(case(
                    next_id,
                    domain,
                    &[
                        BOUNDED_CAPACITY,
                        DEMAND_EXCEEDS_RELEASE,
                        QUEUE_AMPLIFICATION,
                    ],
                    Some(BACKPRESSURE),
                    DiscoveryOutcome::Successful,
                    EvidencePartition::Validation,
                ));
                next_id += 1;
            }
            cases.push(case(
                next_id,
                domain,
                &[FeatureId(200 + domain)],
                Some(BACKPRESSURE),
                DiscoveryOutcome::Failed,
                EvidencePartition::Validation,
            ));
            next_id += 1;
        }

        let shared_roles = BTreeSet::from([StructuralRole {
            relation_arity: 3,
            role_ordinal: 1,
            peer_role_count: 2,
            temporal_position: 0,
        }]);
        let concept_profiles = (1..=3u64)
            .map(|domain| ConceptProfile {
                domain: DomainId(domain),
                concept: ConceptId(1_000 + domain),
                capabilities: BTreeSet::from([BOUNDED_CAPACITY]),
                roles: shared_roles.clone(),
                empirical_roots: BTreeSet::from([EmpiricalRootId(10_000 + domain)]),
                certified_evidence: true,
            })
            .collect();

        DiscoveryCorpus {
            cases,
            concept_profiles,
        }
    }

    fn governed_policy() -> DiscoveryPolicy {
        DiscoveryPolicy {
            mining: MotifMinerConfig {
                max_features_per_case: 8,
                max_condition_terms: 3,
                min_successes: 5,
                min_domains: 3,
                max_motifs: 32,
            },
            validation: OperatorValidationPolicy {
                min_evaluated_cases: 6,
                min_supporting_domains: 3,
                min_independent_roots: 6,
                min_precision_q32: (0.95 * (1u64 << 32) as f64) as i64,
                min_lift_q32: (0.25 * (1u64 << 32) as f64) as i64,
                max_contradiction_ratio_q32: 0,
                min_held_out_domain_passes: 3,
            },
            schema_min_domains: 3,
            schema_min_members: 3,
            max_experiments: 8,
            governance: DiscoveryGovernance::PolicyAuthorized {
                policy_id: 77,
                version: 1,
            },
        }
    }

    #[test]
    fn discovers_cross_domain_schema_and_non_scripted_resolution() {
        let corpus = cross_domain_corpus();
        let report = GovernedDiscoveryEngine::new(governed_policy()).discover(&corpus);
        assert_eq!(report.schemas.len(), 1);
        assert_eq!(report.schemas[0].supporting_domains.len(), 3);

        let candidate = report
            .operators
            .iter()
            .find(|assessment| {
                assessment.operator.lifecycle == OperatorLifecycle::Shadow
                    && assessment.operator.proposed_resolution() == BACKPRESSURE
            })
            .expect("cross-domain law should satisfy all governance gates");
        assert_eq!(candidate.validation.supporting_domains.len(), 3);
        assert_eq!(candidate.validation.independent_roots.len(), 6);
        assert!(candidate.validation.predictive_lift_q32 > 0);

        // Persist the exact lifecycle through replicated mutations. Direct
        // registration as Admitted is intentionally forbidden by the catalog.
        let learning = Arc::new(LearningSegment::new(1));
        let relations = RelationSegment::new(1, 1);
        let mut provisional = candidate.operator.clone();
        provisional.lifecycle = OperatorLifecycle::Provisional;
        provisional.admission_authority = None;
        LearningMutation::UpsertDiscoveredOperator {
            operator: provisional,
            expected_previous: None,
        }
        .apply(&learning, &relations, 100)
        .unwrap();

        let mut falsification = candidate.operator.clone();
        falsification.lifecycle = OperatorLifecycle::FalsificationTesting;
        falsification.admission_authority = None;
        LearningMutation::UpsertDiscoveredOperator {
            operator: falsification,
            expected_previous: Some(OperatorLifecycle::Provisional),
        }
        .apply(&learning, &relations, 101)
        .unwrap();

        let mut shadow = candidate.operator.clone();
        shadow.lifecycle = OperatorLifecycle::Shadow;
        shadow.admission_authority = None;
        LearningMutation::UpsertDiscoveredOperator {
            operator: shadow,
            expected_previous: Some(OperatorLifecycle::FalsificationTesting),
        }
        .apply(&learning, &relations, 102)
        .unwrap();

        let mut shadow_validated = candidate.operator.clone();
        shadow_validated.lifecycle = OperatorLifecycle::ShadowValidated;
        shadow_validated.admission_authority = None;
        LearningMutation::UpsertDiscoveredOperator {
            operator: shadow_validated,
            expected_previous: Some(OperatorLifecycle::Shadow),
        }
        .apply(&learning, &relations, 103)
        .unwrap();

        let mut admitted = candidate.operator.clone();
        admitted.lifecycle = OperatorLifecycle::Admitted;
        admitted.admission_authority = Some(GovernanceAuthority::ReplicatedPolicy {
            policy_id: 77,
            version: 1,
        });
        LearningMutation::UpsertDiscoveredOperator {
            operator: admitted,
            expected_previous: Some(OperatorLifecycle::ShadowValidated),
        }
        .apply(&learning, &relations, 104)
        .unwrap();

        let new_problem = case(
            999,
            99,
            &[
                BOUNDED_CAPACITY,
                DEMAND_EXCEEDS_RELEASE,
                QUEUE_AMPLIFICATION,
            ],
            None,
            DiscoveryOutcome::Unknown,
            EvidencePartition::Validation,
        );
        let recommendations = learning.discovery.recommend(&new_problem);
        assert!(
            recommendations
                .iter()
                .any(|candidate| candidate.resolution == BACKPRESSURE)
        );
        let at_shadow = learning.read_snapshot(102).discovered_operators();
        assert_eq!(at_shadow.len(), 1);
        assert_eq!(at_shadow[0].lifecycle, OperatorLifecycle::Shadow);
        let at_admission = learning.read_snapshot(104).discovered_operators();
        assert_eq!(at_admission.len(), 1);
        assert_eq!(at_admission[0].lifecycle, OperatorLifecycle::Admitted);

        let compacted = learning.compact(2);
        assert_eq!(
            compacted.read_snapshot(102).discovered_operators()[0].lifecycle,
            OperatorLifecycle::Shadow
        );
    }

    #[test]
    fn propose_only_mode_cannot_admit_its_own_operator() {
        let corpus = cross_domain_corpus();
        let mut policy = governed_policy();
        policy.governance = DiscoveryGovernance::ProposeOnly;
        let report = GovernedDiscoveryEngine::new(policy).discover(&corpus);
        assert!(report.operators.iter().any(|assessment| {
            assessment.operator.lifecycle == OperatorLifecycle::Shadow
                && assessment.operator.admission_authority.is_none()
        }));
        assert!(
            report
                .operators
                .iter()
                .all(|assessment| assessment.operator.lifecycle != OperatorLifecycle::Admitted)
        );
    }

    #[test]
    fn training_correlation_is_rejected_when_reserved_evidence_falsifies_it() {
        let mut corpus = cross_domain_corpus();
        for case in &mut corpus.cases {
            if case.evidence_partition == EvidencePartition::Validation
                && case.features.contains(&BOUNDED_CAPACITY)
            {
                case.outcome = DiscoveryOutcome::Failed;
            }
        }

        let report = GovernedDiscoveryEngine::new(governed_policy()).discover(&corpus);
        assert!(
            !report.motifs.is_empty(),
            "training should still find the motif"
        );
        assert!(report.operators.iter().all(|assessment| {
            assessment.operator.lifecycle == OperatorLifecycle::Rejected
                && assessment.validation.held_out_domain_failures > 0
        }));
    }

    #[test]
    fn validation_reusing_discovery_roots_cannot_admit_an_operator() {
        let mut corpus = cross_domain_corpus();
        for case in &mut corpus.cases {
            if case.evidence_partition == EvidencePartition::Validation {
                case.empirical_roots = BTreeSet::from([EmpiricalRootId(1)]);
            }
        }

        let report = GovernedDiscoveryEngine::new(governed_policy()).discover(&corpus);
        assert!(!report.motifs.is_empty());
        assert!(report.operators.iter().all(|assessment| {
            assessment.operator.lifecycle != OperatorLifecycle::Admitted
                && assessment.validation.evaluated_cases == 0
        }));
    }

    #[test]
    fn catalog_rejects_direct_admission_and_definition_tampering() {
        let report =
            GovernedDiscoveryEngine::new(governed_policy()).discover(&cross_domain_corpus());
        let mut admitted = report
            .operators
            .iter()
            .find(|assessment| assessment.operator.lifecycle == OperatorLifecycle::Shadow)
            .unwrap()
            .operator
            .clone();
        admitted.lifecycle = OperatorLifecycle::Admitted;
        admitted.admission_authority = Some(GovernanceAuthority::ReplicatedPolicy {
            policy_id: 77,
            version: 1,
        });
        let learning = Arc::new(LearningSegment::new(1));
        let relations = RelationSegment::new(1, 1);

        let direct = LearningMutation::UpsertDiscoveredOperator {
            operator: admitted.clone(),
            expected_previous: None,
        }
        .apply(&learning, &relations, 100);
        assert!(direct.is_err());
        assert!(learning.discovery.snapshot().is_empty());

        let mut tampered = admitted;
        tampered.lifecycle = OperatorLifecycle::Provisional;
        tampered.admission_authority = None;
        tampered
            .predicates
            .push(OperatorPredicate::HasFeature(FeatureId(999)));
        let rejected = LearningMutation::UpsertDiscoveredOperator {
            operator: tampered,
            expected_previous: None,
        }
        .apply(&learning, &relations, 101);
        assert!(rejected.is_err());
        assert!(learning.discovery.snapshot().is_empty());
    }

    #[test]
    fn competing_rules_generate_bounded_shadow_experiment() {
        let motif_a = DiscoveredMotif {
            id: MotifId([1; 32]),
            conditions: vec![BOUNDED_CAPACITY],
            resolution: ResolutionId(10),
            successes: 2,
            contradictions: 1,
            supporting_domains: BTreeSet::from([DomainId(1)]),
            empirical_roots: BTreeSet::new(),
            precision_q32: 1 << 31,
        };
        let motif_b = DiscoveredMotif {
            resolution: ResolutionId(20),
            id: MotifId([2; 32]),
            ..motif_a.clone()
        };
        let operators = vec![
            DeclarativeOperator::from_motif(&motif_a),
            DeclarativeOperator::from_motif(&motif_b),
        ];
        let validations = BTreeMap::from([
            (
                operators[0].id,
                OperatorValidation {
                    supporting_domains: BTreeSet::from([DomainId(1)]),
                    ..OperatorValidation::default()
                },
            ),
            (
                operators[1].id,
                OperatorValidation {
                    supporting_domains: BTreeSet::from([DomainId(1)]),
                    ..OperatorValidation::default()
                },
            ),
        ]);
        let experiments = plan_experiments(
            &operators,
            &validations,
            &BTreeSet::from([DomainId(1), DomainId(2)]),
            1,
        );
        assert_eq!(experiments.len(), 1);
        assert_eq!(experiments[0].kind, ExperimentKind::ShadowReplay);
        assert_eq!(experiments[0].candidate_resolutions.len(), 2);
        assert_eq!(experiments[0].target_domains, BTreeSet::from([DomainId(2)]));
        assert!(!experiments[0].requires_external_authorization);
    }
}
