use std::collections::{BTreeMap, BTreeSet};

use super::*;
use crate::learning::integrity::EmpiricalRootId;

const CAPACITY: FeatureId = FeatureId(1);
const PRESSURE: FeatureId = FeatureId(2);
const RESOLUTION_A: ResolutionId = ResolutionId(10);
const RESOLUTION_B: ResolutionId = ResolutionId(20);

fn profile(domain: u64, concept: u64, capabilities: &[FeatureId], root: u64) -> ConceptProfile {
    ConceptProfile {
        domain: DomainId(domain),
        concept: ConceptId(concept),
        capabilities: capabilities.iter().copied().collect(),
        roles: BTreeSet::from([StructuralRole {
            relation_arity: 3,
            role_ordinal: 1,
            peer_role_count: 2,
            temporal_position: 0,
        }]),
        empirical_roots: BTreeSet::from([EmpiricalRootId(root)]),
        certified_evidence: true,
    }
}

fn edge(
    id: u64,
    domain: u64,
    concepts: &[u64],
    roles: &[u16],
    from: u64,
    outcome: DiscoveryOutcome,
    resolution: Option<ResolutionId>,
) -> TemporalHyperedge {
    TemporalHyperedge {
        id,
        domain: DomainId(domain),
        relation_type: domain as u32 + 100,
        members: concepts
            .iter()
            .zip(roles)
            .map(|(concept, role)| HyperedgeMember {
                concept: ConceptId(*concept),
                role: *role,
            })
            .collect(),
        interval: TemporalInterval {
            valid_from_lsn: from,
            valid_until_lsn: None,
        },
        causal_predecessors: BTreeSet::new(),
        context_features: BTreeSet::from([CAPACITY, PRESSURE]),
        numeric_context_q32: BTreeMap::new(),
        observed_resolution: resolution,
        outcome,
        empirical_roots: BTreeSet::from([EmpiricalRootId(id)]),
        certified_evidence: true,
    }
}

#[test]
fn acceptance_induces_classes_relations_roles_hierarchies_and_competing_mappings() {
    let mut snapshot = KnowledgeSnapshot {
        lsn: 1_000,
        concept_profiles: vec![
            profile(1, 101, &[CAPACITY], 1),
            profile(2, 201, &[CAPACITY], 2),
            profile(3, 301, &[CAPACITY], 3),
            profile(4, 401, &[CAPACITY, PRESSURE], 4),
            profile(5, 501, &[CAPACITY, PRESSURE], 5),
        ],
        ..KnowledgeSnapshot::default()
    };
    snapshot.hyperedges = vec![
        edge(
            11,
            1,
            &[101, 102, 103],
            &[10, 20, 20],
            10,
            DiscoveryOutcome::Successful,
            Some(RESOLUTION_A),
        ),
        edge(
            12,
            2,
            &[201, 202, 203],
            &[30, 40, 40],
            10,
            DiscoveryOutcome::Successful,
            Some(RESOLUTION_A),
        ),
        edge(
            13,
            3,
            &[301, 302, 303],
            &[50, 60, 60],
            10,
            DiscoveryOutcome::Successful,
            Some(RESOLUTION_A),
        ),
    ];

    let schemas = induce_evolved_schemas(
        &snapshot,
        SchemaInductionPolicy {
            min_domains: 2,
            min_members: 2,
            min_independent_roots: 2,
            max_proposals: 100,
        },
    );
    assert!(
        schemas
            .iter()
            .all(|schema| schema.state == SchemaProposalState::Proposed)
    );
    assert!(
        schemas
            .iter()
            .any(|schema| matches!(schema.kind, EvolvedSchemaKind::EntityClass { .. }))
    );
    let relation = schemas
        .iter()
        .find(|schema| matches!(schema.kind, EvolvedSchemaKind::RelationType { .. }))
        .expect("N-ary relation type must be induced");
    let materialized = materialize_proposed_relation_type(relation, 99).unwrap();
    assert_eq!(
        materialized.state,
        crate::relation::RelationTypeState::Proposed
    );
    assert_eq!(materialized.roles.len(), 2);
    assert!(
        schemas
            .iter()
            .any(|schema| matches!(schema.kind, EvolvedSchemaKind::ConceptEquivalence { .. }))
    );
    assert!(
        schemas
            .iter()
            .any(|schema| matches!(schema.kind, EvolvedSchemaKind::Generalization { .. }))
    );
    assert!(
        schemas
            .iter()
            .any(|schema| matches!(schema.kind, EvolvedSchemaKind::Specialization { .. }))
    );

    let behaviors = derive_concept_behaviors(&snapshot);
    let mappings = learn_concept_mappings(
        &behaviors,
        MappingInductionPolicy {
            min_role_similarity_q32: 0,
            min_outcome_similarity_q32: 0,
            min_total_score_q32: 0,
            min_independent_roots: 2,
            max_hypotheses: 100,
        },
    );
    assert!(mappings.len() >= 3);
    assert!(
        mappings
            .iter()
            .all(|mapping| mapping.lifecycle == MappingLifecycle::Proposed)
    );
    assert!(
        mappings
            .iter()
            .any(|mapping| !mapping.competing_hypotheses.is_empty())
    );
}

#[test]
fn acceptance_mines_nary_causal_temporal_invariant_and_anomalous_motifs() {
    let mut edges = Vec::new();
    for domain in 1..=4u64 {
        let before_id = domain * 10;
        let after_id = before_id + 1;
        edges.push(edge(
            before_id,
            domain,
            &[domain * 100 + 1, domain * 100 + 2],
            &[domain as u16 * 10, domain as u16 * 10 + 1],
            10,
            DiscoveryOutcome::Failed,
            None,
        ));
        let mut after = edge(
            after_id,
            domain,
            &[domain * 100 + 1, domain * 100 + 2],
            &[domain as u16 * 10 + 2, domain as u16 * 10 + 3],
            20,
            DiscoveryOutcome::Successful,
            Some(RESOLUTION_A),
        );
        after.causal_predecessors.insert(before_id);
        edges.push(after);
        edges.push(edge(
            100 + domain,
            domain,
            &[domain * 100 + 3, domain * 100 + 4, domain * 100 + 5],
            &[1, 2, 2],
            30,
            DiscoveryOutcome::Successful,
            Some(RESOLUTION_A),
        ));
    }
    edges.push(edge(
        999,
        5,
        &[901, 902, 903],
        &[7, 8, 8],
        30,
        DiscoveryOutcome::Failed,
        Some(RESOLUTION_A),
    ));
    let snapshot = KnowledgeSnapshot {
        lsn: 1_000,
        hyperedges: edges,
        ..KnowledgeSnapshot::default()
    };
    let motifs = mine_temporal_hypergraph_motifs(
        &snapshot,
        HypergraphMotifPolicy {
            min_support: 2,
            min_domains: 2,
            min_independent_roots: 2,
            anomaly_min_baseline: 4,
            max_temporal_gap_lsn: 100,
            max_motifs: 100,
        },
    );
    assert!(motifs.iter().any(|motif| matches!(
        motif.kind,
        HypergraphMotifKind::RepeatedNaryStructure { arity: 3, .. }
    )));
    assert!(
        motifs
            .iter()
            .any(|motif| matches!(motif.kind, HypergraphMotifKind::CausalSequence { .. }))
    );
    assert!(
        motifs
            .iter()
            .any(|motif| matches!(motif.kind, HypergraphMotifKind::BeforeAfterOutcome { .. }))
    );
    assert!(motifs.iter().any(|motif| matches!(
        motif.kind,
        HypergraphMotifKind::DomainInvariantRoleArrangement { .. }
    )));
    assert!(
        motifs
            .iter()
            .any(|motif| matches!(motif.kind, HypergraphMotifKind::OutcomeAnomaly { .. }))
    );
}

fn reasoning_case(id: u64, domain: u64) -> DiscoveryCase {
    DiscoveryCase {
        id: DiscoveryCaseId(id),
        domain: DomainId(domain),
        snapshot_lsn: id,
        features: BTreeSet::from([CAPACITY, PRESSURE]),
        observed_resolution: Some(RESOLUTION_A),
        outcome: DiscoveryOutcome::Successful,
        evidence_partition: EvidencePartition::Validation,
        empirical_roots: BTreeSet::from([EmpiricalRootId(id)]),
        certified_evidence: true,
    }
}

#[test]
fn acceptance_dsl_supports_numeric_temporal_causal_constraints_and_is_sandboxed() {
    let demand = NumericAttributeId(1);
    let release = NumericAttributeId(2);
    let derived = NumericAttributeId(3);
    let motif = HypergraphMotifId([9; 32]);
    let constraint = FeatureId(99);
    let program = OperatorProgram {
        condition: ConditionExpression::All(vec![
            ConditionExpression::FeaturePresent(CAPACITY),
            ConditionExpression::NumericCompare {
                left: NumericExpression::Attribute(demand),
                operator: ComparisonOperator::Greater,
                right: NumericExpression::Attribute(release),
            },
            ConditionExpression::FeaturePersists {
                feature: PRESSURE,
                minimum_duration_lsn: 5,
            },
            ConditionExpression::CausalMotifPresent(motif),
        ]),
        effects: vec![
            DslEffect::PredictOutcome(DiscoveryOutcome::Failed),
            DslEffect::ProposeResolution(RESOLUTION_A),
            DslEffect::ProposeResolution(RESOLUTION_B),
            DslEffect::SetDerivedNumeric {
                attribute: derived,
                value: NumericExpression::Subtract(
                    Box::new(NumericExpression::Attribute(demand)),
                    Box::new(NumericExpression::Attribute(release)),
                ),
            },
            DslEffect::RequireConstraint(constraint),
            DslEffect::ProposeHypergraphTransformation(
                HypergraphTransformation::InstantiateCanonicalMotif { motif },
            ),
        ],
        bounds: ResourceCostBounds::default(),
    };
    let mut context = ReasoningContext {
        case: Some(reasoning_case(1, 1)),
        numeric_values_q32: BTreeMap::from([(demand, 10 << 32), (release, 4 << 32)]),
        feature_durations_lsn: BTreeMap::from([(PRESSURE, 10)]),
        causal_motifs: BTreeSet::from([motif]),
        ..ReasoningContext::default()
    };
    let constrained = execute_program(&program, &context).unwrap();
    assert!(constrained.matched);
    assert!(constrained.proposed_resolutions.is_empty());
    context.satisfied_constraints.insert(constraint);
    let result = execute_program(&program, &context).unwrap();
    assert_eq!(result.proposed_resolutions.len(), 2);
    assert_eq!(result.derived_numeric_q32[&derived], 6 << 32);
    assert_eq!(result.proposed_hypergraph_transformations.len(), 1);

    let mut too_small = program.clone();
    too_small.bounds.max_ast_nodes = 1;
    assert!(matches!(
        validate_program(&too_small),
        Err(OperatorSandboxError::AstNodeLimit { .. })
    ));
}

fn evaluation_program(outcome: DiscoveryOutcome, resolution: ResolutionId) -> OperatorProgram {
    OperatorProgram {
        condition: ConditionExpression::True,
        effects: vec![
            DslEffect::PredictOutcome(outcome),
            DslEffect::ProposeResolution(resolution),
        ],
        bounds: ResourceCostBounds::default(),
    }
}

fn evaluation_observations(program_context: ReasoningContext) -> Vec<EvaluationObservation> {
    let roles = [
        EvaluationRole::Counterfactual,
        EvaluationRole::CausalIntervention,
        EvaluationRole::Adversarial,
        EvaluationRole::Shadow,
    ];
    (1..=12u64)
        .map(|id| EvaluationObservation {
            case_id: DiscoveryCaseId(id),
            domain: DomainId((id % 3) + 1),
            empirical_root: EmpiricalRootId(1_000 + id),
            role: roles[(id as usize - 1) % roles.len()],
            context: program_context.clone(),
            actual_outcome: DiscoveryOutcome::Successful,
            actual_resolution: Some(RESOLUTION_A),
            predicted_confidence_q32: 1i64 << 32,
            adversarial_group: (roles[(id as usize - 1) % roles.len()]
                == EvaluationRole::Adversarial)
                .then_some(id),
        })
        .collect()
}

fn permissive_competitive_policy() -> CompetitiveEvaluationPolicy {
    CompetitiveEvaluationPolicy {
        min_observations: 8,
        min_domains: 3,
        min_independent_roots: 8,
        min_accuracy_q32: q32(9, 10),
        min_incumbent_improvement_q32: q32(1, 10),
        min_counterfactual_accuracy_q32: q32(9, 10),
        min_intervention_accuracy_q32: q32(9, 10),
        min_transfer_accuracy_q32: q32(9, 10),
        max_calibration_error_q32: 0,
        min_adversarial_robustness_q32: q32(9, 10),
        max_description_bytes: 16_384,
    }
}

#[test]
fn acceptance_evaluation_competes_on_prediction_causality_mdl_calibration_and_robustness() {
    let observations = evaluation_observations(ReasoningContext::default());
    let good = evaluate_program_competitively(
        &evaluation_program(DiscoveryOutcome::Successful, RESOLUTION_A),
        &observations,
        q32(1, 2),
        permissive_competitive_policy(),
    );
    assert!(good.passed);
    assert_eq!(good.counterfactual_accuracy_q32, 1i64 << 32);
    assert_eq!(good.intervention_accuracy_q32, 1i64 << 32);
    assert_eq!(good.adversarial_robustness_q32, 1i64 << 32);
    assert!(good.incumbent_improvement_q32 > 0);
    assert!(good.minimum_description_length_score_q32 > 0);

    let bad = evaluate_program_competitively(
        &evaluation_program(DiscoveryOutcome::Failed, RESOLUTION_B),
        &observations,
        q32(1, 2),
        permissive_competitive_policy(),
    );
    assert!(!bad.passed);
    assert_eq!(bad.counterexamples.len(), observations.len());
}

#[test]
fn acceptance_active_planner_selects_safe_and_authorized_experiments() {
    let motif_a = DiscoveredMotif {
        id: MotifId([1; 32]),
        conditions: vec![CAPACITY],
        resolution: RESOLUTION_A,
        successes: 4,
        contradictions: 0,
        supporting_domains: BTreeSet::from([DomainId(1)]),
        empirical_roots: BTreeSet::from([EmpiricalRootId(1)]),
        precision_q32: 1i64 << 32,
    };
    let motif_b = DiscoveredMotif {
        id: MotifId([2; 32]),
        resolution: RESOLUTION_B,
        ..motif_a.clone()
    };
    let operators = vec![
        DeclarativeOperator::from_motif(&motif_a),
        DeclarativeOperator::from_motif(&motif_b),
    ];
    let evaluation = CompetitiveOperatorEvaluation {
        domain_accuracy_q32: BTreeMap::from([(DomainId(1), 1i64 << 32)]),
        ..CompetitiveOperatorEvaluation::default()
    };
    let evaluations = BTreeMap::from([
        (operators[0].id, evaluation.clone()),
        (operators[1].id, evaluation),
    ]);
    let planning_policy = ExperimentPlanningPolicy {
        maximum_risk: RiskLevel::High,
        allow_live_interventions: true,
        maximum_trials: 20,
        max_proposals: 30,
        min_information_gain_q32: 0,
    };
    let mut proposals = plan_active_experiments(
        &operators,
        &evaluations,
        &BTreeSet::from([DomainId(1), DomainId(2)]),
        planning_policy,
    );
    assert!(
        proposals
            .iter()
            .any(|proposal| proposal.kind == ActiveExperimentKind::Simulation)
    );
    assert!(
        proposals
            .iter()
            .any(|proposal| proposal.kind == ActiveExperimentKind::ShadowReplay)
    );
    assert!(
        proposals
            .iter()
            .any(|proposal| proposal.kind == ActiveExperimentKind::AbTest)
    );
    assert!(
        proposals
            .iter()
            .any(|proposal| proposal.kind == ActiveExperimentKind::DiagnosticObservation)
    );
    assert!(
        proposals
            .iter()
            .any(|proposal| proposal.kind == ActiveExperimentKind::ControlledConfigurationChange)
    );

    let controlled = proposals
        .iter_mut()
        .find(|proposal| proposal.kind == ActiveExperimentKind::ControlledConfigurationChange)
        .unwrap();
    assert!(execute_sandbox_experiment(controlled, &BTreeMap::new(), &[]).is_err());
    authorize_experiment(
        controlled,
        ExperimentAuthorization {
            authority_id: 1,
            policy_id: 2,
            authorized_at_lsn: 10,
            expires_at_lsn: 20,
        },
        10,
        planning_policy,
    )
    .unwrap();
    start_experiment(controlled, 10).unwrap();
    assert_eq!(
        execute_sandbox_experiment(controlled, &BTreeMap::new(), &[]),
        Err(ExperimentExecutionError::ExternalExecutionRequired)
    );

    let mut simulation = proposals
        .iter()
        .find(|proposal| proposal.kind == ActiveExperimentKind::Simulation)
        .unwrap()
        .clone();
    authorize_experiment(
        &mut simulation,
        ExperimentAuthorization {
            authority_id: 1,
            policy_id: 2,
            authorized_at_lsn: 10,
            expires_at_lsn: 20,
        },
        10,
        planning_policy,
    )
    .unwrap();
    start_experiment(&mut simulation, 10).unwrap();
    let programs = operators
        .iter()
        .map(|operator| (operator.id, operator.program.clone()))
        .collect();
    let observations = vec![EvaluationObservation {
        case_id: DiscoveryCaseId(1),
        domain: DomainId(1),
        empirical_root: EmpiricalRootId(1),
        role: EvaluationRole::Shadow,
        context: ReasoningContext {
            case: Some(reasoning_case(1, 1)),
            ..ReasoningContext::default()
        },
        actual_outcome: DiscoveryOutcome::Successful,
        actual_resolution: Some(RESOLUTION_A),
        predicted_confidence_q32: 1i64 << 32,
        adversarial_group: None,
    }];
    let result = execute_sandbox_experiment(&simulation, &programs, &observations).unwrap();
    assert_eq!(result.trials, 1);
    assert_eq!(result.disagreements, 1);
    complete_experiment(&mut simulation, result).unwrap();
    assert_eq!(simulation.status, ExperimentStatus::Completed);
    assert!(simulation.result.is_some());

    let replicated = GovernedDiscoveryState::new();
    replicated
        .apply(
            DiscoveryStateMutation::InstallSafetyKernel {
                kernel: ImmutableSafetyKernel::v1(256),
            },
            1,
        )
        .unwrap();
    let mut proposed = simulation.clone();
    proposed.status = ExperimentStatus::Proposed;
    proposed.authorization = None;
    proposed.result = None;
    replicated
        .apply(
            DiscoveryStateMutation::UpsertExperiment {
                experiment: proposed.clone(),
                expected_previous: None,
            },
            2,
        )
        .unwrap();
    let mut authorized = proposed;
    authorize_experiment(
        &mut authorized,
        ExperimentAuthorization {
            authority_id: 1,
            policy_id: 2,
            authorized_at_lsn: 3,
            expires_at_lsn: 10,
        },
        3,
        planning_policy,
    )
    .unwrap();
    replicated
        .apply(
            DiscoveryStateMutation::UpsertExperiment {
                experiment: authorized.clone(),
                expected_previous: Some(ExperimentStatus::Proposed),
            },
            3,
        )
        .unwrap();
    let mut running = authorized;
    start_experiment(&mut running, 4).unwrap();
    replicated
        .apply(
            DiscoveryStateMutation::UpsertExperiment {
                experiment: running.clone(),
                expected_previous: Some(ExperimentStatus::Authorized),
            },
            4,
        )
        .unwrap();
    complete_experiment(&mut running, simulation.result.clone().unwrap()).unwrap();
    replicated
        .apply(
            DiscoveryStateMutation::UpsertExperiment {
                experiment: running,
                expected_previous: Some(ExperimentStatus::Running),
            },
            5,
        )
        .unwrap();
    assert_eq!(
        replicated.snapshot_at(5).experiments[0].status,
        ExperimentStatus::Completed
    );
}

#[test]
fn acceptance_continuous_cycle_admits_monitors_revises_retires_audits_and_rolls_back() {
    let mut knowledge = KnowledgeSnapshot {
        lsn: 500,
        concept_profiles: vec![
            profile(1, 101, &[CAPACITY], 1),
            profile(2, 201, &[CAPACITY], 2),
        ],
        ..KnowledgeSnapshot::default()
    };
    knowledge.hyperedges = vec![
        edge(
            1,
            1,
            &[101, 102, 103],
            &[1, 2, 2],
            10,
            DiscoveryOutcome::Successful,
            Some(RESOLUTION_A),
        ),
        edge(
            2,
            2,
            &[201, 202, 203],
            &[7, 8, 8],
            10,
            DiscoveryOutcome::Successful,
            Some(RESOLUTION_A),
        ),
    ];
    let motif_policy = HypergraphMotifPolicy {
        min_support: 2,
        min_domains: 2,
        min_independent_roots: 2,
        anomaly_min_baseline: 10,
        max_temporal_gap_lsn: 100,
        max_motifs: 10,
    };
    let discovered = mine_temporal_hypergraph_motifs(&knowledge, motif_policy);
    let motif_ids: BTreeSet<_> = discovered.iter().map(|motif| motif.id).collect();
    let observations: Vec<_> = (1..=12u64)
        .map(|id| {
            let role = match id {
                1..=8 => EvaluationRole::Shadow,
                9 => EvaluationRole::Counterfactual,
                10 => EvaluationRole::CausalIntervention,
                _ => EvaluationRole::Adversarial,
            };
            EvaluationObservation {
                case_id: DiscoveryCaseId(1_000 + id),
                domain: DomainId((id % 2) + 1),
                empirical_root: EmpiricalRootId(2_000 + id),
                role,
                context: ReasoningContext {
                    case: Some(reasoning_case(1_000 + id, (id % 2) + 1)),
                    present_motifs: motif_ids.clone(),
                    causal_motifs: motif_ids.clone(),
                    ..ReasoningContext::default()
                },
                actual_outcome: DiscoveryOutcome::Successful,
                actual_resolution: Some(RESOLUTION_A),
                predicted_confidence_q32: 1i64 << 32,
                adversarial_group: (role == EvaluationRole::Adversarial).then_some(id),
            }
        })
        .collect();
    let policy = ContinuousDiscoveryPolicy {
        safety_kernel: ImmutableSafetyKernel::v1(512),
        schema: SchemaInductionPolicy {
            min_domains: 2,
            min_members: 2,
            min_independent_roots: 2,
            max_proposals: 100,
        },
        schema_validation: SchemaValidationPolicy {
            min_observations: 2,
            min_domains: 2,
            min_independent_roots: 2,
            min_structural_accuracy_q32: q32(3, 4),
        },
        mappings: MappingInductionPolicy {
            min_role_similarity_q32: 0,
            min_outcome_similarity_q32: 0,
            min_total_score_q32: 0,
            min_independent_roots: 2,
            max_hypotheses: 100,
        },
        mapping_validation: MappingValidationPolicy {
            min_independent_roots: 2,
            min_role_similarity_q32: 0,
            min_outcome_similarity_q32: 0,
            min_total_score_q32: 0,
        },
        motifs: motif_policy,
        evaluation: CompetitiveEvaluationPolicy {
            min_observations: 8,
            min_domains: 2,
            min_independent_roots: 8,
            min_accuracy_q32: q32(9, 10),
            min_incumbent_improvement_q32: q32(1, 10),
            min_counterfactual_accuracy_q32: q32(9, 10),
            min_intervention_accuracy_q32: q32(9, 10),
            min_transfer_accuracy_q32: q32(9, 10),
            max_calibration_error_q32: 0,
            min_adversarial_robustness_q32: q32(9, 10),
            max_description_bytes: 16_384,
        },
        experiments: ExperimentPlanningPolicy::default(),
        monitoring: MonitoringPolicy {
            min_observations: 4,
            min_accuracy_q32: q32(3, 4),
            max_calibration_error_q32: q32(1, 4),
            max_failure_ratio_q32: q32(1, 4),
        },
        admission_authority: Some(GovernanceAuthority::ReplicatedPolicy {
            policy_id: 77,
            version: 1,
        }),
    };
    let mut engine = ContinuousDiscoveryEngine::new(policy.clone()).unwrap();
    let mut validation_knowledge = knowledge.clone();
    validation_knowledge.lsn += 1;
    for profile in &mut validation_knowledge.concept_profiles {
        profile.empirical_roots = profile
            .empirical_roots
            .iter()
            .map(|root| EmpiricalRootId(root.0 + 50_000))
            .collect();
    }
    for edge in &mut validation_knowledge.hyperedges {
        edge.empirical_roots = edge
            .empirical_roots
            .iter()
            .map(|root| EmpiricalRootId(root.0 + 50_000))
            .collect();
    }
    for case in &mut validation_knowledge.cases {
        case.empirical_roots = case
            .empirical_roots
            .iter()
            .map(|root| EmpiricalRootId(root.0 + 50_000))
            .collect();
    }
    let first = engine
        .run_cycle(&ContinuousDiscoveryInput {
            knowledge: knowledge.clone(),
            validation_knowledge: Some(validation_knowledge),
            evaluation_observations: observations.clone(),
            incumbent_accuracy_q32: q32(1, 2),
            ..ContinuousDiscoveryInput::default()
        })
        .unwrap();
    assert!(first.audit.verify());
    assert!(!first.schemas.is_empty());
    assert!(!first.mappings.is_empty());
    assert!(first.schema_transition_plans.iter().any(|plans| {
        plans
            .last()
            .is_some_and(|plan| plan.record.proposal.state == SchemaProposalState::Admitted)
    }));
    assert!(first.mapping_transition_plans.iter().any(|plans| {
        plans
            .last()
            .is_some_and(|plan| plan.record.hypothesis.lifecycle == MappingLifecycle::Confirmed)
    }));
    let transition_plan = first
        .transition_plans
        .iter()
        .find(|plan| {
            plan.last()
                .is_some_and(|step| step.operator.lifecycle == OperatorLifecycle::Admitted)
        })
        .expect("validated shadow operator must be admitted by external policy");
    assert_eq!(
        transition_plan
            .iter()
            .map(|step| step.operator.lifecycle)
            .collect::<Vec<_>>(),
        vec![
            OperatorLifecycle::Provisional,
            OperatorLifecycle::FalsificationTesting,
            OperatorLifecycle::Shadow,
            OperatorLifecycle::ShadowValidated,
            OperatorLifecycle::Admitted,
        ]
    );
    let admitted = transition_plan.last().unwrap().operator.clone();
    assert!(!admitted.epistemic.validation_evidence.is_empty());
    assert!(!admitted.epistemic.provenance_roots.is_empty());
    let first_known_operators: Vec<_> = first
        .transition_plans
        .iter()
        .filter_map(|plans| plans.last().map(|plan| plan.operator.clone()))
        .collect();

    // The report is not merely descriptive: every state change is emitted as
    // an ordered consensus action and becomes visible in one pinned world view.
    let replicated_snapshot = {
        use std::sync::Arc;

        use crate::cluster::{DataMutation, ReplicatedStateMachine, ShardStateMachine};
        use crate::learning::LearningSegment;
        use crate::relation::{RelationSegment, RelationTypeState};
        use crate::storage::segment::SegmentedEngine;

        let learning = Arc::new(LearningSegment::new(1));
        let relations = Arc::new(RelationSegment::new(1, 1));
        let state = ShardStateMachine::new(1, Arc::new(SegmentedEngine::new(8, 1_000)))
            .with_learning_discovery(Arc::clone(&learning))
            .with_evolved_relation_catalog(Arc::clone(&relations));
        let actions = first.replicated_actions(Some(policy.safety_kernel.clone()));
        for (offset, action) in actions.into_iter().enumerate() {
            let lsn = offset as u64 + 1;
            let audit_head = learning
                .governed_discovery
                .snapshot_at(lsn.saturating_sub(1))
                .audit_entries
                .last()
                .map_or([0; 32], |entry| entry.entry_hash);
            state
                .apply(lsn, &DataMutation::new_discovery_action(action, audit_head))
                .unwrap();
        }
        let snapshot = state.pin_physical_snapshot();
        let governed = snapshot.governed_discovery.unwrap();
        assert!(governed.safety_kernel.is_some());
        assert!(
            governed
                .schemas
                .iter()
                .any(|record| record.proposal.state == SchemaProposalState::Admitted)
        );
        assert!(
            governed
                .mappings
                .iter()
                .any(|record| { record.hypothesis.lifecycle == MappingLifecycle::Confirmed })
        );
        assert!(!governed.evaluations.is_empty());
        assert!(!governed.audit_entries.is_empty());
        assert!(snapshot.evolved_relation_types.iter().any(|rtype| {
            rtype.state == RelationTypeState::Admitted
                && rtype.name.starts_with("induced_relation_")
        }));
        let mapping_index = ConfirmedConceptMappingIndex::from_confirmed(
            &governed
                .mappings
                .iter()
                .map(|record| record.hypothesis.clone())
                .collect::<Vec<_>>(),
        );
        let confirmed = governed
            .mappings
            .iter()
            .find(|record| record.hypothesis.lifecycle == MappingLifecycle::Confirmed)
            .unwrap();
        assert!(mapping_index.equivalent(confirmed.hypothesis.left, confirmed.hypothesis.right));
        let checkpoint = GovernedDiscoveryCheckpoint::capture(&learning, snapshot.lsn).unwrap();
        let mut encoded = checkpoint.encode().unwrap();
        let decoded = GovernedDiscoveryCheckpoint::decode(&encoded).unwrap();
        let recovered = LearningSegment::new(2);
        let recovered_relations = RelationSegment::new(2, 1);
        decoded
            .restore_into_with_relations(&recovered, &recovered_relations)
            .unwrap();
        assert_eq!(
            recovered.governed_discovery.snapshot_at(snapshot.lsn),
            governed
        );
        assert_eq!(
            recovered.discovery.snapshot_at(snapshot.lsn),
            snapshot.discovered_operators
        );
        assert!(
            recovered_relations
                .types
                .read()
                .iter()
                .any(|rtype| rtype.state == RelationTypeState::Admitted)
        );
        let last = encoded.len() - 1;
        encoded[last] ^= 0x80;
        assert!(GovernedDiscoveryCheckpoint::decode(&encoded).is_err());
        governed
    };

    let bad_monitoring: Vec<_> = observations
        .iter()
        .take(8)
        .cloned()
        .map(|mut observation| {
            observation.actual_outcome = DiscoveryOutcome::Failed;
            observation.actual_resolution = Some(RESOLUTION_B);
            observation.empirical_root = EmpiricalRootId(observation.empirical_root.0 + 10_000);
            if let Some(case) = &mut observation.context.case {
                case.features.insert(FeatureId(999));
            }
            observation
        })
        .collect();
    let second = engine
        .run_cycle(&ContinuousDiscoveryInput {
            knowledge: knowledge.clone(),
            validation_knowledge: None,
            evaluation_observations: observations.clone(),
            incumbent_accuracy_q32: q32(1, 2),
            admitted_operators: vec![admitted.clone()],
            pending_operators: Vec::new(),
            monitoring_observations: BTreeMap::from([(admitted.id, bad_monitoring)]),
            prior_discovery_state: Some(replicated_snapshot.clone()),
            known_operators: first_known_operators.clone(),
        })
        .unwrap();
    assert!(second.schema_transition_plans.is_empty());
    assert!(second.mapping_transition_plans.is_empty());
    assert!(second.experiments.is_empty());
    assert!(
        second
            .monitoring_transitions
            .iter()
            .any(|transition| transition.operator.lifecycle == OperatorLifecycle::Deprecated)
    );
    assert!(
        second
            .revisions
            .iter()
            .any(|revision| revision.previous == admitted.id
                && revision.revision.version == admitted.version + 1)
    );

    let revision = second
        .revisions
        .iter()
        .find(|revision| revision.previous == admitted.id)
        .unwrap();
    let mut pending_revision = revision.revision.clone();
    pending_revision.lifecycle = OperatorLifecycle::Provisional;
    let mut deprecated = admitted.clone();
    deprecated.lifecycle = OperatorLifecycle::Deprecated;
    let third = engine
        .run_cycle(&ContinuousDiscoveryInput {
            knowledge,
            validation_knowledge: None,
            evaluation_observations: observations,
            incumbent_accuracy_q32: q32(1, 2),
            admitted_operators: vec![deprecated],
            pending_operators: vec![pending_revision.clone()],
            monitoring_observations: BTreeMap::new(),
            prior_discovery_state: Some(replicated_snapshot),
            known_operators: first_known_operators
                .into_iter()
                .chain([pending_revision.clone()])
                .collect(),
        })
        .unwrap();
    assert!(third.transition_plans.iter().any(|plans| {
        plans.last().is_some_and(|plan| {
            plan.operator.id == pending_revision.id
                && plan.operator.lifecycle == OperatorLifecycle::Admitted
        })
    }));
    assert!(third.monitoring_transitions.iter().any(|transition| {
        transition.operator.id == admitted.id
            && transition.operator.lifecycle == OperatorLifecycle::Superseded
            && transition.expected_previous == Some(OperatorLifecycle::Deprecated)
    }));

    let mut rollback_audit = DiscoveryAuditLog::default();
    let rollback = propose_compensating_rollback(
        &second.revisions[0].revision,
        &admitted,
        &mut rollback_audit,
        600,
    )
    .unwrap();
    assert_eq!(rollback.previous, second.revisions[0].revision.id);
    assert_eq!(rollback.revision.lifecycle, OperatorLifecycle::Generated);
    assert!(rollback_audit.verify());

    let mut encoded = serde_json::to_value(policy.safety_kernel).unwrap();
    encoded["sandbox_required"] = serde_json::Value::Bool(false);
    let tampered: ImmutableSafetyKernel = serde_json::from_value(encoded).unwrap();
    assert_eq!(
        tampered.verify(),
        Err(SafetyKernelViolation::ModifiedOrInvalid)
    );
}

fn q32(numerator: usize, denominator: usize) -> i64 {
    (((numerator as i128) << 32) / denominator as i128) as i64
}
