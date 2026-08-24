/* holosphere/src/learning/mod.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Learning & Deterministic Adjudication Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates empirical experience to derive deterministic epistemic state transitions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod adjudication;
pub mod collective;
pub mod discovery;
pub mod evidence;
pub mod id;
pub mod inference;
pub mod integrity;
pub mod mutation;
pub mod query;
pub mod read;
pub mod synthesis;

pub use adjudication::{
    AdjudicationDecisionCode, AdjudicationDisposition, AdjudicationPolicy, AdjudicationRecord,
    evaluate_adjudication,
};
pub use collective::{
    AgentBelief, AgentId, AgentMeta, ConflictPair, ConflictResolution, ConsensusResult,
    SWARM_CONSENSUS_METHOD_ID, SWARM_CONSENSUS_METHOD_VERSION, compute_swarm_consensus,
    materialize_collective_hypothesis,
};
pub use discovery::*;
pub use evidence::{
    ContextClassRegistry, EvidenceAccumulator, EvidenceDirection, EvidenceKey, EvidenceRecord,
    EvidenceSummary, FixedUtility, MetricDirection, MetricEvaluationRule, NormalizationRule,
    compute_evidence_digest,
};
pub use id::{AdjudicationId, ContextClassId, EvidenceId, EvidenceSummaryId};
pub use inference::{
    BIVECTOR_DIM, BarycentricWeightSemantics, CandidateEntityId, CandidateEntityRef,
    CandidateRoleBinding, CausalOrientation, Cl24BasisError, Cl24Blade, Cl24CompositionArtifact,
    Cl24EntityBasis, ClosureCandidate, ClosureKind, CompositionRule, CompositionRuleRegistry,
    CompositionSemantics, DEFAULT_MAX_TRUNCATION_LOSS_RATIO, DEFAULT_TRUNCATION_TOPK,
    DerivedEntityProposal, DirectedWedgeArtifact, DirectedWedgeRequest, DurableEvidenceRef,
    EvolutionArtifact, EvolutionHistoryView, EvolutionProposal, InferenceCandidate,
    InferenceCandidateId, InferenceError, InferenceGeometryArtifact, InferenceMethod,
    InferenceMethodId, InferenceMode, InferenceProposal, InferenceProposalBundle,
    InferenceRegistry, InferenceRequest, InferenceScope, InferenceScore, InferenceSeed,
    InferenceTrace, MAX_OPERATOR_CHAIN, MultivectorCl24Sparse, PARALLEL_BLEND_THRESHOLD,
    PhaseShift, PhaseShiftArtifact, RUNE_ANALOGY_METHOD_ID, RUNE_ANALOGY_METHOD_VERSION,
    RUNE_BARYCENTRIC_METHOD_ID, RUNE_BARYCENTRIC_METHOD_VERSION, RUNE_CLOSURE_METHOD_ID,
    RUNE_CLOSURE_METHOD_VERSION, RUNE_DIRECTED_WEDGE_METHOD_ID, RUNE_DIRECTED_WEDGE_METHOD_VERSION,
    RUNE_EVOLUTION_METHOD_ID, RUNE_EVOLUTION_METHOD_VERSION, ReasoningOperator,
    ReasoningOperatorId, RelationProposal, RotorAlignmentResult, RuneAnalogyConfig,
    RuneBarycentricConfig, RuneBarycentricInsight, RuneBarycentricV1, RuneCl24CompositionConfig,
    RuneClosureEvidenceV1, RuneOperatorClass, RunePhaseEvolutionV1, RuneStructuralAnalogyV1,
    SemanticFingerprint, align_regions, apply_givens_rotation, apply_phase_shift, apply_rotation,
    bivector_contract, bivector_strength, blade_product_sign, build_directed_wedge_edge,
    causal_bivector, compile_closure, dot8, euclidean_dist_8, execute_operator_chain,
    geometric_counterfactual_projection, gram_schmidt_tangent, identity_rotation, infer_between,
    l2_sq_8, leech_to_e8_f32, mean_alignment_residual, normalise_weights, normalize_vector_8,
    optimal_givens_angle, parallel_centroid, region_centroid, resolve_barycentric,
    sequential_centroid, snap_to_e8_lattice,
};
pub use integrity::{
    CanonicalLearningAuditDigest, CircularityCheck, EmpiricalRootId, EpistemicLineageGraph,
    EvidenceIndependenceReport, LineageNodeKind, PlanAttributionMethod, PlanAttributionRecord,
    ProposalStalenessCheck, ResolutionSemanticKey, SemanticCandidateRegistry, SynthesisCandidateId,
    SynthesisDependencyDigest, SynthesisOccurrence, SynthesisRunId, check_epistemic_circularity,
    compute_audit_digest, compute_plan_attribution, evaluate_evidence_independence,
};
pub use mutation::{LearningMutation, LearningMutationError};
pub use query::AdjudicationQuery;
pub use read::{AdjudicationExplanation, LearningReadSnapshot, LearningSegment};
pub use synthesis::{
    ActionComposition, ActionConstraint, ActionPlan, ActionPlanStep, CandidateActionStepId,
    CandidateResolutionState, ClosureArtifactId, ConstraintCheck, ConstraintCode, ConstraintResult,
    ContextApplicability, ContextDifference, Precedent, PrecedentDisposition, ResolutionCandidate,
    ResolutionCandidateId, StructuralAnalogyArtifact, StructuralSynthesisTrace, SynthesisAttempt,
    SynthesisBasis, SynthesisGoal, SynthesisKnowledgeBase, SynthesisPolicy, SynthesisPolicyId,
    SynthesisRequest, SynthesisResult, SynthesisScores, synthesize,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::entity::segment::EntitySegment;
    use crate::entity::status::EpistemicStatus;
    use crate::experience::id::{AttemptId, ContextId, EvaluationPolicyId, MetricId};
    use crate::experience::metric::MetricValue;
    use crate::learning::evidence::stats::{
        MetricDirection, MetricEvaluationRule, NormalizationRule,
    };
    use crate::relation::read::RelationSegment;
    use crate::relation::schema::{RelationType, RelationTypeState, RoleSchema};

    #[test]
    fn test_phase6_fixed_point_utility_and_normalization_determinism() {
        let rule = MetricEvaluationRule {
            metric_id: MetricId(1),
            weight_q32: (1i64 << 32), // 1.0 weight
            direction: MetricDirection::LowerIsBetter,
            normalization: NormalizationRule::RelativeDelta,
        };

        let baseline = MetricValue::Unsigned(8400);
        let observed = MetricValue::Unsigned(3100);

        // (8400 - 3100) / 8400 = 5300 / 8400 = 0.63095238...
        let util = rule.evaluate(&baseline, &observed).expect("must evaluate");
        assert!((util.to_f64() - 0.630952).abs() < 0.001);
    }

    #[test]
    fn test_phase6_idempotency_and_order_independent_summary_fold() {
        let acc = EvidenceAccumulator::new();
        let rel_id = 701;
        let ctx_id = ContextId(801);
        let pol_id = EvaluationPolicyId(901);

        let ev1 = EvidenceRecord {
            evidence_id: EvidenceId(1),
            target_relation: rel_id,
            attempt_id: AttemptId(101),
            context_id: ctx_id,
            policy_id: pol_id,
            direction: EvidenceDirection::Supports,
            utility_q32: (1i64 << 32),
            provenance_id: 1,
            evaluated_at_lsn: 100,
        };

        let ev2 = EvidenceRecord {
            evidence_id: EvidenceId(2),
            target_relation: rel_id,
            attempt_id: AttemptId(102),
            context_id: ctx_id,
            policy_id: pol_id,
            direction: EvidenceDirection::Supports,
            utility_q32: (1i64 << 32),
            provenance_id: 1,
            evaluated_at_lsn: 110,
        };

        // 1. Enforce deduplication / idempotency
        let (id1, inserted1) = acc.record(ev1.clone());
        assert!(inserted1);
        assert_eq!(id1, EvidenceId(1));

        let (id1_dup, inserted1_dup) = acc.record(ev1.clone());
        assert!(!inserted1_dup);
        assert_eq!(id1_dup, EvidenceId(1));

        let (_id2, inserted2) = acc.record(ev2.clone());
        assert!(inserted2);

        // 2. Incremental summary matches full rebuild
        let summary_incremental = acc.build_summary(rel_id, ContextClassId(1), 200);
        assert_eq!(summary_incremental.observation_count, 2);
        assert_eq!(summary_incremental.support_count, 2);
        assert_eq!(summary_incremental.contradiction_count, 0);

        // 3. Permuted fold yields identical summary
        let mut permuted_summary = EvidenceSummary {
            relation_id: rel_id,
            context_class_id: ContextClassId(1),
            ..Default::default()
        };
        permuted_summary.accumulate(&ev2);
        permuted_summary.accumulate(&ev1);

        assert_eq!(summary_incremental, permuted_summary);
    }

    #[test]
    fn test_phase6_epistemic_transition_boundaries_and_forbidden_observed() {
        let policy = AdjudicationPolicy {
            id: EvaluationPolicyId(1),
            version: 1,
            min_observations: 2,
            min_support: 2,
            max_contradictions: 0,
            promote_utility_q32: (1i64 << 32),
            falsify_utility_q32: -(1i64 << 32),
            rules: vec![],
        };

        // 1. Insufficient evidence -> Pending
        let mut summary_sparse = EvidenceSummary {
            observation_count: 1,
            support_count: 1,
            utility_sum_q32: 1i64 << 32,
            ..Default::default()
        };
        let (st, code, disp) =
            evaluate_adjudication(&summary_sparse, EpistemicStatus::Provisional, &policy);
        assert_eq!(st, EpistemicStatus::Provisional);
        assert_eq!(code, AdjudicationDecisionCode::InsufficientEvidence);
        assert_eq!(disp, AdjudicationDisposition::Pending);

        // 2. Sufficient support -> Inferred
        summary_sparse.observation_count = 2;
        summary_sparse.support_count = 2;
        let (st2, code2, disp2) =
            evaluate_adjudication(&summary_sparse, EpistemicStatus::Provisional, &policy);
        assert_eq!(st2, EpistemicStatus::Inferred);
        assert_eq!(code2, AdjudicationDecisionCode::SupportThresholdReached);
        assert_eq!(disp2, AdjudicationDisposition::Supported);

        // 3. Machine learning adjudication cannot promote to Observed (hardcoded invariant)
        let (st3, _, _) =
            evaluate_adjudication(&summary_sparse, EpistemicStatus::Inferred, &policy);
        assert_eq!(st3, EpistemicStatus::Inferred); // Inferred stays Inferred, never becomes Observed
    }

    #[test]
    fn test_phase6_outage_adjudication_and_compaction_scenario() {
        let ent_seg = Arc::new(EntitySegment::new(1, 1));
        let rel_seg = Arc::new(RelationSegment::new(1, 1));
        let learn_seg = Arc::new(LearningSegment::new(1));

        let rel_id = 5001;
        let type_mitigates = 10;
        let role_action = 1;
        let role_problem = 2;

        let roles = vec![
            RoleSchema {
                role_id: role_action,
                name: Arc::from("Action"),
                min_count: 1,
                max_count: 1,
                required: true,
            },
            RoleSchema {
                role_id: role_problem,
                name: Arc::from("Problem"),
                min_count: 1,
                max_count: 1,
                required: true,
            },
        ];
        let fp = RelationType::compute_structural_fingerprint(type_mitigates, 1, &roles);

        // Register RelationType MITIGATES(Action, Problem)
        let schema = RelationType {
            id: type_mitigates,
            name: Arc::from("MITIGATES"),
            roles,
            schema_version: 1,
            binary_projection: None,
            state: RelationTypeState::Admitted,
            provenance_id: 1,
            structural_fingerprint: fp,
        };
        rel_seg.register_type(schema);

        // Create entity placeholders for Action and Problem
        let act_ent_idx = ent_seg
            .arena
            .bind(910, crate::entity::header::EntityHeader::default());
        let prob_ent_idx = ent_seg
            .arena
            .bind(901, crate::entity::header::EntityHeader::default());

        // Create Provisional relation instance
        let rel_header = crate::relation::header::RelationHeader {
            relation_type_id: type_mitigates,
            binding_start: 0,
            version_row: 0,
            provenance_row: 0,
            binding_len: 2,
            schema_version: 1,
            epistemic_status: EpistemicStatus::Provisional as u8,
            lifecycle_status: crate::entity::status::LifecycleStatus::Active as u8,
            flags: 1,
            reserved: [0u8; 8],
        };
        let bindings = vec![
            crate::relation::binding::SegmentRoleBinding {
                entity: act_ent_idx,
                role_id: role_action,
                flags: 0,
            },
            crate::relation::binding::SegmentRoleBinding {
                entity: prob_ent_idx,
                role_id: role_problem,
                flags: 0,
            },
        ];
        rel_seg.arena.bind(rel_id, rel_header, &bindings);

        let policy = AdjudicationPolicy {
            id: EvaluationPolicyId(1),
            version: 1,
            min_observations: 2,
            min_support: 2,
            max_contradictions: 0,
            promote_utility_q32: (1i64 << 32),
            falsify_utility_q32: -(1i64 << 32),
            rules: vec![],
        };
        LearningMutation::RegisterPolicy {
            policy: policy.clone(),
        }
        .apply(&learn_seg, &rel_seg, 50)
        .unwrap();

        let ctx_c1 = ContextId(801); // NVMe small writes
        let ctx_c2 = ContextId(802); // CPU-bound giant batches

        // Record 2 supporting attempts under C1
        let ev1 = EvidenceRecord {
            evidence_id: EvidenceId(1),
            target_relation: rel_id,
            attempt_id: AttemptId(401),
            context_id: ctx_c1,
            policy_id: EvaluationPolicyId(1),
            direction: EvidenceDirection::Supports,
            utility_q32: (1i64 << 32),
            provenance_id: 1,
            evaluated_at_lsn: 100,
        };
        let ev2 = EvidenceRecord {
            evidence_id: EvidenceId(2),
            target_relation: rel_id,
            attempt_id: AttemptId(402),
            context_id: ctx_c1,
            policy_id: EvaluationPolicyId(1),
            direction: EvidenceDirection::Supports,
            utility_q32: (1i64 << 32),
            provenance_id: 1,
            evaluated_at_lsn: 110,
        };
        LearningMutation::RecordEvidence {
            evidence: ev1.clone(),
        }
        .apply(&learn_seg, &rel_seg, 100)
        .unwrap();
        LearningMutation::RecordEvidence {
            evidence: ev2.clone(),
        }
        .apply(&learn_seg, &rel_seg, 110)
        .unwrap();

        // 1. Adjudicate under C1 evidence -> Promote to Inferred
        let evidence_list_1 = vec![ev1, ev2];
        let digest_1 = compute_evidence_digest(&evidence_list_1);

        let adj1 = AdjudicationRecord {
            id: AdjudicationId(1),
            target_relation: rel_id,
            policy_id: EvaluationPolicyId(1),
            evidence_snapshot_lsn: 150,
            previous_status: EpistemicStatus::Provisional,
            resulting_status: EpistemicStatus::Inferred,
            evidence_summary_id: EvidenceSummaryId(1),
            decision_code: AdjudicationDecisionCode::SupportThresholdReached,
            disposition: AdjudicationDisposition::Supported,
            committed_lsn: 150,
        };
        LearningMutation::ApplyAdjudication {
            adjudication: adj1,
            expected_evidence_digest: digest_1,
        }
        .apply(&learn_seg, &rel_seg, 150)
        .unwrap();

        // Verify relation is now Inferred
        let (_, cur_header) = rel_seg.arena.get_by_id(rel_id).unwrap();
        assert_eq!(cur_header.epistemic(), EpistemicStatus::Inferred);

        // 2. Record contradictory attempt under C2
        let ev3 = EvidenceRecord {
            evidence_id: EvidenceId(3),
            target_relation: rel_id,
            attempt_id: AttemptId(501),
            context_id: ctx_c2,
            policy_id: EvaluationPolicyId(1),
            direction: EvidenceDirection::Contradicts,
            utility_q32: -(1i64 << 32),
            provenance_id: 1,
            evaluated_at_lsn: 200,
        };
        LearningMutation::RecordEvidence { evidence: ev3 }
            .apply(&learn_seg, &rel_seg, 200)
            .unwrap();

        // 3. Re-evaluate joint summary: mixed evidence yields ContextDependent disposition
        let snap = learn_seg.read_snapshot(250);
        let ev_c1 = snap.evidence_for_context(rel_id, ctx_c1);
        let ev_c2 = snap.evidence_for_context(rel_id, ctx_c2);
        assert_eq!(ev_c1.len(), 2);
        assert_eq!(ev_c2.len(), 1);

        let all_ev = snap.evidence_for(rel_id);
        let mut joint_summary = EvidenceSummary {
            relation_id: rel_id,
            context_class_id: ContextClassId(1),
            ..Default::default()
        };
        for r in &all_ev {
            joint_summary.accumulate(r);
        }

        let (st_joint, code_joint, disp_joint) =
            evaluate_adjudication(&joint_summary, EpistemicStatus::Inferred, &policy);
        assert_eq!(st_joint, EpistemicStatus::Inferred);
        assert_eq!(code_joint, AdjudicationDecisionCode::ContextDependent);
        assert_eq!(disp_joint, AdjudicationDisposition::ContextDependent);

        // 4. Verify physical compaction preserves all evidence and adjudications
        let compacted_learn = learn_seg.compact(2);
        let snap_compacted = compacted_learn.read_snapshot(250);
        assert_eq!(snap_compacted.evidence_for(rel_id).len(), 3);
        assert_eq!(snap_compacted.adjudication_history(rel_id).len(), 1);
    }
}
