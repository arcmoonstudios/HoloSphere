/* holosphere/src/learning/inference/mod.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Inference Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Subsystem responsible for generating provisional relational hypotheses,
//! derived concept candidates, and evolutionary phase transitions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod candidate;
pub mod contract;
pub mod registry;
pub mod rune_evo;
pub mod trace;

pub use candidate::{
    CandidateEntityId, CandidateEntityRef, CandidateRoleBinding, DerivedEntityProposal,
    EvolutionArtifact, EvolutionProposal, InferenceCandidate, InferenceCandidateId,
    InferenceGeometryArtifact, InferenceProposal, InferenceProposalBundle, InferenceScore,
    PhaseShiftArtifact, RelationProposal,
};
pub use contract::{
    InferenceError, InferenceMethod, InferenceMethodId, InferenceMode, InferenceRequest,
    InferenceScope, InferenceSeed,
};
pub use registry::InferenceRegistry;
pub use rune_evo::{
    BIVECTOR_DIM, BarycentricWeightSemantics, CausalOrientation, Cl24BasisError, Cl24Blade,
    Cl24CompositionArtifact, Cl24EntityBasis, ClosureCandidate, ClosureKind, CompositionRule,
    CompositionRuleRegistry, CompositionSemantics, DEFAULT_MAX_TRUNCATION_LOSS_RATIO,
    DEFAULT_TRUNCATION_TOPK, DirectedWedgeArtifact, DirectedWedgeRequest, DurableEvidenceRef,
    EvolutionHistoryView, MAX_OPERATOR_CHAIN, MultivectorCl24Sparse, PARALLEL_BLEND_THRESHOLD,
    PhaseShift, RUNE_ANALOGY_METHOD_ID, RUNE_ANALOGY_METHOD_VERSION, RUNE_BARYCENTRIC_METHOD_ID,
    RUNE_BARYCENTRIC_METHOD_VERSION, RUNE_CLOSURE_METHOD_ID, RUNE_CLOSURE_METHOD_VERSION,
    RUNE_DIRECTED_WEDGE_METHOD_ID, RUNE_DIRECTED_WEDGE_METHOD_VERSION, RUNE_EVOLUTION_METHOD_ID,
    RUNE_EVOLUTION_METHOD_VERSION, ReasoningOperator, ReasoningOperatorId, RotorAlignmentResult,
    RuneAnalogyConfig, RuneBarycentricConfig, RuneBarycentricInsight, RuneBarycentricV1,
    RuneCl24CompositionConfig, RuneClosureEvidenceV1, RuneOperatorClass, RunePhaseEvolutionV1,
    RuneStructuralAnalogyV1, align_regions, apply_givens_rotation, apply_phase_shift,
    apply_rotation, bivector_contract, bivector_strength, blade_product_sign,
    build_directed_wedge_edge, causal_bivector, compile_closure, dot8, euclidean_dist_8,
    execute_operator_chain, geometric_counterfactual_projection, gram_schmidt_tangent,
    identity_rotation, infer_between, l2_sq_8, leech_to_e8_f32, mean_alignment_residual,
    normalise_weights, normalize_vector_8, optimal_givens_angle, parallel_centroid,
    region_centroid, resolve_barycentric, sequential_centroid, snap_to_e8_lattice,
};
pub use trace::{InferenceTrace, SemanticFingerprint};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use crate::entity::id::{EntityId, VersionId};
    use crate::entity::status::EpistemicStatus;
    use crate::learning::read::LearningSegment;

    fn make_test_region_e8() -> Vec<[f32; 8]> {
        let mut r = Vec::new();
        for i in 0..4 {
            let mut v = [0.0f32; 8];
            v[i] = 1.0;
            r.push(v);
        }
        r
    }

    #[test]
    fn test_phase7_reference_equivalence_givens_orthogonal_invariance() {
        let mut rot = *identity_rotation();
        apply_givens_rotation(&mut rot, 0, 1, 0.3);
        for i in 0..8usize {
            for j in 0..8usize {
                let dot: f32 = (0..8).map(|k| rot[i * 8 + k] * rot[j * 8 + k]).sum();
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-5,
                    "R^T R [{i}][{j}] = {dot:.6}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_phase7_reference_equivalence_identical_regions_zero_residual() {
        let region = make_test_region_e8();
        let result = align_regions(&region, &region, 48).expect("alignment must succeed");
        assert!(
            result.residual < 0.01,
            "identical regions must align with near-zero residual; got {}",
            result.residual
        );
    }

    #[test]
    fn test_phase7_inference_candidate_strict_provisional_boundary() {
        let method = RuneStructuralAnalogyV1::default();
        let learn_seg = Arc::new(LearningSegment::new(1));
        let snap = learn_seg.read_snapshot(100);

        let req = InferenceRequest {
            learning_snapshot: &snap,
            scope: InferenceScope::Region {
                entities: vec![101, 102, 101, 102],
            },
            seed: InferenceSeed::default(),
            max_candidates: 10,
        };

        let candidates = method.infer(&req).expect("must infer");
        assert_eq!(candidates.len(), 1);
        let candidate = &candidates[0];

        // Invariant: MUST strictly be Provisional
        assert_eq!(candidate.epistemic_status, EpistemicStatus::Provisional);
        assert_eq!(candidate.trace.method, RUNE_ANALOGY_METHOD_ID);
        assert_eq!(candidate.trace.method_version, RUNE_ANALOGY_METHOD_VERSION);
        assert_eq!(candidate.trace.snapshot_lsn, 100);
        assert_eq!(candidate.trace.source_entities, vec![101, 102, 101, 102]);
    }

    #[test]
    fn test_phase7_deterministic_reproducibility_and_registry() {
        let registry = InferenceRegistry::new();
        let method = Arc::new(RuneStructuralAnalogyV1::default());
        registry.register(method.clone());

        assert_eq!(registry.list_methods(), vec![RUNE_ANALOGY_METHOD_ID]);
        let resolved = registry.get(RUNE_ANALOGY_METHOD_ID).expect("found");
        assert_eq!(resolved.name(), "RuneStructuralAnalogyV1");

        let learn_seg = Arc::new(LearningSegment::new(1));
        let snap = learn_seg.read_snapshot(100);
        let req = InferenceRequest {
            learning_snapshot: &snap,
            scope: InferenceScope::Region {
                entities: vec![201, 202, 201, 202],
            },
            seed: InferenceSeed::default(),
            max_candidates: 5,
        };

        let res1 = resolved.infer(&req).unwrap();
        let res2 = resolved.infer(&req).unwrap();
        assert_eq!(res1, res2); // Bit-for-bit deterministic reproducibility
    }

    #[test]
    fn test_rune_barycentric_reference_two_point_blend_and_weights_divergence() {
        let a = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let coords = vec![a, b];

        // 1. Two-point [0.7, 0.3] blend
        let cfg = RuneBarycentricConfig::default();
        let insight_1 = resolve_barycentric(&coords, &[0.7, 0.3], &cfg).expect("resolve");
        assert_eq!(insight_1.centroid, [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(insight_1.normalized_weights, vec![0.7, 0.3]);

        // 2. Scaled weights [7.0, 3.0]: same centroid, but different Rune reference friction!
        let insight_2 = resolve_barycentric(&coords, &[7.0, 3.0], &cfg).expect("resolve");
        assert_eq!(insight_2.centroid, [0.7, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(insight_2.normalized_weights, vec![0.7, 0.3]);
        assert_ne!(insight_1.reference_friction, insight_2.reference_friction);
    }

    #[test]
    fn test_rune_barycentric_reference_n_scale_and_parallel_invariance() {
        let cfg = RuneBarycentricConfig::default();
        for &n in &[255, 256, 257, 1024] {
            let coords: Vec<[f32; 8]> = (0..n)
                .map(|i| {
                    let mut v = [0.0f32; 8];
                    v[i % 8] = 1.0;
                    v
                })
                .collect();
            let weights = vec![1.0f32; n];

            let insight1 = resolve_barycentric(&coords, &weights, &cfg).expect("resolve 1");
            let insight2 = resolve_barycentric(&coords, &weights, &cfg).expect("resolve 2");
            assert_eq!(insight1.centroid, insight2.centroid);
            assert_eq!(insight1.reference_friction, insight2.reference_friction);
        }
    }

    #[test]
    fn test_rune_barycentric_negative_extrapolating_weights_and_bundle_synthesis() {
        let a = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let coords = vec![a, b];

        // Extrapolating weights [2.0, -1.0] (sum = 1.0 != 0)
        let cfg = RuneBarycentricConfig::default();
        let insight = resolve_barycentric(&coords, &[2.0, -1.0], &cfg).expect("resolve");
        assert_eq!(insight.centroid, [2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // Synthesize proposal bundle
        let engine = RuneBarycentricV1::new(cfg);
        let parents = vec![1001, 1002];
        let bundle = engine
            .synthesize_bundle(&parents, &coords, &[2.0, -1.0], 500)
            .expect("bundle");

        assert_eq!(bundle.entities.len(), 1);
        assert_eq!(
            bundle.entities[0].epistemic_status,
            EpistemicStatus::Provisional
        );
        assert_eq!(bundle.relations.len(), 2);
        assert_eq!(
            bundle.relations[0].bindings[0].entity,
            CandidateEntityRef::Proposed(CandidateEntityId(1))
        );
        assert_eq!(
            bundle.relations[0].bindings[1].entity,
            CandidateEntityRef::Existing(1001)
        );
    }

    #[test]
    fn test_rune_directed_wedge_reference_antisymmetry_and_counterfactual() {
        let e1 = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let e2 = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // 1. Antisymmetry: A ∧ B = -(B ∧ A)
        let bv_ab = causal_bivector(&e1, &e2);
        let bv_ba = causal_bivector(&e2, &e1);
        for (a, b) in bv_ab.iter().zip(bv_ba.iter()) {
            assert!((a + b).abs() < 1e-6);
        }

        // 2. Self-wedge is zero: A ∧ A = 0
        let bv_self = causal_bivector(&e1, &e1);
        for &c in &bv_self {
            assert!(c.abs() < 1e-6);
        }

        // 3. Orthogonal pair strength
        let s = bivector_strength(&bv_ab);
        assert!(s > 0.0 && s <= 1.0);

        // 4. Counterfactual geometric projection diverges
        let cf = geometric_counterfactual_projection(&e1, &e2, &bv_ab);
        let diff: f32 = e2.iter().zip(cf.iter()).map(|(x, y)| (x - y).abs()).sum();
        assert!(diff > 1e-4);

        // 5. Hypothesis relation context integration
        let artifact = build_directed_wedge_edge(e1, e2, CausalOrientation::Forward, 7001, 100);
        assert_eq!(artifact.orientation, CausalOrientation::Forward);
        assert_eq!(artifact.trace.source_relations, vec![7001]);
    }

    #[test]
    fn test_phase7_evolution_reference_identity_and_clamped_gain() {
        let initial = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        // 1. Identity shift produces identical coords
        let id_shift = PhaseShift::identity();
        let evolved_id = apply_phase_shift(&initial, &id_shift).expect("apply");
        assert_eq!(evolved_id, initial);

        // 2. Gain = 0 returns original
        let zero_gain_shift = PhaseShift::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.5, 0.0);
        let evolved_zero = apply_phase_shift(&initial, &zero_gain_shift).expect("apply");
        assert_eq!(evolved_zero, initial);

        // 3. Clamped gain: gain = 2.0 clamps to 1.0
        let shift_over = PhaseShift::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5, 2.0);
        let shift_one = PhaseShift::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5, 1.0);
        let evolved_over = apply_phase_shift(&initial, &shift_over).expect("apply");
        let evolved_one = apply_phase_shift(&initial, &shift_one).expect("apply");
        assert_eq!(evolved_over, evolved_one);
    }

    #[test]
    fn test_phase7_evolution_history_view_drift_and_depth() {
        let entity_id: EntityId = 42;
        let layer0 = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let layer1 = [0.7071f32, 0.7071, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let layer2 = [0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let history = EvolutionHistoryView::new(entity_id, vec![layer0, layer1, layer2]);

        assert_eq!(history.origin(), Some(layer0));
        assert_eq!(history.surface(), Some(layer2));
        assert_eq!(history.depth(), 2);
        assert_eq!(history.at_depth(1), Some(layer1));

        // Semantic drift = ||layer2 - layer0|| = sqrt(1^2 + 1^2) = sqrt(2) ~ 1.4142
        let drift = history.semantic_drift();
        assert!((drift - std::f32::consts::SQRT_2).abs() < 1e-4);

        // Incremental drifts
        let inc_drifts = history.incremental_drifts();
        assert_eq!(inc_drifts.len(), 2);
        assert!(inc_drifts[0] > 0.0);
        assert!(inc_drifts[1] > 0.0);
    }

    #[test]
    fn test_phase7_evolution_proposal_provisional_and_apollo_scenario() {
        let engine = RunePhaseEvolutionV1;
        let apollo_id: EntityId = 9001;
        let version_0: VersionId = 1;
        let x0 = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let shift = PhaseShift::new([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.785398, 1.0);

        let proposal = engine
            .propose_evolution(apollo_id, version_0, &x0, shift, 100)
            .expect("proposal");

        // Epistemic invariant: strictly begins Provisional
        assert_eq!(proposal.epistemic_status, EpistemicStatus::Provisional);
        // EntityId before == EntityId after (EntityId durability)
        assert_eq!(proposal.entity_id, apollo_id);
        assert_eq!(proposal.source_version, version_0);
        assert_eq!(proposal.artifact.source_coords, x0);
        assert_ne!(proposal.artifact.resulting_coords, x0);

        // State fingerprint represents this geometric state
        assert_ne!(proposal.artifact.state_fingerprint, [0u8; 32]);
    }

    #[test]
    fn test_batch_7d_cl24_closure_composition_system_scenario() {
        // System Scenario:
        // A --PART_OF--> B (op1)
        // B --PART_OF--> C (op2)
        // Rule: PART_OF ∘ PART_OF → PART_OF (DeclaredExact)
        let entity_a: EntityId = 101;
        let entity_b: EntityId = 102;
        let entity_c: EntityId = 103;
        let rel_part_of = 1001u32;

        let op1 = ReasoningOperator {
            operator_id: ReasoningOperatorId(1),
            from_entity: entity_a,
            to_entity: entity_b,
            relation_type: rel_part_of,
            from_coords: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            to_coords: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            transform: vec![Cl24Blade::new(1, 1.0)],
            evidence: vec![DurableEvidenceRef::Relation(501)],
            provenance_id: 10,
            reference_confidence: 0.90,
        };

        let op2 = ReasoningOperator {
            operator_id: ReasoningOperatorId(2),
            from_entity: entity_b,
            to_entity: entity_c,
            relation_type: rel_part_of,
            from_coords: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            to_coords: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            transform: vec![Cl24Blade::new(2, 1.0)],
            evidence: vec![DurableEvidenceRef::Relation(502)],
            provenance_id: 11,
            reference_confidence: 0.95,
        };

        let mut rule_registry = CompositionRuleRegistry::new();
        rule_registry.register(CompositionRule {
            lhs: rel_part_of,
            rhs: rel_part_of,
            result: rel_part_of,
            semantics: CompositionSemantics::DeclaredExact,
        });

        let config = RuneCl24CompositionConfig::default();
        let closure =
            compile_closure(&[op1, op2], &rule_registry, &config, 500).expect("compiled closure");

        // Assert 1: Semantic endpoint MUST be C, NOT the Cl(24) projected coords
        assert_eq!(closure.start_entity, entity_a);
        assert_eq!(closure.semantic_endpoint, entity_c);
        assert_eq!(closure.chain_relations, vec![rel_part_of, rel_part_of]);
        assert_eq!(closure.chain_entities, vec![entity_a, entity_b, entity_c]);
        assert_eq!(closure.result_relation_type, Some(rel_part_of));
        assert_eq!(closure.closure_kind, ClosureKind::ComposedReasoning);
        assert_eq!(closure.epistemic_status, EpistemicStatus::Provisional);

        // Assert 2: Cl24 projected coords are retained as artifact
        assert_eq!(closure.composition_artifact.chain_depth, 2);
        assert_ne!(closure.composition_artifact.semantic_fingerprint, [0u8; 32]);

        // Separately: Unmapped relation pair produces no authoritative result relation
        let rel_correlated = 2001u32;
        let rel_precedes = 2002u32;
        let op_unmapped1 = ReasoningOperator {
            operator_id: ReasoningOperatorId(3),
            from_entity: entity_a,
            to_entity: entity_b,
            relation_type: rel_correlated,
            from_coords: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            to_coords: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            transform: vec![Cl24Blade::new(1, 1.0)],
            evidence: vec![],
            provenance_id: 10,
            reference_confidence: 0.80,
        };
        let op_unmapped2 = ReasoningOperator {
            operator_id: ReasoningOperatorId(4),
            from_entity: entity_b,
            to_entity: entity_c,
            relation_type: rel_precedes,
            from_coords: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            to_coords: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            transform: vec![Cl24Blade::new(2, 1.0)],
            evidence: vec![],
            provenance_id: 11,
            reference_confidence: 0.85,
        };

        let unmapped_closure =
            compile_closure(&[op_unmapped1, op_unmapped2], &rule_registry, &config, 500)
                .expect("exploratory closure");
        assert_eq!(unmapped_closure.result_relation_type, None); // No admitted rule
        assert_eq!(
            unmapped_closure.epistemic_status,
            EpistemicStatus::Provisional
        );
    }
}
