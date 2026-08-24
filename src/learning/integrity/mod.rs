/* holosphere/src/learning/integrity/mod.rs */
//!▫~•◦-------------------------------‣
//! # Epistemic Integrity & Long-Horizon Learning Validation Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Enforces evidential accounting invariants across long-horizon continuous learning:
//! prevents evidence multiplication, blocks circular self-reinforcement, guards multi-action
//! attribution, rejects stale proposals, and deduplicates semantic candidate hypotheses.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod attribution;
pub mod audit;
pub mod dedup;
pub mod dependency;
pub mod independence;
pub mod lineage;
pub mod staleness;

pub use attribution::{PlanAttributionMethod, PlanAttributionRecord, compute_plan_attribution};
pub use audit::{CanonicalLearningAuditDigest, compute_audit_digest};
pub use dedup::{
    ResolutionSemanticKey, SemanticCandidateRegistry, SynthesisCandidateId, SynthesisOccurrence,
    SynthesisRunId,
};
pub use dependency::{CircularityCheck, check_epistemic_circularity};
pub use independence::{EvidenceIndependenceReport, evaluate_evidence_independence};
pub use lineage::{EmpiricalRootId, EpistemicLineageGraph, LineageNodeKind};
pub use staleness::{ProposalStalenessCheck, SynthesisDependencyDigest};

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    use crate::experience::id::{ActionId, AttemptId, ContextId, ProblemId};
    use crate::learning::collective::AgentId;
    use crate::learning::synthesis::composition::{
        ActionComposition, ActionPlan, ActionPlanStep, CandidateActionStepId,
    };

    #[test]
    fn test_phase9_anti_multiplication_swarm_echo_collapse() {
        let mut lineage = EpistemicLineageGraph::new();

        // Observation O1 is an empirical root
        let o1 = EmpiricalRootId(101);
        let entity_o1 = 1001;
        lineage.register_observation(entity_o1, o1);

        // Agent A believes H1 derived directly from O1
        let entity_h1 = 2001;
        lineage.register_derivation(
            entity_h1,
            LineageNodeKind::InferredHypothesis(entity_h1),
            &[entity_o1],
        );

        // Agent B copies A (H2 derived from H1)
        let entity_h2 = 2002;
        lineage.register_derivation(
            entity_h2,
            LineageNodeKind::InferredHypothesis(entity_h2),
            &[entity_h1],
        );

        // Agent C copies B (H3 derived from H2)
        let entity_h3 = 2003;
        lineage.register_derivation(
            entity_h3,
            LineageNodeKind::InferredHypothesis(entity_h3),
            &[entity_h2],
        );

        // Independent empirical observation O2
        let o2 = EmpiricalRootId(102);
        let entity_o2 = 1002;
        lineage.register_observation(entity_o2, o2);

        // Agent D derives H4 from independent observation O2
        let entity_h4 = 2004;
        lineage.register_derivation(
            entity_h4,
            LineageNodeKind::InferredHypothesis(entity_h4),
            &[entity_o2],
        );

        // Swarm claims: 3 agents echo O1, 1 agent independently measured O2 (total 4 assertions)
        let claims = vec![
            (AgentId(1), entity_h1),
            (AgentId(2), entity_h2),
            (AgentId(3), entity_h3),
            (AgentId(4), entity_h4),
        ];

        let report = evaluate_evidence_independence(&lineage, &claims);

        assert_eq!(report.total_assertions, 4);
        assert_eq!(report.reporting_agents_count, 4);
        // CRITICAL INVARIANT: 3 echoed assertions collapse to 1 root, plus 1 independent root = 2 total
        assert_eq!(report.independent_root_count, 2);
        assert_eq!(report.empirical_roots, vec![o1, o2]);
    }

    #[test]
    fn test_phase9_nary_fanout_collapses_to_empirical_roots() {
        let mut lineage = EpistemicLineageGraph::new();
        let root = EmpiricalRootId(500);
        let observation = 10_000;
        lineage.register_observation(observation, root);

        // Model a 24-member inferred hyperrelation whose members all ultimately
        // echo the same observation. Arity and assertion count must not multiply
        // the independent evidential support.
        let mut claims = Vec::new();
        let mut member_entities = Vec::new();
        for member in 0..24u64 {
            let entity = 20_000 + member;
            lineage.register_derivation(
                entity,
                LineageNodeKind::InferredHypothesis(entity),
                &[observation],
            );
            member_entities.push(entity);
            claims.push((AgentId(member + 1), entity));
        }
        let hyperrelation = 30_000;
        lineage.register_derivation(
            hyperrelation,
            LineageNodeKind::InferredHypothesis(hyperrelation),
            &member_entities,
        );
        claims.push((AgentId(99), hyperrelation));

        let report = evaluate_evidence_independence(&lineage, &claims);
        assert_eq!(report.total_assertions, 25);
        assert_eq!(report.reporting_agents_count, 25);
        assert_eq!(report.independent_root_count, 1);
        assert_eq!(report.empirical_roots, vec![root]);

        // Feeding the derived hyperrelation back into one of its own members is circular.
        assert_eq!(
            check_epistemic_circularity(&lineage, member_entities[0], hyperrelation),
            CircularityCheck::CircularDependencyDetected {
                target: member_entities[0],
                reinforcing_source: hyperrelation,
            }
        );
    }

    #[test]
    fn test_phase9_circular_support_prevention() {
        let mut lineage = EpistemicLineageGraph::new();

        let o1 = EmpiricalRootId(101);
        let entity_o1 = 1001;
        lineage.register_observation(entity_o1, o1);

        // H1 depends on O1
        let h1 = 2001;
        lineage.register_derivation(h1, LineageNodeKind::InferredHypothesis(h1), &[entity_o1]);

        // H2 depends on H1
        let h2 = 2002;
        lineage.register_derivation(h2, LineageNodeKind::InferredHypothesis(h2), &[h1]);

        // H3 depends on H2
        let h3 = 2003;
        lineage.register_derivation(h3, LineageNodeKind::InferredHypothesis(h3), &[h2]);

        // Attempting to reinforce H1 from its own descendant H3 MUST be flagged as circular
        let check = check_epistemic_circularity(&lineage, h1, h3);
        assert_eq!(
            check,
            CircularityCheck::CircularDependencyDetected {
                target: h1,
                reinforcing_source: h3,
            }
        );

        // Independent entity H4 depending on O1 is acyclic relative to H1
        let h4 = 2004;
        lineage.register_derivation(h4, LineageNodeKind::InferredHypothesis(h4), &[entity_o1]);
        assert_eq!(
            check_epistemic_circularity(&lineage, h1, h4),
            CircularityCheck::Acyclic
        );
    }

    #[test]
    fn test_phase9_multi_action_attribution_joint_plan_only() {
        let attempt = AttemptId(901);
        let action_a = ActionId(3001);
        let action_b = ActionId(3002);

        let standalone_evidence = HashSet::new(); // No standalone trials for A or B

        let record = compute_plan_attribution(attempt, &[action_a, action_b], &standalone_evidence);

        // When [A, B] succeeds without standalone ablation, credit is JointPlanOnly
        assert_eq!(
            record.attribution_method,
            PlanAttributionMethod::JointPlanOnly
        );
        assert!(record.justified_credited_actions.is_empty());
        assert_eq!(record.withheld_standalone_actions, vec![action_a, action_b]);

        // If A had standalone confirmation, it becomes justified
        let mut standalone_with_a = HashSet::new();
        standalone_with_a.insert(action_a);
        let record2 = compute_plan_attribution(attempt, &[action_a, action_b], &standalone_with_a);
        assert_eq!(record2.justified_credited_actions, vec![action_a]);
        assert_eq!(record2.withheld_standalone_actions, vec![action_b]);
    }

    #[test]
    fn test_phase9_proposal_staleness_rejection() {
        let digest = SynthesisDependencyDigest {
            snapshot_lsn: 100,
            problem_version: 5,
            context_fingerprint: [1u8; 32],
            precedent_digest: [2u8; 32],
            relation_digest: [3u8; 32],
            policy_version: 1,
        };

        // Matching world: Fresh
        assert_eq!(
            digest.validate(5, &[1u8; 32], &[2u8; 32], 1),
            ProposalStalenessCheck::Fresh
        );

        // Stale problem version
        assert_eq!(
            digest.validate(6, &[1u8; 32], &[2u8; 32], 1),
            ProposalStalenessCheck::StaleProblemVersion {
                expected: 5,
                actual: 6,
            }
        );

        // Stale policy version
        assert_eq!(
            digest.validate(5, &[1u8; 32], &[2u8; 32], 2),
            ProposalStalenessCheck::StalePolicyVersion {
                expected: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn test_phase9_semantic_candidate_dedup() {
        let mut registry = SemanticCandidateRegistry::new();

        let problem = ProblemId(101);
        let context = ContextId(201);
        let plan = ActionPlan {
            steps: vec![ActionPlanStep {
                step_id: CandidateActionStepId(0),
                action: ActionId(301),
                parameters: vec![],
                depends_on: vec![],
                composition_mode: ActionComposition::Sequential,
            }],
        };

        let semantic_key = ResolutionSemanticKey::compute(problem, context, &plan);

        // Record 10 independent synthesis run events proposing this exact same plan
        for run_idx in 1..=10 {
            registry.record_occurrence(
                semantic_key,
                SynthesisOccurrence {
                    run_id: SynthesisRunId(run_idx),
                    candidate_id: SynthesisCandidateId(1),
                    snapshot_lsn: 100 + run_idx,
                    ranking_score_q32: 65536,
                },
            );
        }

        // Must maintain exactly 1 distinct semantic hypothesis entity and 10 occurrence derivations
        assert_eq!(registry.unique_semantic_hypotheses_count(), 1);
        assert_eq!(registry.total_occurrences_count(), 10);
        assert_eq!(registry.get_occurrences(&semantic_key).unwrap().len(), 10);
    }

    #[test]
    fn test_phase9_long_horizon_1000_cycles_state_equivalence() {
        let mut incremental_lineage = EpistemicLineageGraph::new();
        let mut incremental_registry = SemanticCandidateRegistry::new();

        // Canonical log of events
        struct CanonicalLogEntry {
            cycle: u64,
            problem: ProblemId,
            context: ContextId,
            plan: ActionPlan,
            root: EmpiricalRootId,
        }

        let mut canonical_log = Vec::new();

        // Simulate 1,000 learning cycles
        for cycle in 1..=1000u64 {
            let problem = ProblemId(100 + (cycle % 10)); // 10 recurring problem classes
            let context = ContextId(200 + (cycle % 5)); // 5 execution contexts
            let action = ActionId(300 + (cycle % 7)); // 7 discrete actions
            let plan = ActionPlan {
                steps: vec![ActionPlanStep {
                    step_id: CandidateActionStepId(0),
                    action,
                    parameters: vec![],
                    depends_on: vec![],
                    composition_mode: ActionComposition::Sequential,
                }],
            };
            let root = EmpiricalRootId(cycle);

            // 1. Incremental update
            incremental_lineage.register_observation(cycle, root);
            let sem_key = ResolutionSemanticKey::compute(problem, context, &plan);
            incremental_registry.record_occurrence(
                sem_key,
                SynthesisOccurrence {
                    run_id: SynthesisRunId(cycle),
                    candidate_id: SynthesisCandidateId(1),
                    snapshot_lsn: cycle,
                    ranking_score_q32: 65536,
                },
            );

            canonical_log.push(CanonicalLogEntry {
                cycle,
                problem,
                context,
                plan,
                root,
            });

            // Checkpoints at cycles 1, 10, 100, 500, 1000: test rebuild bit-equivalence
            if cycle == 1 || cycle == 10 || cycle == 100 || cycle == 500 || cycle == 1000 {
                let mut rebuilt_lineage = EpistemicLineageGraph::new();
                let mut rebuilt_registry = SemanticCandidateRegistry::new();

                for entry in &canonical_log {
                    rebuilt_lineage.register_observation(entry.cycle, entry.root);
                    let k =
                        ResolutionSemanticKey::compute(entry.problem, entry.context, &entry.plan);
                    rebuilt_registry.record_occurrence(
                        k,
                        SynthesisOccurrence {
                            run_id: SynthesisRunId(entry.cycle),
                            candidate_id: SynthesisCandidateId(1),
                            snapshot_lsn: entry.cycle,
                            ranking_score_q32: 65536,
                        },
                    );
                }

                let inc_digest = compute_audit_digest(&incremental_lineage, &incremental_registry);
                let reb_digest = compute_audit_digest(&rebuilt_lineage, &rebuilt_registry);

                // Incremental == Rebuilt invariant across all checkpoints
                assert_eq!(
                    inc_digest, reb_digest,
                    "Rebuild mismatch at cycle {}",
                    cycle
                );
            }
        }
    }
}
