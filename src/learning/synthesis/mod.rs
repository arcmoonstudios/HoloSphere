/* holosphere/src/learning/synthesis/mod.rs */
//!▫~•◦-------------------------------‣
//! # End-to-End Structural Synthesis & Learning Loop
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Synthesizes provisional resolution action plans from empirical precedents,
//! structural analogies, and algebraic composition without executing actions directly.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod alignment;
pub mod candidate;
pub mod composition;
pub mod constraint;
pub mod planner;
pub mod precedent;
pub mod request;
pub mod trace;

pub use alignment::{ContextApplicability, ContextDifference, StructuralAnalogyArtifact};
pub use candidate::{
    CandidateResolutionState, ResolutionCandidate, ResolutionCandidateId, SynthesisScores,
};
pub use composition::{
    ActionComposition, ActionPlan, ActionPlanStep, CandidateActionStepId, SynthesisBasis,
};
pub use constraint::{ActionConstraint, ConstraintCheck, ConstraintCode, ConstraintResult};
pub use planner::{
    SynthesisAttempt, SynthesisKnowledgeBase, SynthesisPolicy, SynthesisResult, synthesize,
};
pub use precedent::{Precedent, PrecedentDisposition};
pub use request::{SynthesisGoal, SynthesisPolicyId, SynthesisRequest};
pub use trace::{ClosureArtifactId, StructuralSynthesisTrace};

#[cfg(test)]
mod tests {
    use super::*;

    use crate::entity::status::EpistemicStatus;
    use crate::experience::action::{
        ActionInvocation, ActionParameterValue, DurableActionParameter,
    };
    use crate::experience::id::{ActionId, AttemptId, ContextId, ProblemId};

    fn make_test_e8(offset: f32) -> Vec<[f32; 8]> {
        vec![
            [1.0 + offset, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0 + offset, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0 + offset, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    }

    #[test]
    fn test_phase8_write_latency_end_to_end_synthesis_scenario() {
        let mut kb = SynthesisKnowledgeBase::new(100);

        // Problems: P1 (HighWriteLatency), P2 (ReplicatedMetadataWriteStalls)
        let p1 = ProblemId(1001);
        let p2 = ProblemId(1002);
        kb.register_problem(p1, make_test_e8(0.0));
        kb.register_problem(p2, make_test_e8(0.01)); // Structurally isomorphic

        // Contexts: C1 (NVMe, 5-node, tiny batches), C2 (NVMe, 5-node, tiny updates)
        let c1 = ContextId(2001);
        let c2 = ContextId(2002);
        kb.register_context(
            c1,
            vec![
                ("storage".into(), "nvme".into()),
                ("cluster_size".into(), "5".into()),
            ],
        );
        kb.register_context(
            c2,
            vec![
                ("storage".into(), "nvme".into()),
                ("cluster_size".into(), "5".into()),
            ],
        );

        // Actions: Action 1 (IncreaseWorkerCount), Action 2 (GroupCommit), Action 3 (LargeBatch)
        let action_workers = ActionId(3001);
        let action_group_commit = ActionId(3002);
        let action_large_batch = ActionId(3003);

        // Historical Attempt 118: IncreaseWorkerCount on P1 -> failed (Contradicting, utility -50)
        kb.register_attempt(SynthesisAttempt {
            attempt_id: AttemptId(118),
            problem_id: p1,
            context_id: c1,
            actions: vec![ActionInvocation {
                invocation_id: 1,
                attempt_id: AttemptId(118),
                action_id: action_workers,
                ordinal: 0,
                parameters: vec![DurableActionParameter {
                    key: "workers".into(),
                    value: ActionParameterValue::Integer(64),
                }],
                started_lsn: 10,
                completed_lsn: 20,
                provenance_id: 1,
            }],
            outcome_utility_q32: -3276800, // -50 in Q16
            disposition: PrecedentDisposition::Contradicting,
        });

        // Historical Attempt 402: GroupCommit on P1 -> succeeded (Supporting, utility +80)
        kb.register_attempt(SynthesisAttempt {
            attempt_id: AttemptId(402),
            problem_id: p1,
            context_id: c1,
            actions: vec![ActionInvocation {
                invocation_id: 2,
                attempt_id: AttemptId(402),
                action_id: action_group_commit,
                ordinal: 0,
                parameters: vec![DurableActionParameter {
                    key: "window_ms".into(),
                    value: ActionParameterValue::Integer(3),
                }],
                started_lsn: 30,
                completed_lsn: 40,
                provenance_id: 2,
            }],
            outcome_utility_q32: 5242880, // +80 in Q16
            disposition: PrecedentDisposition::Supporting,
        });

        // Negative Combination: GroupCommit + LargeBatch causes severe tail latency
        kb.register_negative_combination(vec![action_group_commit, action_large_batch]);

        // Synthesize for new problem P2 in Context C2
        let request = SynthesisRequest {
            problem: p2,
            context: c2,
            snapshot_lsn: 100,
            goal: SynthesisGoal::MitigateProblem(p2),
            policy: SynthesisPolicyId(1),
        };

        let policy = SynthesisPolicy::default();
        let result = synthesize(&kb, &request, &policy).expect("synthesis result");

        assert!(!result.candidates.is_empty());

        // Top-ranked candidate MUST be GroupCommit due to positive empirical evidence on P1
        let top = &result.candidates[0];
        assert_eq!(top.action_plan.steps.len(), 1);
        assert_eq!(top.action_plan.steps[0].action, action_group_commit);
        assert_eq!(top.supporting_precedents, vec![AttemptId(402)]);
        assert!(top.scores.aggregate_ranking_score_q32 > 0);

        // HARD INVARIANT: Epistemic status is strictly Provisional
        assert_eq!(top.epistemic_status, EpistemicStatus::Provisional);
        // Execution Boundary: Candidate resolution state is Proposed, not auto-executed
        assert_eq!(top.resolution_state, CandidateResolutionState::Proposed);

        // Verify that IncreaseWorkerCount is ranked lower / penalized
        let worker_cand = result
            .candidates
            .iter()
            .find(|c| c.action_plan.steps[0].action == action_workers);
        assert!(worker_cand.is_some());
        assert!(
            worker_cand.unwrap().scores.aggregate_ranking_score_q32
                < top.scores.aggregate_ranking_score_q32
        );
        assert_eq!(
            worker_cand.unwrap().contradicting_precedents,
            vec![AttemptId(118)]
        );
    }

    #[test]
    fn test_phase8_negative_combination_constraint_suppression() {
        let mut kb = SynthesisKnowledgeBase::new(100);
        let p1 = ProblemId(1001);
        let c1 = ContextId(2001);
        let action_a = ActionId(3001);
        let action_b = ActionId(3002);

        // Attempt using both Action A and Action B
        kb.register_attempt(SynthesisAttempt {
            attempt_id: AttemptId(550),
            problem_id: p1,
            context_id: c1,
            actions: vec![
                ActionInvocation {
                    invocation_id: 1,
                    attempt_id: AttemptId(550),
                    action_id: action_a,
                    ordinal: 0,
                    parameters: vec![],
                    started_lsn: 10,
                    completed_lsn: 20,
                    provenance_id: 1,
                },
                ActionInvocation {
                    invocation_id: 2,
                    attempt_id: AttemptId(550),
                    action_id: action_b,
                    ordinal: 1,
                    parameters: vec![],
                    started_lsn: 20,
                    completed_lsn: 30,
                    provenance_id: 1,
                },
            ],
            outcome_utility_q32: -6553600,
            disposition: PrecedentDisposition::Contradicting,
        });

        // Register negative combination
        kb.register_negative_combination(vec![action_a, action_b]);

        let request = SynthesisRequest {
            problem: p1,
            context: c1,
            snapshot_lsn: 100,
            goal: SynthesisGoal::MitigateProblem(p1),
            policy: SynthesisPolicyId(1),
        };

        let result =
            synthesize(&kb, &request, &SynthesisPolicy::default()).expect("synthesis result");

        // The combination A+B MUST be suppressed/rejected by constraint check
        assert!(result.candidates.is_empty());
        assert!(!result.trace.constraint_checks.is_empty());
        assert_eq!(
            result.trace.constraint_checks[0].code,
            ConstraintCode::FailedCombinationPrecedent
        );
    }
}
