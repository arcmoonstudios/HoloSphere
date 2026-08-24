/* holosphere/src/learning/synthesis/planner.rs */
//!▫~•◦-------------------------------‣
//! # End-to-End Structural Synthesis Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Synthesizes provisional resolution action plans from pinned empirical precedents,
//! structural analogies, and algebraic composition without executing actions directly.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::entity::status::EpistemicStatus;
use crate::experience::action::ActionInvocation;
use crate::experience::id::{ActionId, AttemptId, ContextId, ProblemId};
use crate::learning::inference::contract::InferenceError;
use crate::learning::inference::rune_evo::analogy::align_regions;
use crate::learning::synthesis::alignment::{
    ContextApplicability, ContextDifference, StructuralAnalogyArtifact,
};
use crate::learning::synthesis::candidate::{
    CandidateResolutionState, ResolutionCandidate, ResolutionCandidateId, SynthesisScores,
};
use crate::learning::synthesis::composition::{
    ActionComposition, ActionPlan, ActionPlanStep, CandidateActionStepId,
};
use crate::learning::synthesis::constraint::{ConstraintCheck, ConstraintCode, ConstraintResult};
use crate::learning::synthesis::precedent::{Precedent, PrecedentDisposition};
use crate::learning::synthesis::request::{SynthesisPolicyId, SynthesisRequest};
use crate::learning::synthesis::trace::StructuralSynthesisTrace;

/// Deterministic, versioned synthesis evaluation policy.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SynthesisPolicy {
    pub id: SynthesisPolicyId,
    pub version: u32,
    pub structural_weight_q32: i64,
    pub context_weight_q32: i64,
    pub empirical_weight_q32: i64,
    pub contradiction_penalty_q32: i64,
    pub novelty_weight_q32: i64,
}

impl Default for SynthesisPolicy {
    fn default() -> Self {
        Self {
            id: SynthesisPolicyId(1),
            version: 1,
            structural_weight_q32: 65536,      // 1.0 in Q16
            context_weight_q32: 65536,         // 1.0 in Q16
            empirical_weight_q32: 131072,      // 2.0 in Q16
            contradiction_penalty_q32: 262144, // 4.0 in Q16
            novelty_weight_q32: 16384,         // 0.25 in Q16
        }
    }
}

/// A registered empirical attempt record in the synthesis knowledge base.
#[derive(Clone, Debug, PartialEq)]
pub struct SynthesisAttempt {
    pub attempt_id: AttemptId,
    pub problem_id: ProblemId,
    pub context_id: ContextId,
    pub actions: Vec<ActionInvocation>,
    pub outcome_utility_q32: i64,
    pub disposition: PrecedentDisposition,
}

/// Snapshot view over the pinned empirical universe.
#[derive(Clone, Debug, Default)]
pub struct SynthesisKnowledgeBase {
    pub snapshot_lsn: u64,
    pub problem_geometries: std::collections::HashMap<ProblemId, Vec<[f32; 8]>>,
    pub context_properties: std::collections::HashMap<ContextId, Vec<(Arc<str>, Arc<str>)>>,
    pub attempts: Vec<SynthesisAttempt>,
    pub negative_combinations: Vec<Vec<ActionId>>,
}

impl SynthesisKnowledgeBase {
    pub fn new(snapshot_lsn: u64) -> Self {
        Self {
            snapshot_lsn,
            problem_geometries: std::collections::HashMap::new(),
            context_properties: std::collections::HashMap::new(),
            attempts: Vec::new(),
            negative_combinations: Vec::new(),
        }
    }

    pub fn register_problem(&mut self, problem: ProblemId, region: Vec<[f32; 8]>) {
        self.problem_geometries.insert(problem, region);
    }

    pub fn register_context(&mut self, context: ContextId, props: Vec<(Arc<str>, Arc<str>)>) {
        self.context_properties.insert(context, props);
    }

    pub fn register_attempt(&mut self, attempt: SynthesisAttempt) {
        self.attempts.push(attempt);
    }

    pub fn register_negative_combination(&mut self, combo: Vec<ActionId>) {
        self.negative_combinations.push(combo);
    }
}

/// Result returned by synthesis containing synthesized candidates and the execution trace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SynthesisResult {
    pub candidates: Vec<ResolutionCandidate>,
    pub trace: StructuralSynthesisTrace,
}

/// Pure synthesis entry point: plans candidate solutions strictly within the pinned snapshot.
pub fn synthesize(
    kb: &SynthesisKnowledgeBase,
    request: &SynthesisRequest,
    policy: &SynthesisPolicy,
) -> Result<SynthesisResult, InferenceError> {
    let target_geom = kb
        .problem_geometries
        .get(&request.problem)
        .cloned()
        .unwrap_or_default();
    let target_ctx_props = kb
        .context_properties
        .get(&request.context)
        .cloned()
        .unwrap_or_default();

    let mut precedents: Vec<Precedent> = Vec::new();
    let analogy_traces: Vec<crate::learning::inference::candidate::InferenceCandidateId> =
        Vec::new();
    let mut constraint_checks: Vec<ConstraintCheck> = Vec::new();

    // 1. Structural Analogy & Precedent Retrieval
    for attempt in &kb.attempts {
        let hist_geom = kb
            .problem_geometries
            .get(&attempt.problem_id)
            .cloned()
            .unwrap_or_default();
        let analogy_artifact = if !target_geom.is_empty() && !hist_geom.is_empty() {
            align_regions(&target_geom, &hist_geom, 32).map(|rotor_res| StructuralAnalogyArtifact {
                source_problem: attempt.problem_id,
                target_problem: request.problem,
                residual: rotor_res.residual,
                alignment: rotor_res,
                trace: crate::learning::inference::trace::InferenceTrace {
                    method: crate::learning::inference::rune_evo::analogy::RUNE_ANALOGY_METHOD_ID,
                    method_version:
                        crate::learning::inference::rune_evo::analogy::RUNE_ANALOGY_METHOD_VERSION,
                    source_entities: vec![attempt.problem_id.0, request.problem.0],
                    source_relations: Vec::new(),
                    source_attempts: vec![attempt.attempt_id],
                    snapshot_lsn: request.snapshot_lsn,
                    seed: crate::learning::inference::contract::InferenceSeed::default(),
                    parameter_digest: [0u8; 32],
                },
            })
        } else {
            None
        };

        // Context applicability
        let hist_ctx_props = kb
            .context_properties
            .get(&attempt.context_id)
            .cloned()
            .unwrap_or_default();
        let exact_ctx = attempt.context_id == request.context;
        let mut differing_dims = Vec::new();
        if !exact_ctx {
            for (k, v_target) in &target_ctx_props {
                if let Some((_, v_hist)) = hist_ctx_props.iter().find(|(hk, _)| hk == k) {
                    if v_hist != v_target {
                        differing_dims.push(ContextDifference {
                            dimension_name: k.clone(),
                            source_value: v_hist.clone(),
                            target_value: v_target.clone(),
                        });
                    }
                }
            }
        }

        let ctx_applicability = ContextApplicability {
            source_context: attempt.context_id,
            target_context: request.context,
            exact_match: exact_ctx,
            inferred_similarity: None,
            differing_dimensions: differing_dims,
        };

        precedents.push(Precedent {
            problem: attempt.problem_id,
            attempt: attempt.attempt_id,
            context: attempt.context_id,
            actions: attempt.actions.clone(),
            outcome: crate::experience::id::OutcomeId(attempt.attempt_id.0),
            adjudication: None,
            analogy: analogy_artifact,
            context_applicability: ctx_applicability,
            evidence_disposition: attempt.disposition,
            measured_utility_q32: attempt.outcome_utility_q32,
        });
    }

    // 2. Candidate Action Plan Generation
    let mut candidate_plans: Vec<(ActionPlan, Vec<AttemptId>, Vec<AttemptId>, SynthesisScores)> =
        Vec::new();

    // Group positive and negative precedents by action signature
    for precedent in &precedents {
        if precedent.actions.is_empty() {
            continue;
        }
        let action_ids: Vec<ActionId> = precedent.actions.iter().map(|a| a.action_id).collect();

        // Constraint check for known negative combinations
        let mut rejected = false;
        for neg_combo in &kb.negative_combinations {
            if neg_combo.iter().all(|na| action_ids.contains(na)) {
                constraint_checks.push(ConstraintCheck {
                    code: ConstraintCode::FailedCombinationPrecedent,
                    result: ConstraintResult::Rejected("Known failed action combination".into()),
                    evidence: vec![crate::entity::id::DurableEvidenceRef::Attempt(
                        precedent.attempt.0,
                    )],
                });
                rejected = true;
                break;
            }
        }
        if rejected {
            continue;
        }

        let mut steps = Vec::new();
        for (idx, action_inv) in precedent.actions.iter().enumerate() {
            steps.push(ActionPlanStep {
                step_id: CandidateActionStepId(idx as u32),
                action: action_inv.action_id,
                parameters: action_inv.parameters.clone(),
                depends_on: if idx > 0 {
                    vec![CandidateActionStepId((idx - 1) as u32)]
                } else {
                    Vec::new()
                },
                composition_mode: ActionComposition::Sequential,
            });
        }

        let plan = ActionPlan { steps };

        let supporting: Vec<AttemptId> = precedents
            .iter()
            .filter(|p| {
                p.actions.iter().map(|a| a.action_id).collect::<Vec<_>>() == action_ids
                    && p.evidence_disposition == PrecedentDisposition::Supporting
            })
            .map(|p| p.attempt)
            .collect();

        let contradicting: Vec<AttemptId> = precedents
            .iter()
            .filter(|p| {
                p.actions.iter().map(|a| a.action_id).collect::<Vec<_>>() == action_ids
                    && p.evidence_disposition == PrecedentDisposition::Contradicting
            })
            .map(|p| p.attempt)
            .collect();

        let structural_align = precedent
            .analogy
            .as_ref()
            .map_or(0.0, |a| 1.0 - a.residual)
            .clamp(0.0, 1.0);
        let context_match = if precedent.context_applicability.exact_match {
            1.0
        } else {
            0.5
        };

        let structural_q32 = (structural_align * 65536.0) as i64;
        let context_q32 = (context_match * 65536.0) as i64;
        let utility_q32 = precedent.measured_utility_q32;

        let mut agg_score = (structural_q32 * policy.structural_weight_q32 / 65536)
            + (context_q32 * policy.context_weight_q32 / 65536)
            + (utility_q32 * policy.empirical_weight_q32 / 65536)
            - ((contradicting.len() as i64) * policy.contradiction_penalty_q32);

        if precedent.evidence_disposition == PrecedentDisposition::Contradicting {
            agg_score -= policy.contradiction_penalty_q32;
        }

        let scores = SynthesisScores {
            structural_alignment_q32: structural_q32,
            context_applicability_q32: context_q32,
            supporting_precedent_count: supporting.len() as u32,
            contradicting_precedent_count: contradicting.len() as u32,
            historical_utility_q32: utility_q32,
            cl24_reference_novelty: 0.0,
            cl24_truncation_loss: 0.0,
            aggregate_ranking_score_q32: agg_score,
        };

        // Avoid duplicate plans in candidate list
        if !candidate_plans
            .iter()
            .any(|(p, _, _, _)| p.steps == plan.steps)
        {
            candidate_plans.push((plan, supporting, contradicting, scores));
        }
    }

    // 3. Deterministic Sorting: score DESC, candidate fingerprint ASC
    candidate_plans.sort_by(|a, b| {
        b.3.aggregate_ranking_score_q32
            .cmp(&a.3.aggregate_ranking_score_q32)
            .then_with(|| a.1.len().cmp(&b.1.len()))
    });

    let mut candidates = Vec::new();
    for (idx, (plan, supporting, contradicting, scores)) in candidate_plans.into_iter().enumerate()
    {
        let candidate_id = ResolutionCandidateId((idx + 1) as u64);
        let trace = StructuralSynthesisTrace {
            snapshot_lsn: request.snapshot_lsn,
            target_problem: request.problem,
            target_context: request.context,
            precedent_attempts: supporting
                .iter()
                .chain(contradicting.iter())
                .copied()
                .collect(),
            precedent_relations: Vec::new(),
            analogy_artifacts: analogy_traces.clone(),
            closure_artifacts: Vec::new(),
            constraint_checks: constraint_checks.clone(),
            synthesis_policy: policy.id,
            method_fingerprint: [idx as u8; 32],
        };

        candidates.push(ResolutionCandidate {
            candidate_id,
            problem: request.problem,
            context: request.context,
            action_plan: plan,
            supporting_precedents: supporting,
            contradicting_precedents: contradicting,
            scores,
            structural_trace: trace,
            epistemic_status: EpistemicStatus::Provisional,
            resolution_state: CandidateResolutionState::Proposed,
        });
    }

    let global_trace = StructuralSynthesisTrace {
        snapshot_lsn: request.snapshot_lsn,
        target_problem: request.problem,
        target_context: request.context,
        precedent_attempts: precedents.iter().map(|p| p.attempt).collect(),
        precedent_relations: Vec::new(),
        analogy_artifacts: analogy_traces,
        closure_artifacts: Vec::new(),
        constraint_checks,
        synthesis_policy: policy.id,
        method_fingerprint: [42u8; 32],
    };

    Ok(SynthesisResult {
        candidates,
        trace: global_trace,
    })
}
