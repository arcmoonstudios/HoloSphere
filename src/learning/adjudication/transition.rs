/* holosphere/src/learning/adjudication/transition.rs */
//!▫~•◦-------------------------------‣
//! # Epistemic Adjudication Transition Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates consolidated evidence summaries against versioned policies to produce
//! deterministic, replay-safe epistemic transitions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::entity::status::EpistemicStatus;
use crate::learning::adjudication::policy::{
    AdjudicationDecisionCode, AdjudicationDisposition, AdjudicationPolicy,
};
use crate::learning::evidence::accumulator::EvidenceSummary;

/// Evaluates consolidated evidence to determine the justified epistemic transition.
pub fn evaluate_adjudication(
    summary: &EvidenceSummary,
    current_status: EpistemicStatus,
    policy: &AdjudicationPolicy,
) -> (
    EpistemicStatus,
    AdjudicationDecisionCode,
    AdjudicationDisposition,
) {
    if summary.observation_count < policy.min_observations as u64 {
        return (
            current_status,
            AdjudicationDecisionCode::InsufficientEvidence,
            AdjudicationDisposition::Pending,
        );
    }

    // Check for contradiction threshold
    if summary.contradiction_count > policy.max_contradictions as u64
        && summary.utility_sum_q32 <= policy.falsify_utility_q32
    {
        return (
            EpistemicStatus::Contradicted,
            AdjudicationDecisionCode::ContradictionThresholdReached,
            AdjudicationDisposition::Contradicted,
        );
    }

    // Check for promotion to Inferred
    if summary.support_count >= policy.min_support as u64
        && summary.contradiction_count <= policy.max_contradictions as u64
        && summary.utility_sum_q32 >= policy.promote_utility_q32
    {
        // Strictly only promote to Inferred from Provisional
        let next_status = match current_status {
            EpistemicStatus::Provisional => EpistemicStatus::Inferred,
            other => other, // Asserted and Observed remain unchanged
        };
        return (
            next_status,
            AdjudicationDecisionCode::SupportThresholdReached,
            AdjudicationDisposition::Supported,
        );
    }

    // Mixed evidence or context-dependent
    if summary.support_count > 0 && summary.contradiction_count > 0 {
        return (
            current_status,
            AdjudicationDecisionCode::ContextDependent,
            AdjudicationDisposition::ContextDependent,
        );
    }

    (
        current_status,
        AdjudicationDecisionCode::MixedEvidence,
        AdjudicationDisposition::Pending,
    )
}
