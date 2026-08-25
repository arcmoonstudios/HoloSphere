/* hnsqr/src/learning/discovery/distillation.rs */
//!▫~•◦-------------------------------------------------------------------‣
//! # Autonomous Cognitive Distillation (RLVR & DPO Generator)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! This module synthesizes high-fidelity Direct Preference Optimization (DPO) and
//! Reinforcement Learning from Verified Reasoning (RLVR) training pairs directly
//! from HoloSphere's empirical discovery catalog.
//!
//! ## Mechanism
//! - **Chosen Response:** Mathematically verified declarative operator plans with
//!   admitted empirical utility and $Cl(24)$ Clifford causal proofs.
//! - **Rejected Response:** Falsified, unverified, or counterfactually invalidated
//!   candidate plans with recorded error traces.
//! - **Direct Export:** Generates JSONL format consumable by Axolotl, TRL, and vLLM.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// A single candidate plan representation in the preference pair.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DpoCandidate {
    /// Declarative plan or operator representation.
    pub plan_text: String,
    /// Measured empirical utility score.
    pub utility_score: f32,
    /// Formal verification status or proof certificate.
    pub verification_proof: String,
}

/// A complete Direct Preference Optimization (DPO) training sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DpoReasoningPair {
    /// Unique identifier of the distillation instance.
    pub id: String,
    /// Structured problem prompt and environmental context.
    pub prompt: String,
    /// Chosen (empirically verified) action trajectory.
    pub chosen: DpoCandidate,
    /// Rejected (falsified / suboptimal) action trajectory.
    pub rejected: DpoCandidate,
    /// Utility margin $\Delta U = U(\text{chosen}) - U(\text{rejected})$.
    pub margin: f32,
    /// Epistemic status confirmation.
    pub epistemic_status: String,
}

/// Exporter for autonomous model distillation datasets.
pub struct AutonomousDistillationExporter;

impl AutonomousDistillationExporter {
    /// Synthesizes a DPO reasoning pair from empirical problem, chosen, and rejected plans.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn create_pair(
        id: impl Into<String>,
        prompt: impl Into<String>,
        chosen_plan: impl Into<String>,
        chosen_utility: f32,
        chosen_proof: impl Into<String>,
        rejected_plan: impl Into<String>,
        rejected_utility: f32,
        rejected_trace: impl Into<String>,
    ) -> DpoReasoningPair {
        let chosen = DpoCandidate {
            plan_text: chosen_plan.into(),
            utility_score: chosen_utility,
            verification_proof: chosen_proof.into(),
        };
        let rejected = DpoCandidate {
            plan_text: rejected_plan.into(),
            utility_score: rejected_utility,
            verification_proof: rejected_trace.into(),
        };
        let margin = chosen_utility - rejected_utility;

        DpoReasoningPair {
            id: id.into(),
            prompt: prompt.into(),
            chosen,
            rejected,
            margin,
            epistemic_status: "MathematicallyFalsifiedComparison".to_string(),
        }
    }

    /// Serializes a batch of DPO reasoning pairs into standard JSONL format.
    #[must_use]
    pub fn serialize_to_jsonl(pairs: &[DpoReasoningPair]) -> String {
        let mut buffer = String::new();
        for pair in pairs {
            if let Ok(json) = serde_json::to_string(pair) {
                buffer.push_str(&json);
                buffer.push('\n');
            }
        }
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dpo_pair_creation_and_serialization() {
        let pair = AutonomousDistillationExporter::create_pair(
            "distill-001",
            "Resolve high write latency under 5-node Raft consensus cluster",
            "GroupCommitBatch(3ms) + LinearizableReadIndexBarrier",
            0.92,
            "Cl(24) Causal Invariant Verified: LSN monotonic",
            "DirectFsyncOnEveryWrite + SpinlockWait",
            -0.45,
            "Falsified under multi_process_chaos.rs: Disk stall detected",
        );

        assert_eq!(pair.margin, 0.92 - (-0.45));
        let jsonl = AutonomousDistillationExporter::serialize_to_jsonl(&[pair]);
        assert!(jsonl.contains("GroupCommitBatch"));
        assert!(jsonl.contains("DirectFsyncOnEveryWrite"));
        assert!(jsonl.contains("distill-001"));
    }
}
