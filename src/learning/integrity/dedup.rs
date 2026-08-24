/* holosphere/src/learning/integrity/dedup.rs */
//!▫~•◦-------------------------------‣
//! # Semantic Candidate Identity & Generation Deduplication
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Separates durable semantic hypothesis identity from individual transient synthesis
//! run events, preventing ontology and graph entity explosion when identical plans
//! are proposed repeatedly across learning cycles.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

use crate::experience::id::{ContextId, ProblemId};
use crate::learning::synthesis::composition::ActionPlan;

/// Unique identifier for a particular synthesis invocation event.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct SynthesisRunId(pub u64);

/// Unique identifier for a candidate within a specific synthesis run.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct SynthesisCandidateId(pub u64);

/// Deterministic content-addressed semantic key of a problem, context, and plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResolutionSemanticKey(pub [u8; 32]);

impl ResolutionSemanticKey {
    /// Computes the canonical deterministic hash of a problem, context, and action plan.
    pub fn compute(problem: ProblemId, context: ContextId, plan: &ActionPlan) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"RESOL_SEMANTIC_KEY_V1");
        hasher.update(&problem.0.to_le_bytes());
        hasher.update(&context.0.to_le_bytes());

        for step in &plan.steps {
            hasher.update(&step.step_id.0.to_le_bytes());
            hasher.update(&step.action.0.to_le_bytes());
            hasher.update(&(step.composition_mode as u8).to_le_bytes());
            for dep in &step.depends_on {
                hasher.update(&dep.0.to_le_bytes());
            }
            for param in &step.parameters {
                hasher.update(param.key.as_bytes());
            }
        }

        let mut key = [0u8; 32];
        key.copy_from_slice(&hasher.finalize());
        Self(key)
    }
}

/// Evidential occurrence of a semantic plan being synthesized.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SynthesisOccurrence {
    pub run_id: SynthesisRunId,
    pub candidate_id: SynthesisCandidateId,
    pub snapshot_lsn: u64,
    pub ranking_score_q32: i64,
}

/// Deduplicating registry mapping semantic keys to their derivation histories.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SemanticCandidateRegistry {
    entries: HashMap<ResolutionSemanticKey, Vec<SynthesisOccurrence>>,
}

impl SemanticCandidateRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records a synthesis occurrence for a semantic key.
    pub fn record_occurrence(
        &mut self,
        key: ResolutionSemanticKey,
        occurrence: SynthesisOccurrence,
    ) {
        self.entries.entry(key).or_default().push(occurrence);
    }

    /// Returns the number of distinct semantic hypotheses registered.
    pub fn unique_semantic_hypotheses_count(&self) -> usize {
        self.entries.len()
    }

    /// Returns the total occurrence count across all registered hypotheses.
    pub fn total_occurrences_count(&self) -> usize {
        self.entries.values().map(|v| v.len()).sum()
    }

    /// Retrieves all recorded occurrences for a given semantic key.
    pub fn get_occurrences(&self, key: &ResolutionSemanticKey) -> Option<&[SynthesisOccurrence]> {
        self.entries.get(key).map(|v| v.as_slice())
    }
}
