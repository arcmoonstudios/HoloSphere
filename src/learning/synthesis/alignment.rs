/* holosphere/src/learning/synthesis/alignment.rs */
//!▫~•◦-------------------------------‣
//! # Structural Analogy & Context Applicability Alignment
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates structural isomorphism between problem regions and context
//! applicability dimensions without conflating analogy with equivalence.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::experience::id::{ContextId, ProblemId};
use crate::learning::inference::candidate::InferenceScore;
use crate::learning::inference::rune_evo::analogy::RotorAlignmentResult;
use crate::learning::inference::trace::InferenceTrace;

/// Explicit dimension difference between two execution contexts.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextDifference {
    pub dimension_name: Arc<str>,
    pub source_value: Arc<str>,
    pub target_value: Arc<str>,
}

/// Evaluated applicability between a historical precedent context and target context.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextApplicability {
    pub source_context: ContextId,
    pub target_context: ContextId,
    pub exact_match: bool,
    pub inferred_similarity: Option<InferenceScore>,
    pub differing_dimensions: Vec<ContextDifference>,
}

/// Structural analogy artifact recording the $SO(8)$ Givens rotor alignment between problems.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuralAnalogyArtifact {
    pub source_problem: ProblemId,
    pub target_problem: ProblemId,
    pub alignment: RotorAlignmentResult,
    pub residual: f32,
    pub trace: InferenceTrace,
}
