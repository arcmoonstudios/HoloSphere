/* holosphere/src/learning/synthesis/composition.rs */
//!▫~•◦-------------------------------‣
//! # Structured Action Plans & Synthesis Basis
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Models structured action plans with explicit dependency relationships
//! (sequential, parallel, conditional) and synthesis basis derivations.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::experience::action::DurableActionParameter;
use crate::experience::id::{ActionId, AttemptId};
use crate::learning::inference::candidate::InferenceCandidateId;
use crate::learning::inference::rune_evo::reasoning::closure::CompositionRule;

/// Unique identifier for an action step within a candidate plan.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct CandidateActionStepId(pub u32);

/// Explicit composition semantics between action steps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ActionComposition {
    /// Steps execute in strict sequential order (A then B).
    Sequential,
    /// Steps execute concurrently without ordering dependencies.
    Parallel,
    /// Steps must both succeed together to deliver utility (A and B).
    Conjunctive,
    /// Step executes only if predecessor steps succeed.
    Conditional,
    /// Step is an alternative/fallback if predecessor fails.
    Alternative,
}

/// A concrete action invocation step within a proposed plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionPlanStep {
    pub step_id: CandidateActionStepId,
    pub action: ActionId,
    pub parameters: Vec<DurableActionParameter>,
    pub depends_on: Vec<CandidateActionStepId>,
    pub composition_mode: ActionComposition,
}

/// A complete structured action plan.
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ActionPlan {
    pub steps: Vec<ActionPlanStep>,
}

/// The formal empirical or algebraic basis justifying a synthesized composition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SynthesisBasis {
    /// Derived from an admitted transitive relation composition rule.
    DeclaredRule(CompositionRule),
    /// Transferred from an $SO(8)$-aligned structural analogy candidate.
    StructuralAnalogy(InferenceCandidateId),
    /// Composed from historical co-success across empirical attempts.
    HistoricalCoSuccess(Vec<AttemptId>),
    /// Generated via $Cl(24)$ multivector exploratory product.
    ExploratoryGeometry(InferenceCandidateId),
}
