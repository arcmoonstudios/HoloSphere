/* holosphere/src/learning/inference/contract.rs */
//!▫~•◦-------------------------------‣
//! # Universal Inference Contracts & Method Traits
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the universal request, scope, and trait interfaces for all
//! hypothesis generators.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::entity::id::EntityId;
use crate::experience::id::ContextId;
use crate::learning::inference::candidate::InferenceCandidate;
use crate::learning::read::LearningReadSnapshot;
use crate::relation::id::RelationId;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum InferenceError {
    #[error("Inference method is disabled")]
    Disabled,
    #[error("Invalid request parameters: {0}")]
    InvalidParameters(String),
    #[error("Computation failure: {0}")]
    ComputationFailed(String),
    #[error("Input entity {0} not found in pinned snapshot")]
    EntityNotFound(EntityId),
    #[error("Input relation {0} not found in pinned snapshot")]
    RelationNotFound(RelationId),
}

/// Execution mode controlling hypothesis generation during queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum InferenceMode {
    /// No inference candidate generation (default).
    #[default]
    Disabled,
    /// Generate candidates in read-only mode for exploration without proposals.
    ReadOnlyCandidates,
    /// Allow candidate proposals to be submitted to consensus validation.
    AllowProposals,
}

/// Unique identifier for an inference method.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct InferenceMethodId(pub u32);

/// Explicit deterministic seed for reproducible inference execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct InferenceSeed(pub [u8; 32]);

impl Default for InferenceSeed {
    fn default() -> Self {
        Self([
            0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x13, 0x37, 0x01, 0x23, 0x45, 0x67, 0x89, 0xAB,
            0xCD, 0xEF, 0xFE, 0xDC, 0xBA, 0x98, 0x76, 0x54, 0x32, 0x10, 0x55, 0xAA, 0x55, 0xAA,
            0x33, 0xCC, 0x33, 0xCC,
        ])
    }
}

/// Bounded scope for an inference request.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceScope {
    Entity(EntityId),
    Relation(RelationId),
    Context(ContextId),
    Region { entities: Vec<EntityId> },
    Global,
}

/// Universal request payload for hypothesis generation against a pinned snapshot.
pub struct InferenceRequest<'a> {
    pub learning_snapshot: &'a LearningReadSnapshot,
    pub scope: InferenceScope,
    pub seed: InferenceSeed,
    pub max_candidates: usize,
}

/// Universal trait implemented by all hypothesis generation algorithms.
pub trait InferenceMethod: Send + Sync {
    /// Unique identifier for this inference method.
    fn id(&self) -> InferenceMethodId;

    /// Monotonic version of this inference algorithm.
    fn version(&self) -> u32;

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Executes hypothesis generation against a pinned snapshot.
    fn infer(
        &self,
        request: &InferenceRequest<'_>,
    ) -> Result<Vec<InferenceCandidate>, InferenceError>;
}
