/* holosphere/src/learning/collective/conflict.rs */
//!▫~•◦-------------------------------‣
//! # Swarm Belief Conflict Detection & Retention
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Preserves unresolved inter-agent disagreements explicitly as relational audit state.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::id::EntityId;
use crate::learning::collective::belief::AgentId;

/// A pair of conflicting agent beliefs in the same semantic neighbourhood.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConflictPair {
    pub belief_a: EntityId,
    pub author_a: AgentId,
    pub belief_b: EntityId,
    pub author_b: AgentId,
    pub distance: f32,
}

/// Status of an identified inter-agent belief conflict.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ConflictResolution {
    /// Conflict is actively preserved as an empirical disagreement.
    #[default]
    Preserved,
    /// Conflict was reconciled by subsequent empirical evidence or consensus.
    Reconciled,
}
