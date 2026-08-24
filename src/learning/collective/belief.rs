/* holosphere/src/learning/collective/belief.rs */
//!▫~•◦-------------------------------‣
//! # Agent Identity & Decayed Belief Confidence
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides multi-agent belief representation with time-dependent confidence decay.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::id::EntityId;

/// Unique identifier for an autonomous agent participant in collective belief synthesis.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct AgentId(pub u64);

/// Metadata associated with a registered swarm agent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentMeta {
    pub id: AgentId,
    pub name: String,
    pub registered_at_lsn: u64,
    pub default_decay_rate: f32,
    pub default_confidence: f32,
}

/// An individual agent belief anchored in 8D semantic space.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AgentBelief {
    pub belief_id: EntityId,
    pub author_agent: AgentId,
    pub initial_confidence: f32,
    pub decay_rate: f32,
    pub reinforcement_count: u32,
    pub committed_at_ms: u64,
    pub last_reinforced_ms: u64,
    pub coords: [f32; 8],
}

impl AgentBelief {
    /// Computes the time-decayed effective confidence of this belief:
    /// $$c(t) = c_0 \cdot \exp\left(-\frac{\lambda}{\max(1, R)} \cdot \Delta t_{\text{sec}}\right)$$
    pub fn effective_confidence(&self, current_time_ms: u64) -> f32 {
        let delta_sec = (current_time_ms.saturating_sub(self.last_reinforced_ms) as f64) / 1000.0;
        let r = self.reinforcement_count.max(1) as f64;
        let lambda_eff = (self.decay_rate as f64) / r;
        let decay = (-lambda_eff * delta_sec).exp() as f32;
        (self.initial_confidence * decay).clamp(0.0, 1.0)
    }
}
