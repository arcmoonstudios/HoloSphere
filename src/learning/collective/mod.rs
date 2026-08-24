/* holosphere/src/learning/collective/mod.rs */
//!▫~•◦-------------------------------‣
//! # Swarm Belief Consensus & Multi-Agent Epistemic Arbitration
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides multi-agent epistemic arbitration, decayed provenance confidence weighting,
//! and explicit disagreement conflict preservation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod belief;
pub mod conflict;
pub mod consensus;

pub use belief::{AgentBelief, AgentId, AgentMeta};
pub use conflict::{ConflictPair, ConflictResolution};
pub use consensus::{
    ConsensusResult, SWARM_CONSENSUS_METHOD_ID, SWARM_CONSENSUS_METHOD_VERSION,
    compute_swarm_consensus, materialize_collective_hypothesis,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::status::EpistemicStatus;
    use crate::learning::inference::contract::InferenceSeed;

    #[test]
    fn test_swarm_consensus_decayed_centroid_and_conflict_preservation() {
        let agent1 = AgentId(1);
        let agent2 = AgentId(2);

        // Belief 1: [1.0, 0.0, ...] written at t=0 with high decay rate
        let b1 = AgentBelief {
            belief_id: 101,
            author_agent: agent1,
            initial_confidence: 1.0,
            decay_rate: 0.01,
            reinforcement_count: 1,
            committed_at_ms: 0,
            last_reinforced_ms: 0,
            coords: [1.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };

        // Belief 2: [1.0, -0.05, ...] written at t=5000 with 5 reinforcements (slow decay)
        let b2 = AgentBelief {
            belief_id: 102,
            author_agent: agent2,
            initial_confidence: 1.0,
            decay_rate: 0.001,
            reinforcement_count: 5,
            committed_at_ms: 5000,
            last_reinforced_ms: 5000,
            coords: [1.0, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };

        let beliefs = vec![b1, b2];
        let region = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let current_time_ms = 10_000;

        let result = compute_swarm_consensus(&beliefs, &region, 0.5, 0.15, current_time_ms);

        assert_eq!(result.participating_beliefs.len(), 2);
        assert_eq!(result.contributing_agents.len(), 2);
        // Distance is ~0.10 <= 0.15, and authored by different agents => 1 conflict pair preserved
        assert_eq!(result.conflict_pairs.len(), 1);
        assert_eq!(result.conflict_pairs[0].belief_a, 101);
        assert_eq!(result.conflict_pairs[0].author_a, agent1);
        assert_eq!(result.conflict_pairs[0].belief_b, 102);
        assert_eq!(result.conflict_pairs[0].author_b, agent2);

        // Materialize into collective hypothesis
        let hypothesis = materialize_collective_hypothesis(
            &result,
            301, // CONSENSUS_HYPOTHESIS relation type
            250, // Snapshot LSN
            InferenceSeed::default(),
        )
        .expect("must materialize");

        // Hard epistemic boundary: begins Provisional
        assert_eq!(hypothesis.epistemic_status, EpistemicStatus::Provisional);
        assert_eq!(hypothesis.trace.method, SWARM_CONSENSUS_METHOD_ID);
        assert_eq!(hypothesis.trace.source_entities, vec![101, 102]);
        assert_eq!(hypothesis.bindings.len(), 2);
    }
}
