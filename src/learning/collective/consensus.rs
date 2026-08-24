/* holosphere/src/learning/collective/consensus.rs */
//!▫~•◦-------------------------------‣
//! # Swarm Belief Arbitration & Consensus Synthesis Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Direct reference-equivalent port of Rune-EVO's `hive/consensus.rs`.
//! Arbitrates between proximate agent beliefs via decayed provenance-weighted centroids
//! while explicitly retaining unresolved inter-agent conflicts.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::entity::id::EntityId;
use crate::learning::collective::belief::{AgentBelief, AgentId};
use crate::learning::collective::conflict::ConflictPair;
use crate::learning::inference::candidate::{
    InferenceCandidate, InferenceCandidateId, InferenceScore,
};
use crate::learning::inference::contract::{InferenceMethodId, InferenceSeed};
use crate::learning::inference::rune_evo::analogy::{euclidean_dist_8, normalize_vector_8};
use crate::learning::inference::trace::InferenceTrace;

pub const SWARM_CONSENSUS_METHOD_ID: InferenceMethodId = InferenceMethodId(201);
pub const SWARM_CONSENSUS_METHOD_VERSION: u32 = 1;

/// Summary result of a multi-agent belief consensus scan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConsensusResult {
    pub region: [f32; 8],
    pub radius: f32,
    pub participating_beliefs: Vec<EntityId>,
    pub contributing_agents: Vec<AgentId>,
    pub centroid: [f32; 8],
    pub merged_confidence: f32,
    pub conflict_pairs: Vec<ConflictPair>,
}

/// Scans agent beliefs within `radius` of `region` and arbitrates a weighted consensus centroid.
pub fn compute_swarm_consensus(
    beliefs: &[AgentBelief],
    region: &[f32; 8],
    radius: f32,
    conflict_epsilon: f32,
    current_time_ms: u64,
) -> ConsensusResult {
    let mut participating_beliefs = Vec::new();
    let mut contributing_agents = Vec::new();
    let mut in_region_entries = Vec::new();
    let mut acc = [0.0f32; 8];
    let mut weight_sum = 0.0f32;
    let mut total_conf = 0.0f32;
    let mut conf_count = 0u32;

    for b in beliefs {
        let dist = euclidean_dist_8(&b.coords, region);
        if dist > radius {
            continue;
        }

        let eff_conf = b.effective_confidence(current_time_ms);
        for i in 0..8 {
            acc[i] += b.coords[i] * eff_conf;
        }
        weight_sum += eff_conf;
        total_conf += eff_conf;
        conf_count += 1;

        if !contributing_agents.contains(&b.author_agent) {
            contributing_agents.push(b.author_agent);
        }
        participating_beliefs.push(b.belief_id);
        in_region_entries.push((b.belief_id, b.author_agent, b.coords, eff_conf));
    }

    if participating_beliefs.is_empty() {
        return ConsensusResult {
            region: *region,
            radius,
            participating_beliefs: Vec::new(),
            contributing_agents: Vec::new(),
            centroid: *region,
            merged_confidence: 1.0,
            conflict_pairs: Vec::new(),
        };
    }

    let centroid = if weight_sum > 1e-9 {
        let raw: [f32; 8] = std::array::from_fn(|i| acc[i] / weight_sum);
        normalize_vector_8(&raw)
    } else {
        *region
    };

    let merged_confidence = if conf_count > 0 {
        total_conf / (conf_count as f32)
    } else {
        1.0
    };

    // Detect inter-agent belief conflict pairs within conflict_epsilon
    let mut conflict_pairs = Vec::new();
    let n = in_region_entries.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let dist = euclidean_dist_8(&in_region_entries[i].2, &in_region_entries[j].2);
            if dist > conflict_epsilon {
                continue;
            }
            if in_region_entries[i].1 != in_region_entries[j].1 {
                conflict_pairs.push(ConflictPair {
                    belief_a: in_region_entries[i].0,
                    author_a: in_region_entries[i].1,
                    belief_b: in_region_entries[j].0,
                    author_b: in_region_entries[j].1,
                    distance: dist,
                });
            }
        }
    }

    ConsensusResult {
        region: *region,
        radius,
        participating_beliefs,
        contributing_agents,
        centroid,
        merged_confidence,
        conflict_pairs,
    }
}

/// Materializes a consensus result into an epistemically Provisional collective hypothesis.
pub fn materialize_collective_hypothesis(
    result: &ConsensusResult,
    relation_type_id: u32,
    snapshot_lsn: u64,
    seed: InferenceSeed,
) -> Option<InferenceCandidate> {
    if result.participating_beliefs.is_empty() {
        return None;
    }

    let mut hasher = Sha256::new();
    hasher.update(&result.merged_confidence.to_le_bytes());
    for &c in result.centroid.iter() {
        hasher.update(&c.to_le_bytes());
    }
    for &b in &result.participating_beliefs {
        hasher.update(&b.to_le_bytes());
    }
    let digest = hasher.finalize();
    let mut param_digest = [0u8; 32];
    param_digest.copy_from_slice(&digest);

    let confidence_q32 =
        ((result.merged_confidence.clamp(0.0, 1.0) * (1u64 << 32) as f32) as i64).max(0);

    let trace = InferenceTrace {
        method: SWARM_CONSENSUS_METHOD_ID,
        method_version: SWARM_CONSENSUS_METHOD_VERSION,
        source_entities: result.participating_beliefs.clone(),
        source_relations: Vec::new(),
        source_attempts: Vec::new(),
        snapshot_lsn,
        seed,
        parameter_digest: param_digest,
    };

    let bindings: Vec<crate::learning::inference::candidate::CandidateRoleBinding> = result
        .participating_beliefs
        .iter()
        .map(
            |&belief_id| crate::learning::inference::candidate::CandidateRoleBinding {
                entity: crate::learning::inference::candidate::CandidateEntityRef::Existing(
                    belief_id,
                ),
                role_id: 1, // ParticipantBelief role
            },
        )
        .collect();

    Some(InferenceCandidate::new_provisional(
        InferenceCandidateId(1),
        relation_type_id,
        bindings,
        InferenceScore {
            confidence_q32,
            raw_floating: result.merged_confidence,
        },
        trace,
    ))
}
