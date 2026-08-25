/* holosphere/src/cluster/topology.rs */
//!▫~•◦-------------------------------‣
//! # Cluster Topology & Partition Mapping
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::ring::{ConsistentHashRing, ShardId};

pub type NodeAddress = String;

/// Health and replication status of a shard.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShardRole {
    Leader,
    Follower,
}

/// Shard replica descriptor.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ShardReplica {
    pub shard_id: ShardId,
    pub node_addr: NodeAddress,
    pub role: ShardRole,
}

/// Cluster topology mapping partitions and replicas across nodes.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ClusterTopology {
    pub epoch: u64,
    pub num_shards: u32,
    pub ring: ConsistentHashRing,
    pub shard_replicas: HashMap<ShardId, Vec<ShardReplica>>,
}

impl ClusterTopology {
    pub fn new(num_shards: u32) -> Self {
        let mut ring = ConsistentHashRing::new(ConsistentHashRing::DEFAULT_VNODES_PER_SHARD);
        let mut shard_replicas = HashMap::new();
        for s in 0..num_shards {
            ring.add_shard(s);
            shard_replicas.insert(
                s,
                vec![ShardReplica {
                    shard_id: s,
                    node_addr: "127.0.0.1:8080".to_string(),
                    role: ShardRole::Leader,
                }],
            );
        }
        Self {
            epoch: 1,
            num_shards,
            ring,
            shard_replicas,
        }
    }

    /// Maps a node ID or tenant key to its target shard ID via consistent hashing.
    #[inline(always)]
    pub fn shard_for_key(&self, key: &str) -> ShardId {
        self.ring.shard_for_key(key).unwrap_or(0)
    }

    /// Adds a new shard dynamically to the topology, remapping only ~1/N of keys.
    pub fn add_shard(&mut self, shard_id: ShardId, replicas: Vec<ShardReplica>) {
        self.ring.add_shard(shard_id);
        self.shard_replicas.insert(shard_id, replicas);
        self.num_shards = self.shard_replicas.len() as u32;
        self.epoch = self.epoch.wrapping_add(1);
    }
}

use super::world_digest::WorldStateDigest;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Node liveness state in cluster topology.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeLiveness {
    Healthy,
    Degraded,
    Dead,
}

/// Cluster heartbeat frame carrying liveness and world state digest.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterHeartbeat {
    pub node_id: String,
    pub term: u64,
    pub world_digest: Option<WorldStateDigest>,
}

/// Topology manager coordinating cluster membership and anti-entropy reconciliation.
pub struct TopologyManager {
    pub topology: parking_lot::RwLock<ClusterTopology>,
    pub local_digest: parking_lot::RwLock<Option<WorldStateDigest>>,
    pub anti_entropy_triggers: AtomicUsize,
}

impl TopologyManager {
    pub fn new(num_shards: u32) -> Self {
        Self {
            topology: parking_lot::RwLock::new(ClusterTopology::new(num_shards)),
            local_digest: parking_lot::RwLock::new(None),
            anti_entropy_triggers: AtomicUsize::new(0),
        }
    }

    pub fn update_local_digest(&self, digest: WorldStateDigest) {
        *self.local_digest.write() = Some(digest);
    }

    /// Handles an incoming heartbeat and triggers anti-entropy if digests diverge.
    pub fn handle_heartbeat(&self, heartbeat: ClusterHeartbeat) -> bool {
        if let Some(remote_digest) = &heartbeat.world_digest {
            let local_opt = *self.local_digest.read();
            if let Some(local_digest) = local_opt {
                if local_digest.combined_digest != remote_digest.combined_digest {
                    self.trigger_anti_entropy_reconciliation(&heartbeat.node_id, remote_digest);
                    return false;
                }
            }
        }
        true
    }

    pub fn trigger_anti_entropy_reconciliation(
        &self,
        _node_id: &str,
        _remote_digest: &WorldStateDigest,
    ) {
        self.anti_entropy_triggers.fetch_add(1, Ordering::Relaxed);
    }
}
