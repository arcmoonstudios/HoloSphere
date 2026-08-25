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
