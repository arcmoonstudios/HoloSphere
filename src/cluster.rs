/* hnsqr/src/cluster.rs */
//!▫~•◦-------------------------------‣
//! # Distributed Cluster Control Plane & Partitioned Sharding Architecture
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides distributed partition routing, per-shard replication, scatter-gather query execution,
//! and global Top-K finalist merging.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::lutz::SemanticRerankPlan;
use crate::segment::SegmentedEngine;
use crate::{HNSQRError, HNSQRResult, NodeId, SimilarityScore, VectorEmbedding};

pub type ShardId = u32;
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
    pub shard_replicas: HashMap<ShardId, Vec<ShardReplica>>,
}

impl ClusterTopology {
    pub fn new(num_shards: u32) -> Self {
        let mut shard_replicas = HashMap::new();
        for s in 0..num_shards {
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
            shard_replicas,
        }
    }

    /// Maps a node ID or tenant hash to its target shard ID.
    #[inline(always)]
    pub fn shard_for_key(&self, key: &str) -> ShardId {
        let mut hash = 5381u64;
        for byte in key.bytes() {
            hash = ((hash << 5).wrapping_add(hash)).wrapping_add(byte as u64);
        }
        (hash % (self.num_shards as u64)) as ShardId
    }
}

/// Shard container hosting local storage engines.
pub struct LocalShard {
    pub shard_id: ShardId,
    pub engine: SegmentedEngine,
}

/// Distributed Coordinator managing shard routing, parallel scatter-gather, and Top-K merging.
pub struct DistributedCoordinator {
    pub dimension: usize,
    topology: RwLock<ClusterTopology>,
    local_shards: RwLock<HashMap<ShardId, Arc<LocalShard>>>,
    epoch: AtomicU64,
}

impl DistributedCoordinator {
    /// Creates a distributed coordinator.
    pub fn new(dimension: usize, num_shards: u32, max_mutable_capacity: usize) -> Self {
        let topology = ClusterTopology::new(num_shards);
        let mut local_shards = HashMap::new();

        for s in 0..num_shards {
            local_shards.insert(
                s,
                Arc::new(LocalShard {
                    shard_id: s,
                    engine: SegmentedEngine::new(dimension, max_mutable_capacity),
                }),
            );
        }

        Self {
            dimension,
            topology: RwLock::new(topology),
            local_shards: RwLock::new(local_shards),
            epoch: AtomicU64::new(1),
        }
    }

    /// Returns the current cluster topology epoch.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Inserts a vector by routing it to its assigned partition shard.
    pub fn insert(&self, id: impl Into<NodeId>, vector: VectorEmbedding) -> HNSQRResult<ShardId> {
        let node_id: Arc<str> = id.into();
        let shard_id = { self.topology.read().shard_for_key(&node_id) };

        let shards = self.local_shards.read();
        if let Some(shard) = shards.get(&shard_id) {
            shard.engine.insert(node_id, vector)?;
            Ok(shard_id)
        } else {
            Err(HNSQRError::SearchError(format!(
                "Shard {shard_id} not hosted locally"
            )))
        }
    }

    /// Deletes a vector across the cluster.
    pub fn delete(&self, id: &str) -> bool {
        let shard_id = { self.topology.read().shard_for_key(id) };
        let shards = self.local_shards.read();
        if let Some(shard) = shards.get(&shard_id) {
            shard.engine.delete(id)
        } else {
            false
        }
    }

    /// Scatter-gather query across all shards with global Top-K merging.
    pub fn search(
        &self,
        query: &VectorEmbedding,
        k: usize,
        rerank_plan: SemanticRerankPlan,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if k == 0 {
            return Vec::new();
        }

        let shards: Vec<Arc<LocalShard>> = {
            let guard = self.local_shards.read();
            guard.values().cloned().collect()
        };

        // Scatter search across shards
        let mut global_candidates = Vec::with_capacity(shards.len() * k);
        for shard in shards {
            let shard_topk = shard.engine.search(query, k, rerank_plan);
            global_candidates.extend(shard_topk);
        }

        // Global merge and deduplication
        global_candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut unique = Vec::with_capacity(k);
        let mut seen = std::collections::HashSet::with_capacity(k * 2);

        for (id, score) in global_candidates {
            if seen.insert(id.clone()) {
                unique.push((id, score));
                if unique.len() >= k {
                    break;
                }
            }
        }

        unique
    }

    /// Triggers online compaction across all shards.
    pub fn compact_all(&self) -> HNSQRResult<usize> {
        let shards: Vec<Arc<LocalShard>> = {
            let guard = self.local_shards.read();
            guard.values().cloned().collect()
        };

        let mut total_purged = 0usize;
        for shard in shards {
            total_purged += shard.engine.compact()?;
        }
        Ok(total_purged)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    #[test]
    fn test_distributed_sharding_and_scatter_gather() {
        let dim = 8;
        let num_shards = 4;
        let coordinator = DistributedCoordinator::new(dim, num_shards, 10);

        for i in 0..40 {
            let v = VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 7 + d) as f32, (i * 3 + d) as f32))
                    .collect(),
            )
            .into_normalized();
            coordinator.insert(format!("user_doc_{i}"), v).unwrap();
        }

        let query = VectorEmbedding::from_complex(
            (0..dim)
                .map(|d| Complex32::new((12 * 7 + d) as f32, (12 * 3 + d) as f32))
                .collect(),
        )
        .into_normalized();

        let topk = coordinator.search(&query, 5, SemanticRerankPlan::ExactSimd);
        assert_eq!(topk.len(), 5);
        assert_eq!(topk[0].0.as_ref(), "user_doc_12");

        // Delete user_doc_12
        assert!(coordinator.delete("user_doc_12"));
        let topk_after = coordinator.search(&query, 5, SemanticRerankPlan::ExactSimd);
        assert_ne!(topk_after[0].0.as_ref(), "user_doc_12");
    }
}
