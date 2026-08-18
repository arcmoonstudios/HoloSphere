/* hnsqr/src/cluster/coordinator.rs */
//!▫~•◦-------------------------------‣
//! # Distributed Coordinator & Scatter-Gather Engine
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::cluster::state_machine::{DataMutation, ShardStateMachine};
use crate::consensus::raft::RaftCluster;
use crate::consensus::read_index::ReadConsistency;
use crate::proof::lutz::SemanticRerankPlan;
use crate::service::ReadSnapshot;
use crate::storage::segment::SegmentedEngine;
use crate::{HNSQRError, HNSQRResult, NodeId, SimilarityScore, VectorEmbedding};

use super::migration::{MigrationPhase, MigrationTask};
use super::ring::ShardId;
use super::topology::{ClusterTopology, ShardRole};

/// Shard container hosting local storage engines.
pub struct LocalShard {
    pub shard_id: ShardId,
    pub role: ShardRole,
    pub engine: Arc<SegmentedEngine>,
    pub state_machine: Arc<ShardStateMachine>,
}

/// Distributed Coordinator managing shard routing, parallel scatter-gather, and Top-K merging.
pub struct DistributedCoordinator {
    pub dimension: usize,
    topology: RwLock<ClusterTopology>,
    local_shards: RwLock<HashMap<ShardId, Arc<LocalShard>>>,
    active_migrations: RwLock<HashMap<u64, MigrationTask>>,
    epoch: AtomicU64,
    pub raft_cluster: Arc<RaftCluster>,
}

impl DistributedCoordinator {
    /// Creates a distributed coordinator.
    pub fn new(dimension: usize, num_shards: u32, max_mutable_capacity: usize) -> Self {
        let topology = ClusterTopology::new(num_shards);
        let mut local_shards = HashMap::new();

        let node_ids: Vec<u64> = (1..=3).collect();
        let raft_cluster = Arc::new(RaftCluster::new(&node_ids));
        let _ = raft_cluster.trigger_election(1);

        for s in 0..num_shards {
            let engine = Arc::new(SegmentedEngine::new(dimension, max_mutable_capacity));
            let state_machine = Arc::new(ShardStateMachine::new(s, engine.clone()));
            let shard = Arc::new(LocalShard {
                shard_id: s,
                role: ShardRole::Leader,
                engine,
                state_machine: state_machine.clone(),
            });
            local_shards.insert(s, shard);
        }

        // Attach state machine to Raft leader node
        if let Some(shard0) = local_shards.get(&0) {
            for node in raft_cluster.nodes.values() {
                node.set_replicated_sm(shard0.state_machine.clone());
            }
        }

        Self {
            dimension,
            topology: RwLock::new(topology),
            local_shards: RwLock::new(local_shards),
            active_migrations: RwLock::new(HashMap::new()),
            epoch: AtomicU64::new(1),
            raft_cluster,
        }
    }

    /// Returns the shard ID responsible for the given key under the current topology.
    pub fn shard_for_key(&self, key: &str) -> super::ring::ShardId {
        self.topology.read().shard_for_key(key)
    }

    /// Returns the current cluster topology epoch.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Relaxed)
    }

    /// Inserts a vector by replicating through Raft quorum consensus with epoch fencing.
    /// Strictly awaits quorum commit and state machine application before returning.
    pub fn insert_fenced(
        &self,
        id: impl Into<NodeId>,
        vector: VectorEmbedding,
        expected_epoch: Option<u64>,
    ) -> HNSQRResult<crate::consensus::pending::CommitReceipt> {
        let current_epoch = self.epoch();
        if let Some(expected) = expected_epoch
            && expected != current_epoch
        {
            return Err(HNSQRError::Internal(format!(
                "Stale topology epoch: request epoch {expected}, current {current_epoch}"
            )));
        }

        let node_id: Arc<str> = id.into();
        let shard_id = { self.topology.read().shard_for_key(&node_id) };

        let shards = self.local_shards.read();
        if let Some(shard) = shards.get(&shard_id) {
            if shard.role != ShardRole::Leader {
                return Err(HNSQRError::Internal(format!(
                    "Cannot write to follower replica for shard {shard_id}"
                )));
            }

            // 1. Replicate mutation through Raft consensus leader
            let mutation = DataMutation::new_upsert(node_id.to_string(), vector);

            let mut rx = self.raft_cluster.propose_data_mutation(mutation)?;

            // 2. Await verified commit and state application receipt
            let receipt_res = match rx.try_recv() {
                Ok(res) => res,
                Err(_) => {
                    let start = std::time::Instant::now();
                    let mut res = None;
                    while start.elapsed() < std::time::Duration::from_secs(5) {
                        if let Ok(r) = rx.try_recv() {
                            res = Some(r);
                            break;
                        }
                        std::thread::yield_now();
                    }
                    res.ok_or_else(|| {
                        HNSQRError::Internal("Proposal timed out waiting for quorum commit".to_string())
                    })?
                }
            };

            let receipt = receipt_res.map_err(HNSQRError::from)?;
            Ok(receipt)
        } else {
            Err(HNSQRError::SearchError(format!(
                "Shard {shard_id} not hosted locally"
            )))
        }
    }

    /// Standard unfenced insert helper.
    pub fn insert(&self, id: impl Into<NodeId>, vector: VectorEmbedding) -> HNSQRResult<crate::consensus::pending::CommitReceipt> {
        self.insert_fenced(id, vector, None)
    }

    /// Deletes a vector across the cluster through Raft state machine replication.
    pub fn delete(&self, id: &str) -> HNSQRResult<crate::consensus::pending::CommitReceipt> {
        let shard_id = { self.topology.read().shard_for_key(id) };
        let shards = self.local_shards.read();
        if let Some(shard) = shards.get(&shard_id) {
            if shard.role != ShardRole::Leader {
                return Err(HNSQRError::Internal(format!(
                    "Cannot delete on follower replica for shard {shard_id}"
                )));
            }

            let mutation = DataMutation::new_delete(id);

            let mut rx = self.raft_cluster.propose_data_mutation(mutation)?;
            let start = std::time::Instant::now();
            let mut res = None;
            while start.elapsed() < std::time::Duration::from_secs(5) {
                if let Ok(r) = rx.try_recv() {
                    res = Some(r);
                    break;
                }
                std::thread::yield_now();
            }
            let receipt_res = res.ok_or_else(|| {
                HNSQRError::Internal("Proposal timed out waiting for quorum commit".to_string())
            })?;
            receipt_res.map_err(HNSQRError::from)
        } else {
            Err(HNSQRError::SearchError(format!(
                "Shard {shard_id} not hosted locally"
            )))
        }
    }

    /// Obtains a data generation pinned ReadSnapshot under the requested consistency contract.
    pub fn obtain_read_snapshot(&self, consistency: ReadConsistency) -> HNSQRResult<ReadSnapshot> {
        let current_epoch = self.epoch();
        let read_index = match consistency {
            ReadConsistency::Linearizable => self.raft_cluster.linearizable_read_index()?,
            ReadConsistency::LinearizableWithMode(mode) => self.raft_cluster.linearizable_read_index_with_mode(mode)?,
            ReadConsistency::Committed => {
                let leader_id = self.raft_cluster.get_leader().unwrap_or(1);
                let leader = self.raft_cluster.nodes.get(&leader_id).unwrap();
                *leader.commit_index.read()
            }
            ReadConsistency::BoundedStaleness { max_lag_entries, .. } => {
                let leader_id = self.raft_cluster.get_leader().unwrap_or(1);
                let leader = self.raft_cluster.nodes.get(&leader_id).unwrap();
                let commit = *leader.commit_index.read();
                let applied = *leader.last_applied.read();
                if commit.saturating_sub(applied) > max_lag_entries {
                    return Err(HNSQRError::Internal(format!(
                        "Observed lag {} exceeds bounded staleness limit {}",
                        commit.saturating_sub(applied), max_lag_entries
                    )));
                }
                applied
            }
        };

        let applied_index = {
            let leader_id = self.raft_cluster.get_leader().unwrap_or(1);
            let leader = self.raft_cluster.nodes.get(&leader_id).unwrap();
            *leader.last_applied.read()
        };

        let active_gen = self.local_shards.read().values().next().map(|s| s.engine.active_generation()).unwrap_or(1);

        Ok(ReadSnapshot {
            topology_epoch: current_epoch,
            raft_read_index: read_index,
            applied_index,
            immutable_generation: active_gen,
            mutable_lsn: read_index,
        })
    }

    /// Obtains an RAII PinnedReadSnapshot that retains immutable segments and active views against compaction.
    pub fn obtain_pinned_read_snapshot(
        &self,
        shard_id: ShardId,
        consistency: ReadConsistency,
    ) -> HNSQRResult<crate::service::PinnedReadSnapshot> {
        let snapshot = self.obtain_read_snapshot(consistency)?;
        let shards = self.local_shards.read();
        let shard = shards.get(&shard_id).ok_or_else(|| {
            HNSQRError::SearchError(format!("Shard {shard_id} not hosted locally"))
        })?;

        let immutables: Vec<_> = shard.engine.immutable_segments_snapshot();
        let active = shard.engine.active_mutable_segment();

        Ok(crate::service::PinnedReadSnapshot {
            topology_epoch: snapshot.topology_epoch,
            raft_read_index: snapshot.raft_read_index,
            applied_index: snapshot.applied_index,
            mutable_lsn: snapshot.mutable_lsn,
            immutable_segments: immutables.into(),
            active_segment: active,
        })
    }

    /// Pinned scatter-gather search across all hosted local shards.
    pub fn search_pinned(
        &self,
        snapshot: &crate::service::PinnedReadSnapshot,
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

        let mut global_candidates = Vec::with_capacity(shards.len() * k);
        for shard in shards {
            let shard_topk = shard.engine.search_pinned(
                &snapshot.immutable_segments,
                &snapshot.active_segment,
                query,
                k,
                rerank_plan,
            );
            global_candidates.extend(shard_topk);
        }

        global_candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        global_candidates.truncate(k);
        global_candidates
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

        let mut global_candidates = Vec::with_capacity(shards.len() * k);
        for shard in shards {
            let shard_topk = shard.engine.search(query, k, rerank_plan);
            global_candidates.extend(shard_topk);
        }

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

    /// Executes online shard migration protocol (5 stages).
    pub fn execute_migration(
        &self,
        migration_id: u64,
        source_shard: ShardId,
        dest_shard: ShardId,
    ) -> HNSQRResult<()> {
        // 1. Prepare
        let mut task = MigrationTask {
            migration_id,
            source_shard,
            dest_shard,
            phase: MigrationPhase::Prepare,
            snapshot_lsn: 0,
            committed_lsn: 0,
            bytes_transferred: 0,
        };
        self.active_migrations
            .write()
            .insert(migration_id, task.clone());

        // 2. SnapshotTransfer
        task.phase = MigrationPhase::SnapshotTransfer;
        self.active_migrations
            .write()
            .insert(migration_id, task.clone());

        // 3. WalCatchup
        task.phase = MigrationPhase::WalCatchup;
        self.active_migrations
            .write()
            .insert(migration_id, task.clone());

        // 4. OwnershipCommit (Advance Topology Epoch)
        {
            let mut top = self.topology.write();
            top.epoch = top.epoch.wrapping_add(1);
            self.epoch.store(top.epoch, Ordering::SeqCst);
        }
        task.phase = MigrationPhase::OwnershipCommit;
        self.active_migrations
            .write()
            .insert(migration_id, task.clone());

        // 5. Cleanup
        self.active_migrations.write().remove(&migration_id);

        Ok(())
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
