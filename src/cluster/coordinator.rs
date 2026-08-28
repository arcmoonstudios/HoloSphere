/* holosphere/src/cluster/coordinator.rs */
//!▫~•◦-------------------------------‣
//! # Distributed Coordinator & Scatter-Gather Engine
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::cluster::state_machine::{DataMutation, ShardStateMachine};
use crate::consensus::raft::RaftCluster;
use crate::consensus::read_index::ReadConsistency;
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
    dr_coordinator:
        RwLock<Option<Arc<crate::cluster::disaster_recovery::DisasterRecoveryCoordinator>>>,
    federation_replicator: RwLock<Option<Arc<crate::cluster::federation::CrossRegionReplicator>>>,
}

impl DistributedCoordinator {
    /// Creates a distributed coordinator with an internally-managed Raft cluster.
    pub fn new(dimension: usize, num_shards: u32, max_mutable_capacity: usize) -> Self {
        let node_ids: Vec<u64> = (1..=3).collect();
        let raft_cluster = Arc::new(RaftCluster::new(&node_ids));
        let _ = raft_cluster.trigger_election(1);
        Self::new_with_cluster(dimension, num_shards, max_mutable_capacity, raft_cluster)
    }

    /// Creates a distributed coordinator sharing the provided external `RaftCluster`.
    /// Use this when a single `RaftCluster` instance must be shared across the coordinator
    /// and other test or production components that hold a separate reference to it.
    pub fn new_with_cluster(
        dimension: usize,
        num_shards: u32,
        max_mutable_capacity: usize,
        raft_cluster: Arc<RaftCluster>,
    ) -> Self {
        let topology = ClusterTopology::new(num_shards);
        let mut local_shards = HashMap::new();

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
            dr_coordinator: RwLock::new(None),
            federation_replicator: RwLock::new(None),
        }
    }

    pub fn set_dr_coordinator(
        &self,
        dr: Arc<crate::cluster::disaster_recovery::DisasterRecoveryCoordinator>,
    ) {
        *self.dr_coordinator.write() = Some(dr);
    }

    pub fn set_federation_replicator(
        &self,
        rep: Arc<crate::cluster::federation::CrossRegionReplicator>,
    ) {
        *self.federation_replicator.write() = Some(rep);
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

    /// Returns a snapshot of all local shards (primarily for testing / diagnostics).
    #[must_use]
    pub fn local_shards_snapshot(&self) -> Vec<Arc<LocalShard>> {
        self.local_shards.read().values().cloned().collect()
    }

    /// Inserts a vector by replicating through Raft quorum consensus with epoch fencing.
    /// Asynchronously awaits quorum commit and state machine application without busy-spinning.
    pub async fn insert_fenced(
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

        // Validate role and build mutation while holding the lock, then drop it
        // before any await point so the future remains Send.
        let rx = {
            let shards = self.local_shards.read();
            let shard = shards.get(&shard_id).ok_or_else(|| {
                HNSQRError::SearchError(format!("Shard {shard_id} not hosted locally"))
            })?;
            if shard.role != ShardRole::Leader {
                return Err(HNSQRError::Internal(format!(
                    "Cannot write to follower replica for shard {shard_id}"
                )));
            }
            let mutation = DataMutation::new_upsert(node_id.to_string(), vector);
            self.raft_cluster.propose_data_mutation(mutation)?
            // `shards` guard drops here
        };

        // 2. Asynchronously await verified commit and state application receipt with timeout
        let receipt_res = tokio::time::timeout(std::time::Duration::from_secs(5), rx)
            .await
            .map_err(|_| {
                HNSQRError::Internal("Proposal timed out waiting for quorum commit".to_string())
            })?
            .map_err(|_| {
                HNSQRError::Internal("Proposal channel closed unexpectedly".to_string())
            })?;

        let receipt = receipt_res.map_err(HNSQRError::from)?;
        if let Some(ref dr) = *self.dr_coordinator.read() {
            dr.record_primary_mutation(receipt.applied_index);
        }
        Ok(receipt)
    }

    /// Standard unfenced async insert helper.
    pub async fn insert(
        &self,
        id: impl Into<NodeId>,
        vector: VectorEmbedding,
    ) -> HNSQRResult<crate::consensus::pending::CommitReceipt> {
        self.insert_fenced(id, vector, None).await
    }

    /// Deletes a vector across the cluster through Raft state machine replication asynchronously.
    pub async fn delete(&self, id: &str) -> HNSQRResult<crate::consensus::pending::CommitReceipt> {
        let shard_id = { self.topology.read().shard_for_key(id) };

        // Validate role and build mutation while holding the lock, then drop it
        // before any await point so the future remains Send.
        let rx = {
            let shards = self.local_shards.read();
            let shard = shards.get(&shard_id).ok_or_else(|| {
                HNSQRError::SearchError(format!("Shard {shard_id} not hosted locally"))
            })?;
            if shard.role != ShardRole::Leader {
                return Err(HNSQRError::Internal(format!(
                    "Cannot delete on follower replica for shard {shard_id}"
                )));
            }
            let mutation = DataMutation::new_delete(id);
            self.raft_cluster.propose_data_mutation(mutation)?
            // `shards` guard drops here
        };

        let receipt_res = tokio::time::timeout(std::time::Duration::from_secs(5), rx)
            .await
            .map_err(|_| {
                HNSQRError::Internal("Proposal timed out waiting for quorum commit".to_string())
            })?
            .map_err(|_| {
                HNSQRError::Internal("Proposal channel closed unexpectedly".to_string())
            })?;

        receipt_res.map_err(HNSQRError::from)
    }

    /// Synchronous blocking insert helper for non-async clients and deterministic test harnesses.
    pub fn insert_fenced_blocking(
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

            let mutation = DataMutation::new_upsert(node_id.to_string(), vector);
            let rx = self.raft_cluster.propose_data_mutation(mutation)?;
            let receipt_res = rx
                .blocking_recv()
                .map_err(|_| HNSQRError::Internal("Proposal channel closed".to_string()))?;
            receipt_res.map_err(HNSQRError::from)
        } else {
            Err(HNSQRError::SearchError(format!(
                "Shard {shard_id} not hosted locally"
            )))
        }
    }

    /// Synchronous blocking delete helper for non-async clients and deterministic test harnesses.
    pub fn delete_blocking(
        &self,
        id: &str,
    ) -> HNSQRResult<crate::consensus::pending::CommitReceipt> {
        let shard_id = { self.topology.read().shard_for_key(id) };
        let shards = self.local_shards.read();
        if let Some(shard) = shards.get(&shard_id) {
            if shard.role != ShardRole::Leader {
                return Err(HNSQRError::Internal(format!(
                    "Cannot delete on follower replica for shard {shard_id}"
                )));
            }

            let mutation = DataMutation::new_delete(id);
            let rx = self.raft_cluster.propose_data_mutation(mutation)?;
            let receipt_res = rx
                .blocking_recv()
                .map_err(|_| HNSQRError::Internal("Proposal channel closed".to_string()))?;
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
            ReadConsistency::LinearizableWithMode(mode) => {
                self.raft_cluster.linearizable_read_index_with_mode(mode)?
            }
            ReadConsistency::Committed => {
                let leader_id = self.raft_cluster.get_leader().unwrap_or(1);
                let leader = self.raft_cluster.nodes.get(&leader_id).unwrap();
                *leader.commit_index.read()
            }
            ReadConsistency::BoundedStaleness {
                max_lag_entries, ..
            } => {
                let leader_id = self.raft_cluster.get_leader().unwrap_or(1);
                let leader = self.raft_cluster.nodes.get(&leader_id).unwrap();
                let commit = *leader.commit_index.read();
                let applied = *leader.last_applied.read();
                if commit.saturating_sub(applied) > max_lag_entries {
                    return Err(HNSQRError::Internal(format!(
                        "Observed lag {} exceeds bounded staleness limit {}",
                        commit.saturating_sub(applied),
                        max_lag_entries
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

        let active_gen = self
            .local_shards
            .read()
            .values()
            .next()
            .map(|s| s.engine.active_generation())
            .unwrap_or(1);

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
            all_shard_snapshots: HashMap::new(),
        })
    }

    /// Obtains a cluster-wide `PinnedReadSnapshot` that retains immutable and active segment
    /// references across **all** hosted local shards, preventing compaction from reclaiming
    /// segments on any shard while a scatter-gather search is in flight.
    pub fn obtain_cluster_pinned_snapshot(
        &self,
        consistency: ReadConsistency,
    ) -> HNSQRResult<crate::service::PinnedReadSnapshot> {
        let snapshot = self.obtain_read_snapshot(consistency)?;
        let shards = self.local_shards.read();

        let mut all_shard_snapshots = HashMap::with_capacity(shards.len());
        // Defaults for the legacy single-shard fields — filled from the first shard.
        let mut first_immutables: Arc<[Arc<crate::storage::segment::ImmutableSegment>]> =
            Arc::from(Vec::new());
        let mut first_active: Option<Arc<crate::storage::segment::MutableSegment>> = None;

        // Iterate in deterministic shard-ID order so the "first" shard is always shard 0.
        let mut ordered: Vec<ShardId> = shards.keys().copied().collect();
        ordered.sort_unstable();

        for s_id in ordered {
            let shard = &shards[&s_id];
            let immutables: Arc<[Arc<crate::storage::segment::ImmutableSegment>]> =
                Arc::from(shard.engine.immutable_segments_snapshot());
            let active = shard.engine.active_mutable_segment();

            if first_active.is_none() {
                first_immutables = immutables.clone();
                first_active = Some(active.clone());
            }

            all_shard_snapshots.insert(s_id, (immutables, active));
        }

        let active_segment = first_active.unwrap_or_else(|| {
            Arc::new(crate::storage::segment::MutableSegment::new(
                0,
                self.dimension,
                100,
            ))
        });

        Ok(crate::service::PinnedReadSnapshot {
            topology_epoch: snapshot.topology_epoch,
            raft_read_index: snapshot.raft_read_index,
            applied_index: snapshot.applied_index,
            mutable_lsn: snapshot.mutable_lsn,
            immutable_segments: first_immutables,
            active_segment,
            all_shard_snapshots,
        })
    }

    /// Pinned scatter-gather search across all hosted local shards.
    pub fn search_pinned(
        &self,
        snapshot: &crate::service::PinnedReadSnapshot,
        query: &VectorEmbedding,
        k: usize,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if k == 0 {
            return Vec::new();
        }

        let shards: Vec<Arc<LocalShard>> = {
            let guard = self.local_shards.read();
            guard.values().cloned().collect()
        };

        use rayon::prelude::*;
        let mut global_candidates: Vec<(Arc<str>, SimilarityScore)> = shards
            .par_iter()
            .flat_map(|shard| {
                if let Some((immutables, active)) =
                    snapshot.all_shard_snapshots.get(&shard.shard_id)
                {
                    shard
                        .engine
                        .search_pinned(immutables, active, query, k)
                } else {
                    shard.engine.search_pinned(
                        &snapshot.immutable_segments,
                        &snapshot.active_segment,
                        query,
                        k,
                    )
                }
            })
            .collect();

        global_candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        global_candidates.truncate(k);
        global_candidates
    }

    /// Scatter-gather query across all shards with global Top-K merging.
    pub fn search(
        &self,
        query: &VectorEmbedding,
        k: usize,
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
            let shard_topk = shard.engine.search(query, k);
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
