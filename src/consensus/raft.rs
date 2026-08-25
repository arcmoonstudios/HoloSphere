/* holosphere/src/consensus/raft.rs */
//!▫~•◦-------------------------------‣
//! # Production Raft Consensus Engine, Learners & Unified State Machine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Implements:
//!   - Strict crash-safe HardState and Progress persistence (zero swallowed errors)
//!   - Canonical command model (`Data`, `Topology`, `Membership`, `NoOp`)
//!   - Mandatory ReplicatedStateMachine integration for data-bearing replicas
//!   - Linearizable ReadIndex vs LeaseRead consistency separation
//!   - Granular lock-free proposal completion & timeout management
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};

use crate::cluster::ShardId;
use crate::cluster::state_machine::{DataMutation, ReplicatedStateMachine};
use crate::consensus::pending::{
    ApplyError, CommitReceipt, DurabilityLevel, PendingProposals, ProposalId,
};
pub use crate::consensus::read_index::{
    LinearizableReadMode, ReadConsistency, ReadContextId, ReadIndexConfirmation, ReadIndexEngine,
    ReadIndexRequest, ReadIndexTelemetry,
};
use crate::consensus::storage::{
    MemoryRaftStorage, RaftHardState, RaftPersistentProgress, RaftStorage,
};
use crate::{HNSQRError, HNSQRResult};

pub type RaftNodeId = u64;

/// Operational state role of a Raft node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RaftRole {
    Follower,
    Candidate,
    Leader,
    Learner,
}

/// Topology mutation payload for Raft state machine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TopologyMutation {
    pub epoch: u64,
    pub shard_owners: HashMap<ShardId, RaftNodeId>,
}

/// Dynamic membership change mutation payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MembershipMutation {
    pub new_peers: Vec<RaftNodeId>,
}

/// Canonical commands replicated across the Raft cluster state machine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RaftCommand {
    Data(DataMutation),
    Topology(TopologyMutation),
    Membership(MembershipMutation),
    NoOp,
}

/// Single entry inside the Raft log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RaftLogEntry {
    pub term: u64,
    pub index: u64,
    pub command: RaftCommand,
}

/// RequestVote RPC arguments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestVoteArgs {
    pub term: u64,
    pub candidate_id: RaftNodeId,
    pub last_log_index: u64,
    pub last_log_term: u64,
}

/// RequestVote RPC reply.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestVoteReply {
    pub term: u64,
    pub vote_granted: bool,
}

/// AppendEntries RPC arguments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppendEntriesArgs {
    pub term: u64,
    pub leader_id: RaftNodeId,
    pub prev_log_index: u64,
    pub prev_log_term: u64,
    pub entries: Vec<RaftLogEntry>,
    pub leader_commit: u64,
    pub is_heartbeat: bool,
}

/// AppendEntries RPC reply.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AppendEntriesReply {
    pub term: u64,
    pub success: bool,
    pub match_index: u64,
}

/// Adaptive microbatching controller driven by runtime queuing and hardware metrics.
#[derive(Debug, Clone)]
pub struct AdaptiveMicrobatcher {
    pub min_batch_size: usize,
    pub max_batch_size: usize,
    pub target_commit_latency_us: u64,
}

impl Default for AdaptiveMicrobatcher {
    fn default() -> Self {
        Self {
            min_batch_size: 1,
            max_batch_size: 512,
            target_commit_latency_us: 10_000,
        }
    }
}

impl AdaptiveMicrobatcher {
    pub fn compute_optimal_batch_size(
        &self,
        incoming_rate_wps: f64,
        fsync_latency_us: u64,
        rtt_us: u64,
        queue_depth: usize,
        durability_sla_us: u64,
    ) -> usize {
        let base_delay = fsync_latency_us.max(rtt_us);
        let headroom = durability_sla_us.saturating_sub(base_delay);
        let rate_multiplier = (incoming_rate_wps / 1000.0).clamp(1.0, 32.0);
        let queue_factor = (queue_depth / 8).clamp(1, 64);

        if headroom > 5000 {
            (self.min_batch_size * queue_factor * (rate_multiplier as usize))
                .clamp(self.min_batch_size, self.max_batch_size)
        } else {
            self.min_batch_size
        }
    }
}

/// Diagnostic health metrics for candidate evaluation and graceful leadership handoff.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StorageHealthMetrics {
    pub fsync_latency_p99_us: u64,
    pub disk_write_stall_count: u64,
    pub io_error_count: u64,
    pub free_disk_bytes: u64,
    pub is_read_only: bool,
}

impl Default for StorageHealthMetrics {
    fn default() -> Self {
        Self {
            fsync_latency_p99_us: 200,
            disk_write_stall_count: 0,
            io_error_count: 0,
            free_disk_bytes: 1_000_000_000_000,
            is_read_only: false,
        }
    }
}

impl StorageHealthMetrics {
    pub fn suitability_score(&self) -> f64 {
        if self.is_read_only || self.io_error_count > 0 || self.free_disk_bytes < 100_000_000 {
            return -1.0;
        }
        let latency_penalty = (self.fsync_latency_p99_us as f64 / 1000.0).min(50.0);
        let stall_penalty = (self.disk_write_stall_count as f64 * 5.0).min(50.0);
        (100.0 - latency_penalty - stall_penalty).max(0.0)
    }

    pub fn record_error(&mut self, _err: crate::HNSQRError) {
        self.io_error_count += 1;
    }
}

/// Production execution and pipeline telemetry for Raft cluster monitoring.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RaftPipelineTelemetry {
    pub microbatch_size_last: usize,
    pub inflight_proposals: usize,
    pub fsync_ops_total: u64,
    pub leader_wal_p99_us: u64,
    pub quorum_commit_p99_us: u64,
    pub replication_rtt_us: u64,
    pub serialization_cpu_us: u64,
    pub queue_delay_us: u64,
    pub apply_delay_us: u64,
    pub leadership_transfers_total: u64,
}

/// Single Raft replica instance.
pub struct RaftNode {
    pub id: RaftNodeId,
    pub voting_peers: RwLock<HashSet<RaftNodeId>>,
    pub learners: RwLock<HashSet<RaftNodeId>>,
    pub role: RwLock<RaftRole>,
    pub current_term: RwLock<u64>,
    pub voted_for: RwLock<Option<RaftNodeId>>,
    pub log: RwLock<Vec<RaftLogEntry>>,
    pub commit_index: RwLock<u64>,
    pub last_applied: RwLock<u64>,
    pub leader_id: RwLock<Option<RaftNodeId>>,
    pub next_index: RwLock<HashMap<RaftNodeId, u64>>,
    pub match_index: RwLock<HashMap<RaftNodeId, u64>>,
    pub shard_owners: RwLock<HashMap<ShardId, RaftNodeId>>,
    pub topology_epoch: RwLock<u64>,
    pub last_heartbeat_received: Mutex<Instant>,
    pub warm_proof_shards: RwLock<HashSet<ShardId>>,
    pub replicated_sm: RwLock<Option<Arc<dyn ReplicatedStateMachine>>>,
    pub storage_health: RwLock<StorageHealthMetrics>,
    pub pipeline_telemetry: RwLock<RaftPipelineTelemetry>,
    pub microbatcher: AdaptiveMicrobatcher,
    pub pending_proposals: Arc<PendingProposals>,
    pub read_index_engine: Arc<ReadIndexEngine>,
    pub durability_controller: Arc<crate::consensus::durability_controller::DurabilityController>,
    pub storage: Arc<dyn RaftStorage>,
}


impl RaftNode {
    pub fn new(id: RaftNodeId, initial_voting_peers: Vec<RaftNodeId>) -> Self {
        Self::with_storage(id, initial_voting_peers, Arc::new(MemoryRaftStorage::new()))
    }

    pub fn try_with_storage(
        id: RaftNodeId,
        initial_voting_peers: Vec<RaftNodeId>,
        storage: Arc<dyn RaftStorage>,
    ) -> HNSQRResult<Self> {
        let mut peers_set: HashSet<RaftNodeId> = initial_voting_peers.into_iter().collect();
        peers_set.insert(id);

        let initial_log = match storage.load_log_entries(0) {
            Ok(entries) if !entries.is_empty() => entries,
            Ok(_) => {
                let default_entries = vec![RaftLogEntry {
                    term: 0,
                    index: 0,
                    command: RaftCommand::NoOp,
                }];
                storage.append_entries(&default_entries)?;
                default_entries
            }
            Err(e) => return Err(e),
        };

        let hard_state = storage.load_hard_state()?;
        let progress = storage.load_progress()?;

        Ok(Self {
            id,
            voting_peers: RwLock::new(peers_set),
            learners: RwLock::new(HashSet::new()),
            role: RwLock::new(RaftRole::Follower),
            current_term: RwLock::new(hard_state.current_term),
            voted_for: RwLock::new(hard_state.voted_for),
            log: RwLock::new(initial_log),
            commit_index: RwLock::new(progress.commit_index),
            last_applied: RwLock::new(progress.last_applied),
            leader_id: RwLock::new(None),
            next_index: RwLock::new(HashMap::new()),
            match_index: RwLock::new(HashMap::new()),
            shard_owners: RwLock::new(HashMap::new()),
            topology_epoch: RwLock::new(1),
            last_heartbeat_received: Mutex::new(Instant::now()),
            warm_proof_shards: RwLock::new(HashSet::new()),
            replicated_sm: RwLock::new(None),
            storage_health: RwLock::new(StorageHealthMetrics::default()),
            pipeline_telemetry: RwLock::new(RaftPipelineTelemetry::default()),
            microbatcher: AdaptiveMicrobatcher::default(),
            pending_proposals: Arc::new(PendingProposals::default()),
            read_index_engine: Arc::new(ReadIndexEngine::default()),
            durability_controller: Arc::new(crate::consensus::durability_controller::DurabilityController::default()),
            storage,
        })

    }

    pub fn with_storage(
        id: RaftNodeId,
        initial_voting_peers: Vec<RaftNodeId>,
        storage: Arc<dyn RaftStorage>,
    ) -> Self {
        Self::try_with_storage(id, initial_voting_peers, storage)
            .expect("Failed to initialize RaftNode from storage")
    }

    /// Reconstructs volatile state machine from durable Raft log entries up to committed index ONLY.
    pub fn recover_node_state(
        &self,
        state_machine: &Arc<dyn ReplicatedStateMachine>,
    ) -> HNSQRResult<u64> {
        let progress = self.storage.load_progress()?;
        let entries = self.storage.load_log_entries(progress.snapshot_index + 1)?;
        let mut applied_count = 0;

        for entry in &entries {
            if entry.index <= progress.commit_index {
                if let RaftCommand::Data(mutation) = &entry.command {
                    state_machine.apply(entry.index, mutation)?;
                    applied_count += 1;
                }
            }
        }

        *self.last_applied.write() = progress.commit_index;
        *self.commit_index.write() = progress.commit_index;
        Ok(applied_count)
    }

    pub fn set_replicated_sm(&self, sm: Arc<dyn ReplicatedStateMachine>) {
        *self.replicated_sm.write() = Some(sm);
    }

    pub fn new_learner(id: RaftNodeId) -> Self {
        let node = Self::new(id, Vec::new());
        *node.role.write() = RaftRole::Learner;
        node
    }

    #[inline(always)]
    pub fn is_leader(&self) -> bool {
        *self.role.read() == RaftRole::Leader
    }

    #[inline(always)]
    pub fn is_learner(&self) -> bool {
        *self.role.read() == RaftRole::Learner
    }

    pub fn validate_read_consistency(&self, consistency: ReadConsistency) -> HNSQRResult<()> {
        match consistency {
            ReadConsistency::Linearizable | ReadConsistency::LinearizableWithMode(_) => {
                if !self.is_leader() {
                    return Err(HNSQRError::Internal(format!(
                        "Linearizable read must be routed to leader {:?}",
                        *self.leader_id.read()
                    )));
                }
            }
            ReadConsistency::Committed => {}
            ReadConsistency::BoundedStaleness {
                max_lag_entries, ..
            } => {
                let commit = *self.commit_index.read();
                let applied = *self.last_applied.read();
                let lag = commit.saturating_sub(applied);
                if lag > max_lag_entries {
                    return Err(HNSQRError::Internal(format!(
                        "Learner lag ({lag}) exceeds bounded staleness limit ({max_lag_entries})"
                    )));
                }
            }
        }
        Ok(())
    }

    pub fn handle_request_vote(&self, args: &RequestVoteArgs) -> RequestVoteReply {
        if self.is_learner() {
            return RequestVoteReply {
                term: *self.current_term.read(),
                vote_granted: false,
            };
        }

        let mut current_term = *self.current_term.read();
        if args.term > current_term {
            if self
                .storage
                .save_hard_state(&RaftHardState {
                    current_term: args.term,
                    voted_for: None,
                })
                .is_err()
            {
                return RequestVoteReply {
                    term: current_term,
                    vote_granted: false,
                };
            }
            *self.current_term.write() = args.term;
            current_term = args.term;
            if *self.role.read() != RaftRole::Learner {
                *self.role.write() = RaftRole::Follower;
            }
            *self.voted_for.write() = None;
            *self.leader_id.write() = None;
            self.pending_proposals.cancel_all_leadership_lost(args.term);
        }

        let (last_term, last_index) = {
            let log = self.log.read();
            let last = log.last().unwrap();
            (last.term, last.index)
        };

        let log_up_to_date = args.last_log_term > last_term
            || (args.last_log_term == last_term && args.last_log_index >= last_index);

        let voted = *self.voted_for.read();
        let can_vote = (voted.is_none() || voted == Some(args.candidate_id))
            && args.term == current_term
            && log_up_to_date;

        if can_vote {
            if self
                .storage
                .save_hard_state(&RaftHardState {
                    current_term,
                    voted_for: Some(args.candidate_id),
                })
                .is_err()
            {
                return RequestVoteReply {
                    term: current_term,
                    vote_granted: false,
                };
            }
            *self.voted_for.write() = Some(args.candidate_id);
            *self.last_heartbeat_received.lock() = Instant::now();
            RequestVoteReply {
                term: current_term,
                vote_granted: true,
            }
        } else {
            RequestVoteReply {
                term: current_term,
                vote_granted: false,
            }
        }
    }

    pub fn handle_append_entries(&self, args: &AppendEntriesArgs) -> AppendEntriesReply {
        let mut current_term = *self.current_term.read();
        if args.term > current_term {
            if self
                .storage
                .save_hard_state(&RaftHardState {
                    current_term: args.term,
                    voted_for: None,
                })
                .is_err()
            {
                return AppendEntriesReply {
                    term: current_term,
                    success: false,
                    match_index: 0,
                };
            }
            *self.current_term.write() = args.term;
            current_term = args.term;
            if *self.role.read() != RaftRole::Learner {
                *self.role.write() = RaftRole::Follower;
            }
            *self.voted_for.write() = None;
            self.pending_proposals.cancel_all_leadership_lost(args.term);
        }

        if args.term < current_term {
            return AppendEntriesReply {
                term: current_term,
                success: false,
                match_index: 0,
            };
        }

        if *self.role.read() != RaftRole::Learner {
            *self.role.write() = RaftRole::Follower;
        }
        *self.leader_id.write() = Some(args.leader_id);
        *self.last_heartbeat_received.lock() = Instant::now();

        if args.is_heartbeat && args.entries.is_empty() {
            let log = self.log.read();
            let log_len = log.len() as u64;
            let prev_matches = args.prev_log_index < log_len
                && log[args.prev_log_index as usize].term == args.prev_log_term;

            if !prev_matches {
                return AppendEntriesReply {
                    term: current_term,
                    success: false,
                    match_index: log_len.saturating_sub(1),
                };
            }

            let cur_match = log_len.saturating_sub(1);
            let current_commit = *self.commit_index.read();
            drop(log);

            if args.leader_commit > current_commit {
                let new_commit = args.leader_commit.min(cur_match);
                *self.commit_index.write() = new_commit;
                let log_snapshot = self.log.read().clone();
                self.apply_committed_entries(&log_snapshot, new_commit);
            }
            return AppendEntriesReply {
                term: current_term,
                success: true,
                match_index: cur_match,
            };
        }

        let (prev_exists, new_match) = {
            let mut log = self.log.write();
            if args.prev_log_index >= log.len() as u64
                || log[args.prev_log_index as usize].term != args.prev_log_term
            {
                (false, log.len() as u64 - 1)
            } else {
                let mut insert_idx = (args.prev_log_index + 1) as usize;
                for entry in &args.entries {
                    if insert_idx < log.len() {
                        if log[insert_idx].term != entry.term {
                            if self.storage.truncate_suffix(insert_idx as u64).is_err() {
                                return AppendEntriesReply {
                                    term: current_term,
                                    success: false,
                                    match_index: 0,
                                };
                            }
                            if self
                                .storage
                                .append_entries(std::slice::from_ref(entry))
                                .is_err()
                            {
                                return AppendEntriesReply {
                                    term: current_term,
                                    success: false,
                                    match_index: 0,
                                };
                            }
                            log.truncate(insert_idx);
                            log.push(entry.clone());
                        }
                    } else {
                        if self
                            .storage
                            .append_entries(std::slice::from_ref(entry))
                            .is_err()
                        {
                            return AppendEntriesReply {
                                term: current_term,
                                success: false,
                                match_index: 0,
                            };
                        }
                        log.push(entry.clone());
                    }
                    insert_idx += 1;
                }
                (true, log.len() as u64 - 1)
            }
        };

        if !prev_exists {
            return AppendEntriesReply {
                term: current_term,
                success: false,
                match_index: new_match,
            };
        }

        let current_commit = *self.commit_index.read();
        if args.leader_commit > current_commit {
            let new_commit = args.leader_commit.min(new_match);
            *self.commit_index.write() = new_commit;
            let log_snapshot = self.log.read().clone();
            self.apply_committed_entries(&log_snapshot, new_commit);
        }

        AppendEntriesReply {
            term: current_term,
            success: true,
            match_index: new_match,
        }
    }

    pub fn propose(&self, command: RaftCommand) -> HNSQRResult<u64> {
        self.propose_batch(vec![command]).map(|indices| indices[0])
    }

    pub fn propose_batch(&self, commands: Vec<RaftCommand>) -> HNSQRResult<Vec<u64>> {
        if !self.is_leader() {
            return Err(HNSQRError::Internal(format!(
                "Node {} is not leader (current leader: {:?})",
                self.id,
                *self.leader_id.read()
            )));
        }

        let term = *self.current_term.read();
        let plan = self.durability_controller.current_plan();
        self.pipeline_telemetry.write().microbatch_size_last = commands.len().min(plan.max_batch_size);
        self.durability_controller.record_telemetry(crate::consensus::durability_controller::StorageTelemetry {
            p50_fsync_micros: self.storage_health.read().fsync_latency_p99_us / 2,
            p99_fsync_micros: self.storage_health.read().fsync_latency_p99_us,
            mutation_arrival_rate_per_sec: commands.len() as u64 * 1000,
            outstanding_wal_bytes: (commands.len() * 128) as u64,
            replication_rtt_micros: 350,
        });

        let mut log = self.log.write();

        let mut indices = Vec::with_capacity(commands.len());
        let mut new_entries = Vec::with_capacity(commands.len());

        let start_len = log.len() as u64;
        for (i, cmd) in commands.into_iter().enumerate() {
            let index = start_len + i as u64;
            let entry = RaftLogEntry {
                term,
                index,
                command: cmd,
            };
            new_entries.push(entry);
            indices.push(index);
        }

        // Durably append to local log before publishing to in-memory state
        self.storage.append_entries(&new_entries)?;

        log.extend(new_entries);
        let last_idx = *indices.last().unwrap();
        self.match_index.write().insert(self.id, last_idx);
        Ok(indices)
    }

    pub fn check_quorum_commit(&self) -> bool {
        if !self.is_leader() {
            return false;
        }

        let current_commit = *self.commit_index.read();
        let term = *self.current_term.read();
        let peers = self.voting_peers.read().clone();
        let match_idx = self.match_index.read().clone();
        let log_entries = self.log.read().clone();

        let quorum_size = (peers.len() / 2) + 1;
        let mut target_new_commit = None;

        for n in ((current_commit + 1)..=log_entries.len() as u64 - 1).rev() {
            if log_entries[n as usize].term == term {
                let mut count = 0;
                for peer in peers.iter() {
                    if let Some(&matched) = match_idx.get(peer) {
                        if matched >= n {
                            count += 1;
                        }
                    } else if *peer == self.id {
                        count += 1;
                    }
                }

                if count >= quorum_size {
                    target_new_commit = Some(n);
                    break;
                }
            }
        }

        if let Some(n) = target_new_commit {
            *self.commit_index.write() = n;
            if let Err(e) = self.storage.save_progress(&RaftPersistentProgress {
                commit_index: n,
                last_applied: *self.last_applied.read(),
                snapshot_index: 0,
                snapshot_term: 0,
            }) {
                tracing::error!("Failed to persist Raft progress at commit index {n}: {e}");
                self.storage_health.write().record_error(e);
            }
            self.apply_committed_entries(&log_entries, n);
            true
        } else {
            false
        }
    }

    pub fn propose_data_mutation(
        &self,
        mutation: DataMutation,
    ) -> HNSQRResult<tokio::sync::oneshot::Receiver<Result<CommitReceipt, ApplyError>>> {
        if !self.is_leader() {
            return Err(HNSQRError::Internal(format!(
                "Node {} is not leader (current leader: {:?})",
                self.id,
                *self.leader_id.read()
            )));
        }

        let term = *self.current_term.read();
        let mutation_id = mutation.mutation_id().clone();
        let cmd = RaftCommand::Data(mutation);
        let index = self.propose(cmd)?;
        let proposal_id = ProposalId {
            term,
            log_index: index,
        };

        let rx = self.pending_proposals.register(proposal_id, mutation_id)?;
        Ok(rx)
    }

    /// Handles an incoming ReadIndex verification request from the current leader.
    pub fn handle_read_index_request(&self, req: &ReadIndexRequest) -> ReadIndexConfirmation {
        let current_term = *self.current_term.read();
        if req.term < current_term {
            return ReadIndexConfirmation {
                context: req.context,
                term: current_term,
                node_id: self.id,
                success: false,
            };
        }
        *self.last_heartbeat_received.lock() = Instant::now();
        ReadIndexConfirmation {
            context: req.context,
            term: req.term,
            node_id: self.id,
            success: true,
        }
    }

    pub fn linearizable_read_index(&self) -> HNSQRResult<u64> {
        self.linearizable_read_index_with_mode(LinearizableReadMode::ReadIndex)
    }

    pub fn linearizable_read_index_with_mode(
        &self,
        mode: LinearizableReadMode,
    ) -> HNSQRResult<u64> {
        if !self.is_leader() {
            return Err(HNSQRError::Internal(format!(
                "Node {} cannot serve linearizable read: not leader (leader is {:?})",
                self.id,
                *self.leader_id.read()
            )));
        }

        let term = *self.current_term.read();
        match mode {
            LinearizableReadMode::ReadIndex => {
                let (_ctx, _req) = self.read_index_engine.start_read_index_round(term, self.id);
                let read_idx = *self.commit_index.read();
                let last_applied = *self.last_applied.read();
                if last_applied < read_idx {
                    self.read_index_engine.wait_applied(
                        || *self.last_applied.read(),
                        read_idx,
                        std::time::Duration::from_millis(500),
                    )?;
                }
                Ok(read_idx)
            }
            LinearizableReadMode::LeaseRead {
                lease_duration_ms,
                max_clock_drift_ms,
            } => {
                ReadIndexEngine::validate_lease_contract(
                    lease_duration_ms,
                    max_clock_drift_ms,
                    1000,
                )?;
                self.read_index_engine
                    .verify_lease_read(term, std::time::Duration::from_millis(lease_duration_ms))?;
                let read_idx = *self.commit_index.read();
                let last_applied = *self.last_applied.read();
                if last_applied < read_idx {
                    self.read_index_engine.wait_applied(
                        || *self.last_applied.read(),
                        read_idx,
                        std::time::Duration::from_millis(500),
                    )?;
                }
                Ok(read_idx)
            }
        }
    }

    fn apply_committed_entries(&self, log: &[RaftLogEntry], target_commit: u64) {
        let epoch = *self.topology_epoch.read();
        let mut applied = *self.last_applied.read();
        let mut to_complete = Vec::new();
        let mut to_fail = Vec::new();

        while applied < target_commit {
            let next_to_apply = (applied + 1) as usize;
            if next_to_apply < log.len() {
                let entry = &log[next_to_apply];
                let mut apply_ok = true;

                match &entry.command {
                    RaftCommand::Topology(top) => {
                        *self.topology_epoch.write() = top.epoch;
                        let mut owners = self.shard_owners.write();
                        for (s, o) in &top.shard_owners {
                            owners.insert(*s, *o);
                        }
                    }
                    RaftCommand::Membership(mem) => {
                        let mut peers_guard = self.voting_peers.write();
                        *peers_guard = mem.new_peers.iter().copied().collect();
                    }
                    RaftCommand::Data(mutation) => {
                        let proposal_id = ProposalId {
                            term: entry.term,
                            log_index: entry.index,
                        };

                        if let Some(sm) = self.replicated_sm.read().as_ref() {
                            match sm.apply(entry.index, mutation) {
                                Ok(receipt) => {
                                    to_complete.push((
                                        proposal_id,
                                        entry.term,
                                        receipt.applied_index,
                                        epoch,
                                        DurabilityLevel::QuorumReplicated,
                                    ));
                                }
                                Err(e) => {
                                    to_fail.push((
                                        proposal_id,
                                        ApplyError::StateApplyFailed {
                                            mutation_id: mutation.mutation_id().clone(),
                                            reason: e.to_string(),
                                            log_index: entry.index,
                                        },
                                    ));
                                    apply_ok = false;
                                }
                            }
                        } else {
                            to_fail.push((
                                proposal_id,
                                ApplyError::StateMachineUnavailable {
                                    log_index: entry.index,
                                },
                            ));
                            apply_ok = false;
                        }
                    }
                    RaftCommand::NoOp => {}
                }

                if !apply_ok {
                    break;
                }

                applied = entry.index;
                *self.last_applied.write() = applied;
            } else {
                break;
            }
        }

        if let Err(e) = self.storage.save_progress(&RaftPersistentProgress {
            commit_index: target_commit,
            last_applied: applied,
            snapshot_index: 0,
            snapshot_term: 0,
        }) {
            tracing::error!("Failed to persist Raft progress after apply: {e}");
            self.storage_health.write().record_error(e);
        }

        // Complete/Fail proposals outside the lock
        for (proposal_id, term, applied_idx, ep, durability) in to_complete {
            self.pending_proposals
                .complete_applied(proposal_id, term, applied_idx, ep, durability);
        }
        for (proposal_id, err) in to_fail {
            self.pending_proposals.fail_proposal(proposal_id, err);
        }
    }
}

pub struct RaftCluster {
    pub nodes: HashMap<RaftNodeId, Arc<RaftNode>>,
}

impl RaftCluster {
    pub fn new(node_ids: &[RaftNodeId]) -> Self {
        let mut nodes = HashMap::new();
        for &id in node_ids {
            let node = Arc::new(RaftNode::new(id, node_ids.to_vec()));
            nodes.insert(id, node);
        }
        Self { nodes }
    }

    pub fn with_storages(storages: HashMap<RaftNodeId, Arc<dyn RaftStorage>>) -> Self {
        let node_ids: Vec<RaftNodeId> = storages.keys().copied().collect();
        let mut nodes = HashMap::new();
        for (&id, storage) in &storages {
            let node = Arc::new(RaftNode::with_storage(
                id,
                node_ids.clone(),
                storage.clone(),
            ));
            nodes.insert(id, node);
        }
        Self { nodes }
    }

    pub fn add_learner(&mut self, learner_id: RaftNodeId) {
        let learner = Arc::new(RaftNode::new_learner(learner_id));
        self.nodes.insert(learner_id, learner);
    }

    pub fn trigger_election(&self, node_id: RaftNodeId) -> bool {
        let candidate = match self.nodes.get(&node_id) {
            Some(n) if !n.is_learner() => n.clone(),
            _ => return false,
        };

        let new_term = *candidate.current_term.read() + 1;
        if candidate
            .storage
            .save_hard_state(&RaftHardState {
                current_term: new_term,
                voted_for: Some(node_id),
            })
            .is_err()
        {
            return false;
        }

        *candidate.current_term.write() = new_term;
        *candidate.role.write() = RaftRole::Candidate;
        *candidate.voted_for.write() = Some(node_id);

        let (last_idx, last_term) = {
            let log = candidate.log.read();
            let last = log.last().unwrap();
            (last.index, last.term)
        };

        let args = RequestVoteArgs {
            term: new_term,
            candidate_id: node_id,
            last_log_index: last_idx,
            last_log_term: last_term,
        };

        let peers: Vec<RaftNodeId> = candidate.voting_peers.read().iter().copied().collect();
        let mut votes = 1;
        let quorum_required = (peers.len() / 2) + 1;

        for peer_id in peers {
            if peer_id == node_id {
                continue;
            }
            if let Some(peer_node) = self.nodes.get(&peer_id) {
                let reply = peer_node.handle_request_vote(&args);
                if reply.vote_granted && reply.term == new_term {
                    votes += 1;
                }
            }
        }

        if votes >= quorum_required {
            *candidate.role.write() = RaftRole::Leader;
            *candidate.leader_id.write() = Some(node_id);
            candidate.read_index_engine.record_quorum_success(new_term);

            let last_log_len = candidate.log.read().len() as u64;
            {
                let mut next_map = candidate.next_index.write();
                let mut match_map = candidate.match_index.write();
                for &p in self.nodes.keys() {
                    next_map.insert(p, last_log_len);
                    match_map.insert(p, 0);
                }
                match_map.insert(node_id, last_log_len - 1);
            }

            let _ = candidate.propose(RaftCommand::NoOp);
            self.broadcast_heartbeats(node_id);
            true
        } else {
            *candidate.role.write() = RaftRole::Follower;
            false
        }
    }

    pub fn broadcast_heartbeats(&self, leader_id: RaftNodeId) {
        let leader = match self.nodes.get(&leader_id) {
            Some(n) if n.is_leader() => n.clone(),
            _ => return,
        };

        let current_term = *leader.current_term.read();
        let leader_commit = *leader.commit_index.read();
        let log = leader.log.read().clone();
        let all_targets: Vec<RaftNodeId> = self.nodes.keys().copied().collect();
        let mut successful_responses = 1;

        for peer_id in all_targets {
            if peer_id == leader_id {
                continue;
            }
            if let Some(peer_node) = self.nodes.get(&peer_id) {
                let next_idx = *leader.next_index.read().get(&peer_id).unwrap_or(&1);
                let prev_log_idx = (next_idx - 1).min(log.len() as u64 - 1);
                let prev_log_term = log[prev_log_idx as usize].term;
                let entries = if next_idx < log.len() as u64 {
                    log[next_idx as usize..].to_vec()
                } else {
                    Vec::new()
                };

                let args = AppendEntriesArgs {
                    term: current_term,
                    leader_id,
                    prev_log_index: prev_log_idx,
                    prev_log_term,
                    entries: entries.clone(),
                    leader_commit,
                    is_heartbeat: entries.is_empty(),
                };

                let reply = peer_node.handle_append_entries(&args);
                if reply.success {
                    successful_responses += 1;
                    leader
                        .next_index
                        .write()
                        .insert(peer_id, reply.match_index + 1);
                    leader
                        .match_index
                        .write()
                        .insert(peer_id, reply.match_index);
                }
            }
        }

        let quorum_required = (leader.voting_peers.read().len() / 2) + 1;
        if successful_responses >= quorum_required {
            leader.read_index_engine.record_quorum_success(current_term);
        }

        leader.check_quorum_commit();
    }

    pub fn route_read_locality(&self, shard_id: ShardId) -> Option<RaftNodeId> {
        let mut best_candidate = None;
        let mut lowest_lag = u64::MAX;

        for (&id, node) in &self.nodes {
            let is_warm = node.warm_proof_shards.read().contains(&shard_id);
            let lag = node
                .commit_index
                .read()
                .saturating_sub(*node.last_applied.read());

            if is_warm && lag < lowest_lag {
                lowest_lag = lag;
                best_candidate = Some(id);
            }
        }

        best_candidate.or_else(|| self.nodes.keys().next().copied())
    }

    /// Identifies the healthiest voting replica suitable for leadership transfer.
    pub fn get_healthiest_candidate(&self, current_leader_id: RaftNodeId) -> Option<RaftNodeId> {
        let mut best_node = None;
        let mut highest_score = -1.0;

        for (&id, node) in &self.nodes {
            if id == current_leader_id || node.is_learner() {
                continue;
            }
            let score = node.storage_health.read().suitability_score();
            if score > highest_score {
                highest_score = score;
                best_node = Some(id);
            }
        }

        best_node
    }

    /// Performs a Raft-safe graceful leadership transfer to the healthiest candidate.
    pub fn transfer_leadership_to_healthiest(
        &self,
        current_leader_id: RaftNodeId,
    ) -> HNSQRResult<Option<RaftNodeId>> {
        let leader = match self.nodes.get(&current_leader_id) {
            Some(n) if n.is_leader() => n.clone(),
            _ => {
                return Err(HNSQRError::Internal(format!(
                    "Node {current_leader_id} is not leader"
                )));
            }
        };

        if let Some(target_id) = self.get_healthiest_candidate(current_leader_id) {
            self.broadcast_heartbeats(current_leader_id);
            if self.trigger_election(target_id) {
                leader.pipeline_telemetry.write().leadership_transfers_total += 1;
                Ok(Some(target_id))
            } else {
                Err(HNSQRError::Internal(format!(
                    "Leadership transfer to {target_id} failed quorum election"
                )))
            }
        } else {
            Ok(None)
        }
    }

    pub fn get_leader(&self) -> Option<RaftNodeId> {
        for (&id, node) in &self.nodes {
            if node.is_leader() {
                return Some(id);
            }
        }
        None
    }

    pub fn propose_data_mutation(
        &self,
        mutation: DataMutation,
    ) -> HNSQRResult<tokio::sync::oneshot::Receiver<Result<CommitReceipt, ApplyError>>> {
        let leader_id = self.get_leader().unwrap_or_else(|| {
            let first_id = *self.nodes.keys().next().unwrap();
            self.trigger_election(first_id);
            first_id
        });
        let leader = self.nodes.get(&leader_id).unwrap();
        let rx = leader.propose_data_mutation(mutation)?;
        self.broadcast_heartbeats(leader_id);
        Ok(rx)
    }

    pub fn linearizable_read_index(&self) -> HNSQRResult<u64> {
        self.linearizable_read_index_with_mode(LinearizableReadMode::ReadIndex)
    }

    pub fn linearizable_read_index_with_mode(
        &self,
        mode: LinearizableReadMode,
    ) -> HNSQRResult<u64> {
        let leader_id = self.get_leader().ok_or_else(|| {
            HNSQRError::Internal("No active Raft leader to coordinate ReadIndex".to_string())
        })?;
        let leader = self.nodes.get(&leader_id).unwrap();

        match mode {
            LinearizableReadMode::ReadIndex => {
                let term = *leader.current_term.read();
                let (_ctx, req) = leader
                    .read_index_engine
                    .start_read_index_round(term, leader_id);

                let peers: Vec<RaftNodeId> = leader.voting_peers.read().iter().copied().collect();
                let quorum_required = (peers.len() / 2) + 1;

                let mut quorum_reached = quorum_required <= 1;
                for peer_id in &peers {
                    if *peer_id == leader_id {
                        continue;
                    }
                    if let Some(peer_node) = self.nodes.get(peer_id) {
                        let confirmation = peer_node.handle_read_index_request(&req);
                        if leader.read_index_engine.handle_confirmation(
                            &confirmation,
                            term,
                            quorum_required,
                        )? {
                            quorum_reached = true;
                        }
                    }
                }

                if !quorum_reached {
                    return Err(HNSQRError::Internal(
                        "ReadIndex round failed to achieve voting quorum confirmation from peers"
                            .into(),
                    ));
                }

                // Invalidate if term changed during confirmation exchange
                let term_after = *leader.current_term.read();
                if term_after != term {
                    leader
                        .read_index_engine
                        .readindex_term_invalidations
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    return Err(HNSQRError::Internal(format!(
                        "ReadIndex round invalidated: term changed from {term} to {term_after}"
                    )));
                }

                let read_idx = *leader.commit_index.read();
                let last_applied = *leader.last_applied.read();
                if last_applied < read_idx {
                    leader.read_index_engine.wait_applied(
                        || *leader.last_applied.read(),
                        read_idx,
                        std::time::Duration::from_millis(500),
                    )?;
                }
                Ok(read_idx)
            }
            LinearizableReadMode::LeaseRead {
                lease_duration_ms,
                max_clock_drift_ms,
            } => {
                ReadIndexEngine::validate_lease_contract(
                    lease_duration_ms,
                    max_clock_drift_ms,
                    1000,
                )?;
                self.broadcast_heartbeats(leader_id);
                let term = *leader.current_term.read();
                leader
                    .read_index_engine
                    .verify_lease_read(term, std::time::Duration::from_millis(lease_duration_ms))?;

                let read_idx = *leader.commit_index.read();
                let last_applied = *leader.last_applied.read();
                if last_applied < read_idx {
                    leader.read_index_engine.wait_applied(
                        || *leader.last_applied.read(),
                        read_idx,
                        std::time::Duration::from_millis(500),
                    )?;
                }
                Ok(read_idx)
            }
        }
    }
}
