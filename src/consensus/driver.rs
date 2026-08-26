/* holosphere/src/consensus/driver.rs */
//!▫~•◦---------------------------------‣
//! # RaftDriver — Decoupled Actor Runtime for Raft Consensus
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Separates three execution concerns that must never block each other:
//!
//! ```text
//!  ┌─────────────────────────────────────────────────────────────┐
//!  │ hnsqr-raft-timer  (heartbeat ticks + election watchdog)     │
//!  │  Never touches disk. Posts DriverEvent::Tick to event loop. │
//!  ├─────────────────────────────────────────────────────────────┤
//!  │ hnsqr-raft-driver (Raft state machine event loop)           │
//!  │  Drives RaftCluster / RaftNode synchronously.               │
//!  │  Posts WalFlushBatch to WAL worker; never calls fsync.      │
//!  ├─────────────────────────────────────────────────────────────┤
//!  │ hnsqr-wal-worker  (group-commit fsync lane)                 │
//!  │  Drains WalFlushBatch channel, coalesces, calls             │
//!  │  WalManager::sync_target_lsn once per group.                │
//!  │  Posts DriverEvent::WalFlushed on completion.               │
//!  └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Invariants
//! - Heartbeat ticks are posted on a fixed interval regardless of disk pressure.
//! - WAL fsyncs are offloaded; the driver event loop is never blocked on I/O.
//! - `RaftNode::last_heartbeat_received` is the sole election-timeout signal.
//! - Election timeout uses base + random jitter to avoid split-vote thundering.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crossbeam_channel::{Sender, TryRecvError, bounded};
use parking_lot::RwLock;

use crate::cluster::state_machine::DataMutation;
use crate::consensus::pending::{ApplyError, CommitReceipt, MutationId};
use crate::consensus::raft::{RaftCluster, RaftMessage, RaftNode, RaftNodeId};
use crate::storage::wal::{DurabilityPolicy, WalManager, WalMutation};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// Error variants returned through client proposal response channels.
#[derive(Debug, thiserror::Error)]
pub enum RaftDriverError {
    #[error("Node {0} is not the current cluster leader")]
    NotLeader(RaftNodeId),

    #[error("Raft driver is shutting down")]
    ShuttingDown,

    #[error("Proposal channel is full — driver is overloaded")]
    ChannelFull,

    #[error("Internal driver error: {0}")]
    Internal(String),
}

/// A client mutation request. The response is delivered through a one-shot
/// Tokio channel, consistent with [`RaftNode::propose_data_mutation`].
pub struct ClientProposal {
    pub mutation: DataMutation,
    /// Response channel to the caller. On success, yields the [`CommitReceipt`]
    /// once the entry is applied to the replicated state machine.
    pub response_tx: tokio::sync::oneshot::Sender<Result<CommitReceipt, ApplyError>>,
}

impl std::fmt::Debug for ClientProposal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClientProposal")
            .field("mutation_id", &self.mutation.mutation_id())
            .finish_non_exhaustive()
    }
}

/// All events multiplexed into the Raft driver event loop.
#[derive(Debug)]
pub enum DriverEvent {
    /// Periodic tick from the timer lane — drives heartbeats and election timeouts.
    Tick,
    /// Inbound peer RPC (AppendEntries, RequestVote, their replies).
    PeerMessage {
        /// Target node within the local cluster (for multi-node test clusters).
        target_node: RaftNodeId,
        message: RaftMessage,
    },
    /// Client mutation proposal.
    Propose(ClientProposal),
    /// WAL worker confirms that entries up to `lsn` are durably on disk.
    WalFlushed { lsn: u64 },
    /// Graceful shutdown signal.
    Shutdown,
}

/// Shared liveness markers readable from any execution lane without locks.
pub struct ClusterLiveness {
    /// Highest WAL LSN that has been durably fsynced on the local node.
    pub flushed_lsn: AtomicU64,
    /// Highest commit index advanced by quorum across all nodes.
    pub committed_index: AtomicU64,
    /// Approximate last-seen leader (advisory; not authoritative under partition).
    pub last_known_leader: RwLock<Option<RaftNodeId>>,
    /// Set to `false` by the WAL worker on any unrecoverable I/O error.
    /// The driver event loop observes this flag and steps the local node down
    /// from leadership to prevent accepting mutations it cannot persist.
    pub is_storage_healthy: AtomicBool,
}

impl Default for ClusterLiveness {
    fn default() -> Self {
        Self {
            flushed_lsn: AtomicU64::new(0),
            committed_index: AtomicU64::new(0),
            last_known_leader: RwLock::new(None),
            is_storage_healthy: AtomicBool::new(true),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal WAL flush batch
// ─────────────────────────────────────────────────────────────────────────────

/// A serialized cluster-state payload queued for the group-commit WAL worker.
struct WalFlushBatch {
    /// Serialized `WalMutation::ClusterState` payload to persist.
    mutation: WalMutation,
}

// ─────────────────────────────────────────────────────────────────────────────
// RaftDriver
// ─────────────────────────────────────────────────────────────────────────────

/// Actor runtime that drives a [`RaftCluster`] with decoupled timer, compute,
/// and I/O execution lanes.
///
/// # Usage
/// ```no_run
/// # use std::path::PathBuf;
/// # use std::time::Duration;
/// # use holosphere::consensus::raft::RaftCluster;
/// # use holosphere::consensus::driver::RaftDriver;
/// let cluster = RaftCluster::new(&[1, 2, 3]);
/// let driver = RaftDriver::new(
///     cluster,
///     1,                                    // local node id
///     PathBuf::from("/var/db/wal"),
///     Duration::from_millis(150),           // heartbeat interval
///     Duration::from_millis(300),           // election timeout base
/// );
/// // Propose a mutation (non-blocking)
/// // let rx = driver.propose(my_mutation)?;
/// // later: rx.await?
/// driver.shutdown();
/// ```
pub struct RaftDriver {
    event_tx: Sender<DriverEvent>,
    pub liveness: Arc<ClusterLiveness>,
    local_node_id: RaftNodeId,
    driver_thread: Option<JoinHandle<()>>,
    wal_thread: Option<JoinHandle<()>>,
    // timer thread is detached — it exits when the event channel is dropped
}

impl RaftDriver {
    /// Constructs and starts the three execution lanes.
    ///
    /// - `cluster` — the deterministic Raft state machine to drive
    /// - `local_node_id` — which node in the cluster is local to this driver
    /// - `wal_dir` — directory for the [`WalManager`] used by the WAL worker lane
    /// - `heartbeat_interval` — how often `Tick` is posted (should be ≤ election_timeout/2)
    /// - `election_timeout_base` — minimum quiet time before a follower starts a new election;
    ///   actual timeout is `base + rand(0..base)` for split-vote avoidance
    pub fn new(
        cluster: RaftCluster,
        local_node_id: RaftNodeId,
        wal_dir: PathBuf,
        heartbeat_interval: Duration,
        election_timeout_base: Duration,
    ) -> Self {
        // Fix 1: Capture the Tokio runtime handle here, in the caller's context,
        // where a runtime is guaranteed to be active. Panics immediately with a
        // clear message if called outside a Tokio runtime — far better than a
        // silent panic on the first proposal deep inside the driver thread.
        let rt_handle = tokio::runtime::Handle::try_current()
            .expect("RaftDriver::new must be called from within a Tokio runtime context");

        let (event_tx, event_rx) = bounded::<DriverEvent>(10_000);
        let (wal_tx, wal_rx) = bounded::<WalFlushBatch>(1_000);

        let liveness = Arc::new(ClusterLiveness::default());

        // ─────────────────────────────────────────────────────────────────────
        // LANE 1 — WAL Group-Commit Worker (dedicated OS thread, disk I/O only)
        // Fix 3: Worker holds Arc<ClusterLiveness> and marks is_storage_healthy
        // false before returning on any unrecoverable I/O error, then sends
        // Shutdown so the driver loop stops accepting mutations immediately.
        // ─────────────────────────────────────────────────────────────────────
        let event_tx_wal = event_tx.clone();
        let liveness_wal = Arc::clone(&liveness);
        let wal_thread = thread::Builder::new()
            .name("hnsqr-wal-worker".into())
            .spawn(move || {
                let poison =
                    |liveness: &ClusterLiveness, event_tx: &Sender<DriverEvent>, msg: &str| {
                        // Mark storage unhealthy and signal the driver to shut down.
                        // Called on any unrecoverable WAL I/O error.
                        eprintln!("[hnsqr-wal-worker] FATAL: {msg}");
                        liveness.is_storage_healthy.store(false, Ordering::Release);
                        // Best-effort: if the channel is already gone this is a no-op.
                        let _ = event_tx.send(DriverEvent::Shutdown);
                    };

                let wal = match WalManager::open(&wal_dir) {
                    Ok(w) => w,
                    Err(e) => {
                        poison(
                            &liveness_wal,
                            &event_tx_wal,
                            &format!("failed to open WAL at {}: {e}", wal_dir.display()),
                        );
                        return;
                    }
                };

                // Receive first batch, then drain all immediately available
                // batches into a single coalesced fsync (group commit).
                while let Ok(first) = wal_rx.recv() {
                    if let Err(e) = wal.append(&first.mutation, DurabilityPolicy::WalGroupCommit) {
                        poison(
                            &liveness_wal,
                            &event_tx_wal,
                            &format!("WAL append failed: {e}"),
                        );
                        return;
                    }

                    // Drain remaining pending batches without blocking.
                    loop {
                        match wal_rx.try_recv() {
                            Ok(batch) => {
                                if let Err(e) =
                                    wal.append(&batch.mutation, DurabilityPolicy::WalGroupCommit)
                                {
                                    poison(
                                        &liveness_wal,
                                        &event_tx_wal,
                                        &format!("WAL append failed: {e}"),
                                    );
                                    return;
                                }
                            }
                            Err(TryRecvError::Empty) => break,
                            Err(TryRecvError::Disconnected) => return,
                        }
                    }

                    // Single fsync for the entire collected group.
                    let target_lsn = wal.current_lsn();
                    if let Err(e) = wal.sync_target_lsn(target_lsn) {
                        poison(
                            &liveness_wal,
                            &event_tx_wal,
                            &format!("fsync failed at LSN {target_lsn}: {e}"),
                        );
                        return;
                    }

                    // Notify the driver that disk durability has advanced.
                    let _ = event_tx_wal.send(DriverEvent::WalFlushed { lsn: target_lsn });
                }
            })
            .expect("failed to spawn hnsqr-wal-worker thread");

        // ─────────────────────────────────────────────────────────────────────
        // Timer tick thread — detached, exits when event_tx is dropped.
        // Never touches disk; only posts Tick at the fixed heartbeat interval.
        // ─────────────────────────────────────────────────────────────────────
        let event_tx_timer = event_tx.clone();
        thread::Builder::new()
            .name("hnsqr-raft-timer".into())
            .spawn(move || {
                loop {
                    thread::sleep(heartbeat_interval);
                    if event_tx_timer.send(DriverEvent::Tick).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn hnsqr-raft-timer thread");

        // ─────────────────────────────────────────────────────────────────────
        // LANE 2 — Raft Driver Event Loop (high-priority compute lane).
        // Fix 1: rt_handle cloned in so tokio::spawn calls go through the
        //        captured handle rather than implicitly requiring a runtime context.
        // Fix 2: Propose only routes to the local node when it IS the leader;
        //        any other leader ID is a redirect rejection.
        // Fix 3: WalFlushed checks is_storage_healthy and steps down leadership
        //        if disk durability is dead.
        // ─────────────────────────────────────────────────────────────────────
        let liveness_driver = Arc::clone(&liveness);
        let rt_for_driver = rt_handle.clone();
        let driver_thread = thread::Builder::new()
            .name("hnsqr-raft-driver".into())
            .spawn(move || {
                let jitter_seed = local_node_id
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .wrapping_add(0x6c62_272e_07bb_0142);
                let mut jitter_state = jitter_seed;

                let next_election_timeout = |state: &mut u64| -> Duration {
                    *state ^= *state << 13;
                    *state ^= *state >> 7;
                    *state ^= *state << 17;
                    let frac = (*state >> 33) as u32;
                    let jitter_ns = (frac as u64 * election_timeout_base.as_nanos() as u64) >> 31;
                    election_timeout_base + Duration::from_nanos(jitter_ns)
                };

                let mut current_election_timeout = next_election_timeout(&mut jitter_state);

                while let Ok(event) = event_rx.recv() {
                    match event {
                        DriverEvent::Tick => {
                            if let Some(leader_id) = cluster.get_leader() {
                                if leader_id == local_node_id {
                                    cluster.broadcast_heartbeats(local_node_id);
                                }
                                *liveness_driver.last_known_leader.write() = Some(leader_id);
                            }

                            if let Some(local_node) = cluster.nodes.get(&local_node_id) {
                                if !local_node.is_leader() && !local_node.is_learner() {
                                    let elapsed = {
                                        let hb = local_node.last_heartbeat_received.lock();
                                        hb.elapsed()
                                    };
                                    if elapsed >= current_election_timeout {
                                        cluster.trigger_election(local_node_id);
                                        current_election_timeout =
                                            next_election_timeout(&mut jitter_state);
                                    }
                                }
                            }
                        }

                        DriverEvent::PeerMessage {
                            target_node,
                            message,
                        } => {
                            let Some(node) = cluster.nodes.get(&target_node) else {
                                continue;
                            };
                            dispatch_peer_message(node, message, &cluster, &wal_tx);
                        }

                        DriverEvent::Propose(proposal) => {
                            match cluster.get_leader() {
                                // Fix 2: Only accept the proposal if THIS node is the leader.
                                // If another node is leader, reject with redirect info so the
                                // client can route to the correct host.
                                Some(leader_id) if leader_id == local_node_id => {
                                    if let Some(local_node) = cluster.nodes.get(&local_node_id) {
                                        match local_node.propose_data_mutation(proposal.mutation) {
                                            Ok(rx) => {
                                                let caller_tx = proposal.response_tx;
                                                // Fix 1: spawn through the captured handle,
                                                // not via the ambient runtime context (which
                                                // does not exist inside std::thread).
                                                rt_for_driver.spawn(async move {
                                                    let result = rx.await.unwrap_or_else(|_| {
                                                        Err(ApplyError::StateMachineUnavailable {
                                                            log_index: 0,
                                                        })
                                                    });
                                                    let _ = caller_tx.send(result);
                                                });
                                            }
                                            Err(e) => {
                                                let _ = proposal.response_tx.send(Err(
                                                    ApplyError::StateApplyFailed {
                                                        mutation_id: MutationId(
                                                            "proposal-failed".into(),
                                                        ),
                                                        reason: e.to_string(),
                                                        log_index: 0,
                                                    },
                                                ));
                                            }
                                        }
                                    }
                                }
                                Some(leader_id) => {
                                    // Fix 2: We are not the leader — reject with redirect.
                                    let _ = proposal.response_tx.send(Err(
                                        ApplyError::StateApplyFailed {
                                            mutation_id: MutationId("not-leader".into()),
                                            reason: format!(
                                                "node {local_node_id} is not the leader; \
                                                 redirect to node {leader_id}"
                                            ),
                                            log_index: 0,
                                        },
                                    ));
                                }
                                None => {
                                    let _ = proposal.response_tx.send(Err(
                                        ApplyError::StateMachineUnavailable { log_index: 0 },
                                    ));
                                }
                            }
                        }

                        DriverEvent::WalFlushed { lsn } => {
                            liveness_driver
                                .flushed_lsn
                                .fetch_max(lsn, Ordering::Release);

                            // Fix 3: If the WAL worker has poisoned the health flag,
                            // step the local node down from leadership immediately so
                            // no new mutations are accepted against a dead disk lane.
                            if !liveness_driver.is_storage_healthy.load(Ordering::Acquire) {
                                if let Some(local_node) = cluster.nodes.get(&local_node_id) {
                                    if local_node.is_leader() {
                                        *local_node.role.write() =
                                            crate::consensus::raft::RaftRole::Follower;
                                        tracing::error!(
                                            node = local_node_id,
                                            "WAL worker reported I/O failure; \
                                             stepping down from leadership"
                                        );
                                    }
                                }
                                break;
                            }

                            if let Some(leader_id) = cluster.get_leader() {
                                if let Some(leader) = cluster.nodes.get(&leader_id) {
                                    if leader.check_quorum_commit() {
                                        let committed = *leader.commit_index.read();
                                        liveness_driver
                                            .committed_index
                                            .fetch_max(committed, Ordering::Release);
                                    }
                                }
                            }
                        }

                        DriverEvent::Shutdown => break,
                    }
                }
            })
            .expect("failed to spawn hnsqr-raft-driver thread");

        Self {
            event_tx,
            liveness,
            local_node_id,
            driver_thread: Some(driver_thread),
            wal_thread: Some(wal_thread),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Public interface
    // ─────────────────────────────────────────────────────────────────────────

    /// Non-blocking proposal entry point for client mutations.
    ///
    /// Returns the response receiver immediately. The actual commit confirmation
    /// arrives asynchronously once the entry clears Raft quorum and is applied
    /// to the replicated state machine.
    pub fn propose(
        &self,
        mutation: DataMutation,
    ) -> Result<tokio::sync::oneshot::Receiver<Result<CommitReceipt, ApplyError>>, RaftDriverError>
    {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        let event = DriverEvent::Propose(ClientProposal {
            mutation,
            response_tx,
        });
        self.event_tx
            .try_send(event)
            .map_err(|_| RaftDriverError::ChannelFull)?;
        Ok(response_rx)
    }

    /// Non-blocking ingress for peer RPC packets (from network layer or in-process cluster).
    ///
    /// Returns `false` if the driver event channel is full or the driver has shut down.
    #[inline(always)]
    pub fn ingest_peer_message(&self, target_node: RaftNodeId, message: RaftMessage) -> bool {
        self.event_tx
            .try_send(DriverEvent::PeerMessage {
                target_node,
                message,
            })
            .is_ok()
    }

    /// Returns the highest WAL LSN durably fsynced on the local node.
    #[inline(always)]
    pub fn flushed_lsn(&self) -> u64 {
        self.liveness.flushed_lsn.load(Ordering::Acquire)
    }

    /// Returns the highest commit index advanced by quorum.
    #[inline(always)]
    pub fn committed_index(&self) -> u64 {
        self.liveness.committed_index.load(Ordering::Acquire)
    }

    /// Returns the local node ID this driver is managing.
    #[inline(always)]
    pub fn local_node_id(&self) -> RaftNodeId {
        self.local_node_id
    }

    /// Sends a graceful shutdown signal and joins all execution lanes.
    ///
    /// Blocks until both the driver thread and WAL worker have exited.
    pub fn shutdown(mut self) {
        let _ = self.event_tx.send(DriverEvent::Shutdown);
        if let Some(h) = self.driver_thread.take() {
            let _ = h.join();
        }
        if let Some(h) = self.wal_thread.take() {
            let _ = h.join();
        }
    }
}

impl Drop for RaftDriver {
    fn drop(&mut self) {
        // Best-effort shutdown if the driver is dropped without calling shutdown().
        let _ = self.event_tx.try_send(DriverEvent::Shutdown);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Dispatches a [`RaftMessage`] to the correct handler on `node` and queues
/// any resulting cluster-state changes for WAL persistence.
fn dispatch_peer_message(
    node: &Arc<RaftNode>,
    message: RaftMessage,
    cluster: &RaftCluster,
    wal_tx: &Sender<WalFlushBatch>,
) {
    match message {
        RaftMessage::RequestVote(args) => {
            let reply = node.handle_request_vote(&args);
            // Persist hard-state change (term / voted_for) via WAL worker.
            // The hard state is already durably written inside handle_request_vote
            // through RaftStorage::save_hard_state; we additionally record the
            // cluster-state epoch in the WAL for cross-engine recovery.
            queue_cluster_state_wal(node, wal_tx);
            // Drive any quorum advancement triggered by the vote.
            if reply.vote_granted {
                if let Some(leader_id) = cluster.get_leader() {
                    cluster.broadcast_heartbeats(leader_id);
                }
            }
        }

        RaftMessage::RequestVoteReply(_reply) => {
            // Replies are handled passively — the state machine already advanced
            // in trigger_election. No further action needed from the driver.
        }

        RaftMessage::AppendEntries(args) => {
            let _reply = node.handle_append_entries(&args);
            // Persist the applied cluster state after entries land.
            queue_cluster_state_wal(node, wal_tx);
        }

        RaftMessage::AppendEntriesReply(_reply) => {
            // As with RequestVoteReply — RaftCluster::broadcast_heartbeats already
            // updated next_index / match_index synchronously. No further state change.
        }
    }
}

/// Enqueues a `WalMutation::ClusterState` snapshot of the node's current epoch
/// into the WAL worker for durable persistence.
///
/// This is a best-effort, non-blocking push. If the WAL worker channel is full
/// (which indicates severe I/O backpressure), the push is silently dropped —
/// the entry will be re-created on the next event that calls this function.
fn queue_cluster_state_wal(node: &Arc<RaftNode>, wal_tx: &Sender<WalFlushBatch>) {
    let epoch = *node.topology_epoch.read();
    let term = *node.current_term.read();
    // Encode a compact cluster-state record: epoch (8 bytes) | term (8 bytes)
    let mut state_data = Vec::with_capacity(16);
    state_data.extend_from_slice(&epoch.to_le_bytes());
    state_data.extend_from_slice(&term.to_le_bytes());

    let mutation = WalMutation::ClusterState { epoch, state_data };
    let _ = wal_tx.try_send(WalFlushBatch { mutation });
}

// ─────────────────────────────────────────────────────────────────────────────
// Re-export for ergonomic use from parent module
// ─────────────────────────────────────────────────────────────────────────────
// RaftMessage is already imported at the top of this file from crate::consensus::raft.
