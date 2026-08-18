/* hnsqr/src/consensus/read_index.rs */
//!▫~•◦-------------------------------‣
//! # Linearizable ReadIndex & Explicit Read Consistency Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides strict linearizable reads without writing to disk by cleanly separating
//! pure consensus ReadIndex (conservative default) from clock-dependent LeaseRead.
//! Guarantees non-stale reads across dynamic partitions and leader re-elections.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::consensus::raft::RaftNodeId;
use crate::{HNSQRError, HNSQRResult};

/// Unique monotonic or random identifier for a specific ReadIndex verification round.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ReadContextId(pub u64);

impl ReadContextId {
    pub fn generate() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let id = COUNTER.fetch_add(1, Ordering::Relaxed);
        let random_salt = (rand::random::<u32>() as u64) << 32;
        Self(id ^ random_salt)
    }
}

/// Request sent by a leader to followers to confirm its leadership for a ReadIndex round.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadIndexRequest {
    pub context: ReadContextId,
    pub term: u64,
}

/// Confirmation returned by a follower acknowledging leader authority for a ReadIndex round.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadIndexConfirmation {
    pub context: ReadContextId,
    pub term: u64,
    pub node_id: RaftNodeId,
    pub success: bool,
}

/// Exact execution strategy for serving linearizable reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LinearizableReadMode {
    /// Conservative default: requires active round-bound quorum confirmation for current term.
    #[default]
    ReadIndex,
    /// High-throughput lease read: valid only under bounded clock drift assumptions.
    LeaseRead {
        lease_duration_ms: u64,
        max_clock_drift_ms: u64,
    },
}

/// Read consistency level requested by caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReadConsistency {
    /// Strict linearizability under declared mode (ReadIndex by default).
    #[default]
    Linearizable,
    /// Explicit linearizable read with mode configuration.
    LinearizableWithMode(LinearizableReadMode),
    /// Reads locally applied state up to current `commit_index`.
    Committed,
    /// Bounded staleness allowing up to `max_lag_entries` or `max_age_ms`.
    BoundedStaleness {
        max_lag_entries: u64,
        max_age_ms: u64,
    },
}

/// Active in-flight state for a pending ReadIndex quorum round.
#[derive(Debug)]
struct ReadIndexRoundState {
    term: u64,
    confirmations: HashSet<RaftNodeId>,
    created_at: Instant,
}

/// Telemetry metrics produced by a ReadIndex query evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadIndexTelemetry {
    pub read_index: u64,
    pub applied_index: u64,
    pub observed_lag_entries: u64,
    pub quorum_check_latency_us: u64,
    pub wait_applied_latency_us: u64,
}

/// Consolidated engine managing ReadIndex rounds, LeaseRead validity, and read telemetry.
pub struct ReadIndexEngine {
    last_quorum_check: RwLock<Instant>,
    lease_duration: Duration,
    lease_term: AtomicU64,
    pending_rounds: RwLock<HashMap<ReadContextId, ReadIndexRoundState>>,

    // Telemetry counters
    pub readindex_requests_total: AtomicU64,
    pub readindex_quorum_latency_us: AtomicU64,
    pub readindex_term_invalidations: AtomicU64,
    pub lease_reads_total: AtomicU64,
    pub lease_read_rejections: AtomicU64,
    pub lease_remaining_us: AtomicU64,
}

impl Default for ReadIndexEngine {
    fn default() -> Self {
        Self::new(Duration::from_millis(250))
    }
}

impl ReadIndexEngine {
    pub fn new(lease_duration: Duration) -> Self {
        Self {
            last_quorum_check: RwLock::new(Instant::now() - lease_duration * 2),
            lease_duration,
            lease_term: AtomicU64::new(0),
            pending_rounds: RwLock::new(HashMap::new()),
            readindex_requests_total: AtomicU64::new(0),
            readindex_quorum_latency_us: AtomicU64::new(0),
            readindex_term_invalidations: AtomicU64::new(0),
            lease_reads_total: AtomicU64::new(0),
            lease_read_rejections: AtomicU64::new(0),
            lease_remaining_us: AtomicU64::new(0),
        }
    }

    /// Registers a new active ReadIndex round for the specified term.
    pub fn start_read_index_round(&self, term: u64, leader_id: RaftNodeId) -> (ReadContextId, ReadIndexRequest) {
        self.readindex_requests_total.fetch_add(1, Ordering::Relaxed);
        let ctx = ReadContextId::generate();
        let mut confirmations = HashSet::new();
        confirmations.insert(leader_id); // Leader votes for itself

        let state = ReadIndexRoundState {
            term,
            confirmations,
            created_at: Instant::now(),
        };

        self.pending_rounds.write().insert(ctx, state);
        (ctx, ReadIndexRequest { context: ctx, term })
    }

    /// Ingests a follower confirmation for an active ReadIndex round.
    /// Returns `Ok(true)` if the round has now achieved the required voting quorum.
    pub fn handle_confirmation(
        &self,
        conf: &ReadIndexConfirmation,
        current_term: u64,
        quorum_size: usize,
    ) -> HNSQRResult<bool> {
        if conf.term != current_term {
            self.readindex_term_invalidations.fetch_add(1, Ordering::Relaxed);
            return Err(HNSQRError::Internal(format!(
                "ReadIndex confirmation term mismatch: received {}, current {current_term}",
                conf.term
            )));
        }

        let mut rounds = self.pending_rounds.write();

        // Check for stale term before mutating confirmations
        if let Some(state) = rounds.get(&conf.context) {
            if state.term != current_term {
                let stale_term = state.term;
                rounds.remove(&conf.context);
                self.readindex_term_invalidations.fetch_add(1, Ordering::Relaxed);
                return Err(HNSQRError::Internal(format!(
                    "ReadIndex round invalidated due to term change from {stale_term} to {current_term}"
                )));
            }
        }

        if let Some(state) = rounds.get_mut(&conf.context) {
            if conf.success {
                state.confirmations.insert(conf.node_id);
            }

            if state.confirmations.len() >= quorum_size {
                let elapsed_us = state.created_at.elapsed().as_micros() as u64;
                self.readindex_quorum_latency_us.store(elapsed_us, Ordering::Relaxed);
                rounds.remove(&conf.context);
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Validates the LeaseRead safety contract against election timeout.
    /// Rejects configurations where lease_duration + max_clock_drift >= election_timeout.
    pub fn validate_lease_contract(
        lease_duration_ms: u64,
        max_clock_drift_ms: u64,
        election_timeout_ms: u64,
    ) -> HNSQRResult<()> {
        if lease_duration_ms == 0 {
            return Err(HNSQRError::InvalidConfig(
                "Lease duration must be greater than 0 ms".to_string(),
            ));
        }
        let total_safety_window = lease_duration_ms.saturating_add(max_clock_drift_ms);
        if total_safety_window >= election_timeout_ms {
            return Err(HNSQRError::InvalidConfig(format!(
                "Unsafe LeaseRead configuration: lease_duration ({lease_duration_ms}ms) + max_clock_drift ({max_clock_drift_ms}ms) = {total_safety_window}ms >= election_timeout ({election_timeout_ms}ms)"
            )));
        }
        Ok(())
    }

    /// Evaluates LeaseRead validity against monotonic clock and configured boundaries.
    pub fn verify_lease_read(
        &self,
        current_term: u64,
        lease_duration: Duration,
    ) -> HNSQRResult<()> {
        self.lease_reads_total.fetch_add(1, Ordering::Relaxed);

        if self.lease_term.load(Ordering::SeqCst) != current_term {
            self.lease_read_rejections.fetch_add(1, Ordering::Relaxed);
            return Err(HNSQRError::Internal(format!(
                "LeaseRead rejected: lease term {} != current term {current_term}",
                self.lease_term.load(Ordering::SeqCst)
            )));
        }

        let elapsed = self.last_quorum_check.read().elapsed();
        if elapsed >= lease_duration {
            self.lease_read_rejections.fetch_add(1, Ordering::Relaxed);
            return Err(HNSQRError::Internal(format!(
                "LeaseRead expired: elapsed {:?} >= allowed lease {:?}",
                elapsed, lease_duration
            )));
        }

        let remaining = lease_duration.saturating_sub(elapsed).as_micros() as u64;
        self.lease_remaining_us.store(remaining, Ordering::Relaxed);
        Ok(())
    }

    /// Records a successful quorum heartbeat exchange for the given term.
    pub fn record_quorum_success(&self, term: u64) {
        self.lease_term.store(term, Ordering::SeqCst);
        *self.last_quorum_check.write() = Instant::now();
    }

    /// Checks whether the leader holds a valid unexpired leader lease.
    pub fn has_valid_lease(&self, current_term: u64) -> bool {
        if self.lease_term.load(Ordering::SeqCst) != current_term {
            return false;
        }
        let elapsed = self.last_quorum_check.read().elapsed();
        elapsed < self.lease_duration
    }

    /// Polls until `last_applied >= target_index`, sleeping briefly between checks.
    ///
    /// This is a synchronous blocking call suitable for use outside the Tokio runtime
    /// (e.g. the read-snapshot path). For async callers, prefer the notify-based
    /// async variant once P0-19 lands.
    pub fn wait_applied(
        &self,
        get_last_applied: impl Fn() -> u64,
        target_index: u64,
        timeout: Duration,
    ) -> HNSQRResult<u64> {
        const POLL_INTERVAL: Duration = Duration::from_micros(100);
        let start = Instant::now();
        loop {
            let cur = get_last_applied();
            if cur >= target_index {
                return Ok(cur);
            }
            if start.elapsed() > timeout {
                return Err(HNSQRError::Internal(format!(
                    "Timed out waiting for applied index (current {cur}, target {target_index})"
                )));
            }
            std::thread::sleep(POLL_INTERVAL);
        }
    }
}

