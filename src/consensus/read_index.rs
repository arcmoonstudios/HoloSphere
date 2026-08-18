/* hnsqr/src/consensus/read_index.rs */
//!▫~•◦-------------------------------‣
//! # Linearizable ReadIndex & Explicit Read Consistency Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides strict linearizable reads without writing to disk by cleanly separating
//! pure consensus ReadIndex (conservative default) from clock-dependent LeaseRead.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{HNSQRError, HNSQRResult};

/// Exact execution strategy for serving linearizable reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LinearizableReadMode {
    /// Conservative default: requires active quorum confirmation for current term.
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

/// Telemetry metrics produced by a ReadIndex query evaluation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReadIndexTelemetry {
    pub read_index: u64,
    pub applied_index: u64,
    pub observed_lag_entries: u64,
    pub quorum_check_latency_us: u64,
    pub wait_applied_latency_us: u64,
}

pub struct ReadIndexEngine {
    last_quorum_check: RwLock<Instant>,
    lease_duration: Duration,
    lease_term: AtomicU64,
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
        }
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

    /// Waits asynchronously or spin-sleeps until `last_applied >= target_index`.
    pub fn wait_applied(
        &self,
        get_last_applied: impl Fn() -> u64,
        target_index: u64,
        timeout: Duration,
    ) -> HNSQRResult<u64> {
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
            std::thread::yield_now();
        }
    }
}
