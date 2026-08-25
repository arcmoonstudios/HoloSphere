/* holosphere/src/storage/io_budget.rs */
//!▫~•◦-------------------------------‣
//! # Maintenance I/O Isolation & Self-Throttling Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides token-bucket bandwidth limiting for background operations (compaction,
//! snapshot generation, backup upload, shard migration) while ensuring zero
//! interference with foreground query and WAL durability lanes.
//! Automatically self-throttles when foreground latency spikes, with hysteresis.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

/// Maintenance I/O category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IoMaintenanceClass {
    Compaction,
    SnapshotGeneration,
    BackupUpload,
    ShardMigration,
    ColdFetch,
}

/// Token-bucket I/O bandwidth controller.
pub struct IoBudgetManager {
    /// Maximum allowed maintenance bandwidth in bytes per second (default: 100 MB/s).
    pub max_bandwidth_bytes_per_sec: AtomicU64,
    /// Available tokens in bytes.
    available_tokens: AtomicU64,
    last_refill: Mutex<Instant>,
    /// Whether background maintenance is currently throttled due to foreground latency pressure.
    pub is_throttled: AtomicBool,
    pub p95_latency_threshold_micros: u64,
    pub p95_recovery_threshold_micros: u64,
}

impl Default for IoBudgetManager {
    fn default() -> Self {
        Self::new(100 * 1024 * 1024) // 100 MB/s default
    }
}

impl IoBudgetManager {
    pub fn new(max_bytes_per_sec: u64) -> Self {
        Self {
            max_bandwidth_bytes_per_sec: AtomicU64::new(max_bytes_per_sec),
            available_tokens: AtomicU64::new(max_bytes_per_sec),
            last_refill: Mutex::new(Instant::now()),
            is_throttled: AtomicBool::new(false),
            p95_latency_threshold_micros: 5000, // 5ms throttle trigger
            p95_recovery_threshold_micros: 2000, // 2ms recovery with hysteresis
        }
    }

    /// Feeds foreground p95 query latency to trigger self-throttling with hysteresis.
    pub fn report_foreground_latency(&self, p95_micros: u64) {
        if p95_micros > self.p95_latency_threshold_micros {
            self.is_throttled.store(true, Ordering::Release);
        } else if p95_micros < self.p95_recovery_threshold_micros {
            self.is_throttled.store(false, Ordering::Release);
        }
    }

    /// Requests permit to write/read `bytes` for a background maintenance task.
    pub fn acquire_maintenance_budget(&self, bytes: u64) -> u64 {
        if self.is_throttled.load(Ordering::Relaxed) {
            // Throttled: scale down permit to 10% rate
            return (bytes / 10).max(1);
        }

        let mut lock = self.last_refill.lock();
        let elapsed = lock.elapsed().as_secs_f64();
        if elapsed >= 0.1 {
            let max_bw = self.max_bandwidth_bytes_per_sec.load(Ordering::Relaxed);
            let refill = (elapsed * max_bw as f64) as u64;
            self.available_tokens.fetch_add(refill, Ordering::Relaxed);
            *lock = Instant::now();
        }

        let available = self.available_tokens.load(Ordering::Relaxed);
        let granted = bytes.min(available).max(1);
        self.available_tokens.fetch_sub(granted, Ordering::Relaxed);
        granted
    }
}
