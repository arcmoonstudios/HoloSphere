/* holosphere/src/storage/backpressure.rs */
//!▫~•◦-------------------------------‣
//! # Bounded Ingestion Queues, Load-Shedding & Backpressure
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Protects against out-of-memory cascading failures, unbounded queue growth,
//! and low-disk exhaustion by enforcing deterministic backpressure rejection
//! and automated read-only fail-safe mode.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::{HNSQRError, HNSQRResult};

/// Configuration for backpressure and queue bounding.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BackpressureConfig {
    /// Maximum concurrent in-flight mutation operations allowed (default: 10,000).
    pub max_inflight_mutations: usize,
    /// Minimum required free disk headroom in bytes before triggering read-only mode (default: 1 GB).
    pub min_disk_headroom_bytes: u64,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            max_inflight_mutations: 10_000,
            min_disk_headroom_bytes: 1024 * 1024 * 1024,
        }
    }
}

/// Backpressure admission controller.
#[derive(Debug)]
pub struct BackpressureController {
    config: BackpressureConfig,
    current_inflight: AtomicUsize,
    read_only_mode: AtomicBool,
}

impl BackpressureController {
    pub fn new(config: BackpressureConfig) -> Self {
        Self {
            config,
            current_inflight: AtomicUsize::new(0),
            read_only_mode: AtomicBool::new(false),
        }
    }

    /// Attempts to admit a new mutation under capacity limits.
    pub fn try_admit_mutation(&self) -> HNSQRResult<MutationPermit<'_>> {
        if self.read_only_mode.load(Ordering::Relaxed) {
            return Err(HNSQRError::Internal(
                "Engine is in read-only fail-safe mode due to storage resource pressure"
                    .to_string(),
            ));
        }

        let cur = self.current_inflight.load(Ordering::Relaxed);
        if cur >= self.config.max_inflight_mutations {
            return Err(HNSQRError::Internal(format!(
                "Backpressure limit reached: {} active in-flight mutations",
                cur
            )));
        }

        self.current_inflight.fetch_add(1, Ordering::Relaxed);
        Ok(MutationPermit { controller: self })
    }

    /// Sets or clears read-only emergency state.
    pub fn set_read_only(&self, enable: bool) {
        self.read_only_mode.store(enable, Ordering::Release);
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only_mode.load(Ordering::Acquire)
    }

    pub fn inflight_count(&self) -> usize {
        self.current_inflight.load(Ordering::Relaxed)
    }
}

/// RAII Permit releasing in-flight slot upon drop.
pub struct MutationPermit<'a> {
    controller: &'a BackpressureController,
}

impl<'a> Drop for MutationPermit<'a> {
    fn drop(&mut self) {
        self.controller
            .current_inflight
            .fetch_sub(1, Ordering::Relaxed);
    }
}
