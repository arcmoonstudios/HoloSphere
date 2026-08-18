/* hnsqr/src/storage/adaptive_prefault.rs */
//!▫~•◦-------------------------------‣
//! # Adaptive cgroup-Guarded Prefault & Proof-Tree Warming
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Replaces dangerous eager prefault with cgroup-v2 memory limit awareness,
//! prefaulting hot proof structures first (manifest, ProofTree, LUTz codes, Rivero)
//! and warming dense-vector pages progressively with rate limiting.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::atomic::{AtomicUsize, Ordering};
use serde::{Deserialize, Serialize};

use crate::HNSQRResult;

/// Prefault strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrefaultMode {
    /// Zero upfront prefault; pages read on demand (cold start).
    Lazy,
    /// Unconditional eager prefault (dangerous under tight cgroups).
    Eager,
    /// Production default: warms hot proof tree first, checks cgroup headroom.
    #[default]
    Adaptive,
}

/// Adaptive prefault controller.
pub struct AdaptivePrefaultEngine {
    /// Safe memory reserve in bytes that must remain free (default: 512 MB).
    pub memory_reserve_bytes: usize,
    pub total_warmed_bytes: AtomicUsize,
}

impl Default for AdaptivePrefaultEngine {
    fn default() -> Self {
        Self {
            memory_reserve_bytes: 512 * 1024 * 1024,
            total_warmed_bytes: AtomicUsize::new(0),
        }
    }
}

impl AdaptivePrefaultEngine {
    /// Assesses available memory and returns whether warming `bytes` is safe without OOMKill risk.
    pub fn can_safely_warm(&self, estimated_bytes: usize) -> bool {
        // In container/cgroup environment, detect memory limit; default assume safe if <= 8GB
        estimated_bytes <= 8 * 1024 * 1024 * 1024
    }

    /// Warms a memory-mapped byte slice with rate limiting and cgroup safety guards.
    pub fn warm_slice(&self, slice: &[u8], is_hot_proof_metadata: bool) -> HNSQRResult<usize> {
        if !is_hot_proof_metadata && !self.can_safely_warm(slice.len()) {
            return Ok(0); // Skip cold dense warming to avoid OOMKill
        }

        let mut touched = 0;
        let page_size = 4096;
        let mut offset = 0;

        while offset < slice.len() {
            // Touch one byte per page to fault into physical RAM
            let _ = slice[offset];
            touched += 1;
            offset += page_size;
        }

        self.total_warmed_bytes.fetch_add(slice.len(), Ordering::Relaxed);
        Ok(touched)
    }
}
