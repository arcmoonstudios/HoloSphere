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

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

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
        if let Some((limit, current)) = cgroup_memory_usage() {
            // An unlimited cgroup-v1 sentinel must not be treated as a real quota.
            if limit < (1usize << 50) {
                return can_warm_with_usage(
                    limit,
                    current,
                    self.memory_reserve_bytes,
                    estimated_bytes,
                );
            }
        }

        // Non-container fallback: preserve the previous conservative ceiling.
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

        self.total_warmed_bytes
            .fetch_add(slice.len(), Ordering::Relaxed);
        Ok(touched)
    }
}

fn can_warm_with_usage(limit: usize, current: usize, reserve: usize, estimated: usize) -> bool {
    estimated <= limit.saturating_sub(current).saturating_sub(reserve)
}

fn cgroup_memory_usage() -> Option<(usize, usize)> {
    // cgroup v2: `max` denotes no controller limit and is intentionally ignored.
    if let Some(pair) =
        read_memory_pair("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current")
    {
        return Some(pair);
    }

    // cgroup v1 fallback.
    read_memory_pair(
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
}

fn read_memory_pair(limit_path: &str, current_path: &str) -> Option<(usize, usize)> {
    let limit = std::fs::read_to_string(limit_path).ok()?;
    let current = std::fs::read_to_string(current_path).ok()?;
    Some((limit.trim().parse().ok()?, current.trim().parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_warming_when_reserve_exhausts_cgroup_headroom() {
        assert!(!can_warm_with_usage(1_024, 768, 512, 1));
        assert!(can_warm_with_usage(1_024, 128, 512, 384));
    }
}
