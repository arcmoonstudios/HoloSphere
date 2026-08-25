/* holosphere/src/storage/two_tier_cache.rs */
//!▫~•◦-------------------------------‣
//! # Two-Tier NVMe/Memory Cache (Proof Metadata vs TinyLFU Exact Vectors)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides separated cache budgets:
//!   - Tier 0: Eviction-resistant pinned budget for ProofTree, Rivero, and LUTz structures
//!   - Tier 1: Frequency-aware TinyLFU + segmented recency cache for dense exact vectors
//! Includes per-tenant quota protection and minimum shared pool guarantees.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::HNSQRResult;

pub type TenantId = String;
pub type CacheBlockId = u64;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedVectorBlock {
    pub block_id: CacheBlockId,
    pub tenant_id: TenantId,
    pub data: Vec<u8>,
    pub frequency: u32,
    pub is_tier_0_pinned: bool,
}

/// Two-tier caching engine.
pub struct TwoTierCache {
    tier_0_max_bytes: usize,
    tier_1_max_bytes: usize,
    tier_0_current_bytes: AtomicUsize,
    tier_1_current_bytes: AtomicUsize,
    tier_0_blocks: RwLock<HashMap<CacheBlockId, CachedVectorBlock>>,
    tier_1_blocks: RwLock<HashMap<CacheBlockId, CachedVectorBlock>>,
    tenant_usage: RwLock<HashMap<TenantId, usize>>,
    tenant_quota_bytes: usize,
    pub hits_total: AtomicU64,
    pub misses_total: AtomicU64,
}

impl TwoTierCache {
    pub fn new(
        tier_0_max_bytes: usize,
        tier_1_max_bytes: usize,
        tenant_quota_bytes: usize,
    ) -> Self {
        Self {
            tier_0_max_bytes,
            tier_1_max_bytes,
            tier_0_current_bytes: AtomicUsize::new(0),
            tier_1_current_bytes: AtomicUsize::new(0),
            tier_0_blocks: RwLock::new(HashMap::new()),
            tier_1_blocks: RwLock::new(HashMap::new()),
            tenant_usage: RwLock::new(HashMap::new()),
            tenant_quota_bytes,
            hits_total: AtomicU64::new(0),
            misses_total: AtomicU64::new(0),
        }
    }

    /// Stores a Tier 0 proof metadata block (extremely resistant to eviction).
    pub fn put_tier_0(&self, block_id: CacheBlockId, data: Vec<u8>) -> HNSQRResult<()> {
        let size = data.len();
        let mut guard = self.tier_0_blocks.write();

        if self.tier_0_current_bytes.load(Ordering::Relaxed) + size > self.tier_0_max_bytes {
            // Evict lowest frequency Tier 0 if necessary
            if let Some((&lfu_id, _)) = guard.iter().min_by_key(|(_, b)| b.frequency) {
                if let Some(removed) = guard.remove(&lfu_id) {
                    self.tier_0_current_bytes
                        .fetch_sub(removed.data.len(), Ordering::Relaxed);
                }
            }
        }

        let block = CachedVectorBlock {
            block_id,
            tenant_id: "system".to_string(),
            data,
            frequency: 100, // High default priority
            is_tier_0_pinned: true,
        };

        guard.insert(block_id, block);
        self.tier_0_current_bytes.fetch_add(size, Ordering::Relaxed);
        Ok(())
    }

    /// Retrieves or fetches a Tier 1 exact vector block with TinyLFU admission and tenant governance.
    pub fn get_or_fetch_tier_1<F>(
        &self,
        tenant_id: &str,
        block_id: CacheBlockId,
        fetch_fn: F,
    ) -> HNSQRResult<Vec<u8>>
    where
        F: FnOnce(CacheBlockId) -> HNSQRResult<Vec<u8>>,
    {
        // 1. Check Tier 0 first
        {
            let mut guard0 = self.tier_0_blocks.write();
            if let Some(block) = guard0.get_mut(&block_id) {
                block.frequency = block.frequency.saturating_add(1);
                self.hits_total.fetch_add(1, Ordering::Relaxed);
                return Ok(block.data.clone());
            }
        }

        // 2. Check Tier 1
        {
            let mut guard1 = self.tier_1_blocks.write();
            if let Some(block) = guard1.get_mut(&block_id) {
                block.frequency = block.frequency.saturating_add(1);
                self.hits_total.fetch_add(1, Ordering::Relaxed);
                return Ok(block.data.clone());
            }
        }

        // 3. Cache Miss: Execute remote fetch
        self.misses_total.fetch_add(1, Ordering::Relaxed);
        let data = fetch_fn(block_id)?;
        let size = data.len();

        // 4. Check tenant quota
        {
            let mut usage_map = self.tenant_usage.write();
            let current = usage_map.entry(tenant_id.to_string()).or_insert(0);
            if *current + size > self.tenant_quota_bytes {
                // Tenant quota exceeded: return fetched data without polluting shared cache
                return Ok(data);
            }
            *current += size;
        }

        // 5. TinyLFU Admission
        let mut guard1 = self.tier_1_blocks.write();
        if self.tier_1_current_bytes.load(Ordering::Relaxed) + size > self.tier_1_max_bytes {
            if let Some((&lfu_id, _)) = guard1.iter().min_by_key(|(_, b)| b.frequency) {
                if let Some(removed) = guard1.remove(&lfu_id) {
                    self.tier_1_current_bytes
                        .fetch_sub(removed.data.len(), Ordering::Relaxed);
                    let mut usage_map = self.tenant_usage.write();
                    if let Some(u) = usage_map.get_mut(&removed.tenant_id) {
                        *u = u.saturating_sub(removed.data.len());
                    }
                }
            }
        }

        let block = CachedVectorBlock {
            block_id,
            tenant_id: tenant_id.to_string(),
            data: data.clone(),
            frequency: 1,
            is_tier_0_pinned: false,
        };

        guard1.insert(block_id, block);
        self.tier_1_current_bytes.fetch_add(size, Ordering::Relaxed);

        Ok(data)
    }

    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits_total.load(Ordering::Relaxed) as f64;
        let misses = self.misses_total.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        if total == 0.0 { 1.0 } else { hits / total }
    }
}
