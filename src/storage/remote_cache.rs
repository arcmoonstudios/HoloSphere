/* holosphere/src/storage/remote_cache.rs */
//!▫~•◦-------------------------------‣
//! # S3 / Blob Disaggregation & TinyLFU NVMe Range Cache
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Stores dense exact-vector blocks remotely while caching hot range blocks
//! locally using frequency-aware TinyLFU admission to eliminate cache thrashing.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use crate::HNSQRResult;

pub type ChunkId = u64;

/// Cached remote vector chunk descriptor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CachedChunk {
    pub chunk_id: ChunkId,
    pub data: Vec<u8>,
    pub access_frequency: u32,
}

/// TinyLFU frequency-aware local NVMe range cache for remote immutable segments.
pub struct RemoteRangeCache {
    pub max_capacity_bytes: usize,
    pub current_bytes: AtomicUsize,
    chunks: RwLock<HashMap<ChunkId, CachedChunk>>,
    pub remote_fetches_total: AtomicU64,
    pub cache_hits_total: AtomicU64,
}

impl RemoteRangeCache {
    pub fn new(max_capacity_bytes: usize) -> Self {
        Self {
            max_capacity_bytes,
            current_bytes: AtomicUsize::new(0),
            chunks: RwLock::new(HashMap::new()),
            remote_fetches_total: AtomicU64::new(0),
            cache_hits_total: AtomicU64::new(0),
        }
    }

    /// Fetches a chunk from local cache if present, or fetches remotely.
    pub fn get_or_fetch<F>(&self, chunk_id: ChunkId, fetch_fn: F) -> HNSQRResult<Vec<u8>>
    where
        F: FnOnce(ChunkId) -> HNSQRResult<Vec<u8>>,
    {
        {
            let mut guard = self.chunks.write();
            if let Some(chunk) = guard.get_mut(&chunk_id) {
                chunk.access_frequency = chunk.access_frequency.saturating_add(1);
                self.cache_hits_total.fetch_add(1, Ordering::Relaxed);
                return Ok(chunk.data.clone());
            }
        }

        // Fetch remote chunk
        self.remote_fetches_total.fetch_add(1, Ordering::Relaxed);
        let fetched_data = fetch_fn(chunk_id)?;
        let size = fetched_data.len();

        // TinyLFU Admission & Eviction
        let mut guard = self.chunks.write();
        if self.current_bytes.load(Ordering::Relaxed) + size > self.max_capacity_bytes {
            // Evict least frequently used chunk
            if let Some((&lfu_id, _)) = guard.iter().min_by_key(|(_, c)| c.access_frequency) {
                if let Some(removed) = guard.remove(&lfu_id) {
                    self.current_bytes
                        .fetch_sub(removed.data.len(), Ordering::Relaxed);
                }
            }
        }

        let chunk = CachedChunk {
            chunk_id,
            data: fetched_data.clone(),
            access_frequency: 1,
        };

        guard.insert(chunk_id, chunk);
        self.current_bytes.fetch_add(size, Ordering::Relaxed);

        Ok(fetched_data)
    }
}
