/* hnsqr/src/cluster/ring.rs */
//!▫~•◦-------------------------------‣
//! # Consistent Hash Ring with Virtual Nodes
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

pub type ShardId = u32;

/// Consistent Hash Ring with Virtual Nodes for minimal data movement during scaling ($1/N$).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConsistentHashRing {
    /// Number of virtual nodes per physical shard (default: 256)
    pub vnodes_per_shard: u32,
    /// Sorted ring of `(hash_token, shard_id)`
    ring: Vec<(u64, ShardId)>,
}

impl ConsistentHashRing {
    pub const DEFAULT_VNODES_PER_SHARD: u32 = 256;

    /// Creates an empty consistent hash ring.
    pub fn new(vnodes_per_shard: u32) -> Self {
        Self {
            vnodes_per_shard: vnodes_per_shard.max(1),
            ring: Vec::new(),
        }
    }

    /// Fast 64-bit FNV-1a hash function with SplitMix64 avalanche for uniform ring token distribution.
    #[inline(always)]
    pub fn hash_key(key: &[u8]) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        for &byte in key {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        // SplitMix64 bit avalanche
        let mut z = hash.wrapping_add(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    /// Adds a physical shard to the ring by placing `vnodes_per_shard` deterministic tokens.
    pub fn add_shard(&mut self, shard_id: ShardId) {
        for v in 0..self.vnodes_per_shard {
            let vnode_key = format!("shard_{shard_id}_vnode_{v}");
            let token = Self::hash_key(vnode_key.as_bytes());
            self.ring.push((token, shard_id));
        }
        self.ring.sort_unstable_by_key(|&(token, _)| token);
    }

    /// Removes a physical shard from the ring.
    pub fn remove_shard(&mut self, shard_id: ShardId) {
        self.ring.retain(|&(_, s)| s != shard_id);
    }

    /// Maps an arbitrary string key to its target shard via binary search on the ring.
    #[inline(always)]
    pub fn shard_for_key(&self, key: &str) -> Option<ShardId> {
        if self.ring.is_empty() {
            return None;
        }
        let token = Self::hash_key(key.as_bytes());
        match self.ring.binary_search_by_key(&token, |&(t, _)| t) {
            Ok(idx) => Some(self.ring[idx].1),
            Err(idx) => {
                if idx < self.ring.len() {
                    Some(self.ring[idx].1)
                } else {
                    Some(self.ring[0].1) // Wrap around ring
                }
            }
        }
    }
}
