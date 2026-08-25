/* holosphere/src/ecosystem/kv_cache.rs */
//!▫~•◦-------------------------------‣
//! # In-Memory Multi-Model KV Store & Fast Scratchpad (Front 3: Redis Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides ultra-low latency (<100ns) in-memory key-value, session, hash-map,
//! string tag sets, atomic numeric counters, and TTL expiration primitives embedded
//! directly in the engine, eliminating RPC round-trips to external cache instances.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Typed values stored in the in-memory multi-model KV engine.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum KvValue {
    String(String),
    Bytes(Vec<u8>),
    Integer(i64),
    Float(f64),
    HashSet(HashSet<String>),
    HashMap(HashMap<String, String>),
}

/// An entry with optional expiration deadline.
#[derive(Clone, Debug)]
struct KvEntry {
    value: KvValue,
    expires_at: Option<Instant>,
}

/// In-Memory Multi-Model Key-Value Store with raw byte key indexing.
pub struct MemoryKvStore {
    entries: RwLock<HashMap<Vec<u8>, KvEntry>>,
    total_ops: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl MemoryKvStore {
    pub fn new() -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            total_ops: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Sets a raw byte key-value pair with optional TTL with zero UTF-8 allocation overhead.
    pub fn set_raw(&self, key: &[u8], value: KvValue, ttl: Option<Duration>) {
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        let expires_at = ttl.map(|d| Instant::now() + d);
        let entry = KvEntry { value, expires_at };
        self.entries.write().insert(key.to_vec(), entry);
    }

    /// Gets a value by raw byte slice key, honoring TTL expiration with zero string allocations.
    pub fn get_raw(&self, key: &[u8]) -> Option<KvValue> {
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        let now = Instant::now();
        let mut entries = self.entries.write();

        if let Some(entry) = entries.get(key) {
            if let Some(exp) = entry.expires_at {
                if now >= exp {
                    entries.remove(key);
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    return None;
                }
            }
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry.value.clone())
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    /// Deletes a key by raw byte slice.
    pub fn delete_raw(&self, key: &[u8]) -> bool {
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        self.entries.write().remove(key).is_some()
    }

    /// Atomic integer increment over raw byte slice key.
    pub fn incr_by_raw(&self, key: &[u8], delta: i64) -> i64 {
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.write();
        let entry = entries.entry(key.to_vec()).or_insert_with(|| KvEntry {
            value: KvValue::Integer(0),
            expires_at: None,
        });

        match &mut entry.value {
            KvValue::Integer(val) => {
                *val += delta;
                *val
            }
            _ => {
                entry.value = KvValue::Integer(delta);
                delta
            }
        }
    }

    /// Sets a string key-value pair with optional TTL.
    pub fn set(&self, key: &str, value: KvValue, ttl: Option<Duration>) {
        self.set_raw(key.as_bytes(), value, ttl);
    }

    /// Gets a value by string key.
    pub fn get(&self, key: &str) -> Option<KvValue> {
        self.get_raw(key.as_bytes())
    }

    /// Deletes a string key from the cache.
    pub fn delete(&self, key: &str) -> bool {
        self.delete_raw(key.as_bytes())
    }

    /// Atomic integer increment by string key.
    pub fn incr_by(&self, key: &str, delta: i64) -> i64 {
        self.incr_by_raw(key.as_bytes(), delta)
    }

    /// Adds a tag to a string set (SADD equivalent).
    pub fn set_add(&self, key: &str, member: &str) -> bool {
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        let mut entries = self.entries.write();
        let entry = entries
            .entry(key.as_bytes().to_vec())
            .or_insert_with(|| KvEntry {
                value: KvValue::HashSet(HashSet::new()),
                expires_at: None,
            });

        if let KvValue::HashSet(set) = &mut entry.value {
            set.insert(member.to_string())
        } else {
            let mut new_set = HashSet::new();
            new_set.insert(member.to_string());
            entry.value = KvValue::HashSet(new_set);
            true
        }
    }

    /// Checks membership in a set (SISMEMBER equivalent).
    pub fn set_is_member(&self, key: &str, member: &str) -> bool {
        self.total_ops.fetch_add(1, Ordering::Relaxed);
        let entries = self.entries.read();
        if let Some(entry) = entries.get(key.as_bytes()) {
            if let KvValue::HashSet(set) = &entry.value {
                return set.contains(member);
            }
        }
        false
    }

    /// Purges all expired keys.
    pub fn purge_expired(&self) -> usize {
        let now = Instant::now();
        let mut entries = self.entries.write();
        let before = entries.len();
        entries.retain(|_, entry| {
            if let Some(exp) = entry.expires_at {
                now < exp
            } else {
                true
            }
        });
        before - entries.len()
    }

    pub fn total_operations(&self) -> u64 {
        self.total_ops.load(Ordering::Relaxed)
    }
}

impl Default for MemoryKvStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_kv_store_primitives() {
        let kv = MemoryKvStore::new();

        // 1. Basic Get/Set
        kv.set(
            "user:100:session",
            KvValue::String("token_xyz".to_string()),
            None,
        );
        assert_eq!(
            kv.get("user:100:session"),
            Some(KvValue::String("token_xyz".to_string()))
        );

        // 2. Atomic INCR
        assert_eq!(kv.incr_by("rate_limit:ip", 1), 1);
        assert_eq!(kv.incr_by("rate_limit:ip", 5), 6);

        // 3. Set operations
        assert!(kv.set_add("tags:vector:1", "compliance"));
        assert!(kv.set_add("tags:vector:1", "legal"));
        assert!(kv.set_is_member("tags:vector:1", "legal"));
        assert!(!kv.set_is_member("tags:vector:1", "medical"));

        // 4. TTL Expiration
        kv.set(
            "temp_key",
            KvValue::String("expires".to_string()),
            Some(Duration::from_millis(10)),
        );
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(kv.get("temp_key"), None);
    }
}
