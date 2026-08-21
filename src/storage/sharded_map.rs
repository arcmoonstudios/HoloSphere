/* hnsqr/src/storage/sharded_map.rs */
//!▫~•◦-------------------------------‣
//! # Striped Lock-Free Concurrent Hash Map for Batch Ingestion
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a 64-way striped concurrent hash map eliminating coarse write lock
//! serialization on hot index lookups (`id_to_index`, `lutz_codes`).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

const NUM_SHARDS: usize = 64;

/// A 64-way striped concurrent hash map.
pub struct ShardedConcurrentMap<K, V> {
    shards: Vec<RwLock<HashMap<K, V>>>,
    count: AtomicUsize,
}

impl<K: Hash + Eq + Clone, V: Clone> Default for ShardedConcurrentMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone, V: Clone> ShardedConcurrentMap<K, V> {
    /// Creates a new 64-way striped concurrent map.
    pub fn new() -> Self {
        let mut shards = Vec::with_capacity(NUM_SHARDS);
        for _ in 0..NUM_SHARDS {
            shards.push(RwLock::new(HashMap::new()));
        }
        Self {
            shards,
            count: AtomicUsize::new(0),
        }
    }

    #[inline]
    fn shard_index(&self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() as usize) % NUM_SHARDS
    }

    /// Gets a cloned value for a given key.
    pub fn get(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let guard = self.shards[idx].read();
        guard.get(key).cloned()
    }

    /// Checks whether the key exists.
    pub fn contains_key(&self, key: &K) -> bool {
        let idx = self.shard_index(key);
        let guard = self.shards[idx].read();
        guard.contains_key(key)
    }

    /// Inserts a key-value pair, returning the old value if any.
    pub fn insert(&self, key: K, value: V) -> Option<V> {
        let idx = self.shard_index(&key);
        let mut guard = self.shards[idx].write();
        let prev = guard.insert(key, value);
        if prev.is_none() {
            self.count.fetch_add(1, Ordering::Relaxed);
        }
        prev
    }

    /// Removes a key from the map.
    pub fn remove(&self, key: &K) -> Option<V> {
        let idx = self.shard_index(key);
        let mut guard = self.shards[idx].write();
        let prev = guard.remove(key);
        if prev.is_some() {
            self.count.fetch_sub(1, Ordering::Relaxed);
        }
        prev
    }

    /// Returns the total number of items stored.
    pub fn len(&self) -> usize {
        self.count.load(Ordering::Relaxed)
    }

    /// Checks if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clears all shards.
    pub fn clear(&self) {
        for shard in &self.shards {
            let mut guard = shard.write();
            guard.clear();
        }
        self.count.store(0, Ordering::Relaxed);
    }

    /// Iterates over all key-value entries across all shards.
    pub fn to_vec(&self) -> Vec<(K, V)> {
        let mut entries = Vec::with_capacity(self.len());
        for shard in &self.shards {
            let guard = shard.read();
            for (k, v) in guard.iter() {
                entries.push((k.clone(), v.clone()));
            }
        }
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_sharded_concurrent_map_crud_and_concurrency() {
        let map = Arc::new(ShardedConcurrentMap::new());

        let mut handles = Vec::new();
        for t in 0..8 {
            let m = map.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    let key = format!("thread_{t}_key_{i}");
                    m.insert(key.clone(), i);
                    assert_eq!(m.get(&key), Some(i));
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(map.len(), 8000);
        assert!(map.contains_key(&"thread_0_key_42".to_string()));
        assert_eq!(map.remove(&"thread_0_key_42".to_string()), Some(42));
        assert_eq!(map.len(), 7999);
    }
}
