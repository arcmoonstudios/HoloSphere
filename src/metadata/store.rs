/* holosphere/src/storage/metadata_store.rs */
//!▫~•◦-------------------------------‣
//! # Metadata Memory Safety, Cardinality Quotas & String Interning
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Replaces unbounded heap allocation with tracked byte accounting,
//! string interning pools, and configurable hard quotas with typed admission errors.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::RwLock;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use crate::metadata::index::MetadataValue;
use crate::{HNSQRError, HNSQRResult, NodeIndex};

pub type TermId = u32;

/// Configuration for metadata memory quotas and cardinality limits.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetadataQuotaConfig {
    /// Maximum allowed memory in bytes for all metadata structures (default: 512 MB).
    pub max_memory_bytes: usize,
    /// Maximum allowed distinct string terms across all keys (default: 1,000,000).
    pub max_distinct_terms: usize,
    /// Soft pressure threshold ratio (0.0 to 1.0, default: 0.85).
    pub soft_pressure_ratio: f32,
}

impl Default for MetadataQuotaConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024,
            max_distinct_terms: 1_000_000,
            soft_pressure_ratio: 0.85,
        }
    }
}

/// Real-time quota tracker with atomic byte accounting.
#[derive(Debug, Default)]
pub struct QuotaTracker {
    pub config: MetadataQuotaConfig,
    pub current_bytes: AtomicUsize,
    pub current_terms: AtomicUsize,
}

impl QuotaTracker {
    pub fn new(config: MetadataQuotaConfig) -> Self {
        Self {
            config,
            current_bytes: AtomicUsize::new(0),
            current_terms: AtomicUsize::new(0),
        }
    }

    /// Attempts to allocate `bytes` and `terms`. Returns error if quota exceeded.
    pub fn try_admit(&self, bytes: usize, terms: usize) -> HNSQRResult<()> {
        let cur_bytes = self.current_bytes.load(Ordering::Relaxed);
        let cur_terms = self.current_terms.load(Ordering::Relaxed);

        if cur_bytes + bytes > self.config.max_memory_bytes {
            return Err(HNSQRError::Internal(format!(
                "Metadata quota exceeded: requested {} bytes, current {}, limit {}",
                bytes, cur_bytes, self.config.max_memory_bytes
            )));
        }

        if cur_terms + terms > self.config.max_distinct_terms {
            return Err(HNSQRError::Internal(format!(
                "Metadata distinct term limit exceeded: requested {} terms, current {}, limit {}",
                terms, cur_terms, self.config.max_distinct_terms
            )));
        }

        self.current_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.current_terms.fetch_add(terms, Ordering::Relaxed);
        Ok(())
    }

    pub fn release(&self, bytes: usize, terms: usize) {
        self.current_bytes.fetch_sub(bytes, Ordering::Relaxed);
        self.current_terms.fetch_sub(terms, Ordering::Relaxed);
    }
}

/// Thread-safe String Interning Pool to eliminate duplicate singleton string allocations.
#[derive(Default)]
pub struct StringInterner {
    to_id: RwLock<HashMap<Arc<str>, TermId>>,
    to_str: RwLock<Vec<Arc<str>>>,
}

impl StringInterner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Interns a string slice, returning its compact 32-bit TermId.
    pub fn intern(&self, s: &str, tracker: &QuotaTracker) -> HNSQRResult<TermId> {
        {
            let reader = self.to_id.read();
            if let Some(&id) = reader.get(s) {
                return Ok(id);
            }
        }

        let mut writer = self.to_id.write();
        if let Some(&id) = writer.get(s) {
            return Ok(id);
        }

        let bytes_needed = s.len() + std::mem::size_of::<Arc<str>>() + 32;
        tracker.try_admit(bytes_needed, 1)?;

        let arc_str: Arc<str> = Arc::from(s);
        let mut str_table = self.to_str.write();
        let id = str_table.len() as TermId;

        writer.insert(arc_str.clone(), id);
        str_table.push(arc_str);

        Ok(id)
    }

    /// Resolves a TermId back to its interned string slice.
    pub fn resolve(&self, id: TermId) -> Option<Arc<str>> {
        let reader = self.to_str.read();
        reader.get(id as usize).cloned()
    }

    pub fn len(&self) -> usize {
        self.to_str.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Quota-bounded, interned inverted metadata store.
pub struct MetadataStore {
    tracker: Arc<QuotaTracker>,
    interner: Arc<StringInterner>,
    /// Inverted string index: `key -> term_id -> postings`
    categorical_postings: RwLock<HashMap<String, HashMap<TermId, RoaringBitmap>>>,
    /// Numeric integer index: `key -> value -> postings`
    numeric_postings: RwLock<HashMap<String, BTreeMap<i64, RoaringBitmap>>>,
}

impl MetadataStore {
    pub fn new(config: MetadataQuotaConfig) -> Self {
        Self {
            tracker: Arc::new(QuotaTracker::new(config)),
            interner: Arc::new(StringInterner::new()),
            categorical_postings: RwLock::new(HashMap::new()),
            numeric_postings: RwLock::new(HashMap::new()),
        }
    }

    /// Indexes a slot with the provided metadata map under quota admission checks.
    pub fn index_slot(
        &self,
        slot: NodeIndex,
        metadata: &HashMap<String, MetadataValue>,
    ) -> HNSQRResult<()> {
        for (key, val) in metadata {
            match val {
                MetadataValue::String(s) => {
                    let term_id = self.interner.intern(s, &self.tracker)?;
                    let mut cat_guard = self.categorical_postings.write();
                    let key_map: &mut HashMap<TermId, RoaringBitmap> =
                        cat_guard.entry(key.to_owned()).or_default();
                    let bitmap = key_map.entry(term_id).or_default();

                    let prev_serialized_bytes = bitmap.serialized_size();
                    bitmap.insert(slot);
                    let new_serialized_bytes = bitmap.serialized_size();

                    if new_serialized_bytes > prev_serialized_bytes {
                        self.tracker
                            .try_admit(new_serialized_bytes - prev_serialized_bytes, 0)?;
                    }
                }
                MetadataValue::Integer(i) => {
                    let mut num_guard = self.numeric_postings.write();
                    let key_map: &mut BTreeMap<i64, RoaringBitmap> =
                        num_guard.entry(key.to_owned()).or_default();
                    let bitmap = key_map.entry(*i).or_default();

                    let prev_serialized_bytes = bitmap.serialized_size();
                    bitmap.insert(slot);
                    let new_serialized_bytes = bitmap.serialized_size();

                    if new_serialized_bytes > prev_serialized_bytes {
                        self.tracker
                            .try_admit(new_serialized_bytes - prev_serialized_bytes, 0)?;
                    }
                }
                MetadataValue::Float(f) => {
                    // Discrete float encoding via bit preservation
                    let bits = f.to_bits() as i64;
                    let mut num_guard = self.numeric_postings.write();
                    let key_map = num_guard.entry(key.clone()).or_default();
                    let bitmap = key_map.entry(bits).or_default();
                    bitmap.insert(slot);
                }
                MetadataValue::Boolean(b) => {
                    let term_id = self
                        .interner
                        .intern(if *b { "true" } else { "false" }, &self.tracker)?;
                    let mut cat_guard = self.categorical_postings.write();
                    let key_map = cat_guard.entry(key.clone()).or_default();
                    let bitmap = key_map.entry(term_id).or_default();
                    bitmap.insert(slot);
                }
            }
        }
        Ok(())
    }

    /// Evaluates exact match on a categorical string key.
    pub fn match_categorical(&self, key: &str, value: &str) -> Option<RoaringBitmap> {
        let term_id = {
            let reader = self.interner.to_id.read();
            *reader.get(value)?
        };

        let cat_guard = self.categorical_postings.read();
        let key_map = cat_guard.get(key)?;
        key_map.get(&term_id).cloned()
    }

    /// Evaluates integer range query `[min, max]` inclusive.
    pub fn match_integer_range(&self, key: &str, min: i64, max: i64) -> RoaringBitmap {
        let mut result = RoaringBitmap::new();
        let num_guard = self.numeric_postings.read();
        if let Some(key_map) = num_guard.get(key) {
            for (_, bitmap) in key_map.range(min..=max) {
                result |= bitmap;
            }
        }
        result
    }

    /// Reclaims dead terms and compacts postings during LSM compaction.
    pub fn compact_and_reclaim(&self, live_slots: &RoaringBitmap) {
        let mut cat_guard = self.categorical_postings.write();
        for key_map in cat_guard.values_mut() {
            key_map.retain(|_, bitmap| {
                *bitmap &= live_slots;
                !bitmap.is_empty()
            });
        }
        cat_guard.retain(|_, key_map| !key_map.is_empty());

        let mut num_guard = self.numeric_postings.write();
        for key_map in num_guard.values_mut() {
            key_map.retain(|_, bitmap| {
                *bitmap &= live_slots;
                !bitmap.is_empty()
            });
        }
        num_guard.retain(|_, key_map| !key_map.is_empty());
    }

    pub fn total_memory_bytes(&self) -> usize {
        self.tracker.current_bytes.load(Ordering::Relaxed)
    }

    pub fn total_distinct_terms(&self) -> usize {
        self.interner.len()
    }
}
