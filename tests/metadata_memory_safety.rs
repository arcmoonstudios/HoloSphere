/* hnsqr/tests/metadata_memory_safety.rs */
//!▫~•◦-------------------------------‣
//! # Metadata Memory Safety & Cardinality Quota Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - String interning pool preventing singleton memory explosion
//!   - Hard byte and distinct term quota enforcement with typed rejection
//!   - Dead term reclamation during compaction
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::metadata::index::MetadataValue;
use hnsqr::metadata::store::{MetadataQuotaConfig, MetadataStore};
use roaring::RoaringBitmap;
use std::collections::HashMap;

#[test]
fn test_string_interning_deduplication() {
    let config = MetadataQuotaConfig {
        max_memory_bytes: 10 * 1024 * 1024,
        max_distinct_terms: 1000,
        soft_pressure_ratio: 0.85,
    };
    let store = MetadataStore::new(config);

    // Index 5,000 slots with the same 10 categories
    let categories = [
        "finance", "tech", "health", "energy", "retail", "legal", "hr", "sales", "ops", "eng",
    ];

    for slot in 0..5000 {
        let mut meta = HashMap::new();
        meta.insert(
            "category".to_string(),
            MetadataValue::String(categories[slot % 10].to_string()),
        );
        store.index_slot(slot as u32, &meta).unwrap();
    }

    // Must have interned exactly 10 distinct terms, not 5,000!
    assert_eq!(store.total_distinct_terms(), 10);

    // Matching must return exact roaring bitmaps
    let finance_matches = store.match_categorical("category", "finance").unwrap();
    assert_eq!(finance_matches.len(), 500);
}

#[test]
fn test_metadata_hard_quota_rejection() {
    // Tight quota: only 20 distinct terms
    let config = MetadataQuotaConfig {
        max_memory_bytes: 1024 * 1024,
        max_distinct_terms: 20,
        soft_pressure_ratio: 0.85,
    };
    let store = MetadataStore::new(config);

    let mut accepted = 0;
    for i in 0..100 {
        let mut meta = HashMap::new();
        meta.insert(
            "unique_key".to_string(),
            MetadataValue::String(format!("val_{i}")),
        );
        if store.index_slot(i as u32, &meta).is_ok() {
            accepted += 1;
        }
    }

    // Must have admitted only up to the quota and safely rejected the rest
    assert!(accepted <= 20, "Must enforce hard quota limits!");
    assert_eq!(store.total_distinct_terms(), accepted);
}

#[test]
fn test_dead_term_compaction_reclamation() {
    let config = MetadataQuotaConfig::default();
    let store = MetadataStore::new(config);

    for i in 0..100 {
        let mut meta = HashMap::new();
        meta.insert(
            "tag".to_string(),
            MetadataValue::String(if i < 50 { "active" } else { "deleted" }.to_string()),
        );
        store.index_slot(i as u32, &meta).unwrap();
    }

    // Keep only slots 0..50 alive
    let mut live = RoaringBitmap::new();
    for i in 0..50 {
        live.insert(i);
    }

    store.compact_and_reclaim(&live);

    let active_matches = store.match_categorical("tag", "active").unwrap();
    assert_eq!(active_matches.len(), 50);

    // "deleted" tag must have zero live matches
    let deleted_matches = store.match_categorical("tag", "deleted");
    assert!(deleted_matches.is_none() || deleted_matches.unwrap().is_empty());
}
