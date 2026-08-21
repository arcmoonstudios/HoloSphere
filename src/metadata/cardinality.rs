/* hnsqr/src/storage/metadata_cardinality.rs */
//!▫~•◦-------------------------------‣
//! # Multi-Tenant Metadata Cardinality Protection & Adaptive Representation
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Protects against cardinality exhaustion attacks by tracking dictionary bytes,
//! term counts, posting density, and filter complexity per tenant/collection.
//! Adaptively selects optimal representation (Roaring, Postings, Dense Bitmap) at seal.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{HNSQRError, HNSQRResult};

/// Physical index representation adaptively chosen based on term posting density.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostingRepresentation {
    /// Sparse term: sorted array of integers (< 1% density)
    SortedPostings,
    /// Medium density / clustered: Roaring bitmap (1% - 30% density)
    RoaringBitmap,
    /// Dense term: contiguous bitset (> 30% density)
    DenseBitmap,
    /// Low-cardinality dictionary enum (< 256 distinct values)
    CompactDictionary,
}

/// Tenant cardinality budget configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CardinalityBudget {
    pub max_distinct_terms: usize,
    pub max_fields: usize,
    pub max_dictionary_bytes: usize,
    pub max_bitmap_bytes: usize,
}

impl Default for CardinalityBudget {
    fn default() -> Self {
        Self {
            max_distinct_terms: 500_000,
            max_fields: 1_000,
            max_dictionary_bytes: 64 * 1024 * 1024, // 64 MB
            max_bitmap_bytes: 128 * 1024 * 1024,    // 128 MB
        }
    }
}

/// Cardinality tracker per tenant.
#[derive(Default)]
pub struct TenantCardinalityTracker {
    pub term_count: AtomicUsize,
    pub field_count: AtomicUsize,
    pub dictionary_bytes: AtomicUsize,
    pub bitmap_bytes: AtomicUsize,
}

/// Cardinality governance engine.
pub struct CardinalityGuard {
    budgets: RwLock<HashMap<String, CardinalityBudget>>,
    trackers: RwLock<HashMap<String, TenantCardinalityTracker>>,
    default_budget: CardinalityBudget,
}

impl Default for CardinalityGuard {
    fn default() -> Self {
        Self::new(CardinalityBudget::default())
    }
}

impl CardinalityGuard {
    pub fn new(default_budget: CardinalityBudget) -> Self {
        Self {
            budgets: RwLock::new(HashMap::new()),
            trackers: RwLock::new(HashMap::new()),
            default_budget,
        }
    }

    /// Evaluates admission for a new metadata term under tenant budgets.
    pub fn check_admission(
        &self,
        tenant_id: &str,
        is_new_term: bool,
        term_bytes: usize,
    ) -> HNSQRResult<()> {
        let budget = self
            .budgets
            .read()
            .get(tenant_id)
            .cloned()
            .unwrap_or_else(|| self.default_budget.clone());

        let mut trackers = self.trackers.write();
        let tracker = trackers.entry(tenant_id.to_string()).or_default();

        if is_new_term {
            let current_terms = tracker.term_count.load(Ordering::Relaxed);
            if current_terms + 1 > budget.max_distinct_terms {
                return Err(HNSQRError::Internal(format!(
                    "Tenant '{tenant_id}' exceeded maximum distinct metadata terms ({current_terms}/{})",
                    budget.max_distinct_terms
                )));
            }
            tracker.term_count.fetch_add(1, Ordering::Relaxed);
            tracker
                .dictionary_bytes
                .fetch_add(term_bytes, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Selects optimal posting representation at segment seal based on term density.
    pub fn select_representation(
        total_docs: usize,
        matching_docs: usize,
        distinct_values: usize,
    ) -> PostingRepresentation {
        if total_docs == 0 {
            return PostingRepresentation::SortedPostings;
        }

        if distinct_values > 0 && distinct_values <= 256 {
            return PostingRepresentation::CompactDictionary;
        }

        let density = matching_docs as f64 / total_docs as f64;
        if density < 0.01 {
            PostingRepresentation::SortedPostings
        } else if density > 0.30 {
            PostingRepresentation::DenseBitmap
        } else {
            PostingRepresentation::RoaringBitmap
        }
    }
}
