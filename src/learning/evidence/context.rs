/* holosphere/src/learning/evidence/context.rs */
//!▫~•◦-------------------------------‣
//! # Context Classes & Equivalence Mapping
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides deterministic mapping from Phase 5 ContextFingerprints to durable ContextClassIds.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::experience::context::ContextRecord;
use crate::learning::id::ContextClassId;

/// Deterministic registry mapping exact cryptographic context fingerprints to durable `ContextClassId`s.
pub struct ContextClassRegistry {
    next_id: AtomicU64,
    fingerprint_to_class: RwLock<HashMap<[u8; 32], ContextClassId>>,
}

impl Default for ContextClassRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextClassRegistry {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            fingerprint_to_class: RwLock::new(HashMap::new()),
        }
    }

    /// Resolves or registers a durable `ContextClassId` for a Phase 5 `ContextRecord`.
    pub fn get_or_create_class(&self, context: &ContextRecord) -> ContextClassId {
        let fp = context.canonical_fingerprint;
        {
            let map = self.fingerprint_to_class.read();
            if let Some(&class_id) = map.get(&fp) {
                return class_id;
            }
        }

        let mut map = self.fingerprint_to_class.write();
        if let Some(&class_id) = map.get(&fp) {
            return class_id;
        }

        let class_id = ContextClassId(self.next_id.fetch_add(1, Ordering::Relaxed));
        map.insert(fp, class_id);
        class_id
    }

    /// Resolves an existing `ContextClassId` if known.
    pub fn get_class(&self, context: &ContextRecord) -> Option<ContextClassId> {
        self.fingerprint_to_class
            .read()
            .get(&context.canonical_fingerprint)
            .copied()
    }
}
