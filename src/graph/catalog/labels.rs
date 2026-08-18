/* hnsqr/src/graph/catalog/labels.rs */
//!▫~•◦-------------------------------‣
//! # Label Catalog — Interned Node Label Identifiers
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Labels 0–63 are "fast" labels whose presence is encoded as a single bit
//! in `GraphNodeRecord::label_fast_mask`.  Labels ≥ 64 require an overflow
//! lookup through `GraphNodeRecord::label_overflow_ref`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Compact interned label identifier.  Slots 0–63 fit the fast bitmask path.
pub type LabelId = u32;

/// Boundary between fast-mask and overflow-bitmap label storage.
pub const LABEL_FAST_SLOTS: u32 = 64;

/// Result of resolving a label name in the catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LabelResolution {
    /// Label maps to a bit position in the 64-bit fast mask.
    Fast { bit: u8 },
    /// Label index ≥ 64; stored in the per-node overflow bitmap.
    Overflow { index: u32 },
    /// Label has never been registered.
    Unknown,
}

/// Thread-safe label name ↔ [`LabelId`] registry.
pub struct LabelCatalog {
    name_to_id: RwLock<HashMap<String, LabelId>>,
    id_to_name: RwLock<Vec<String>>,
}

impl Default for LabelCatalog {
    fn default() -> Self {
        Self {
            name_to_id: RwLock::new(HashMap::new()),
            id_to_name: RwLock::new(Vec::new()),
        }
    }
}

impl LabelCatalog {
    /// Returns the [`LabelId`] for `name`, registering it if not yet known.
    pub fn get_or_register(&self, name: &str) -> LabelId {
        {
            let guard = self.name_to_id.read();
            if let Some(&id) = guard.get(name) {
                return id;
            }
        }
        let mut guard = self.name_to_id.write();
        // Re-check after acquiring write lock (TOCTOU).
        if let Some(&id) = guard.get(name) {
            return id;
        }
        let id = self.id_to_name.read().len() as LabelId;
        guard.insert(name.to_string(), id);
        self.id_to_name.write().push(name.to_string());
        id
    }

    /// Returns the [`LabelId`] for `name`, or `None` if not registered.
    pub fn get(&self, name: &str) -> Option<LabelId> {
        self.name_to_id.read().get(name).copied()
    }

    /// Returns the string name for a given [`LabelId`].
    pub fn name_of(&self, id: LabelId) -> Option<String> {
        self.id_to_name.read().get(id as usize).cloned()
    }

    /// Classifies a [`LabelId`] for storage routing.
    pub fn resolve(&self, id: LabelId) -> LabelResolution {
        if id < LABEL_FAST_SLOTS {
            LabelResolution::Fast { bit: id as u8 }
        } else {
            LabelResolution::Overflow { index: id }
        }
    }

    /// Total number of registered labels.
    pub fn len(&self) -> usize {
        self.id_to_name.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Compact serializable snapshot of the catalog for Raft-replication.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LabelCatalogSnapshot {
    pub entries: Vec<String>,
}

impl From<&LabelCatalog> for LabelCatalogSnapshot {
    fn from(c: &LabelCatalog) -> Self {
        Self {
            entries: c.id_to_name.read().clone(),
        }
    }
}

impl From<LabelCatalogSnapshot> for LabelCatalog {
    fn from(snap: LabelCatalogSnapshot) -> Self {
        let mut name_to_id = HashMap::with_capacity(snap.entries.len());
        for (id, name) in snap.entries.iter().enumerate() {
            name_to_id.insert(name.clone(), id as LabelId);
        }
        Self {
            name_to_id: RwLock::new(name_to_id),
            id_to_name: RwLock::new(snap.entries),
        }
    }
}
