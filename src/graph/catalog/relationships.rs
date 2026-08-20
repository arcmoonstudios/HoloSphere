/* hnsqr/src/graph/catalog/relationships.rs */
//!▫~•◦-------------------------------‣
//! # Relationship-Type Catalog — Interned Edge Type Identifiers
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

/// Compact interned relationship-type identifier.
pub type RelTypeId = u16;

/// Result of resolving a relationship-type name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelTypeResolution {
    Known(RelTypeId),
    Unknown,
}

/// Thread-safe relationship-type name ↔ [`RelTypeId`] registry.
pub struct RelTypeCatalog {
    name_to_id: RwLock<HashMap<String, RelTypeId>>,
    id_to_name: RwLock<Vec<String>>,
}

impl Default for RelTypeCatalog {
    fn default() -> Self {
        Self {
            name_to_id: RwLock::new(HashMap::new()),
            id_to_name: RwLock::new(Vec::new()),
        }
    }
}

impl RelTypeCatalog {
    /// Returns the [`RelTypeId`] for `name`, registering it if not yet known.
    /// Returns `None` if the catalog would overflow `u16::MAX`.
    pub fn get_or_register(&self, name: &str) -> Option<RelTypeId> {
        {
            let guard = self.name_to_id.read();
            if let Some(&id) = guard.get(name) {
                return Some(id);
            }
        }
        let mut guard = self.name_to_id.write();
        if let Some(&id) = guard.get(name) {
            return Some(id);
        }
        let next = self.id_to_name.read().len();
        if next > u16::MAX as usize {
            return None; // Catalog saturated.
        }
        let id = next as RelTypeId;
        guard.insert(name.to_string(), id);
        self.id_to_name.write().push(name.to_string());
        Some(id)
    }

    pub fn get(&self, name: &str) -> Option<RelTypeId> {
        self.name_to_id.read().get(name).copied()
    }

    pub fn name_of(&self, id: RelTypeId) -> Option<String> {
        self.id_to_name.read().get(id as usize).cloned()
    }

    pub fn resolve(&self, name: &str) -> RelTypeResolution {
        match self.name_to_id.read().get(name).copied() {
            Some(id) => RelTypeResolution::Known(id),
            None => RelTypeResolution::Unknown,
        }
    }

    pub fn len(&self) -> usize {
        self.id_to_name.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Captures a frozen point-in-time snapshot of the relationship type catalog.
    pub fn snapshot(&self) -> RelTypeCatalogSnapshot {
        RelTypeCatalogSnapshot::from(self)
    }
}

/// Compact serializable snapshot for Raft-replication.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RelTypeCatalogSnapshot {
    pub entries: Vec<String>,
}

impl From<&RelTypeCatalog> for RelTypeCatalogSnapshot {
    fn from(c: &RelTypeCatalog) -> Self {
        Self {
            entries: c.id_to_name.read().clone(),
        }
    }
}

impl From<RelTypeCatalogSnapshot> for RelTypeCatalog {
    fn from(snap: RelTypeCatalogSnapshot) -> Self {
        let mut name_to_id = HashMap::with_capacity(snap.entries.len());
        for (id, name) in snap.entries.iter().enumerate() {
            name_to_id.insert(name.clone(), id as RelTypeId);
        }
        Self {
            name_to_id: RwLock::new(name_to_id),
            id_to_name: RwLock::new(snap.entries),
        }
    }
}
