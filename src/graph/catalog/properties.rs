/* holosphere/src/graph/catalog/properties.rs */
//!▫~•◦-------------------------------‣
//! # Property Key Catalog — Interned Graph Property Key Identifiers
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;

use parking_lot::RwLock;

/// Compact interned property-key identifier.
pub type PropertyKey = u32;

/// Thread-safe property-key name ↔ [`PropertyKey`] registry.
pub struct PropertyKeyCatalog {
    name_to_id: RwLock<HashMap<String, PropertyKey>>,
    id_to_name: RwLock<Vec<String>>,
}

impl Default for PropertyKeyCatalog {
    fn default() -> Self {
        Self {
            name_to_id: RwLock::new(HashMap::new()),
            id_to_name: RwLock::new(Vec::new()),
        }
    }
}

impl PropertyKeyCatalog {
    pub fn get_or_register(&self, name: &str) -> PropertyKey {
        {
            let guard = self.name_to_id.read();
            if let Some(&id) = guard.get(name) {
                return id;
            }
        }
        let mut guard = self.name_to_id.write();
        if let Some(&id) = guard.get(name) {
            return id;
        }
        let id = self.id_to_name.read().len() as PropertyKey;
        guard.insert(name.to_string(), id);
        self.id_to_name.write().push(name.to_string());
        id
    }

    pub fn get(&self, name: &str) -> Option<PropertyKey> {
        self.name_to_id.read().get(name).copied()
    }

    pub fn name_of(&self, id: PropertyKey) -> Option<String> {
        self.id_to_name.read().get(id as usize).cloned()
    }

    pub fn len(&self) -> usize {
        self.id_to_name.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
