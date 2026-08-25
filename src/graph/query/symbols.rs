/* holosphere/src/graph/query/symbols.rs */
//!▫~•◦-------------------------------‣
//! # Symbol Table — Alias-to-ID Resolution
//!▫~•◦-------------------------------------------------------------------‣
//!
//! After parsing, every alias (`n`, `r`, `c`) is converted to a compact
//! `SymbolId(u32)` so the executor never compares query strings at runtime.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;

/// Compact interned query alias identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SymbolId(pub u32);

/// Bidirectional alias ↔ `SymbolId` map for one query.
#[derive(Default, Debug)]
pub struct SymbolTable {
    name_to_id: HashMap<String, SymbolId>,
    id_to_name: Vec<String>,
}

impl SymbolTable {
    /// Interns `name` and returns its `SymbolId`, registering if new.
    pub fn intern(&mut self, name: &str) -> SymbolId {
        if let Some(&id) = self.name_to_id.get(name) {
            return id;
        }
        let id = SymbolId(self.id_to_name.len() as u32);
        self.id_to_name.push(name.to_string());
        self.name_to_id.insert(name.to_string(), id);
        id
    }

    /// Returns the `SymbolId` for `name`, or `None` if not registered.
    pub fn get(&self, name: &str) -> Option<SymbolId> {
        self.name_to_id.get(name).copied()
    }

    /// Returns the original alias string for `id`.
    pub fn name_of(&self, id: SymbolId) -> Option<&str> {
        self.id_to_name.get(id.0 as usize).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.id_to_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_name.is_empty()
    }
}
