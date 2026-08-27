/* holosphere/src/codegraph/registry.rs */
//!▫~•◦-------------------------------‣
//! # Hierarchical Symbol Tables & Repository Scope Registry
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Maintains file-level, module-level, and workspace-level symbol registries to support
//! multi-pass deterministic cross-file and local symbol resolution.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use super::parser::ImportItem;
use super::schema::{CodeNode, CodeNodeId, CodeNodeKind};

/// Compact metadata descriptor for indexed code symbols.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SymbolEntry {
    pub id: CodeNodeId,
    pub name: String,
    pub qualified_name: String,
    pub kind: CodeNodeKind,
    pub source_file: PathBuf,
    pub signature: Option<String>,
}

/// Symbol table scoped to one source file.
#[derive(Clone, Debug, Default)]
pub struct FileSymbolTable {
    pub relative_path: PathBuf,
    pub local_symbols: Vec<CodeNodeId>,
    /// Maps local alias/name to imported target path (e.g. "HNSQRIndex" -> "crate::vector::HNSQRIndex")
    pub imports: HashMap<String, String>,
}

/// Workspace-wide authoritative symbol registry.
#[derive(Clone, Debug, Default)]
pub struct WorkspaceSymbolTable {
    pub symbols_by_id: BTreeMap<CodeNodeId, SymbolEntry>,
    pub symbols_by_short_name: BTreeMap<String, Vec<CodeNodeId>>,
    pub symbols_by_qualified: BTreeMap<String, CodeNodeId>,
    pub file_tables: BTreeMap<PathBuf, FileSymbolTable>,
}

impl WorkspaceSymbolTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a code node into the workspace registry.
    pub fn insert_node(&mut self, node: &CodeNode) {
        let entry = SymbolEntry {
            id: node.id.clone(),
            name: node.name.clone(),
            qualified_name: node.qualified_name.clone(),
            kind: node.kind,
            source_file: node.source_file.clone(),
            signature: node.signature.clone(),
        };

        self.symbols_by_id.insert(node.id.clone(), entry);
        self.symbols_by_short_name
            .entry(node.name.clone())
            .or_default()
            .push(node.id.clone());
        self.symbols_by_qualified
            .insert(node.qualified_name.clone(), node.id.clone());

        let file_table = self
            .file_tables
            .entry(node.source_file.clone())
            .or_insert_with(|| FileSymbolTable {
                relative_path: node.source_file.clone(),
                local_symbols: Vec::new(),
                imports: HashMap::new(),
            });
        file_table.local_symbols.push(node.id.clone());
    }

    /// Records an import mapping for a source file.
    pub fn register_import(&mut self, file_path: &Path, import: &ImportItem) {
        let file_table = self
            .file_tables
            .entry(file_path.to_path_buf())
            .or_insert_with(|| FileSymbolTable {
                relative_path: file_path.to_path_buf(),
                local_symbols: Vec::new(),
                imports: HashMap::new(),
            });

        let key = import
            .alias
            .as_ref()
            .unwrap_or(&import.imported_symbol)
            .clone();
        file_table.imports.insert(key, import.import_path.clone());
    }

    /// Removes all symbols and imports registered under a file path (for incremental invalidation).
    pub fn remove_file(&mut self, file_path: &Path) {
        if let Some(file_table) = self.file_tables.remove(file_path) {
            for node_id in file_table.local_symbols {
                if let Some(entry) = self.symbols_by_id.remove(&node_id) {
                    if let Some(ids) = self.symbols_by_short_name.get_mut(&entry.name) {
                        ids.retain(|id| id != &node_id);
                        if ids.is_empty() {
                            self.symbols_by_short_name.remove(&entry.name);
                        }
                    }
                    self.symbols_by_qualified.remove(&entry.qualified_name);
                }
            }
        }
    }

    #[must_use]
    pub fn get(&self, id: &CodeNodeId) -> Option<&SymbolEntry> {
        self.symbols_by_id.get(id)
    }

    #[must_use]
    pub fn lookup_exact(&self, qualified_name: &str) -> Option<&SymbolEntry> {
        let id = self.symbols_by_qualified.get(qualified_name)?;
        self.symbols_by_id.get(id)
    }

    #[must_use]
    pub fn lookup_by_short_name(&self, name: &str) -> Vec<&SymbolEntry> {
        self.symbols_by_short_name
            .get(name)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.symbols_by_id.get(id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Resolves a symbol reference within the context of a source file.
    #[must_use]
    pub fn resolve_reference(&self, file_path: &Path, symbol_ref: &str) -> Vec<&SymbolEntry> {
        let trimmed = symbol_ref.trim();

        // 1. Exact match on qualified name
        if let Some(entry) = self.lookup_exact(trimmed) {
            return vec![entry];
        }

        // 2. File-local import resolution
        if let Some(file_table) = self.file_tables.get(file_path) {
            if let Some(imported_path) = file_table.imports.get(trimmed) {
                // Try resolving imported path directly
                if let Some(entry) = self.lookup_exact(imported_path) {
                    return vec![entry];
                }
                // Try matching by suffix of qualified name
                let suffix_matches: Vec<&SymbolEntry> = self
                    .symbols_by_id
                    .values()
                    .filter(|e| {
                        e.qualified_name.ends_with(imported_path)
                            || imported_path.ends_with(&e.name)
                    })
                    .collect();
                if !suffix_matches.is_empty() {
                    return suffix_matches;
                }
            }

            // 3. File-local symbol definitions
            let local_matches: Vec<&SymbolEntry> = file_table
                .local_symbols
                .iter()
                .filter_map(|id| self.symbols_by_id.get(id))
                .filter(|e| e.name == trimmed || e.qualified_name.ends_with(trimmed))
                .collect();
            if !local_matches.is_empty() {
                return local_matches;
            }
        }

        // 4. Global short name lookup
        self.lookup_by_short_name(trimmed)
    }
}
