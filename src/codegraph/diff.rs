/* holosphere/src/codegraph/diff.rs */
//!▫~•◦-------------------------------‣
//! # Temporal LSN CodeGraph Diff & Structural Blast Radius Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates structural differences between two snapshot states, computing added/removed
//! symbols, relational drift, modified signatures, and impacted tests.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashSet};

use serde::{Deserialize, Serialize};

use super::ingest::CodeGraphStoreState;
use super::schema::{CodeEdge, CodeNode, CodeNodeId, CodeNodeKind};

/// Complete structural diff between two CodeGraph states.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeGraphDiffReport {
    pub from_lsn: u64,
    pub to_lsn: u64,
    pub added_symbols: Vec<CodeNode>,
    pub deleted_symbols: Vec<CodeNode>,
    pub modified_symbols: Vec<ModifiedSymbolPair>,
    pub added_edges_count: usize,
    pub deleted_edges_count: usize,
    pub affected_public_apis: Vec<String>,
    pub affected_tests: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModifiedSymbolPair {
    pub name: String,
    pub qualified_name: String,
    pub before_signature: Option<String>,
    pub after_signature: Option<String>,
}

pub struct CodeGraphDiffEngine;

impl CodeGraphDiffEngine {
    /// Compares two CodeGraph snapshots.
    #[must_use]
    pub fn diff(
        old_state: &CodeGraphStoreState,
        new_state: &CodeGraphStoreState,
    ) -> CodeGraphDiffReport {
        let mut added_symbols = Vec::new();
        let mut deleted_symbols = Vec::new();
        let mut modified_symbols = Vec::new();
        let mut affected_public_apis = Vec::new();
        let mut affected_tests = Vec::new();

        // Detect added & modified
        for (id, new_node) in &new_state.nodes {
            if let Some(old_node) = old_state.nodes.get(id) {
                if old_node.signature != new_node.signature
                    || old_node.symbol_hash != new_node.symbol_hash
                {
                    modified_symbols.push(ModifiedSymbolPair {
                        name: new_node.name.clone(),
                        qualified_name: new_node.qualified_name.clone(),
                        before_signature: old_node.signature.clone(),
                        after_signature: new_node.signature.clone(),
                    });
                }
            } else {
                added_symbols.push(new_node.clone());
                if new_node.kind == CodeNodeKind::Function
                    || new_node.kind == CodeNodeKind::Struct
                    || new_node.kind == CodeNodeKind::Trait
                {
                    affected_public_apis.push(new_node.qualified_name.clone());
                }
                if new_node.kind == CodeNodeKind::Test || new_node.name.starts_with("test_") {
                    affected_tests.push(new_node.qualified_name.clone());
                }
            }
        }

        // Detect deleted
        for (id, old_node) in &old_state.nodes {
            if !new_state.nodes.contains_key(id) {
                deleted_symbols.push(old_node.clone());
                if old_node.kind == CodeNodeKind::Function
                    || old_node.kind == CodeNodeKind::Struct
                    || old_node.kind == CodeNodeKind::Trait
                {
                    affected_public_apis.push(old_node.qualified_name.clone());
                }
            }
        }

        let added_edges_count = new_state
            .edges
            .keys()
            .filter(|id| !old_state.edges.contains_key(*id))
            .count();
        let deleted_edges_count = old_state
            .edges
            .keys()
            .filter(|id| !new_state.edges.contains_key(*id))
            .count();

        added_symbols.sort_by(|a, b| a.qualified_name.cmp(&b.qualified_name));
        deleted_symbols.sort_by(|a, b| a.qualified_name.cmp(&b.qualified_name));
        affected_public_apis.sort();
        affected_public_apis.dedup();
        affected_tests.sort();
        affected_tests.dedup();

        CodeGraphDiffReport {
            from_lsn: old_state.commit_lsn,
            to_lsn: new_state.commit_lsn,
            added_symbols,
            deleted_symbols,
            modified_symbols,
            added_edges_count,
            deleted_edges_count,
            affected_public_apis,
            affected_tests,
        }
    }
}
