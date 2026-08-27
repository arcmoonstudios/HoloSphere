/* holosphere/src/codegraph/resolver.rs */
//!▫~•◦-------------------------------‣
//! # 4-Pass Deterministic Symbol & Call Graph Resolver
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Statically resolves AST-extracted call expressions and type references against
//! local and workspace symbol tables, preserving explicit ambiguity without heuristic guessing.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::path::Path;

use super::parser::{UnresolvedCall, UnresolvedTypeRef};
use super::registry::WorkspaceSymbolTable;
use super::schema::{CodeEdge, CodeEdgeId, CodeNodeId, CodeRelation, RelationOrigin, SourceSpan};

pub struct CodeGraphResolver<'a> {
    symbols: &'a WorkspaceSymbolTable,
}

impl<'a> CodeGraphResolver<'a> {
    #[must_use]
    pub fn new(symbols: &'a WorkspaceSymbolTable) -> Self {
        Self { symbols }
    }

    /// Resolves unresolved calls and type references into authoritative CodeEdges.
    #[must_use]
    pub fn resolve_all(
        &self,
        unresolved_calls: &[UnresolvedCall],
        unresolved_types: &[UnresolvedTypeRef],
    ) -> Vec<CodeEdge> {
        let mut edges = Vec::new();

        // 1. Resolve Function & Method Calls
        for call in unresolved_calls {
            let resolved_edges = self.resolve_call(call);
            edges.extend(resolved_edges);
        }

        // 2. Resolve Type References (Implements, Returns, Accepts, Uses)
        for type_ref in unresolved_types {
            let resolved_edges = self.resolve_type_ref(type_ref);
            edges.extend(resolved_edges);
        }

        edges
    }

    fn resolve_call(&self, call: &UnresolvedCall) -> Vec<CodeEdge> {
        let caller_entry = match self.symbols.get(&call.caller_id) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let file_path = &caller_entry.source_file;
        let mut candidates = Vec::new();

        // Pass B: Local / Receiver resolution
        if let Some(receiver) = &call.receiver {
            if receiver == "self" || receiver == "Self" {
                // Look for method on current file/struct
                if let Some(file_table) = self.symbols.file_tables.get(file_path) {
                    for local_id in &file_table.local_symbols {
                        if let Some(entry) = self.symbols.get(local_id) {
                            if entry.name == call.target_symbol {
                                candidates.push(entry);
                            }
                        }
                    }
                }
            } else {
                // Receiver is a type or module name (e.g. `HNSQRIndex::search`)
                let combined = format!("{receiver}::{}", call.target_symbol);
                let matches = self.symbols.resolve_reference(file_path, &combined);
                if !matches.is_empty() {
                    candidates = matches;
                } else {
                    // Try receiver as type, method as target
                    let type_matches = self.symbols.resolve_reference(file_path, receiver);
                    for tm in type_matches {
                        let method_qual = format!("{}::{}", tm.qualified_name, call.target_symbol);
                        if let Some(me) = self.symbols.lookup_exact(&method_qual) {
                            candidates.push(me);
                        }
                    }
                }
            }
        } else {
            // Pass C: Workspace resolution
            candidates = self
                .symbols
                .resolve_reference(file_path, &call.target_symbol);
        }

        // Pass D: Ambiguity preservation
        if candidates.is_empty() {
            return Vec::new();
        }

        let origin = if candidates.len() == 1 {
            RelationOrigin::Resolved
        } else {
            RelationOrigin::Ambiguous
        };
        let confidence = origin.default_confidence();

        candidates
            .into_iter()
            .map(|target| {
                let edge_id = CodeEdgeId::compute(
                    &call.caller_id,
                    &target.id,
                    CodeRelation::Calls,
                    origin,
                    &call.span,
                );
                CodeEdge {
                    id: edge_id,
                    source: call.caller_id.clone(),
                    target: target.id.clone(),
                    relation: CodeRelation::Calls,
                    origin,
                    confidence,
                    evidence: call.span,
                    attributes: BTreeMap::new(),
                }
            })
            .collect()
    }

    fn resolve_type_ref(&self, type_ref: &UnresolvedTypeRef) -> Vec<CodeEdge> {
        let source_entry = match self.symbols.get(&type_ref.source_id) {
            Some(e) => e,
            None => return Vec::new(),
        };

        let file_path = &source_entry.source_file;
        let candidates = self
            .symbols
            .resolve_reference(file_path, &type_ref.target_type);

        if candidates.is_empty() {
            return Vec::new();
        }

        let origin = if candidates.len() == 1 {
            RelationOrigin::Resolved
        } else {
            RelationOrigin::Ambiguous
        };
        let confidence = origin.default_confidence();

        candidates
            .into_iter()
            .map(|target| {
                let edge_id = CodeEdgeId::compute(
                    &type_ref.source_id,
                    &target.id,
                    type_ref.relation,
                    origin,
                    &type_ref.span,
                );
                CodeEdge {
                    id: edge_id,
                    source: type_ref.source_id.clone(),
                    target: target.id.clone(),
                    relation: type_ref.relation,
                    origin,
                    confidence,
                    evidence: type_ref.span,
                    attributes: BTreeMap::new(),
                }
            })
            .collect()
    }
}
