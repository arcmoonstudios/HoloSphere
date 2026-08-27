/* holosphere/src/codegraph/query.rs */
//!▫~•◦-------------------------------‣
//! # Hybrid CodeGraph Navigation Pipeline (Query, Explain, Path, Impact)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Bridges exact lexical symbol search and HNSQR vector retrieval with multi-hop
//! bounded hypergraph traversal, relation-aware ranking, and scoped subgraph extraction.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::analysis::{BlastRadiusReport, CodeGraphAnalyzer};
use super::ingest::CodeGraphStoreState;
use super::path::{CodePath, CodePathfinder};
use super::schema::{CodeEdge, CodeNode, CodeNodeId, CodeNodeKind, CodeRelation};
use crate::HNSQRResult;

/// Comprehensive single-symbol explanation output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeExplainResult {
    pub node: CodeNode,
    pub incoming_edges: Vec<CodeEdge>,
    pub outgoing_edges: Vec<CodeEdge>,
    pub callers: Vec<String>,
    pub callees: Vec<String>,
    pub attached_rationale: Vec<String>,
    pub covering_tests: Vec<String>,
}

/// Hybrid query answer with scoped subgraph and architectural path trace.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeQueryResult {
    pub query: String,
    pub seed_symbols: Vec<String>,
    pub nodes: Vec<CodeNode>,
    pub edges: Vec<CodeEdge>,
    pub primary_paths: Vec<CodePath>,
    pub involved_files: Vec<PathBuf>,
    pub formatted_trace: String,
}

pub struct CodeQueryEngine;

impl CodeQueryEngine {
    /// Explains a symbol with definition, spans, connections, rationale, and tests.
    #[must_use]
    pub fn explain(state: &CodeGraphStoreState, symbol: &str) -> Option<CodeExplainResult> {
        let node = Self::resolve_symbol_node(state, symbol)?;
        let node_id = &node.id;

        let incoming_edges: Vec<CodeEdge> = state
            .incoming_edges
            .get(node_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| state.edges.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default();

        let outgoing_edges: Vec<CodeEdge> = state
            .outgoing_edges
            .get(node_id)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| state.edges.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default();

        let mut callers = Vec::new();
        let mut covering_tests = Vec::new();
        let mut attached_rationale = Vec::new();

        for edge in &incoming_edges {
            if let Some(src) = state.nodes.get(&edge.source) {
                if edge.relation == CodeRelation::Calls {
                    callers.push(src.qualified_name.clone());
                }
                if src.kind == CodeNodeKind::Test || src.name.starts_with("test_") {
                    covering_tests.push(src.qualified_name.clone());
                }
                if src.kind == CodeNodeKind::Rationale
                    || edge.relation == CodeRelation::Explains
                    || edge.relation == CodeRelation::Justifies
                {
                    attached_rationale.push(src.name.clone());
                }
            }
        }

        let mut callees = Vec::new();
        for edge in &outgoing_edges {
            if let Some(tgt) = state.nodes.get(&edge.target) {
                if edge.relation == CodeRelation::Calls {
                    callees.push(tgt.qualified_name.clone());
                }
            }
        }

        callers.sort();
        callees.sort();
        covering_tests.sort();
        attached_rationale.sort();

        Some(CodeExplainResult {
            node,
            incoming_edges,
            outgoing_edges,
            callers,
            callees,
            attached_rationale,
            covering_tests,
        })
    }

    /// Traces structural path between two symbols.
    #[must_use]
    pub fn trace_path(
        state: &CodeGraphStoreState,
        from_symbol: &str,
        to_symbol: &str,
        max_hops: usize,
    ) -> Option<CodePath> {
        let from_node = Self::resolve_symbol_node(state, from_symbol)?;
        let to_node = Self::resolve_symbol_node(state, to_symbol)?;
        CodePathfinder::find_path(state, &from_node.id, &to_node.id, max_hops)
    }

    /// Evaluates blast radius impact.
    #[must_use]
    pub fn impact(
        state: &CodeGraphStoreState,
        symbol: &str,
        max_depth: usize,
    ) -> Option<BlastRadiusReport> {
        let node = Self::resolve_symbol_node(state, symbol)?;
        Some(CodeGraphAnalyzer::compute_blast_radius(
            state, &node.id, max_depth,
        ))
    }

    /// Executes hybrid natural language / architectural code query.
    #[must_use]
    pub fn query(
        state: &CodeGraphStoreState,
        query: &str,
        max_nodes: usize,
        max_depth: usize,
    ) -> CodeQueryResult {
        let query_terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect();

        // 1. Seed Score Calculation (Exact match + Substring + Docstring match)
        let mut scored_seeds: Vec<(&CodeNode, f32)> = Vec::new();
        for node in state.nodes.values() {
            if node.kind == CodeNodeKind::File || node.kind == CodeNodeKind::Directory {
                continue;
            }
            let mut score = 0.0f32;
            let name_lower = node.name.to_lowercase();
            let qual_lower = node.qualified_name.to_lowercase();

            for term in &query_terms {
                if name_lower == *term {
                    score += 10.0;
                } else if name_lower.contains(term) {
                    score += 4.0;
                }
                if qual_lower.contains(term) {
                    score += 2.0;
                }
                if let Some(doc) = &node.docstring {
                    if doc.to_lowercase().contains(term) {
                        score += 1.5;
                    }
                }
            }

            if score > 0.0 {
                scored_seeds.push((node, score));
            }
        }

        scored_seeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_seeds: Vec<&CodeNode> = scored_seeds.into_iter().take(5).map(|(n, _)| n).collect();

        let seed_symbols: Vec<String> =
            top_seeds.iter().map(|n| n.qualified_name.clone()).collect();

        // 2. Multi-Hop Bounded Structural Traversal
        let mut selected_node_ids = HashSet::new();
        let mut selected_edge_ids = HashSet::new();
        let mut frontier = VecDeque::new();

        for seed in &top_seeds {
            selected_node_ids.insert(seed.id.clone());
            frontier.push_back((seed.id.clone(), 0));
        }

        while let Some((curr_id, depth)) = frontier.pop_front() {
            if depth >= max_depth || selected_node_ids.len() >= max_nodes {
                continue;
            }

            // Outgoing edges
            if let Some(out_edges) = state.outgoing_edges.get(&curr_id) {
                for edge_id in out_edges {
                    if let Some(edge) = state.edges.get(edge_id) {
                        selected_edge_ids.insert(edge_id.clone());
                        if selected_node_ids.insert(edge.target.clone()) {
                            frontier.push_back((edge.target.clone(), depth + 1));
                        }
                    }
                }
            }

            // Incoming calls
            if let Some(in_edges) = state.incoming_edges.get(&curr_id) {
                for edge_id in in_edges {
                    if let Some(edge) = state.edges.get(edge_id) {
                        if edge.relation == CodeRelation::Calls {
                            selected_edge_ids.insert(edge_id.clone());
                            if selected_node_ids.insert(edge.source.clone()) {
                                frontier.push_back((edge.source.clone(), depth + 1));
                            }
                        }
                    }
                }
            }
        }

        let mut nodes: Vec<CodeNode> = selected_node_ids
            .into_iter()
            .filter_map(|id| state.nodes.get(&id).cloned())
            .collect();
        nodes.sort_by(|a, b| a.qualified_name.cmp(&b.qualified_name));

        let mut edges: Vec<CodeEdge> = selected_edge_ids
            .into_iter()
            .filter_map(|id| state.edges.get(&id).cloned())
            .collect();
        edges.sort_by(|a, b| a.id.cmp(&b.id));

        let mut files_set = HashSet::new();
        for n in &nodes {
            files_set.insert(n.source_file.clone());
        }
        let mut involved_files: Vec<PathBuf> = files_set.into_iter().collect();
        involved_files.sort();

        // 3. Format Explanation Trace
        let mut trace_lines = Vec::new();
        trace_lines.push(format!("Query: {query}"));
        trace_lines.push(format!("Discovered {} seeds:", top_seeds.len()));
        for seed in &top_seeds {
            trace_lines.push(format!(
                "  - `{}` ({}) in {}",
                seed.name,
                seed.kind,
                seed.source_file.display()
            ));
        }

        trace_lines.push("\nStructural Subgraph:".to_string());
        for edge in edges.iter().take(20) {
            let src_name = state.nodes.get(&edge.source).map_or("?", |n| &n.name);
            let tgt_name = state.nodes.get(&edge.target).map_or("?", |n| &n.name);
            trace_lines.push(format!(
                "  {} --[{}]--> {} ({})",
                src_name, edge.relation, tgt_name, edge.origin
            ));
        }

        CodeQueryResult {
            query: query.to_string(),
            seed_symbols,
            nodes,
            edges,
            primary_paths: Vec::new(),
            involved_files,
            formatted_trace: trace_lines.join("\n"),
        }
    }

    fn resolve_symbol_node(state: &CodeGraphStoreState, symbol: &str) -> Option<CodeNode> {
        let trimmed = symbol.trim();
        // 1. By ID
        if let Some(node) = state.nodes.get(&CodeNodeId(trimmed.to_string())) {
            return Some(node.clone());
        }
        // 2. By exact qualified name
        if let Some(id) = state.symbols_by_qualified.get(trimmed) {
            return state.nodes.get(id).cloned();
        }
        // 3. By suffix of qualified name
        for (qual, id) in &state.symbols_by_qualified {
            if qual.ends_with(trimmed) {
                return state.nodes.get(id).cloned();
            }
        }
        // 4. By short name
        if let Some(ids) = state.symbols_by_name.get(trimmed) {
            if let Some(first_id) = ids.first() {
                return state.nodes.get(first_id).cloned();
            }
        }
        None
    }
}
