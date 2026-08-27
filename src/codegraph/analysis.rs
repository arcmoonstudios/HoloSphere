/* holosphere/src/codegraph/analysis.rs */
//!▫~•◦-------------------------------‣
//! # Architectural Graph Analytics & Blast Radius Calculations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Evaluates god-nodes, cyclic dependencies, cross-module coupling, unreferenced symbols,
//! and blast radius impact analysis for informed codebase refactoring and navigation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::ingest::CodeGraphStoreState;
use super::schema::{CodeNode, CodeNodeId, CodeNodeKind, CodeRelation};

/// Hub centrality descriptor for dominant symbols.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GodNodeInfo {
    pub id: CodeNodeId,
    pub name: String,
    pub qualified_name: String,
    pub kind: CodeNodeKind,
    pub in_degree: usize,
    pub out_degree: usize,
    pub total_degree: usize,
    pub source_file: PathBuf,
}

/// Detected circular dependency cycle.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DependencyCycle {
    pub symbol_names: Vec<String>,
    pub symbol_ids: Vec<CodeNodeId>,
    pub files: Vec<PathBuf>,
}

/// Transitive impact and blast radius assessment for a target symbol.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlastRadiusReport {
    pub target_id: CodeNodeId,
    pub target_name: String,
    pub total_affected_symbols: usize,
    pub affected_callers: Vec<String>,
    pub affected_files: Vec<PathBuf>,
    pub impacted_tests: Vec<String>,
    pub depth_reached: usize,
}

pub struct CodeGraphAnalyzer;

impl CodeGraphAnalyzer {
    /// Identifies the top god nodes / architectural hubs by total degree.
    #[must_use]
    pub fn find_god_nodes(state: &CodeGraphStoreState, top_k: usize) -> Vec<GodNodeInfo> {
        let mut results = Vec::new();

        for (id, node) in &state.nodes {
            if node.kind == CodeNodeKind::File
                || node.kind == CodeNodeKind::Directory
                || node.kind == CodeNodeKind::Rationale
            {
                continue;
            }
            let in_deg = state.incoming_edges.get(id).map_or(0, |v| v.len());
            let out_deg = state.outgoing_edges.get(id).map_or(0, |v| v.len());
            let total = in_deg + out_deg;

            results.push(GodNodeInfo {
                id: id.clone(),
                name: node.name.clone(),
                qualified_name: node.qualified_name.clone(),
                kind: node.kind,
                in_degree: in_deg,
                out_degree: out_deg,
                total_degree: total,
                source_file: node.source_file.clone(),
            });
        }

        results.sort_by(|a, b| {
            b.total_degree
                .cmp(&a.total_degree)
                .then_with(|| a.name.cmp(&b.name))
        });
        results.truncate(top_k);
        results
    }

    /// Detects circular dependency cycles using Tarjan's strongly connected components algorithm.
    #[must_use]
    pub fn find_dependency_cycles(state: &CodeGraphStoreState) -> Vec<DependencyCycle> {
        let mut adj: HashMap<&CodeNodeId, Vec<&CodeNodeId>> = HashMap::new();
        for edge in state.edges.values() {
            if edge.relation == CodeRelation::Calls
                || edge.relation == CodeRelation::Uses
                || edge.relation == CodeRelation::DependsOn
            {
                adj.entry(&edge.source).or_default().push(&edge.target);
            }
        }

        // Tarjan SCC
        let mut index = 0;
        let mut stack = Vec::new();
        let mut on_stack = HashSet::new();
        let mut indices = HashMap::new();
        let mut lowlink = HashMap::new();
        let mut sccs = Vec::new();

        for node_id in state.nodes.keys() {
            if !indices.contains_key(node_id) {
                Self::tarjan_scc(
                    node_id,
                    &adj,
                    &mut index,
                    &mut stack,
                    &mut on_stack,
                    &mut indices,
                    &mut lowlink,
                    &mut sccs,
                );
            }
        }

        let mut cycles = Vec::new();
        for scc in sccs {
            if scc.len() > 1 {
                let mut symbol_names = Vec::new();
                let mut symbol_ids = Vec::new();
                let mut files = HashSet::new();

                for &id in &scc {
                    if let Some(node) = state.nodes.get(id) {
                        symbol_names.push(node.name.clone());
                        symbol_ids.push(id.clone());
                        files.insert(node.source_file.clone());
                    }
                }

                cycles.push(DependencyCycle {
                    symbol_names,
                    symbol_ids,
                    files: files.into_iter().collect(),
                });
            }
        }

        cycles
    }

    fn tarjan_scc<'a>(
        u: &'a CodeNodeId,
        adj: &HashMap<&'a CodeNodeId, Vec<&'a CodeNodeId>>,
        index: &mut usize,
        stack: &mut Vec<&'a CodeNodeId>,
        on_stack: &mut HashSet<&'a CodeNodeId>,
        indices: &mut HashMap<&'a CodeNodeId, usize>,
        lowlink: &mut HashMap<&'a CodeNodeId, usize>,
        sccs: &mut Vec<Vec<&'a CodeNodeId>>,
    ) {
        indices.insert(u, *index);
        lowlink.insert(u, *index);
        *index += 1;
        stack.push(u);
        on_stack.insert(u);

        if let Some(neighbors) = adj.get(u) {
            for &v in neighbors {
                if !indices.contains_key(v) {
                    Self::tarjan_scc(v, adj, index, stack, on_stack, indices, lowlink, sccs);
                    let v_low = *lowlink.get(v).unwrap();
                    let u_low = lowlink.get_mut(u).unwrap();
                    *u_low = (*u_low).min(v_low);
                } else if on_stack.contains(v) {
                    let v_idx = *indices.get(v).unwrap();
                    let u_low = lowlink.get_mut(u).unwrap();
                    *u_low = (*u_low).min(v_idx);
                }
            }
        }

        if lowlink.get(u) == indices.get(u) {
            let mut scc = Vec::new();
            while let Some(w) = stack.pop() {
                on_stack.remove(w);
                scc.push(w);
                if w == u {
                    break;
                }
            }
            sccs.push(scc);
        }
    }

    /// Computes reverse transitive blast radius (who depends on / calls this symbol?).
    #[must_use]
    pub fn compute_blast_radius(
        state: &CodeGraphStoreState,
        target_id: &CodeNodeId,
        max_depth: usize,
    ) -> BlastRadiusReport {
        let target_node = match state.nodes.get(target_id) {
            Some(n) => n,
            None => {
                return BlastRadiusReport {
                    target_id: target_id.clone(),
                    target_name: "unknown".to_string(),
                    total_affected_symbols: 0,
                    affected_callers: Vec::new(),
                    affected_files: Vec::new(),
                    impacted_tests: Vec::new(),
                    depth_reached: 0,
                };
            }
        };

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((target_id.clone(), 0));
        visited.insert(target_id.clone());

        let mut affected_callers = Vec::new();
        let mut affected_files = HashSet::new();
        let mut impacted_tests = Vec::new();
        let mut max_depth_seen = 0;

        while let Some((curr_id, depth)) = queue.pop_front() {
            max_depth_seen = max_depth_seen.max(depth);
            if depth >= max_depth {
                continue;
            }

            if let Some(incoming_edge_ids) = state.incoming_edges.get(&curr_id) {
                for edge_id in incoming_edge_ids {
                    if let Some(edge) = state.edges.get(edge_id) {
                        let caller_id = &edge.source;
                        if visited.insert(caller_id.clone()) {
                            if let Some(caller_node) = state.nodes.get(caller_id) {
                                affected_callers.push(caller_node.qualified_name.clone());
                                affected_files.insert(caller_node.source_file.clone());
                                if caller_node.kind == CodeNodeKind::Test
                                    || caller_node.name.starts_with("test_")
                                {
                                    impacted_tests.push(caller_node.qualified_name.clone());
                                }
                            }
                            queue.push_back((caller_id.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        affected_callers.sort();
        impacted_tests.sort();
        let mut files_vec: Vec<PathBuf> = affected_files.into_iter().collect();
        files_vec.sort();

        BlastRadiusReport {
            target_id: target_id.clone(),
            target_name: target_node.name.clone(),
            total_affected_symbols: affected_callers.len(),
            affected_callers,
            affected_files: files_vec,
            impacted_tests,
            depth_reached: max_depth_seen,
        }
    }
}
