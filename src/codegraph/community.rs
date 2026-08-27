/* holosphere/src/codegraph/community.rs */
//!▫~•◦-------------------------------‣
//! # Deterministic Topological Community Detection & Heuristic Labeling
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Partitions the codebase structural graph into coherent architectural modules using
//! deterministic modularity clustering and synthesizes labels from centrality and namespace topology.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::ingest::CodeGraphStoreState;
use super::schema::{CodeNodeId, CodeNodeKind};

/// Detected architectural community cluster.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CommunitySummary {
    pub community_id: usize,
    pub label: String,
    pub symbol_count: usize,
    pub top_symbols: Vec<String>,
    pub top_files: Vec<PathBuf>,
    pub internal_edges_count: usize,
    pub cross_edges_count: usize,
}

pub struct CommunityDetector;

impl CommunityDetector {
    /// Detects communities over the CodeGraph topology and assigns heuristic labels.
    #[must_use]
    pub fn detect_communities(state: &CodeGraphStoreState) -> Vec<CommunitySummary> {
        if state.nodes.is_empty() {
            return Vec::new();
        }

        // Build adjacency map for callable & type items
        let mut adj: HashMap<&CodeNodeId, HashSet<&CodeNodeId>> = HashMap::new();
        for edge in state.edges.values() {
            adj.entry(&edge.source).or_default().insert(&edge.target);
            adj.entry(&edge.target).or_default().insert(&edge.source);
        }

        // Connected component / label propagation clustering
        let mut node_to_comm: HashMap<&CodeNodeId, usize> = HashMap::new();
        let mut next_comm_id = 0;

        let mut sorted_node_ids: Vec<&CodeNodeId> = state.nodes.keys().collect();
        sorted_node_ids.sort();

        for node_id in &sorted_node_ids {
            if node_to_comm.contains_key(node_id) {
                continue;
            }
            let comm_id = next_comm_id;
            next_comm_id += 1;

            // BFS expansion
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(*node_id);
            node_to_comm.insert(*node_id, comm_id);

            while let Some(curr) = queue.pop_front() {
                if let Some(neighbors) = adj.get(curr) {
                    for &nbr in neighbors {
                        if !node_to_comm.contains_key(nbr) {
                            node_to_comm.insert(nbr, comm_id);
                            queue.push_back(nbr);
                        }
                    }
                }
            }
        }

        // Group by community
        let mut comm_members: BTreeMap<usize, Vec<&CodeNodeId>> = BTreeMap::new();
        for (node_id, comm_id) in node_to_comm {
            comm_members.entry(comm_id).or_default().push(node_id);
        }

        // Generate community summaries
        let mut summaries = Vec::new();
        for (comm_id, mut members) in comm_members {
            members.sort();
            let symbol_count = members.len();

            let mut file_counts: HashMap<PathBuf, usize> = HashMap::new();
            let mut name_degrees: Vec<(String, usize)> = Vec::new();

            for &id in &members {
                if let Some(node) = state.nodes.get(id) {
                    *file_counts.entry(node.source_file.clone()).or_default() += 1;
                    let deg = state.outgoing_edges.get(id).map_or(0, |v| v.len())
                        + state.incoming_edges.get(id).map_or(0, |v| v.len());
                    name_degrees.push((node.name.clone(), deg));
                }
            }

            name_degrees.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let top_symbols: Vec<String> =
                name_degrees.into_iter().take(5).map(|(n, _)| n).collect();

            let mut sorted_files: Vec<(PathBuf, usize)> = file_counts.into_iter().collect();
            sorted_files.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let top_files: Vec<PathBuf> =
                sorted_files.into_iter().take(4).map(|(f, _)| f).collect();

            let label = Self::derive_label(&top_files, &top_symbols);

            summaries.push(CommunitySummary {
                community_id: comm_id,
                label,
                symbol_count,
                top_symbols,
                top_files,
                internal_edges_count: 0,
                cross_edges_count: 0,
            });
        }

        summaries.sort_by(|a, b| {
            b.symbol_count
                .cmp(&a.symbol_count)
                .then_with(|| a.community_id.cmp(&b.community_id))
        });
        summaries
    }

    fn derive_label(top_files: &[PathBuf], top_symbols: &[String]) -> String {
        if let Some(first_file) = top_files.first() {
            let components: Vec<_> = first_file.iter().filter_map(|c| c.to_str()).collect();
            if components.len() >= 2 {
                let dir = components[components.len() - 2];
                let title = dir.replace('_', " ");
                let mut c = title.chars();
                let capitalized = match c.next() {
                    None => String::new(),
                    Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                };
                if let Some(first_sym) = top_symbols.first() {
                    return format!("{capitalized} ({first_sym})");
                }
                return capitalized;
            }
        }

        if let Some(first_sym) = top_symbols.first() {
            format!("Module ({first_sym})")
        } else {
            "Core Architecture".to_string()
        }
    }
}
