/* holosphere/src/codegraph/community.rs */
//!▫~•◦-------------------------------‣
//! # Deterministic Modularity-Aware Community Detection & Architectural Labeling
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Partitions the codebase structural graph into dense, coherent architectural modules using
//! deterministic Label Propagation / modularity clustering with neighbor density weighting,
//! computing exact internal/cross-community edge density and heuristic architectural labels.
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
    /// Detects communities over the CodeGraph topology and returns both summaries and full node-to-community mapping.
    #[must_use]
    pub fn detect_community_map(
        state: &CodeGraphStoreState,
    ) -> (Vec<CommunitySummary>, HashMap<CodeNodeId, usize>) {
        if state.nodes.is_empty() {
            return (Vec::new(), HashMap::new());
        }

        // Build undirected adjacency graph across structural relations
        let mut adj: HashMap<&CodeNodeId, Vec<&CodeNodeId>> = HashMap::new();
        for edge in state.edges.values() {
            adj.entry(&edge.source).or_default().push(&edge.target);
            adj.entry(&edge.target).or_default().push(&edge.source);
        }

        let mut sorted_node_ids: Vec<&CodeNodeId> = state.nodes.keys().collect();
        sorted_node_ids.sort();

        // Step 1: Initial partition (each node starts in a unique deterministic label)
        let mut node_to_comm: HashMap<&CodeNodeId, usize> = HashMap::new();
        for (idx, &node_id) in sorted_node_ids.iter().enumerate() {
            node_to_comm.insert(node_id, idx);
        }

        // Step 2: Deterministic Label Propagation with self-stabilization (up to 15 iterations)
        let max_iterations = 15;
        for _ in 0..max_iterations {
            let mut changed = false;

            for &node_id in &sorted_node_ids {
                let current_label = match node_to_comm.get(node_id) {
                    Some(&lbl) => lbl,
                    None => continue,
                };

                let neighbors = match adj.get(node_id) {
                    Some(nbrs) if !nbrs.is_empty() => nbrs,
                    _ => continue,
                };

                let mut label_weights: HashMap<usize, usize> = HashMap::new();
                // Self-weight to avoid oscillation
                label_weights.insert(current_label, 1);

                for &nbr in neighbors {
                    if let Some(&nbr_label) = node_to_comm.get(nbr) {
                        *label_weights.entry(nbr_label).or_default() += 2;
                    }
                }

                // Pick highest frequency label, breaking ties deterministically by lowest numeric ID
                let mut best_label = current_label;
                let mut best_weight = 0;

                for (&lbl, &weight) in &label_weights {
                    if weight > best_weight || (weight == best_weight && lbl < best_label) {
                        best_weight = weight;
                        best_label = lbl;
                    }
                }

                if best_label != current_label {
                    node_to_comm.insert(node_id, best_label);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Step 3: Group members by converged community
        let mut raw_comm_members: BTreeMap<usize, Vec<&CodeNodeId>> = BTreeMap::new();
        for (&node_id, &comm_id) in &node_to_comm {
            raw_comm_members.entry(comm_id).or_default().push(node_id);
        }

        // Sort communities by size descending, then by smallest node ID for total determinism
        let mut sorted_comms: Vec<(usize, Vec<&CodeNodeId>)> = raw_comm_members.into_iter().collect();
        for (_, members) in &mut sorted_comms {
            members.sort();
        }
        sorted_comms.sort_by(|a, b| {
            b.1.len()
                .cmp(&a.1.len())
                .then_with(|| a.1.first().cmp(&b.1.first()))
        });

        // Re-index communities consecutively: 0, 1, 2, ...
        let mut final_node_comm_map: HashMap<CodeNodeId, usize> = HashMap::new();
        let mut reindexed_members: Vec<Vec<&CodeNodeId>> = Vec::with_capacity(sorted_comms.len());

        for (new_comm_id, (_old_id, members)) in sorted_comms.into_iter().enumerate() {
            for &id in &members {
                final_node_comm_map.insert(id.clone(), new_comm_id);
            }
            reindexed_members.push(members);
        }

        // Step 4: Compute exact internal and cross-community edge counts
        let mut internal_edges = vec![0usize; reindexed_members.len()];
        let mut cross_edges = vec![0usize; reindexed_members.len()];

        for edge in state.edges.values() {
            let comm_src = final_node_comm_map.get(&edge.source);
            let comm_tgt = final_node_comm_map.get(&edge.target);

            match (comm_src, comm_tgt) {
                (Some(&s), Some(&t)) => {
                    if s == t {
                        if s < internal_edges.len() {
                            internal_edges[s] += 1;
                        }
                    } else {
                        if s < cross_edges.len() {
                            cross_edges[s] += 1;
                        }
                        if t < cross_edges.len() {
                            cross_edges[t] += 1;
                        }
                    }
                }
                (Some(&s), None) => {
                    if s < cross_edges.len() {
                        cross_edges[s] += 1;
                    }
                }
                (None, Some(&t)) => {
                    if t < cross_edges.len() {
                        cross_edges[t] += 1;
                    }
                }
                (None, None) => {}
            }
        }

        // Step 5: Build final summaries
        let mut summaries = Vec::with_capacity(reindexed_members.len());
        for (comm_id, members) in reindexed_members.into_iter().enumerate() {
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
                internal_edges_count: internal_edges.get(comm_id).copied().unwrap_or(0),
                cross_edges_count: cross_edges.get(comm_id).copied().unwrap_or(0),
            });
        }

        (summaries, final_node_comm_map)
    }

    /// Detects communities over the CodeGraph topology and assigns heuristic labels.
    #[must_use]
    pub fn detect_communities(state: &CodeGraphStoreState) -> Vec<CommunitySummary> {
        Self::detect_community_map(state).0
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
