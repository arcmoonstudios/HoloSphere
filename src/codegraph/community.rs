/* holosphere/src/codegraph/community.rs */
//!▫~•◦-------------------------------‣
//! # Deterministic Modularity Community Detection & Architectural Labeling
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Partitions the codebase structural graph into dense, coherent architectural modules using
//! deterministic Newman-Girvan modularity optimization, computing exact internal/cross-community
//! edge density and heuristic architectural labels.
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
        let mut total_edges_count = 0usize;

        for edge in state.edges.values() {
            adj.entry(&edge.source).or_default().push(&edge.target);
            adj.entry(&edge.target).or_default().push(&edge.source);
            total_edges_count += 1;
        }

        let mut sorted_node_ids: Vec<&CodeNodeId> = state.nodes.keys().collect();
        sorted_node_ids.sort();

        let total_m2 = (total_edges_count * 2).max(1) as f64;

        // Step 1: Initial partition (each node starts in its own unique community)
        let mut node_to_comm: HashMap<&CodeNodeId, usize> = HashMap::new();
        let mut comm_tot_degree: HashMap<usize, usize> = HashMap::new();
        let mut node_degrees: HashMap<&CodeNodeId, usize> = HashMap::new();

        for (idx, &node_id) in sorted_node_ids.iter().enumerate() {
            let deg = adj.get(node_id).map_or(0, |nbrs| nbrs.len());
            node_to_comm.insert(node_id, idx);
            comm_tot_degree.insert(idx, deg);
            node_degrees.insert(node_id, deg);
        }

        // Step 2: Deterministic Modularity Optimization (Louvain greedy modularity passes)
        let max_passes = 20;
        for _ in 0..max_passes {
            let mut moved = false;

            for &node_id in &sorted_node_ids {
                let k_i = match node_degrees.get(node_id) {
                    Some(&d) if d > 0 => d as f64,
                    _ => continue,
                };
                let curr_comm = match node_to_comm.get(node_id) {
                    Some(&c) => c,
                    None => continue,
                };

                let neighbors = match adj.get(node_id) {
                    Some(nbrs) => nbrs,
                    None => continue,
                };

                // Compute edge weights to neighboring communities
                let mut comm_edge_weights: HashMap<usize, usize> = HashMap::new();
                for &nbr in neighbors {
                    if let Some(&nbr_comm) = node_to_comm.get(nbr) {
                        *comm_edge_weights.entry(nbr_comm).or_default() += 1;
                    }
                }

                // Remove node from current community totals
                let curr_tot = comm_tot_degree.get(&curr_comm).copied().unwrap_or(0);
                let sigma_tot_curr_without_i = curr_tot.saturating_sub(k_i as usize) as f64;

                let mut best_comm = curr_comm;
                let k_i_in_curr = comm_edge_weights.get(&curr_comm).copied().unwrap_or(0) as f64;
                let mut max_delta_q = k_i_in_curr - (k_i * sigma_tot_curr_without_i) / total_m2;

                for (&target_comm, &k_i_in_target) in &comm_edge_weights {
                    if target_comm == curr_comm {
                        continue;
                    }
                    let sigma_tot_target =
                        comm_tot_degree.get(&target_comm).copied().unwrap_or(0) as f64;
                    let delta_q = (k_i_in_target as f64) - (k_i * sigma_tot_target) / total_m2;

                    if delta_q > max_delta_q
                        || ((delta_q - max_delta_q).abs() < 1e-9 && target_comm < best_comm)
                    {
                        max_delta_q = delta_q;
                        best_comm = target_comm;
                    }
                }

                if best_comm != curr_comm {
                    // Update totals
                    *comm_tot_degree.entry(curr_comm).or_default() -= k_i as usize;
                    *comm_tot_degree.entry(best_comm).or_default() += k_i as usize;
                    node_to_comm.insert(node_id, best_comm);
                    moved = true;
                }
            }

            if !moved {
                break;
            }
        }

        // Step 3: Group members by converged community
        let mut raw_comm_members: BTreeMap<usize, Vec<&CodeNodeId>> = BTreeMap::new();
        for (&node_id, &comm_id) in &node_to_comm {
            raw_comm_members.entry(comm_id).or_default().push(node_id);
        }

        // Sort communities by size descending, then by smallest node ID for total determinism
        let mut sorted_comms: Vec<(usize, Vec<&CodeNodeId>)> =
            raw_comm_members.into_iter().collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegraph::schema::{
        CodeEdge, CodeEdgeId, CodeNode, CodeRelation, Language, RelationOrigin, SourceSpan,
    };

    #[test]
    fn test_community_detection_and_exact_edge_counts() {
        let mut state = CodeGraphStoreState::default();

        // Create 2 dense clusters connected by 1 cross edge:
        // Cluster A: a1, a2, a3 (triangle: a1-a2, a2-a3, a3-a1)
        // Cluster B: b1, b2, b3 (triangle: b1-b2, b2-b3, b3-b1)
        // Bridge: a1 -> b1
        let nodes = ["a1", "a2", "a3", "b1", "b2", "b3"];
        for name in &nodes {
            let id = CodeNodeId(name.to_string());
            state.nodes.insert(
                id.clone(),
                CodeNode {
                    id: id.clone(),
                    name: name.to_string(),
                    qualified_name: name.to_string(),
                    kind: CodeNodeKind::Function,
                    language: Language::Rust,
                    source_file: PathBuf::from(format!("src/{name}.rs")),
                    source_span: SourceSpan::default(),
                    symbol_hash: [0; 32],
                    file_hash: [0; 32],
                    docstring: None,
                    signature: None,
                    attributes: BTreeMap::new(),
                    evidence_class: crate::transport::model_gateway::EvidenceClass::Observation,
                    verification_state:
                        crate::transport::model_gateway::VerificationState::Verified,
                },
            );
        }

        let edges = [
            ("a1", "a2"),
            ("a2", "a3"),
            ("a3", "a1"),
            ("b1", "b2"),
            ("b2", "b3"),
            ("b3", "b1"),
            ("a1", "b1"), // cross-cluster bridge
        ];

        for (src, tgt) in edges {
            let src_id = CodeNodeId(src.to_string());
            let tgt_id = CodeNodeId(tgt.to_string());
            let edge_id = CodeEdgeId(format!("{src}->{tgt}"));

            state.edges.insert(
                edge_id.clone(),
                CodeEdge {
                    id: edge_id.clone(),
                    source: src_id.clone(),
                    target: tgt_id.clone(),
                    relation: CodeRelation::Calls,
                    origin: RelationOrigin::Extracted,
                    confidence: 1.0,
                    evidence: SourceSpan::default(),
                    attributes: BTreeMap::new(),
                },
            );

            state
                .outgoing_edges
                .entry(src_id.clone())
                .or_default()
                .push(edge_id.clone());
            state
                .incoming_edges
                .entry(tgt_id)
                .or_default()
                .push(edge_id);
        }

        let (summaries, node_map) = CommunityDetector::detect_community_map(&state);

        // Should partition into 2 distinct communities of size 3 each
        assert_eq!(summaries.len(), 2);
        assert_eq!(summaries[0].symbol_count, 3);
        assert_eq!(summaries[1].symbol_count, 3);

        // Every node must be mapped to its respective community
        assert_eq!(node_map.len(), 6);

        let comm_a = node_map
            .get(&CodeNodeId("a1".to_string()))
            .copied()
            .unwrap();
        let comm_b = node_map
            .get(&CodeNodeId("b1".to_string()))
            .copied()
            .unwrap();
        assert_ne!(comm_a, comm_b);

        // Verify internal and cross edge counts are exact
        assert_eq!(summaries[0].internal_edges_count, 3);
        assert_eq!(summaries[1].internal_edges_count, 3);
        assert_eq!(summaries[0].cross_edges_count, 1);
        assert_eq!(summaries[1].cross_edges_count, 1);
    }
}
