/* holosphere/src/contextgraph/community.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Scopes & Modularity Clustering
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Partitions multi-domain knowledge graphs into coherent architectural scopes and communities
//! using deterministic Newman-Girvan modularity optimization.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::schema::EntityId;
use super::store::ContextGraphStoreState;

/// Topological scope summary.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScopeSummary {
    pub scope_id: usize,
    pub label: String,
    pub entity_count: usize,
    pub top_entities: Vec<String>,
}

pub struct ScopeClustering;

impl ScopeClustering {
    /// Detects topological scopes/communities over the graph using deterministic Modularity optimization.
    #[must_use]
    pub fn detect_scopes(state: &ContextGraphStoreState) -> Vec<ScopeSummary> {
        if state.entities.is_empty() {
            return Vec::new();
        }

        let mut adj: HashMap<&EntityId, Vec<&EntityId>> = HashMap::new();
        let mut total_edges = 0usize;

        for rel in state.relations.values() {
            for i in 0..rel.participants.len() {
                for j in (i + 1)..rel.participants.len() {
                    let a = &rel.participants[i].entity_id;
                    let b = &rel.participants[j].entity_id;
                    adj.entry(a).or_default().push(b);
                    adj.entry(b).or_default().push(a);
                    total_edges += 1;
                }
            }
        }

        let mut sorted_ids: Vec<&EntityId> = state.entities.keys().collect();
        sorted_ids.sort();

        let total_m2 = (total_edges * 2).max(1) as f64;

        // Step 1: Initial unique partition
        let mut node_to_scope: HashMap<&EntityId, usize> = HashMap::new();
        let mut comm_tot_degree: HashMap<usize, usize> = HashMap::new();
        let mut node_degrees: HashMap<&EntityId, usize> = HashMap::new();

        for (idx, &id) in sorted_ids.iter().enumerate() {
            let deg = adj.get(id).map_or(0, |nbrs| nbrs.len());
            node_to_scope.insert(id, idx);
            comm_tot_degree.insert(idx, deg);
            node_degrees.insert(id, deg);
        }

        // Step 2: Deterministic Modularity Optimization (Louvain greedy modularity passes)
        let max_passes = 20;
        for _ in 0..max_passes {
            let mut moved = false;

            for &id in &sorted_ids {
                let k_i = match node_degrees.get(id) {
                    Some(&d) if d > 0 => d as f64,
                    _ => continue,
                };
                let curr_comm = match node_to_scope.get(id) {
                    Some(&c) => c,
                    None => continue,
                };

                let neighbors = match adj.get(id) {
                    Some(nbrs) => nbrs,
                    None => continue,
                };

                let mut comm_edge_weights: HashMap<usize, usize> = HashMap::new();
                for &nbr in neighbors {
                    if let Some(&nbr_comm) = node_to_scope.get(nbr) {
                        *comm_edge_weights.entry(nbr_comm).or_default() += 1;
                    }
                }

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
                    *comm_tot_degree.entry(curr_comm).or_default() -= k_i as usize;
                    *comm_tot_degree.entry(best_comm).or_default() += k_i as usize;
                    node_to_scope.insert(id, best_comm);
                    moved = true;
                }
            }

            if !moved {
                break;
            }
        }

        // Step 3: Group by converged scope
        let mut raw_scope_members: BTreeMap<usize, Vec<&EntityId>> = BTreeMap::new();
        for (&id, &sid) in &node_to_scope {
            raw_scope_members.entry(sid).or_default().push(id);
        }

        let mut sorted_scopes: Vec<(usize, Vec<&EntityId>)> =
            raw_scope_members.into_iter().collect();
        for (_, members) in &mut sorted_scopes {
            members.sort();
        }
        sorted_scopes.sort_by(|a, b| {
            b.1.len()
                .cmp(&a.1.len())
                .then_with(|| a.1.first().cmp(&b.1.first()))
        });

        let mut summaries = Vec::with_capacity(sorted_scopes.len());
        for (new_scope_id, (_old_id, members)) in sorted_scopes.into_iter().enumerate() {
            let entity_count = members.len();

            let mut top_entities = Vec::new();
            for &id in members.iter().take(5) {
                if let Some(ent) = state.entities.get(id) {
                    top_entities.push(ent.label.clone());
                }
            }

            let label = top_entities
                .first()
                .cloned()
                .unwrap_or_else(|| format!("Scope {new_scope_id}"));

            summaries.push(ScopeSummary {
                scope_id: new_scope_id,
                label: format!("Scope: {label}"),
                entity_count,
                top_entities,
            });
        }

        summaries
    }
}
