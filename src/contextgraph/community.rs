/* holosphere/src/contextgraph/community.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Scopes & Modularity Clustering
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Partitions multi-domain knowledge graphs into coherent architectural scopes and communities
//! using deterministic Label Propagation clustering with neighbor density weighting.
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
    /// Detects topological scopes/communities over the graph using deterministic Label Propagation.
    #[must_use]
    pub fn detect_scopes(state: &ContextGraphStoreState) -> Vec<ScopeSummary> {
        if state.entities.is_empty() {
            return Vec::new();
        }

        let mut adj: HashMap<&EntityId, Vec<&EntityId>> = HashMap::new();
        for rel in state.relations.values() {
            for i in 0..rel.participants.len() {
                for j in (i + 1)..rel.participants.len() {
                    let a = &rel.participants[i].entity_id;
                    let b = &rel.participants[j].entity_id;
                    adj.entry(a).or_default().push(b);
                    adj.entry(b).or_default().push(a);
                }
            }
        }

        let mut sorted_ids: Vec<&EntityId> = state.entities.keys().collect();
        sorted_ids.sort();

        // Step 1: Initial unique partition
        let mut node_to_scope: HashMap<&EntityId, usize> = HashMap::new();
        for (idx, &id) in sorted_ids.iter().enumerate() {
            node_to_scope.insert(id, idx);
        }

        // Step 2: Deterministic Label Propagation (up to 15 iterations)
        let max_iterations = 15;
        for _ in 0..max_iterations {
            let mut changed = false;

            for &id in &sorted_ids {
                let current_label = match node_to_scope.get(id) {
                    Some(&lbl) => lbl,
                    None => continue,
                };

                let neighbors = match adj.get(id) {
                    Some(nbrs) if !nbrs.is_empty() => nbrs,
                    _ => continue,
                };

                let mut label_weights: HashMap<usize, usize> = HashMap::new();
                label_weights.insert(current_label, 1);

                for &nbr in neighbors {
                    if let Some(&nbr_label) = node_to_scope.get(nbr) {
                        *label_weights.entry(nbr_label).or_default() += 2;
                    }
                }

                let mut best_label = current_label;
                let mut best_weight = 0;

                for (&lbl, &weight) in &label_weights {
                    if weight > best_weight || (weight == best_weight && lbl < best_label) {
                        best_weight = weight;
                        best_label = lbl;
                    }
                }

                if best_label != current_label {
                    node_to_scope.insert(id, best_label);
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        // Step 3: Group by converged scope
        let mut raw_scope_members: BTreeMap<usize, Vec<&EntityId>> = BTreeMap::new();
        for (&id, &sid) in &node_to_scope {
            raw_scope_members.entry(sid).or_default().push(id);
        }

        let mut sorted_scopes: Vec<(usize, Vec<&EntityId>)> = raw_scope_members.into_iter().collect();
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
