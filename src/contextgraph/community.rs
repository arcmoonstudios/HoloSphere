/* holosphere/src/contextgraph/community.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Scopes & Community Clustering
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Partitions multi-domain knowledge graphs into coherent scopes (directories, packages,
//! services, detected topological communities) with heuristic label derivation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

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
    /// Detects topological scopes/communities over the graph.
    #[must_use]
    pub fn detect_scopes(state: &ContextGraphStoreState) -> Vec<ScopeSummary> {
        if state.entities.is_empty() {
            return Vec::new();
        }

        let mut adj: HashMap<&EntityId, HashSet<&EntityId>> = HashMap::new();
        for rel in state.relations.values() {
            for i in 0..rel.participants.len() {
                for j in (i + 1)..rel.participants.len() {
                    let a = &rel.participants[i].entity_id;
                    let b = &rel.participants[j].entity_id;
                    adj.entry(a).or_default().insert(b);
                    adj.entry(b).or_default().insert(a);
                }
            }
        }

        let mut node_to_scope: HashMap<&EntityId, usize> = HashMap::new();
        let mut next_scope_id = 0;

        let mut sorted_ids: Vec<&EntityId> = state.entities.keys().collect();
        sorted_ids.sort();

        for id in sorted_ids {
            if node_to_scope.contains_key(id) {
                continue;
            }
            let scope_id = next_scope_id;
            next_scope_id += 1;

            let mut queue = VecDeque::new();
            queue.push_back(id);
            node_to_scope.insert(id, scope_id);

            while let Some(curr) = queue.pop_front() {
                if let Some(neighbors) = adj.get(curr) {
                    for &nbr in neighbors {
                        if !node_to_scope.contains_key(nbr) {
                            node_to_scope.insert(nbr, scope_id);
                            queue.push_back(nbr);
                        }
                    }
                }
            }
        }

        let mut scope_members: BTreeMap<usize, Vec<&EntityId>> = BTreeMap::new();
        for (id, sid) in node_to_scope {
            scope_members.entry(sid).or_default().push(id);
        }

        let mut summaries = Vec::new();
        for (sid, mut members) in scope_members {
            members.sort();
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
                .unwrap_or_else(|| format!("Scope {sid}"));

            summaries.push(ScopeSummary {
                scope_id: sid,
                label: format!("Scope: {label}"),
                entity_count,
                top_entities,
            });
        }

        summaries.sort_by(|a, b| b.entity_count.cmp(&a.entity_count));
        summaries
    }
}
