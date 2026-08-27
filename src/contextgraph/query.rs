/* holosphere/src/contextgraph/query.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Query Engine (Search, Explore, Traverse, Path, Diff, Impact)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Compact, budget-governed universal graph reasoning and retrieval operations
//! over multi-domain entities, hypergraph relations, and temporal snapshots.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use super::planner::{ContextBudget, QueryPlan};
use super::schema::{Entity, EntityId, EntityKind, Relation, RelationId};
use super::store::ContextGraphStoreState;

/// Compact result slice formatted for LLM context bounds.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextSlice {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub summary: String,
    pub total_found: usize,
    pub is_truncated: bool,
    pub plan_used: QueryPlan,
    pub commit_lsn: u64,
}

pub struct ContextQueryEngine;

impl ContextQueryEngine {
    /// Executes budget-bounded multi-domain search.
    #[must_use]
    pub fn search(
        state: &ContextGraphStoreState,
        query: &str,
        kinds: Option<&[String]>,
        budget: &ContextBudget,
    ) -> ContextSlice {
        let terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|s| !s.is_empty())
            .collect();

        let mut scored_entities: Vec<(&Entity, f32)> = Vec::new();

        for entity in state.entities.values() {
            if let Some(filter_kinds) = kinds {
                if !filter_kinds.iter().any(|k| *k == entity.kind.as_str()) {
                    continue;
                }
            }

            let mut score = 0.0f32;
            let label_lower = entity.label.to_lowercase();

            for term in &terms {
                if label_lower == *term {
                    score += 10.0;
                } else if label_lower.contains(term) {
                    score += 4.0;
                }
            }

            if score > 0.0 {
                scored_entities.push((entity, score));
            }
        }

        scored_entities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let total_found = scored_entities.len();
        let is_truncated = total_found > budget.max_results;

        let selected_entities: Vec<Entity> = scored_entities
            .into_iter()
            .take(budget.max_results)
            .map(|(e, _)| e.clone())
            .collect();

        let mut summary_lines = Vec::new();
        summary_lines.push(format!(
            "Found {total_found} matching entities (showing {}):",
            selected_entities.len()
        ));
        for ent in &selected_entities {
            summary_lines.push(format!("- `{}` ({})", ent.label, ent.kind));
        }

        ContextSlice {
            entities: selected_entities,
            relations: Vec::new(),
            summary: summary_lines.join("\n"),
            total_found,
            is_truncated,
            plan_used: QueryPlan::LexicalSearch,
            commit_lsn: state.commit_lsn,
        }
    }

    /// Explores one entity and its surrounding neighborhood.
    #[must_use]
    pub fn explore(
        state: &ContextGraphStoreState,
        entity_id: &EntityId,
        budget: &ContextBudget,
    ) -> Option<ContextSlice> {
        let root_entity = state.entities.get(entity_id)?;
        let mut entities = vec![root_entity.clone()];
        let mut relations = Vec::new();

        if let Some(rel_ids) = state.entity_relations.get(entity_id) {
            for rid in rel_ids.iter().take(budget.max_results) {
                if let Some(rel) = state.relations.get(rid) {
                    relations.push(rel.clone());
                    for p in &rel.participants {
                        if &p.entity_id != entity_id {
                            if let Some(neighbor) = state.entities.get(&p.entity_id) {
                                if !entities.iter().any(|e| e.id == neighbor.id) {
                                    entities.push(neighbor.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        let summary = format!(
            "Entity `{}` ({})\nConnected Relations: {}\nConnected Neighbors: {}",
            root_entity.label,
            root_entity.kind,
            relations.len(),
            entities.len() - 1
        );

        Some(ContextSlice {
            entities,
            relations,
            summary,
            total_found: 1,
            is_truncated: false,
            plan_used: QueryPlan::ExactEntityLookup,
            commit_lsn: state.commit_lsn,
        })
    }

    /// Follows relationships starting from seed entity IDs.
    #[must_use]
    pub fn traverse(
        state: &ContextGraphStoreState,
        seed_ids: &[EntityId],
        relation_kinds: Option<&[String]>,
        budget: &ContextBudget,
    ) -> ContextSlice {
        let mut visited_entities = HashSet::new();
        let mut visited_relations = HashSet::new();
        let mut queue = VecDeque::new();

        for seed in seed_ids {
            visited_entities.insert(seed.clone());
            queue.push_back((seed.clone(), 0));
        }

        while let Some((curr_id, depth)) = queue.pop_front() {
            if depth >= budget.max_depth || visited_entities.len() >= budget.max_results {
                continue;
            }

            if let Some(rel_ids) = state.entity_relations.get(&curr_id) {
                for rid in rel_ids {
                    if let Some(rel) = state.relations.get(rid) {
                        if let Some(kinds) = relation_kinds {
                            if !kinds.iter().any(|k| *k == rel.kind.as_str()) {
                                continue;
                            }
                        }

                        if visited_relations.insert(rid.clone()) {
                            for p in &rel.participants {
                                if visited_entities.insert(p.entity_id.clone()) {
                                    queue.push_back((p.entity_id.clone(), depth + 1));
                                }
                            }
                        }
                    }
                }
            }
        }

        let selected_entities: Vec<Entity> = visited_entities
            .into_iter()
            .filter_map(|id| state.entities.get(&id).cloned())
            .collect();

        let selected_relations: Vec<Relation> = visited_relations
            .into_iter()
            .filter_map(|id| state.relations.get(&id).cloned())
            .collect();

        let summary = format!(
            "Traversed {} entities and {} relations across depth ≤ {}",
            selected_entities.len(),
            selected_relations.len(),
            budget.max_depth
        );

        ContextSlice {
            entities: selected_entities,
            relations: selected_relations,
            summary,
            total_found: seed_ids.len(),
            is_truncated: false,
            plan_used: QueryPlan::GraphTraversal,
            commit_lsn: state.commit_lsn,
        }
    }

    /// Finds the shortest semantic relation path between two entities.
    #[must_use]
    pub fn path(
        state: &ContextGraphStoreState,
        from_id: &EntityId,
        to_id: &EntityId,
        budget: &ContextBudget,
    ) -> Option<ContextSlice> {
        let from_ent = state.entities.get(from_id)?;
        let to_ent = state.entities.get(to_id)?;

        let mut queue = VecDeque::new();
        let mut parent_map: HashMap<EntityId, (EntityId, Relation)> = HashMap::new();
        let mut visited = HashSet::new();

        queue.push_back((from_id.clone(), 0));
        visited.insert(from_id.clone());

        let mut reached = false;

        while let Some((curr_id, depth)) = queue.pop_front() {
            if &curr_id == to_id {
                reached = true;
                break;
            }
            if depth >= budget.max_depth {
                continue;
            }

            if let Some(rel_ids) = state.entity_relations.get(&curr_id) {
                for rid in rel_ids {
                    if let Some(rel) = state.relations.get(rid) {
                        for p in &rel.participants {
                            if visited.insert(p.entity_id.clone()) {
                                parent_map
                                    .insert(p.entity_id.clone(), (curr_id.clone(), rel.clone()));
                                queue.push_back((p.entity_id.clone(), depth + 1));
                            }
                        }
                    }
                }
            }
        }

        if !reached {
            return None;
        }

        let mut entities = Vec::new();
        let mut relations = Vec::new();
        let mut curr = to_id.clone();

        while let Some((prev_id, rel)) = parent_map.get(&curr) {
            if let Some(ent) = state.entities.get(&curr) {
                entities.push(ent.clone());
            }
            relations.push(rel.clone());
            curr = prev_id.clone();
            if curr == *from_id {
                break;
            }
        }

        if let Some(ent) = state.entities.get(from_id) {
            entities.push(ent.clone());
        }

        entities.reverse();
        relations.reverse();

        let summary = format!(
            "Discovered path: `{}` -> `{}` ({} hops)",
            from_ent.label,
            to_ent.label,
            relations.len()
        );

        Some(ContextSlice {
            entities,
            relations,
            summary,
            total_found: 1,
            is_truncated: false,
            plan_used: QueryPlan::PathSearch,
            commit_lsn: state.commit_lsn,
        })
    }

    /// Evaluates blast radius impact for a target entity.
    #[must_use]
    pub fn impact(
        state: &ContextGraphStoreState,
        target_id: &EntityId,
        budget: &ContextBudget,
    ) -> ContextSlice {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((target_id.clone(), 0));
        visited.insert(target_id.clone());

        let mut affected_relations = Vec::new();

        while let Some((curr_id, depth)) = queue.pop_front() {
            if depth >= budget.max_depth {
                continue;
            }

            if let Some(rel_ids) = state.entity_relations.get(&curr_id) {
                for rid in rel_ids {
                    if let Some(rel) = state.relations.get(rid) {
                        for p in &rel.participants {
                            if p.role == "source" || p.role == "caller" {
                                if visited.insert(p.entity_id.clone()) {
                                    affected_relations.push(rel.clone());
                                    queue.push_back((p.entity_id.clone(), depth + 1));
                                }
                            }
                        }
                    }
                }
            }
        }

        let affected_entities: Vec<Entity> = visited
            .into_iter()
            .filter_map(|id| state.entities.get(&id).cloned())
            .collect();

        let summary = format!(
            "Impact Assessment: {} downstream entities affected across depth ≤ {}",
            affected_entities.len(),
            budget.max_depth
        );

        ContextSlice {
            entities: affected_entities,
            relations: affected_relations,
            summary,
            total_found: 1,
            is_truncated: false,
            plan_used: QueryPlan::ImpactTraversal,
            commit_lsn: state.commit_lsn,
        }
    }
}
