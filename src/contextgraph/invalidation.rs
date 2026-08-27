/* holosphere/src/contextgraph/invalidation.rs */
//!▫~•◦-------------------------------‣
//! # Fine-Grained Dependency Invalidation Graph
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Tracks fine-grained dependencies between source locators, entities, and derived
//! relations to allow targeted incremental re-resolution without full graph rebuilds.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{HashMap, HashSet};

use super::schema::{EntityId, RelationId};

/// Tracks fine-grained dependency relationships across source locators and entities.
#[derive(Clone, Debug, Default)]
pub struct InvalidationGraph {
    /// Maps source locator URI -> Entities defined within it
    pub locator_to_entities: HashMap<String, HashSet<EntityId>>,
    /// Maps EntityId -> Source locator URI
    pub entity_to_locator: HashMap<EntityId, String>,
    /// Maps EntityId -> Dependent relations referencing it
    pub entity_to_dependent_relations: HashMap<EntityId, HashSet<RelationId>>,
    /// Maps EntityId -> Downstream entities referencing this entity
    pub entity_dependencies: HashMap<EntityId, HashSet<EntityId>>,
}

impl InvalidationGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers that an entity is defined by a source locator.
    pub fn register_entity_source(&mut self, entity_id: &EntityId, locator_uri: &str) {
        self.locator_to_entities
            .entry(locator_uri.to_string())
            .or_default()
            .insert(entity_id.clone());
        self.entity_to_locator
            .insert(entity_id.clone(), locator_uri.to_string());
    }

    /// Registers a dependency: `referrer` depends on `referenced`.
    pub fn register_dependency(&mut self, referrer: &EntityId, referenced: &EntityId) {
        self.entity_dependencies
            .entry(referenced.clone())
            .or_default()
            .insert(referrer.clone());
    }

    /// Computes all transitively affected entities with a safety cap on cascade depth (default depth 16).
    #[must_use]
    pub fn compute_affected_scope(&self, changed_locators: &[String]) -> HashSet<EntityId> {
        self.compute_affected_scope_bounded(changed_locators, 16)
    }

    /// Computes transitively affected entities bounded by a maximum cascade depth.
    #[must_use]
    pub fn compute_affected_scope_bounded(
        &self,
        changed_locators: &[String],
        max_depth: usize,
    ) -> HashSet<EntityId> {
        let mut affected_entities = HashSet::new();
        let mut queue = std::collections::VecDeque::new();

        // 1. Direct entities in changed locators
        for loc in changed_locators {
            if let Some(entities) = self.locator_to_entities.get(loc) {
                for id in entities {
                    if affected_entities.insert(id.clone()) {
                        queue.push_back((id.clone(), 0));
                    }
                }
            }
        }

        // 2. Transitive dependents bounded by max_depth
        while let Some((curr, depth)) = queue.pop_front() {
            if depth >= max_depth {
                continue;
            }
            if let Some(dependents) = self.entity_dependencies.get(&curr) {
                for dep in dependents {
                    if affected_entities.insert(dep.clone()) {
                        queue.push_back((dep.clone(), depth + 1));
                    }
                }
            }
        }

        affected_entities
    }

    /// Clears records for invalidated locators.
    pub fn invalidate_locators(&mut self, locators: &[String]) {
        for loc in locators {
            if let Some(entities) = self.locator_to_entities.remove(loc) {
                for id in &entities {
                    self.entity_to_locator.remove(id);
                    self.entity_dependencies.remove(id);
                    self.entity_to_dependent_relations.remove(id);
                    for referrers in self.entity_dependencies.values_mut() {
                        referrers.remove(id);
                    }
                }
            }
        }
    }
}
