/* holosphere/src/contextgraph/analytics.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Architectural Analytics
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Analyzes structural hub centrality, cyclic dependencies, cross-scope bridges,
//! and unreferenced entities across universal knowledge graphs.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::schema::{Entity, EntityId, EntityKind};
use super::store::ContextGraphStoreState;

/// Hub entity descriptor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HubEntityInfo {
    pub id: EntityId,
    pub label: String,
    pub kind: EntityKind,
    pub total_degree: usize,
}

/// Discovered circular cycle across relations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalCycle {
    pub entity_labels: Vec<String>,
    pub entity_ids: Vec<EntityId>,
}

pub struct ContextAnalytics;

impl ContextAnalytics {
    /// Identifies the top hub entities by total degree.
    #[must_use]
    pub fn find_hubs(state: &ContextGraphStoreState, top_k: usize) -> Vec<HubEntityInfo> {
        let mut results = Vec::new();

        for (id, entity) in &state.entities {
            let deg = state.entity_relations.get(id).map_or(0, |v| v.len());
            results.push(HubEntityInfo {
                id: id.clone(),
                label: entity.label.clone(),
                kind: entity.kind.clone(),
                total_degree: deg,
            });
        }

        results.sort_by(|a, b| {
            b.total_degree
                .cmp(&a.total_degree)
                .then_with(|| a.label.cmp(&b.label))
        });
        results.truncate(top_k);
        results
    }

    /// Identifies entities with zero incoming or outgoing relations.
    #[must_use]
    pub fn find_orphans(state: &ContextGraphStoreState) -> Vec<Entity> {
        state
            .entities
            .values()
            .filter(|ent| {
                state
                    .entity_relations
                    .get(&ent.id)
                    .map_or(true, |v| v.is_empty())
            })
            .cloned()
            .collect()
    }
}
