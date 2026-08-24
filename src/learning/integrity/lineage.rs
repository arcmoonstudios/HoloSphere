/* holosphere/src/learning/integrity/lineage.rs */
//!▫~•◦-------------------------------‣
//! # Epistemic Lineage & Derivation Graph
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Preserves the complete causal derivation graph of beliefs, hypotheses, and
//! relations, strictly distinguishing direct empirical observation roots from
//! inferred or synthesized descendants.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use crate::entity::id::EntityId;

/// Unique identifier for an empirical observation root (the immutable source event).
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct EmpiricalRootId(pub u64);

/// Classification of an evidential node in the epistemic lineage graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LineageNodeKind {
    /// Direct physical observation or empirical attempt outcome.
    DirectObservation(EmpiricalRootId),
    /// Inferred relation or structural analogy candidate.
    InferredHypothesis(EntityId),
    /// Composed multi-hop closure or synthesis candidate.
    SynthesizedPlan(EntityId),
    /// Swarm multi-agent consensus arbitration.
    SwarmConsensus(EntityId),
}

/// Epistemic lineage graph tracking parent-child dependencies across learning iterations.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct EpistemicLineageGraph {
    parents: HashMap<EntityId, HashSet<EntityId>>,
    empirical_roots: HashMap<EntityId, HashSet<EmpiricalRootId>>,
    node_kinds: HashMap<EntityId, LineageNodeKind>,
}

impl EpistemicLineageGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a direct empirical observation node.
    pub fn register_observation(&mut self, entity: EntityId, root: EmpiricalRootId) {
        self.node_kinds
            .insert(entity, LineageNodeKind::DirectObservation(root));
        let mut roots = HashSet::new();
        roots.insert(root);
        self.empirical_roots.insert(entity, roots);
        self.parents.entry(entity).or_default();
    }

    /// Registers a derived node with explicit parent dependencies.
    pub fn register_derivation(
        &mut self,
        derived_entity: EntityId,
        kind: LineageNodeKind,
        parent_entities: &[EntityId],
    ) {
        self.node_kinds.insert(derived_entity, kind);
        let mut parent_set = HashSet::new();
        let mut accumulated_roots = HashSet::new();

        for &p in parent_entities {
            parent_set.insert(p);
            if let Some(p_roots) = self.empirical_roots.get(&p) {
                accumulated_roots.extend(p_roots.iter().copied());
            }
        }

        self.parents.insert(derived_entity, parent_set);
        self.empirical_roots
            .insert(derived_entity, accumulated_roots);
    }

    /// Returns the set of direct empirical roots underlying a node.
    pub fn get_empirical_roots(&self, entity: EntityId) -> HashSet<EmpiricalRootId> {
        self.empirical_roots
            .get(&entity)
            .cloned()
            .unwrap_or_default()
    }

    /// Returns the number of distinct empirical roots supporting a node.
    pub fn independent_empirical_root_count(&self, entity: EntityId) -> usize {
        self.empirical_roots.get(&entity).map_or(0, |r| r.len())
    }

    /// Checks if adding a parent dependency from `ancestor` to `target` would create a cycle.
    pub fn would_create_cycle(&self, target: EntityId, ancestor: EntityId) -> bool {
        if target == ancestor {
            return true;
        }
        let mut visited = HashSet::new();
        let mut stack = vec![ancestor];

        while let Some(current) = stack.pop() {
            if current == target {
                return true;
            }
            if visited.insert(current) {
                if let Some(pars) = self.parents.get(&current) {
                    for &p in pars {
                        stack.push(p);
                    }
                }
            }
        }
        false
    }
}
