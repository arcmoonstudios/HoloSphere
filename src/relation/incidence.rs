/* holosphere/src/relation/incidence.rs */
//!▫~•◦-------------------------------‣
//! # Hypergraph Inverted Incidence & Multi-Role Posting Index
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Accelerated physical inverted index indexing (type_id, role_id, entity_idx) -> RelationIndex[]
//! for multi-role qualification and exact candidate pruning.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;

use crate::entity::id::EntityIndex;
use crate::relation::id::{RelationIndex, RelationTypeId, RoleId};

/// Key indexing role bindings in the inverted hypergraph posting table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IncidenceKey {
    pub type_id: RelationTypeId,
    pub role_id: RoleId,
    pub entity: EntityIndex,
}

/// Thread-safe inverted posting index accelerating N-ary relation queries.
pub struct IncidenceIndex {
    postings: RwLock<HashMap<IncidenceKey, Vec<RelationIndex>>>,
}

impl Default for IncidenceIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl IncidenceIndex {
    pub fn new() -> Self {
        Self {
            postings: RwLock::new(HashMap::new()),
        }
    }

    /// Inserts a relation index into the incidence posting list for `(type_id, role_id, entity)`.
    pub fn insert(
        &self,
        type_id: RelationTypeId,
        role_id: RoleId,
        entity: EntityIndex,
        relation: RelationIndex,
    ) {
        let key = IncidenceKey {
            type_id,
            role_id,
            entity,
        };
        let mut map = self.postings.write();
        let list = map.entry(key).or_default();
        if let Err(pos) = list.binary_search(&relation) {
            list.insert(pos, relation);
        }
    }

    /// Looks up the sorted list of `RelationIndex`es matching `(type_id, role_id, entity)`.
    pub fn lookup(
        &self,
        type_id: RelationTypeId,
        role_id: RoleId,
        entity: EntityIndex,
    ) -> Vec<RelationIndex> {
        let key = IncidenceKey {
            type_id,
            role_id,
            entity,
        };
        let map = self.postings.read();
        map.get(&key).cloned().unwrap_or_default()
    }

    /// Computes the exact set intersection of multiple sorted posting lists.
    pub fn intersect(postings: &[Vec<RelationIndex>]) -> Vec<RelationIndex> {
        if postings.is_empty() {
            return Vec::new();
        }
        let mut min_idx = 0;
        let mut min_len = postings[0].len();
        for (i, p) in postings.iter().enumerate().skip(1) {
            if p.len() < min_len {
                min_len = p.len();
                min_idx = i;
            }
        }

        let base = &postings[min_idx];
        let mut result = Vec::with_capacity(base.len());

        for &elem in base {
            let mut in_all = true;
            for (i, p) in postings.iter().enumerate() {
                if i != min_idx && p.binary_search(&elem).is_err() {
                    in_all = false;
                    break;
                }
            }
            if in_all {
                result.push(elem);
            }
        }

        result
    }

    /// Clears all postings.
    pub fn clear(&self) {
        self.postings.write().clear();
    }
}

/// Unified traverser over both 2-ary graph CSR/CSC and N-ary hypergraph incidence postings.
pub struct IncidenceTraverser<'a> {
    pub incidence: &'a IncidenceIndex,
}

impl<'a> IncidenceTraverser<'a> {
    pub fn new(incidence: &'a IncidenceIndex) -> Self {
        Self { incidence }
    }

    /// Dispatches query to single or multi-role incidence intersection.
    pub fn query_role_intersection(
        &self,
        type_id: RelationTypeId,
        bindings: &[(RoleId, EntityIndex)],
    ) -> Vec<RelationIndex> {
        if bindings.is_empty() {
            return Vec::new();
        }
        let posting_lists: Vec<Vec<RelationIndex>> = bindings
            .iter()
            .map(|&(role_id, entity)| self.incidence.lookup(type_id, role_id, entity))
            .collect();
        IncidenceIndex::intersect(&posting_lists)
    }
}
