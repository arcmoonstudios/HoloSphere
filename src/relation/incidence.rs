/* holosphere/src/relation/incidence.rs */
//!▫~•◦-------------------------------‣
//! # Hypergraph Inverted Incidence & Multi-Role Posting Index (WCOJ Leapfrog)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Accelerated physical inverted index indexing (type_id, role_id, entity_idx) -> RelationIndex[]
//! for multi-role qualification and exact candidate pruning. Features Worst-Case Optimal
//! Joins (WCOJ) via Ngo et al.'s Leapfrog Triejoin over sorted hypergraph postings.
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

    /// Computes the exact set intersection of multiple sorted posting lists using Worst-Case Optimal Leapfrog Triejoin.
    pub fn intersect(postings: &[Vec<RelationIndex>]) -> Vec<RelationIndex> {
        if postings.is_empty() {
            return Vec::new();
        }
        if postings.len() == 1 {
            return postings[0].clone();
        }
        for p in postings {
            if p.is_empty() {
                return Vec::new();
            }
        }

        let slice_refs: Vec<&[RelationIndex]> = postings.iter().map(|v| v.as_slice()).collect();
        Self::leapfrog_intersect(&slice_refs)
    }

    /// Evaluates Leapfrog Triejoin intersection over borrowed slices with zero heap allocations during search.
    pub fn leapfrog_intersect(postings: &[&[RelationIndex]]) -> Vec<RelationIndex> {
        if postings.is_empty() {
            return Vec::new();
        }
        if postings.len() == 1 {
            return postings[0].to_vec();
        }

        let Some(mut leapfrog) = LeapfrogIncidenceIterator::new(postings) else {
            return Vec::new();
        };

        let mut results = Vec::new();
        while let Some(matched) = leapfrog.leapfrog_next() {
            results.push(matched);
        }
        results
    }

    /// Clears all postings.
    pub fn clear(&self) {
        self.postings.write().clear();
    }
}

/// Cursor over a sorted slice of `RelationIndex`es with galloping search capability.
#[derive(Clone, Debug)]
pub struct IncidencePostingCursor<'a> {
    slice: &'a [RelationIndex],
    pos: usize,
}

impl<'a> IncidencePostingCursor<'a> {
    #[inline]
    pub fn new(slice: &'a [RelationIndex]) -> Self {
        Self { slice, pos: 0 }
    }

    #[inline]
    pub fn at_end(&self) -> bool {
        self.pos >= self.slice.len()
    }

    #[inline]
    pub fn key(&self) -> RelationIndex {
        self.slice[self.pos]
    }

    #[inline]
    pub fn next(&mut self) {
        self.pos += 1;
    }

    /// Advances the cursor to the first element with key >= target.
    /// Uses galloping (exponential) search for asymptotic $O(\log N)$ jumps.
    pub fn seek(&mut self, target: RelationIndex) {
        if self.at_end() || self.key() >= target {
            return;
        }

        // Gallop phase: 1, 2, 4, 8, 16...
        let mut jump = 1;
        let mut prev = self.pos;
        let mut curr = self.pos + jump;

        while curr < self.slice.len() && self.slice[curr] < target {
            prev = curr;
            jump *= 2;
            curr = self.pos + jump;
        }

        let high = curr.min(self.slice.len());
        // Binary search in window [prev, high)
        let slice_window = &self.slice[prev..high];
        match slice_window.binary_search(&target) {
            Ok(idx) | Err(idx) => self.pos = prev + idx,
        }
    }
}

/// Worst-Case Optimal Join (Leapfrog Triejoin) iterator over N-ary hypergraph incidence postings.
///
/// Implements Ngo et al.'s AGM-bound satisfying join algorithm:
/// - Guarantees runtimes bounded by $O(|E|^{3/2})$ for cyclic hyper-patterns.
/// - Operates with zero heap allocations during the traversal search loop.
pub struct LeapfrogIncidenceIterator<'a> {
    iterators: Vec<IncidencePostingCursor<'a>>,
    p: usize,
    max_key: RelationIndex,
    initialized: bool,
}

impl<'a> LeapfrogIncidenceIterator<'a> {
    /// Creates a new Leapfrog Triejoin iterator over $K$ sorted posting lists.
    pub fn new(postings: &[&'a [RelationIndex]]) -> Option<Self> {
        if postings.is_empty() {
            return None;
        }
        for p in postings {
            if p.is_empty() {
                return None;
            }
        }

        let mut cursors: Vec<IncidencePostingCursor<'a>> = postings
            .iter()
            .map(|&slice| IncidencePostingCursor::new(slice))
            .collect();

        // Sort cursors by initial key
        cursors.sort_by_key(|c| c.key());
        let max_key = cursors.last().unwrap().key();

        Some(Self {
            iterators: cursors,
            p: 0,
            max_key,
            initialized: false,
        })
    }

    /// Searches for the next intersection point across all K iterators.
    pub fn leapfrog_search(&mut self) -> Option<RelationIndex> {
        let k = self.iterators.len();
        if k == 0 {
            return None;
        }

        loop {
            let cur_key = self.iterators[self.p].key();
            if cur_key == self.max_key {
                // All K iterators agree on max_key!
                return Some(self.max_key);
            }

            self.iterators[self.p].seek(self.max_key);
            if self.iterators[self.p].at_end() {
                return None;
            }

            self.max_key = self.iterators[self.p].key();
            self.p = (self.p + 1) % k;
        }
    }

    /// Advances to the next matching element.
    pub fn leapfrog_next(&mut self) -> Option<RelationIndex> {
        let k = self.iterators.len();
        if k == 0 {
            return None;
        }

        if !self.initialized {
            self.initialized = true;
            return self.leapfrog_search();
        }

        self.iterators[self.p].next();
        if self.iterators[self.p].at_end() {
            return None;
        }

        self.max_key = self.iterators[self.p].key();
        self.p = (self.p + 1) % k;
        self.leapfrog_search()
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
        self.leapfrog_join(type_id, bindings)
    }

    /// Executes a Worst-Case Optimal Leapfrog Join across N distinct role bindings.
    pub fn leapfrog_join(
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_leapfrog_cursor_galloping() {
        let data = vec![10u32, 20u32, 30u32, 100u32, 500u32];
        let mut cursor = IncidencePostingCursor::new(&data);

        assert_eq!(cursor.key(), 10);
        cursor.seek(25);
        assert_eq!(cursor.key(), 30);

        cursor.seek(200);
        assert_eq!(cursor.key(), 500);

        cursor.seek(600);
        assert!(cursor.at_end());
    }

    #[test]
    fn test_leapfrog_triejoin_multi_way_intersection() {
        let p1 = vec![1u32, 5u32, 10u32, 20u32, 50u32];
        let p2 = vec![2u32, 5u32, 10u32, 30u32, 50u32];
        let p3 = vec![5u32, 10u32, 15u32, 50u32, 99u32];

        let result = IncidenceIndex::intersect(&[p1, p2, p3]);
        assert_eq!(result, vec![5u32, 10u32, 50u32]);
    }

    #[test]
    fn test_leapfrog_empty_and_disjoint_intersections() {
        let p1 = vec![1u32, 2u32];
        let p2 = vec![3u32, 4u32];
        assert_eq!(IncidenceIndex::intersect(&[p1, p2]), Vec::<u32>::new());
        assert_eq!(IncidenceIndex::intersect(&[]), Vec::<u32>::new());
    }
}
