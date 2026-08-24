/* holosphere/src/relation/arena.rs */
//!▫~•◦-------------------------------‣
//! # Hypergraph Relation Arena Storage
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the generation-local physical storage for `RelationHeader`s,
//! `SegmentRoleBinding`s, and bidirectional `RelationId <-> RelationIndex` index tables.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::relation::binding::SegmentRoleBinding;
use crate::relation::header::RelationHeader;
use crate::relation::id::{RelationId, RelationIndex};

/// Thread-safe arena managing localized relation headers and physical role bindings.
pub struct RelationArena {
    next_relation_id: AtomicU64,
    headers: RwLock<Vec<RelationHeader>>,
    bindings: RwLock<Vec<SegmentRoleBinding>>,
    id_to_index: RwLock<HashMap<RelationId, RelationIndex>>,
    index_to_id: RwLock<Vec<RelationId>>,
    live_bitmap: RwLock<roaring::RoaringBitmap>,
}

impl RelationArena {
    pub fn new(start_relation_id: RelationId) -> Self {
        Self {
            next_relation_id: AtomicU64::new(start_relation_id),
            headers: RwLock::new(Vec::new()),
            bindings: RwLock::new(Vec::new()),
            id_to_index: RwLock::new(HashMap::new()),
            index_to_id: RwLock::new(Vec::new()),
            live_bitmap: RwLock::new(roaring::RoaringBitmap::new()),
        }
    }

    /// Allocates a relation header and appends its physical role bindings.
    pub fn alloc(
        &self,
        mut header: RelationHeader,
        role_bindings: &[SegmentRoleBinding],
    ) -> (RelationId, RelationIndex) {
        let id = self.next_relation_id.fetch_add(1, Ordering::Relaxed);

        let mut headers = self.headers.write();
        let mut bindings = self.bindings.write();
        let mut id_map = self.id_to_index.write();
        let mut idx_map = self.index_to_id.write();
        let mut live = self.live_bitmap.write();

        let index = headers.len() as RelationIndex;

        header.binding_start = bindings.len() as u32;
        header.binding_len = role_bindings.len() as u16;

        bindings.extend_from_slice(role_bindings);
        headers.push(header);
        id_map.insert(id, index);
        idx_map.push(id);
        live.insert(index);

        (id, index)
    }

    /// Binds an existing durable `RelationId` to localized physical rows.
    pub fn bind(
        &self,
        id: RelationId,
        mut header: RelationHeader,
        role_bindings: &[SegmentRoleBinding],
    ) -> RelationIndex {
        let mut headers = self.headers.write();
        let mut bindings = self.bindings.write();
        let mut id_map = self.id_to_index.write();
        let mut idx_map = self.index_to_id.write();
        let mut live = self.live_bitmap.write();

        let index = headers.len() as RelationIndex;

        header.binding_start = bindings.len() as u32;
        header.binding_len = role_bindings.len() as u16;

        bindings.extend_from_slice(role_bindings);
        headers.push(header);
        id_map.insert(id, index);
        idx_map.push(id);
        live.insert(index);

        let mut curr = self.next_relation_id.load(Ordering::Relaxed);
        while id >= curr {
            if self
                .next_relation_id
                .compare_exchange_weak(curr, id + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
            curr = self.next_relation_id.load(Ordering::Relaxed);
        }

        index
    }

    pub fn get(&self, index: RelationIndex) -> Option<RelationHeader> {
        let headers = self.headers.read();
        headers.get(index as usize).copied()
    }

    pub fn get_by_id(&self, id: RelationId) -> Option<(RelationIndex, RelationHeader)> {
        let id_map = self.id_to_index.read();
        let &index = id_map.get(&id)?;
        let headers = self.headers.read();
        let header = *headers.get(index as usize)?;
        Some((index, header))
    }

    pub fn get_bindings(&self, header: &RelationHeader) -> Vec<SegmentRoleBinding> {
        let bindings = self.bindings.read();
        let start = header.binding_start as usize;
        let end = start + (header.binding_len as usize);
        if end <= bindings.len() {
            bindings[start..end].to_vec()
        } else {
            Vec::new()
        }
    }

    pub fn update_header(&self, index: RelationIndex, header: RelationHeader) -> bool {
        let mut headers = self.headers.write();
        if let Some(h) = headers.get_mut(index as usize) {
            *h = header;
            true
        } else {
            false
        }
    }

    pub fn id_to_index(&self, id: RelationId) -> Option<RelationIndex> {
        self.id_to_index.read().get(&id).copied()
    }

    pub fn index_to_id(&self, index: RelationIndex) -> Option<RelationId> {
        self.index_to_id.read().get(index as usize).copied()
    }

    pub fn live_ids(&self) -> Vec<RelationId> {
        let live = self.live_bitmap.read();
        let idx_map = self.index_to_id.read();
        live.iter()
            .filter_map(|idx| idx_map.get(idx as usize).copied())
            .collect()
    }

    pub fn live_count(&self) -> usize {
        self.live_bitmap.read().len() as usize
    }

    pub fn delete(&self, id: RelationId) -> bool {
        let mut id_map = self.id_to_index.write();
        if let Some(index) = id_map.remove(&id) {
            self.live_bitmap.write().remove(index);
            let mut headers = self.headers.write();
            if let Some(h) = headers.get_mut(index as usize) {
                h.flags &= !crate::relation::header::RELATION_FLAG_LIVE;
                h.set_lifecycle(crate::entity::status::LifecycleStatus::Tombstoned);
            }
            true
        } else {
            false
        }
    }

    pub fn snapshot_data(
        &self,
    ) -> (
        Vec<RelationHeader>,
        Vec<RelationId>,
        Vec<SegmentRoleBinding>,
    ) {
        (
            self.headers.read().clone(),
            self.index_to_id.read().clone(),
            self.bindings.read().clone(),
        )
    }
}
