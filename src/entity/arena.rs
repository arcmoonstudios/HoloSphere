/* holosphere/src/entity/arena.rs */
//!▫~•◦-------------------------------‣
//! # Entity Arena & Bidirectional Identity Mapping
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides generation-local storage of `EntityHeader` records with O(1)
//! bidirectional translation between durable `EntityId` and physical `EntityIndex`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entity::header::EntityHeader;
use crate::entity::id::{EntityId, EntityIndex, NULL_ROW_REF};
use crate::entity::status::EpistemicStatus;

/// Generation-local entity arena managing physical header allocation
/// and mapping durable `EntityId`s to dense `EntityIndex`es.
pub struct EntityArena {
    headers: RwLock<Vec<EntityHeader>>,
    id_to_index: RwLock<HashMap<EntityId, EntityIndex>>,
    index_to_id: RwLock<Vec<EntityId>>,
    live: RwLock<Vec<bool>>,
    next_id: AtomicU64,
}

impl Default for EntityArena {
    fn default() -> Self {
        Self::new(1)
    }
}

impl EntityArena {
    /// Creates a new `EntityArena` starting monotonic ID allocation from `start_id`.
    pub fn new(start_id: u64) -> Self {
        Self {
            headers: RwLock::new(Vec::new()),
            id_to_index: RwLock::new(HashMap::new()),
            index_to_id: RwLock::new(Vec::new()),
            live: RwLock::new(Vec::new()),
            next_id: AtomicU64::new(start_id),
        }
    }

    /// Allocates a new entity, automatically generating a monotonic `EntityId`.
    pub fn alloc(&self, header: EntityHeader) -> (EntityId, EntityIndex) {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let idx = self.bind(id, header);
        (id, idx)
    }

    /// Binds an existing durable `EntityId` to a new generation-local `EntityIndex`.
    pub fn bind(&self, id: EntityId, header: EntityHeader) -> EntityIndex {
        let mut headers = self.headers.write();
        let mut id_map = self.id_to_index.write();
        let mut idx_vec = self.index_to_id.write();
        let mut live = self.live.write();

        let index = headers.len() as EntityIndex;
        headers.push(header);
        idx_vec.push(id);
        live.push(true);
        id_map.insert(id, index);

        index
    }

    /// Resolves `EntityId` to its current generation `EntityIndex`.
    #[inline]
    pub fn id_to_index(&self, id: EntityId) -> Option<EntityIndex> {
        self.id_to_index.read().get(&id).copied()
    }

    /// Resolves generation `EntityIndex` to its canonical durable `EntityId`.
    #[inline]
    pub fn index_to_id(&self, index: EntityIndex) -> Option<EntityId> {
        let vec = self.index_to_id.read();
        vec.get(index as usize).copied()
    }

    /// Returns a copy of the header for `index`, if live.
    #[inline]
    pub fn get(&self, index: EntityIndex) -> Option<EntityHeader> {
        let idx = index as usize;
        let live = self.live.read();
        if idx < live.len() && live[idx] {
            Some(self.headers.read()[idx])
        } else {
            None
        }
    }

    /// Returns the header and index by durable `EntityId`.
    pub fn get_by_id(&self, id: EntityId) -> Option<(EntityIndex, EntityHeader)> {
        let index = self.id_to_index(id)?;
        let header = self.get(index)?;
        Some((index, header))
    }

    /// Updates the header for an existing entity in-place.
    pub fn update(&self, index: EntityIndex, header: EntityHeader) -> bool {
        let idx = index as usize;
        let live = self.live.read();
        if idx < live.len() && live[idx] {
            drop(live);
            self.headers.write()[idx] = header;
            true
        } else {
            false
        }
    }

    /// Atomically updates the version head, provenance, and epistemic summary on an entity header.
    pub fn publish_version_head(
        &self,
        index: EntityIndex,
        new_version_row: u32,
        new_provenance_row: u32,
        new_property_row: u32,
        new_vector_row: u32,
        epistemic: EpistemicStatus,
    ) -> bool {
        let idx = index as usize;
        let live = self.live.read();
        if idx < live.len() && live[idx] {
            drop(live);
            let mut headers = self.headers.write();
            let h = &mut headers[idx];
            h.version_row = new_version_row;
            if new_provenance_row != NULL_ROW_REF {
                h.provenance_row = new_provenance_row;
            }
            if new_property_row != NULL_ROW_REF {
                h.property_row = new_property_row;
            }
            if new_vector_row != NULL_ROW_REF {
                h.vector_row = new_vector_row;
            }
            h.set_epistemic(epistemic);
            true
        } else {
            false
        }
    }

    /// Marks an entity as tombstoned without reclaiming physical slot.
    pub fn delete(&self, index: EntityIndex) -> bool {
        let idx = index as usize;
        let mut live = self.live.write();
        if idx < live.len() && live[idx] {
            live[idx] = false;
            let mut headers = self.headers.write();
            headers[idx].set_live(false);
            true
        } else {
            false
        }
    }

    #[inline]
    pub fn is_live(&self, index: EntityIndex) -> bool {
        let idx = index as usize;
        let live = self.live.read();
        idx < live.len() && live[idx]
    }

    pub fn live_count(&self) -> usize {
        self.live.read().iter().filter(|&&b| b).count()
    }

    pub fn capacity(&self) -> usize {
        self.headers.read().len()
    }

    /// Returns all live `EntityIndex`es in generation.
    pub fn live_indices(&self) -> Vec<EntityIndex> {
        self.live
            .read()
            .iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { Some(i as EntityIndex) } else { None })
            .collect()
    }

    /// Returns all live `EntityId`s in generation.
    pub fn live_ids(&self) -> Vec<EntityId> {
        let live = self.live.read();
        let idx_vec = self.index_to_id.read();
        live.iter()
            .enumerate()
            .filter_map(|(i, &b)| if b { idx_vec.get(i).copied() } else { None })
            .collect()
    }

    /// Internal snapshot of headers, live flags, and id mappings for serialization.
    pub fn snapshot_data(&self) -> (Vec<EntityHeader>, Vec<bool>, Vec<EntityId>) {
        (
            self.headers.read().clone(),
            self.live.read().clone(),
            self.index_to_id.read().clone(),
        )
    }
}
