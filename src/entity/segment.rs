/* holosphere/src/entity/segment.rs */
//!▫~•◦-------------------------------‣
//! # Unified Entity Segment Structure & Compaction Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the physical `EntitySegment` container grouping entity headers,
//! version histories, provenance rows, and columnar property/vector planes
//! under a unified snapshot generation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::sync::Arc;

use crate::entity::arena::EntityArena;
use crate::entity::header::EntityHeader;
use crate::entity::id::{EntityId, EntityIndex, NULL_ROW_REF, VectorLayout};
use crate::entity::provenance::ProvenanceArena;
use crate::entity::read::EntityReadSnapshot;
use crate::entity::vector::VectorArena;
use crate::entity::version::VersionTable;

/// Represents a single immutable or mutable snapshot generation of the entity universe.
pub struct EntitySegment {
    pub generation_id: u64,
    pub arena: Arc<EntityArena>,
    pub provenance: Arc<ProvenanceArena>,
    pub versions: Arc<VersionTable>,
    pub vector_arena: Arc<VectorArena>,
    pub vector_layouts: RwLock<Vec<VectorLayout>>,
}

impl EntitySegment {
    /// Creates a new mutable entity segment for generation `generation_id` with default vector dimension 128.
    pub fn new(generation_id: u64, start_entity_id: u64) -> Self {
        Self::with_dimension(generation_id, start_entity_id, 128)
    }

    /// Creates a new mutable entity segment with an explicit vector dimension.
    pub fn with_dimension(generation_id: u64, start_entity_id: u64, dimension: usize) -> Self {
        Self {
            generation_id,
            arena: Arc::new(EntityArena::new(start_entity_id)),
            provenance: Arc::new(ProvenanceArena::new(1)),
            versions: Arc::new(VersionTable::new(1)),
            vector_arena: Arc::new(VectorArena::new(dimension)),
            vector_layouts: RwLock::new(Vec::new()),
        }
    }

    /// Creates an immutable point-in-time read snapshot of this segment pinned at committed `lsn`.
    pub fn read_snapshot(self: &Arc<Self>, lsn: u64) -> EntityReadSnapshot {
        EntityReadSnapshot::new(lsn, Arc::clone(self))
    }

    /// Registers a vector storage layout and returns its assigned layout ID.
    pub fn register_vector_layout(&self, layout: VectorLayout) -> u16 {
        let mut layouts = self.vector_layouts.write();
        let id = layouts.len() as u16;
        let mut l = layout;
        l.layout_id = id;
        layouts.push(l);
        id
    }

    /// Allocates an entity within this segment.
    pub fn alloc_entity(&self, header: EntityHeader) -> (EntityId, EntityIndex) {
        self.arena.alloc(header)
    }

    /// Resolves `EntityId` to its header and index.
    pub fn get_entity_by_id(&self, id: EntityId) -> Option<(EntityIndex, EntityHeader)> {
        self.arena.get_by_id(id)
    }

    /// Performs physical compaction, re-allocating dense generation-local `EntityIndex`es
    /// only for live entities while preserving exact durable `EntityId`s, `VersionId`s, and `ProvenanceId`s.
    pub fn compact(&self, new_generation_id: u64) -> Arc<Self> {
        let compacted = Arc::new(Self::with_dimension(
            new_generation_id,
            1,
            self.vector_arena.dimension(),
        ));
        *compacted.vector_layouts.write() = self.vector_layouts.read().clone();

        // 1. Copy over provenance arena
        let (p_rows, p_ids, _p_ev, p_str) = self.provenance.snapshot_data();
        for s in &p_str {
            compacted.provenance.intern_string(s);
        }
        for (i, _prow) in p_rows.iter().enumerate() {
            if i < p_ids.len() {
                if let Some(record) = self.provenance.resolve_record(i as u32) {
                    compacted.provenance.bind(p_ids[i], &record);
                }
            }
        }

        // 2. Copy over version table
        let (v_rows, v_ids) = self.versions.snapshot_data();
        for (i, &vrow) in v_rows.iter().enumerate() {
            if i < v_ids.len() {
                compacted.versions.bind(v_ids[i], vrow);
            }
        }

        // 3. Compact live entities into dense contiguous rows and copy vectors
        for id in self.arena.live_ids() {
            if let Some((_old_idx, mut header)) = self.arena.get_by_id(id) {
                if header.vector_row != NULL_ROW_REF {
                    if let Some(vec_data) = self.vector_arena.get_row(header.vector_row) {
                        if let Some(new_vrow) = compacted.vector_arena.append(&vec_data) {
                            header.vector_row = new_vrow;
                        }
                    }
                }
                compacted.arena.bind(id, header);
            }
        }

        compacted
    }
}
