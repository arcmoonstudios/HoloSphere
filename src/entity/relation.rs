/* holosphere/src/entity/relation.rs */
//!▫~•◦-------------------------------‣
//! # Relation Schemas, Instances, and N-ary Hypergraph Bindings
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides relational ontology structures supporting unscripted relation
//! discovery, N-ary hyperedges, and explicit distinction between durable
//! replicated state and physical generation-local segment bindings.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::entity::id::{EntityId, EntityIndex, NULL_ROW_REF, RelationId, RelationTypeId, RoleId};
use crate::entity::status::{EpistemicStatus, LifecycleStatus};

/// Lifecycle state of a relation type schema in the catalog.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationTypeState {
    /// Candidate relation type proposed by unscripted induction or LLM synthesis.
    Proposed = 0,
    /// Canonicalized, validated, and admitted relation type schema.
    Admitted = 1,
    /// Deprecated relation schema (e.g. merged into a canonical generalization).
    Deprecated = 2,
}

/// Dynamic relation type schema in the entity catalog.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationType {
    pub id: RelationTypeId,
    pub name: String,
    pub arity: u8,
    pub state: RelationTypeState,
    pub provenance_row: u32,
    pub is_learned: bool,
}

/// Durable canonical relation instance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationInstance {
    pub id: RelationId,
    pub relation_type: RelationTypeId,
    pub confidence: f32,
    pub provenance_row: u32,
    pub status: EpistemicStatus,
    pub lifecycle: LifecycleStatus,
    pub valid_from_lsn: u64,
    pub valid_until_lsn: Option<u64>,
}

/// Durable role binding used in replicated Raft mutations and WAL logs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DurableRoleBinding {
    pub relation_id: RelationId,
    pub entity_id: EntityId,
    pub role_id: RoleId,
}

/// Exactly-16-byte, padding-free, cache-aligned segment role binding.
///
/// Layout (16 bytes, 8-byte aligned):
/// ```text
/// offset 0  — relation_id : u64 (8 bytes) ← durable relation ID
/// offset 8  — entity      : u32 (4 bytes) ← generation-local EntityIndex
/// offset 12 — role_id     : u16 (2 bytes) ← RoleId
/// offset 14 — flags       : u16 (2 bytes) ← binding flags
/// total      16 bytes, zero padding
/// ```
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SegmentRoleBinding {
    pub relation_id: u64,
    pub entity: EntityIndex,
    pub role_id: u16,
    pub flags: u16,
}

const _: () = assert!(std::mem::size_of::<SegmentRoleBinding>() == 16);
const _: () = assert!(std::mem::align_of::<SegmentRoleBinding>() == 8);

impl Default for SegmentRoleBinding {
    fn default() -> Self {
        Self {
            relation_id: 0,
            entity: NULL_ROW_REF,
            role_id: 0,
            flags: 0,
        }
    }
}
