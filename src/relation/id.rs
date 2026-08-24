/* holosphere/src/relation/id.rs */
//!▫~•◦-------------------------------‣
//! # Hypergraph Relation Identity Types
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the frozen identity classes for dynamic hypergraph relations:
//! - `RelationId`: 64-bit durable, cluster-wide unique relation identifier.
//! - `RelationIndex`: 32-bit generation-local dense row index.
//! - `RelationTypeId`: 32-bit relation schema type identifier.
//! - `RelationVersionId`: 64-bit durable relation temporal version identifier.
//! - `RoleId`: 16-bit semantic role tag within an N-ary relation schema.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::id::EntityId;

/// Durable 64-bit cluster-wide unique relation identifier.
pub type RelationId = u64;

/// Generation-local 32-bit dense physical row index within a relation segment.
pub type RelationIndex = u32;

/// Schema identifier denoting the hypergraph relation type.
pub type RelationTypeId = u32;

/// Durable 64-bit identifier for relation historical versions.
pub type RelationVersionId = u64;

/// 16-bit semantic role identifier defined within a relation type schema.
pub type RoleId = u16;

/// Durable semantic role binding associating an `EntityId` with a `RoleId`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DurableRoleBinding {
    pub entity_id: EntityId,
    pub role_id: RoleId,
}

impl PartialOrd for DurableRoleBinding {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DurableRoleBinding {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Canonical ordering: role_id ASC, then entity_id ASC
        self.role_id
            .cmp(&other.role_id)
            .then_with(|| self.entity_id.cmp(&other.entity_id))
    }
}
