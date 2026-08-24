/* holosphere/src/relation/binding.rs */
//!▫~•◦-------------------------------‣
//! # Physical Segment Role Binding
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Compact 8-byte Pod struct representing a localized role binding
//! within a relation segment.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::entity::id::EntityIndex;
use crate::relation::id::RoleId;

/// Compact 8-byte generation-local role binding.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable, Serialize, Deserialize)]
pub struct SegmentRoleBinding {
    /// Generation-local entity index bound to this role.
    pub entity: EntityIndex,
    /// Semantic role tag within the parent relation type schema.
    pub role_id: RoleId,
    /// Reserved flags / qualifiers.
    pub flags: u16,
}

const _: () = assert!(std::mem::size_of::<SegmentRoleBinding>() == 8);
const _: () = assert!(std::mem::align_of::<SegmentRoleBinding>() <= 4);
