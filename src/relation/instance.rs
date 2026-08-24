/* holosphere/src/relation/instance.rs */
//!▫~•◦-------------------------------‣
//! # Canonical Hypergraph Relation Instance & Deduplication Fingerprinting
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the durable representation of N-ary relation instances and
//! deterministic canonical fingerprint calculation for deduplication.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::id::ProvenanceId;
use crate::entity::status::EpistemicStatus;
use crate::relation::id::{DurableRoleBinding, RelationId, RelationTypeId};

/// Computes the deterministic canonical fingerprint of a relation instance.
///
/// Bindings are sorted in canonical `(role_id ASC, entity_id ASC)` order before hashing.
pub fn compute_canonical_fingerprint(
    type_id: RelationTypeId,
    schema_version: u16,
    bindings: &[DurableRoleBinding],
) -> u64 {
    let mut sorted = bindings.to_vec();
    sorted.sort_unstable(); // Sorts by role_id ASC, then entity_id ASC

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&type_id.to_le_bytes());
    hasher.update(&schema_version.to_le_bytes());
    for b in &sorted {
        hasher.update(&b.role_id.to_le_bytes());
        hasher.update(&b.entity_id.to_le_bytes());
    }
    hasher.finalize() as u64
}

/// Durable, cluster-wide canonical N-ary hypergraph relation instance.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DurableRelationInstance {
    pub id: RelationId,
    pub type_id: RelationTypeId,
    pub schema_version: u16,
    pub bindings: Vec<DurableRoleBinding>,
    pub provenance_id: ProvenanceId,
    pub epistemic_status: EpistemicStatus,
    pub fingerprint: u64,
}

impl DurableRelationInstance {
    pub fn new(
        id: RelationId,
        type_id: RelationTypeId,
        schema_version: u16,
        mut bindings: Vec<DurableRoleBinding>,
        provenance_id: ProvenanceId,
        epistemic_status: EpistemicStatus,
    ) -> Self {
        bindings.sort_unstable();
        let fingerprint = compute_canonical_fingerprint(type_id, schema_version, &bindings);
        Self {
            id,
            type_id,
            schema_version,
            bindings,
            provenance_id,
            epistemic_status,
            fingerprint,
        }
    }
}
