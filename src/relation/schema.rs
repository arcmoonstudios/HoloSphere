/* holosphere/src/relation/schema.rs */
//!▫~•◦-------------------------------‣
//! # Relation Schemas & Role Specifications
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the explicit semantic schema specifications for N-ary hypergraph
//! relations, supporting dynamic role definitions, admission states, and
//! binary projection directives.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

use crate::entity::id::ProvenanceId;
use crate::relation::id::{DurableRoleBinding, RelationTypeId, RoleId};

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum SchemaValidationError {
    #[error("Missing required role {role_id}")]
    MissingRequiredRole { role_id: RoleId },
    #[error("Unknown role {role_id} not present in relation type schema")]
    UnknownRole { role_id: RoleId },
    #[error("Role {role_id} cardinality {count} violates bounds [{min_count}, {max_count}]")]
    CardinalityViolation {
        role_id: RoleId,
        count: u16,
        min_count: u16,
        max_count: u16,
    },
    #[error("Relation type {type_id} is not admitted (current state: {state:?})")]
    TypeNotAdmitted {
        type_id: RelationTypeId,
        state: RelationTypeState,
    },
}

/// Lifecycle admission state of a relation type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationTypeState {
    /// Proposed by reasoning/learning engine; queryable only when explicitly requested.
    Proposed,
    /// Authoritative schema admitted for production semantic graphs and indexing.
    Admitted,
    /// Deprecated schema preserved for historical provenance resolution.
    Deprecated,
}

/// Directionality of derived binary graph projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionDirection {
    Directed,
    Undirected,
}

/// Optional projection specification allowing an N-ary relation to generate fast binary CSR/CSC graphs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryProjectionSpec {
    pub source_role: RoleId,
    pub target_role: RoleId,
    pub direction: ProjectionDirection,
}

/// Role definition within a hypergraph relation type.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleSchema {
    pub role_id: RoleId,
    pub name: Arc<str>,
    pub min_count: u16,
    pub max_count: u16,
    pub required: bool,
}

/// Complete hypergraph relation type schema.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationType {
    pub id: RelationTypeId,
    pub name: Arc<str>,
    pub schema_version: u16,
    pub state: RelationTypeState,
    pub roles: Vec<RoleSchema>,
    pub binary_projection: Option<BinaryProjectionSpec>,
    pub provenance_id: ProvenanceId,
    pub structural_fingerprint: u64,
}

impl RelationType {
    /// Computes the structural fingerprint of a schema's role layout.
    pub fn compute_structural_fingerprint(
        id: RelationTypeId,
        schema_version: u16,
        roles: &[RoleSchema],
    ) -> u64 {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&id.to_le_bytes());
        hasher.update(&schema_version.to_le_bytes());
        for r in roles {
            hasher.update(&r.role_id.to_le_bytes());
            hasher.update(r.name.as_bytes());
            hasher.update(&r.min_count.to_le_bytes());
            hasher.update(&r.max_count.to_le_bytes());
            hasher.update(&[r.required as u8]);
        }
        hasher.finalize() as u64
    }

    /// Validates a set of durable role bindings against this schema.
    pub fn validate_bindings(
        &self,
        bindings: &[DurableRoleBinding],
    ) -> Result<(), SchemaValidationError> {
        let mut role_counts: HashMap<RoleId, u16> = HashMap::new();
        for b in bindings {
            *role_counts.entry(b.role_id).or_insert(0) += 1;
        }

        let role_map: HashMap<RoleId, &RoleSchema> =
            self.roles.iter().map(|r| (r.role_id, r)).collect();

        // Check for unknown roles
        for &role_id in role_counts.keys() {
            if !role_map.contains_key(&role_id) {
                return Err(SchemaValidationError::UnknownRole { role_id });
            }
        }

        // Check bounds & required roles
        for r in &self.roles {
            let count = role_counts.get(&r.role_id).copied().unwrap_or(0);
            if r.required && count < 1 {
                return Err(SchemaValidationError::MissingRequiredRole { role_id: r.role_id });
            }
            if count < r.min_count || count > r.max_count {
                return Err(SchemaValidationError::CardinalityViolation {
                    role_id: r.role_id,
                    count,
                    min_count: r.min_count,
                    max_count: r.max_count,
                });
            }
        }

        Ok(())
    }
}

/// Scope restricting query execution over relation types.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SchemaScope {
    /// Only production-admitted relation types (default).
    #[default]
    AdmittedOnly,
    /// Include speculative / proposed relation types.
    IncludeProposed,
    /// Historical queries including deprecated relation types.
    Historical,
}
