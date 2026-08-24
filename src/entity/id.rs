/* holosphere/src/entity/id.rs */
//!▫~•◦-------------------------------‣
//! # Entity Kernel Identifiers & Layout Descriptors
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the foundational boundary between durable global identity
//! (`EntityId`, `VersionId`, `ProvenanceId`, `RelationId`: u64) and
//! generation-local dense physical indices (`EntityIndex`, `VersionIndex`, etc.: u32).
//!
//! ## Invariant Guarantees
//! - Replicated state (WAL, Raft logs, snapshots) exclusively references durable IDs.
//! - Generation-local CSRs, bitsets, and SIMD loops operate on dense physical indices.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// Globally unique, persistent monotonic identifier for an entity.
///
/// Stable across compactions, migrations, snapshot generations, and Raft consensus logs.
pub type EntityId = u64;

/// Generation-local, dense contiguous row index inside a physical `EntitySegment`.
///
/// Used for O(1) array lookups, CSR offset indexing, and AVX2/AVX-512 SIMD vector loops.
pub type EntityIndex = u32;

/// Globally unique durable identifier for an entity version in the lineage history.
pub type VersionId = u64;

/// Generation-local dense row index inside the `VersionTable`.
pub type VersionIndex = u32;

/// Globally unique durable identifier for an immutable provenance record.
pub type ProvenanceId = u64;

/// Generation-local dense row index inside the `ProvenanceArena`.
pub type ProvenanceIndex = u32;

/// Globally unique durable identifier for a relation instance / hyperedge.
pub type RelationId = u64;

/// Interned identifier for a canonical relation type schema.
pub type RelationTypeId = u32;

/// Role identifier within an N-ary relation binding.
pub type RoleId = u16;

/// Schema descriptor handle for vector layouts in the vector arena.
pub type VectorLayoutId = u16;

/// Sentinel row index representing `None` / null reference.
pub const NULL_ROW_REF: u32 = u32::MAX;

/// Stable durable reference to an evidence item supporting a claim, version, or relation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DurableEvidenceRef {
    /// Sourced from a specific entity version.
    EntityVersion(EntityId, VersionId),
    /// Sourced from a specific durable entity.
    Entity(EntityId),
    /// Sourced from a durable relation instance.
    Relation(RelationId),
    /// Sourced from an empirical attempt outcome.
    Attempt(u64),
    /// Sourced from an external document / cryptographic signature hash.
    ExternalHash([u8; 32]),
}

/// Scalar data type of vector components in storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum VectorScalarType {
    Float32 = 0,
    Float16 = 1,
    BFloat16 = 2,
    Complex32 = 3,
    PolarQuantized = 4,
    Binary = 5,
}

/// Normalization requirement for vectors in a given layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum VectorNormalization {
    None = 0,
    L2Normalized = 1,
    UnitSphere = 2,
}

/// Descriptive schema for vector storage layouts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorLayout {
    pub layout_id: VectorLayoutId,
    pub dimension: u16,
    pub scalar_type: VectorScalarType,
    pub normalization: VectorNormalization,
    pub stride_bytes: u32,
    pub name: String,
}

impl VectorLayout {
    /// Default FP32 embedding layout.
    pub fn fp32(layout_id: VectorLayoutId, dimension: u16, name: impl Into<String>) -> Self {
        Self {
            layout_id,
            dimension,
            scalar_type: VectorScalarType::Float32,
            normalization: VectorNormalization::L2Normalized,
            stride_bytes: (dimension as u32) * 4,
            name: name.into(),
        }
    }

    /// Complex32 isometric embedding layout.
    pub fn complex32(layout_id: VectorLayoutId, complex_dim: u16, name: impl Into<String>) -> Self {
        Self {
            layout_id,
            dimension: complex_dim,
            scalar_type: VectorScalarType::Complex32,
            normalization: VectorNormalization::L2Normalized,
            stride_bytes: (complex_dim as u32) * 8, // Real + Imag f32
            name: name.into(),
        }
    }
}
