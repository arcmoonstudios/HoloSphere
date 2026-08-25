/* holosphere/src/conformance/version.rs */
//!▫~•◦-------------------------------‣
//! # Canonical Semantic Kernel Versioning Constants
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the immutable, explicit version contracts for HoloSphere v1.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

/// The global authoritative Semantic Kernel version.
pub const SEMANTIC_KERNEL_VERSION: u32 = 1;

/// Format version for universal persistent snapshot archives.
pub const SNAPSHOT_FORMAT_VERSION: u32 = 1;

/// Format version for consensus WAL and Raft replication records.
pub const RAFT_LOG_RECORD_VERSION: u32 = 1;

/// Format version for storage-layout-independent canonical exports.
pub const CANONICAL_EXPORT_VERSION: u32 = 2;

/// Schema serialization version for universal entities and headers.
pub const ENTITY_SCHEMA_VERSION: u32 = 1;

/// Schema serialization version for hypergraph relations and role bindings.
pub const RELATION_SCHEMA_VERSION: u32 = 1;

/// Schema serialization version for empirical problems, attempts, and outcomes.
pub const EXPERIENCE_SCHEMA_VERSION: u32 = 1;

/// Schema serialization version for evidence records and deterministic adjudications.
pub const LEARNING_SCHEMA_VERSION: u32 = 1;

/// Cryptographic digest format version for `WorldStateDigest`.
pub const WORLD_DIGEST_VERSION: u32 = 1;

/// Provenance trace schema version for inference methods.
pub const INFERENCE_TRACE_VERSION: u32 = 1;

/// Provenance trace schema version for structural synthesis plans.
pub const SYNTHESIS_TRACE_VERSION: u32 = 1;
