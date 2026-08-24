/* holosphere/src/conformance/error.rs */
//!▫~•◦-------------------------------‣
//! # Frozen Public Error Taxonomy
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the typed, stable error classification contract for HoloSphere v1 public callers.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;
use thiserror::Error;

use crate::entity::id::EntityId;

/// Authoritative, typed error taxonomy for the HoloSphere v1 public contract.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    #[error("Target entity, relation, or record not found: {id:?}")]
    NotFound { id: EntityId },

    #[error("Write conflict or version concurrency violation: {message}")]
    Conflict { message: Arc<str> },

    #[error("Proposed candidate is stale relative to current world LSN {current_lsn}")]
    StaleProposal {
        synthesized_lsn: u64,
        current_lsn: u64,
    },

    #[error("Snapshot LSN mismatch: expected {expected}, actual {actual}")]
    SnapshotMismatch { expected: u64, actual: u64 },

    #[error("Data corruption or checksum validation failure: {detail}")]
    Corruption { detail: Arc<str> },

    #[error("Reasoning or traversal resource budget exceeded: {budget_type} limit {limit}")]
    ResourceBudgetExceeded {
        budget_type: &'static str,
        limit: usize,
    },

    #[error("Multi-tenant authorization violation or cross-tenant boundary breach")]
    Unauthorized,

    #[error("Unsupported semantic kernel or format version: expected {expected}, got {actual}")]
    UnsupportedVersion { expected: u32, actual: u32 },

    #[error("Operation deadline exceeded after {timeout_ms}ms")]
    DeadlineExceeded { timeout_ms: u64 },

    #[error("Ambiguous commit outcome during leader election / failover")]
    AmbiguousCommitOutcome,
}
