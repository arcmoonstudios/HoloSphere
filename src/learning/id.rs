/* holosphere/src/learning/id.rs */
//!▫~•◦-------------------------------‣
//! # Typed Learning & Adjudication Identifiers
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides durable identifiers for evidence records, summaries, contexts,
//! and adjudication decisions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// Unique durable identifier for an empirical evidence record.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct EvidenceId(pub u64);

/// Unique durable identifier for an adjudication decision record.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct AdjudicationId(pub u64);

/// Unique durable identifier for a context equivalence class.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct ContextClassId(pub u64);

/// Unique identifier for an accumulated evidence summary.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct EvidenceSummaryId(pub u64);
