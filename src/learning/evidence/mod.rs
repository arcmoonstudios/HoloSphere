/* holosphere/src/learning/evidence/mod.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Evidence Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the immutable evidence store, metrics evaluation, and context grouping.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod accumulator;
pub mod context;
pub mod stats;

pub use accumulator::{
    EvidenceAccumulator, EvidenceDirection, EvidenceKey, EvidenceRecord, EvidenceSummary,
    compute_evidence_digest,
};
pub use context::ContextClassRegistry;
pub use stats::{FixedUtility, MetricDirection, MetricEvaluationRule, NormalizationRule};
