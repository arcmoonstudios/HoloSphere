/* holosphere/src/learning/adjudication/mod.rs */
//!▫~•◦-------------------------------‣
//! # Epistemic Adjudication Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides policies, decision audit logs, and transition validation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod decision;
pub mod policy;
pub mod transition;

pub use decision::AdjudicationRecord;
pub use policy::{AdjudicationDecisionCode, AdjudicationDisposition, AdjudicationPolicy};
pub use transition::{evaluate_adjudication, evaluate_adjudication_with_causal};
