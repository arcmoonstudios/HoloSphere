/* holosphere/src/experience/id.rs */
//!▫~•◦-------------------------------‣
//! # Typed Durable Experience Identifiers
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides type-safe wrappers over universal durable `EntityId`s for
//! first-class empirical experience objects.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::entity::id::EntityId;
use serde::{Deserialize, Serialize};

/// Typed identifier for observed problem occurrences.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProblemId(pub EntityId);

/// Typed identifier for empirical attempt instances.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct AttemptId(pub EntityId);

/// Typed identifier for reusable action definitions.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ActionId(pub EntityId);

/// Typed identifier for structured execution contexts.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ContextId(pub EntityId);

/// Typed identifier for measured empirical outcome sets.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct OutcomeId(pub EntityId);

/// Typed identifier for evaluation and scoring policies.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EvaluationPolicyId(pub u64);

/// Typed identifier for metric definitions.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MetricId(pub u32);

/// Typed symbol identifier.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SymbolId(pub u32);
