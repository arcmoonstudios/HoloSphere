/* holosphere/src/experience/action.rs */
//!▫~•◦-------------------------------‣
//! # Action Definitions & Ordered Invocations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the distinction between reusable action definitions and specific
//! parameterized action invocations within an attempt.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::entity::id::ProvenanceId;
use crate::experience::id::{ActionId, AttemptId};

/// Value representation for a typed action parameter.
#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum ActionParameterValue {
    String(Arc<str>),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

/// Key-value parameter configuring an action invocation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurableActionParameter {
    pub key: Arc<str>,
    pub value: ActionParameterValue,
}

/// Reusable action definition.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActionDefinition {
    pub action_id: ActionId,
    pub name: Arc<str>,
    pub description: Arc<str>,
    pub provenance_id: ProvenanceId,
}

/// Concrete, ordered invocation of an action within an empirical attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionInvocation {
    pub invocation_id: u64,
    pub attempt_id: AttemptId,
    pub action_id: ActionId,
    pub ordinal: u32,
    pub parameters: Vec<DurableActionParameter>,
    pub started_lsn: u64,
    pub completed_lsn: u64,
    pub provenance_id: ProvenanceId,
}
