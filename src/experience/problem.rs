/* holosphere/src/experience/problem.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Problem Occurrences
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Stores observed problem occurrences without premature classification.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::entity::id::ProvenanceId;
use crate::experience::id::{ContextId, ProblemId};

/// Concrete observed problem occurrence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProblemOccurrence {
    pub problem_id: ProblemId,
    pub symptom: Arc<str>,
    pub component: Arc<str>,
    pub first_observed_lsn: u64,
    pub provenance_id: ProvenanceId,
    pub context_id: ContextId,
}
