/* holosphere/src/experience/outcome.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Outcomes & Raw Observations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Stores immutable raw metric observations associated with attempts.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::id::ProvenanceId;
use crate::experience::id::{AttemptId, MetricId, OutcomeId};
use crate::experience::metric::MetricValue;

/// Single immutable raw metric observation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurableOutcomeObservation {
    pub metric_id: MetricId,
    pub baseline: MetricValue,
    pub observed: MetricValue,
    pub measurement_start_lsn: u64,
    pub measurement_end_lsn: u64,
    pub provenance_id: ProvenanceId,
}

/// Complete empirical outcome record measured during an attempt.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OutcomeRecord {
    pub outcome_id: OutcomeId,
    pub attempt_id: AttemptId,
    pub observations: Vec<DurableOutcomeObservation>,
    pub commit_lsn: u64,
    pub provenance_id: ProvenanceId,
}
