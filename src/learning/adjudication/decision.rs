/* holosphere/src/learning/adjudication/decision.rs */
//!▫~•◦-------------------------------‣
//! # Durable Adjudication Records
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the durable audit trail recording why an epistemic transition was decided.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::status::EpistemicStatus;
use crate::experience::id::EvaluationPolicyId;
use crate::learning::adjudication::policy::{AdjudicationDecisionCode, AdjudicationDisposition};
use crate::learning::id::{AdjudicationId, EvidenceSummaryId};
use crate::relation::id::RelationId;

/// Durable record documenting an empirical adjudication decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdjudicationRecord {
    pub id: AdjudicationId,
    pub target_relation: RelationId,
    pub policy_id: EvaluationPolicyId,
    pub evidence_snapshot_lsn: u64,
    pub previous_status: EpistemicStatus,
    pub resulting_status: EpistemicStatus,
    pub evidence_summary_id: EvidenceSummaryId,
    pub decision_code: AdjudicationDecisionCode,
    pub disposition: AdjudicationDisposition,
    pub committed_lsn: u64,
}
