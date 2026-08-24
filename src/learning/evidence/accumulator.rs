/* holosphere/src/learning/evidence/accumulator.rs */
//!▫~•◦-------------------------------‣
//! # Evidence Records & Consolidated Summary Accumulator
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the immutable evidence store, idempotent deduplication, and
//! order-independent summary rebuilds.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entity::id::ProvenanceId;
use crate::experience::id::{AttemptId, ContextId, EvaluationPolicyId};
use crate::learning::id::{ContextClassId, EvidenceId};
use crate::relation::id::RelationId;

/// Policy-derived evaluation direction for an empirical attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidenceDirection {
    Supports,
    Contradicts,
    Neutral,
    Inconclusive,
}

/// Immutable empirical evidence record assessing an attempt under a specific policy.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRecord {
    pub evidence_id: EvidenceId,
    pub target_relation: RelationId,
    pub attempt_id: AttemptId,
    pub context_id: ContextId,
    pub policy_id: EvaluationPolicyId,
    pub direction: EvidenceDirection,
    pub utility_q32: i64,
    pub provenance_id: ProvenanceId,
    pub evaluated_at_lsn: u64,
}

/// Unique deduplication key guaranteeing that an attempt is evaluated at most once per policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EvidenceKey {
    pub relation_id: RelationId,
    pub attempt_id: AttemptId,
    pub policy_id: EvaluationPolicyId,
    pub context_id: ContextId,
}

/// Consolidated empirical evidence statistics for a relation under a context class.
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EvidenceSummary {
    pub relation_id: RelationId,
    pub context_class_id: ContextClassId,
    pub observation_count: u64,
    pub support_count: u64,
    pub contradiction_count: u64,
    pub neutral_count: u64,
    pub utility_sum_q32: i64,
    pub utility_sq_sum_q32: u64,
    pub first_evidence_lsn: u64,
    pub last_evidence_lsn: u64,
}

impl EvidenceSummary {
    /// Incorporates a single evidence record into this summary.
    pub fn accumulate(&mut self, record: &EvidenceRecord) {
        self.observation_count += 1;
        match record.direction {
            EvidenceDirection::Supports => self.support_count += 1,
            EvidenceDirection::Contradicts => self.contradiction_count += 1,
            EvidenceDirection::Neutral | EvidenceDirection::Inconclusive => self.neutral_count += 1,
        }

        self.utility_sum_q32 = self.utility_sum_q32.saturating_add(record.utility_q32);
        let u_sq = (record.utility_q32 as i128).saturating_mul(record.utility_q32 as i128) >> 32;
        self.utility_sq_sum_q32 = self.utility_sq_sum_q32.saturating_add(u_sq as u64);

        if self.first_evidence_lsn == 0 || record.evaluated_at_lsn < self.first_evidence_lsn {
            self.first_evidence_lsn = record.evaluated_at_lsn;
        }
        if record.evaluated_at_lsn > self.last_evidence_lsn {
            self.last_evidence_lsn = record.evaluated_at_lsn;
        }
    }
}

/// Computes a deterministic SHA-256 digest over a slice of evidence records.
pub fn compute_evidence_digest(records: &[EvidenceRecord]) -> [u8; 32] {
    let mut sorted = records.to_vec();
    sorted.sort_unstable_by_key(|r| r.evidence_id);

    let mut hasher = Sha256::new();
    for r in &sorted {
        hasher.update(&r.evidence_id.0.to_le_bytes());
        hasher.update(&r.target_relation.to_le_bytes());
        hasher.update(&r.attempt_id.0.to_le_bytes());
        hasher.update(&r.context_id.0.to_le_bytes());
        hasher.update(&r.policy_id.0.to_le_bytes());
        hasher.update(&[r.direction as u8]);
        hasher.update(&r.utility_q32.to_le_bytes());
        hasher.update(&r.evaluated_at_lsn.to_le_bytes());
    }
    hasher.finalize().into()
}

/// In-memory evidence store and summary index.
pub struct EvidenceAccumulator {
    next_id: AtomicU64,
    records: RwLock<HashMap<EvidenceId, EvidenceRecord>>,
    dedup_index: RwLock<HashMap<EvidenceKey, EvidenceId>>,
    relation_index: RwLock<HashMap<RelationId, Vec<EvidenceId>>>,
}

impl Default for EvidenceAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl EvidenceAccumulator {
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            records: RwLock::new(HashMap::new()),
            dedup_index: RwLock::new(HashMap::new()),
            relation_index: RwLock::new(HashMap::new()),
        }
    }

    /// Records an empirical evidence entry, enforcing idempotency.
    pub fn record(&self, mut evidence: EvidenceRecord) -> (EvidenceId, bool) {
        let key = EvidenceKey {
            relation_id: evidence.target_relation,
            attempt_id: evidence.attempt_id,
            policy_id: evidence.policy_id,
            context_id: evidence.context_id,
        };

        {
            let dedup = self.dedup_index.read();
            if let Some(&existing_id) = dedup.get(&key) {
                return (existing_id, false); // Already recorded
            }
        }

        let mut dedup = self.dedup_index.write();
        if let Some(&existing_id) = dedup.get(&key) {
            return (existing_id, false);
        }

        let id = if evidence.evidence_id.0 == 0 {
            EvidenceId(self.next_id.fetch_add(1, Ordering::Relaxed))
        } else {
            evidence.evidence_id
        };
        evidence.evidence_id = id;

        self.records.write().insert(id, evidence.clone());
        dedup.insert(key, id);
        self.relation_index
            .write()
            .entry(evidence.target_relation)
            .or_default()
            .push(id);

        (id, true)
    }

    /// Retrieves all evidence records for a relation up to `cutoff_lsn`.
    pub fn get_evidence_for_relation(
        &self,
        relation_id: RelationId,
        cutoff_lsn: u64,
    ) -> Vec<EvidenceRecord> {
        let rel_map = self.relation_index.read();
        let ids = match rel_map.get(&relation_id) {
            Some(ids) => ids,
            None => return Vec::new(),
        };

        let records = self.records.read();
        ids.iter()
            .filter_map(|id| records.get(id))
            .filter(|r| r.evaluated_at_lsn <= cutoff_lsn)
            .cloned()
            .collect()
    }

    /// Builds a consolidated evidence summary from canonical evidence records.
    pub fn build_summary(
        &self,
        relation_id: RelationId,
        context_class_id: ContextClassId,
        cutoff_lsn: u64,
    ) -> EvidenceSummary {
        let mut summary = EvidenceSummary {
            relation_id,
            context_class_id,
            ..Default::default()
        };

        let evidence_list = self.get_evidence_for_relation(relation_id, cutoff_lsn);
        for ev in &evidence_list {
            summary.accumulate(ev);
        }

        summary
    }

    /// Clones all internal records for physical compaction.
    pub fn snapshot_records(&self) -> Vec<EvidenceRecord> {
        self.records.read().values().cloned().collect()
    }
}
