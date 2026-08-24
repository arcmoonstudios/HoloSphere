/* holosphere/src/learning/read.rs */
//!▫~•◦-------------------------------‣
//! # Pinned Learning Snapshots & Structured Adjudication Explanations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides point-in-time isolated querying of evidence summaries, adjudication
//! histories, and structured explanations.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;

use crate::experience::id::{ContextId, EvaluationPolicyId};
use crate::learning::adjudication::decision::AdjudicationRecord;
use crate::learning::adjudication::policy::AdjudicationPolicy;
use crate::learning::discovery::{
    DeclarativeOperator, DiscoveryCatalog, DiscoveryStateSnapshot, GovernedDiscoveryState,
};
use crate::learning::evidence::accumulator::{
    EvidenceAccumulator, EvidenceRecord, EvidenceSummary,
};
use crate::learning::evidence::context::ContextClassRegistry;
use crate::learning::id::AdjudicationId;
use crate::relation::id::RelationId;

/// Structured audit explanation detailing exactly why an adjudication was decided.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdjudicationExplanation {
    pub adjudication: AdjudicationRecord,
    pub summary: EvidenceSummary,
    pub evidence_records: Vec<EvidenceRecord>,
    pub policy: Option<AdjudicationPolicy>,
}

/// Generation-scoped container managing all empirical learning state.
pub struct LearningSegment {
    pub generation_id: u64,
    pub accumulator: EvidenceAccumulator,
    pub context_registry: ContextClassRegistry,
    pub policies: RwLock<HashMap<EvaluationPolicyId, AdjudicationPolicy>>,
    pub adjudications: RwLock<HashMap<AdjudicationId, AdjudicationRecord>>,
    pub relation_adjudications: RwLock<HashMap<RelationId, Vec<AdjudicationId>>>,
    pub discovery: DiscoveryCatalog,
    pub governed_discovery: GovernedDiscoveryState,
}

impl LearningSegment {
    pub fn new(generation_id: u64) -> Self {
        Self {
            generation_id,
            accumulator: EvidenceAccumulator::new(),
            context_registry: ContextClassRegistry::new(),
            policies: RwLock::new(HashMap::new()),
            adjudications: RwLock::new(HashMap::new()),
            relation_adjudications: RwLock::new(HashMap::new()),
            discovery: DiscoveryCatalog::new(),
            governed_discovery: GovernedDiscoveryState::new(),
        }
    }

    pub fn read_snapshot(self: &Arc<Self>, lsn: u64) -> LearningReadSnapshot {
        LearningReadSnapshot {
            lsn,
            segment: Arc::clone(self),
        }
    }

    /// Performs physical compaction, cloning all durable learning records.
    pub fn compact(&self, new_generation_id: u64) -> Arc<Self> {
        let compacted = Arc::new(Self::new(new_generation_id));
        *compacted.policies.write() = self.policies.read().clone();
        *compacted.adjudications.write() = self.adjudications.read().clone();
        *compacted.relation_adjudications.write() = self.relation_adjudications.read().clone();
        compacted
            .discovery
            .replace_from(self.discovery.history_snapshot());
        self.governed_discovery
            .copy_all_to(&compacted.governed_discovery);

        for rec in self.accumulator.snapshot_records() {
            compacted.accumulator.record(rec);
        }

        compacted
    }
}

/// Point-in-time snapshot pinned at committed LSN `lsn`.
#[derive(Clone)]
pub struct LearningReadSnapshot {
    pub lsn: u64,
    pub segment: Arc<LearningSegment>,
}

impl LearningReadSnapshot {
    /// Returns discovered operators committed no later than this pinned snapshot.
    pub fn discovered_operators(&self) -> Vec<DeclarativeOperator> {
        self.segment.discovery.snapshot_at(self.lsn)
    }

    pub fn governed_discovery(&self) -> DiscoveryStateSnapshot {
        self.segment.governed_discovery.snapshot_at(self.lsn)
    }

    /// Retrieves all evidence records for a relation observed up to this snapshot's LSN.
    pub fn evidence_for(&self, relation_id: RelationId) -> Vec<EvidenceRecord> {
        self.segment
            .accumulator
            .get_evidence_for_relation(relation_id, self.lsn)
    }

    /// Retrieves all evidence records for a relation matching a specific context.
    pub fn evidence_for_context(
        &self,
        relation_id: RelationId,
        context_id: ContextId,
    ) -> Vec<EvidenceRecord> {
        let all = self.evidence_for(relation_id);
        all.into_iter()
            .filter(|e| e.context_id == context_id)
            .collect()
    }

    /// Retrieves the adjudication history for a relation committed up to this snapshot's LSN.
    pub fn adjudication_history(&self, relation_id: RelationId) -> Vec<AdjudicationRecord> {
        let rel_map = self.segment.relation_adjudications.read();
        let ids = match rel_map.get(&relation_id) {
            Some(ids) => ids,
            None => return Vec::new(),
        };

        let adjs = self.segment.adjudications.read();
        ids.iter()
            .filter_map(|id| adjs.get(id))
            .filter(|a| a.committed_lsn <= self.lsn)
            .cloned()
            .collect()
    }

    /// Retrieves the most recent adjudication record committed up to this snapshot's LSN.
    pub fn current_adjudication(&self, relation_id: RelationId) -> Option<AdjudicationRecord> {
        let mut hist = self.adjudication_history(relation_id);
        hist.sort_unstable_by_key(|a| a.committed_lsn);
        hist.pop()
    }

    /// Generates a structured explanation for an adjudication decision.
    pub fn explain_adjudication(
        &self,
        adjudication_id: AdjudicationId,
    ) -> Option<AdjudicationExplanation> {
        let adjs = self.segment.adjudications.read();
        let adjudication = adjs.get(&adjudication_id)?.clone();
        if adjudication.committed_lsn > self.lsn {
            return None;
        }

        let policy = self
            .segment
            .policies
            .read()
            .get(&adjudication.policy_id)
            .cloned();

        let evidence_records = self.segment.accumulator.get_evidence_for_relation(
            adjudication.target_relation,
            adjudication.evidence_snapshot_lsn,
        );

        let mut summary = EvidenceSummary {
            relation_id: adjudication.target_relation,
            context_class_id: crate::learning::id::ContextClassId(1),
            ..Default::default()
        };
        for r in &evidence_records {
            summary.accumulate(r);
        }

        Some(AdjudicationExplanation {
            adjudication,
            summary,
            evidence_records,
            policy,
        })
    }
}
