/* holosphere/src/learning/query.rs */
//!▫~•◦-------------------------------‣
//! # Learning & Adjudication Queries
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides pattern queries over empirical evidence and adjudication histories.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::experience::id::ContextId;
use crate::learning::adjudication::decision::AdjudicationRecord;
use crate::learning::adjudication::policy::AdjudicationDecisionCode;
use crate::learning::read::LearningReadSnapshot;
use crate::relation::id::RelationId;

/// Query predicate matching empirical adjudication decisions.
#[derive(Clone, Debug, Default)]
pub struct AdjudicationQuery {
    pub relation_id: Option<RelationId>,
    pub context_id: Option<ContextId>,
    pub decision_code: Option<AdjudicationDecisionCode>,
}

impl AdjudicationQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_relation(mut self, id: RelationId) -> Self {
        self.relation_id = Some(id);
        self
    }

    pub fn with_decision_code(mut self, code: AdjudicationDecisionCode) -> Self {
        self.decision_code = Some(code);
        self
    }

    /// Executes query against a pinned `LearningReadSnapshot`.
    pub fn execute(&self, snap: &LearningReadSnapshot) -> Vec<AdjudicationRecord> {
        let adjs_map = snap.segment.adjudications.read();
        let mut results = Vec::new();

        for (&_id, adj) in adjs_map.iter() {
            if adj.committed_lsn > snap.lsn {
                continue;
            }
            if let Some(rid) = self.relation_id {
                if adj.target_relation != rid {
                    continue;
                }
            }
            if let Some(code) = self.decision_code {
                if adj.decision_code != code {
                    continue;
                }
            }
            results.push(adj.clone());
        }

        // Deterministic ordering: AdjudicationId ASC
        results.sort_unstable_by_key(|a| a.id);
        results
    }
}
