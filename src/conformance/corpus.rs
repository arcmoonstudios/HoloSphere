/* holosphere/src/conformance/corpus.rs */
//!▫~•◦-------------------------------‣
//! # Deterministic v1 Golden Conformance Corpus
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Generates the canonical, immutable v1 database fixture covering all core concepts.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;

use crate::conformance::export::{
    CanonicalExportArchive, ExportedEntity, ExportedExperience, ExportedLearningRecord,
    ExportedRelation,
};
use crate::conformance::version::CANONICAL_EXPORT_VERSION;
use crate::entity::status::EpistemicStatus;
use crate::experience::id::{ActionId, AttemptId, ContextId, ProblemId};

/// Creates the reference v1 golden archive populated at LSN 10,000.
pub fn create_v1_golden_fixture() -> CanonicalExportArchive {
    let mut entities = Vec::new();
    for i in 1..=50u64 {
        let epistemic_status = match i % 5 {
            0 => EpistemicStatus::Observed,
            1 => EpistemicStatus::Asserted,
            2 => EpistemicStatus::Inferred,
            3 => EpistemicStatus::Provisional,
            _ => EpistemicStatus::Contradicted,
        };

        let mut payload = [0u8; 32];
        payload[0..8].copy_from_slice(&i.to_le_bytes());

        entities.push(ExportedEntity {
            entity_id: i,
            version_id: 1,
            epistemic_status,
            provenance_id: 100 + i,
            payload_digest: payload,
        });
    }

    let mut relations = Vec::new();
    for i in 1..=25u64 {
        relations.push(ExportedRelation {
            relation_id: 1000 + i,
            relation_type: 1,
            role_bindings: vec![(1, i), (2, i + 1)],
            epistemic_status: EpistemicStatus::Observed,
            provenance_id: 500 + i,
        });
    }

    let mut experiences = Vec::new();
    for i in 1..=10u64 {
        experiences.push(ExportedExperience {
            problem_id: ProblemId(2000 + i),
            context_id: ContextId(3000 + i),
            attempt_id: AttemptId(4000 + i),
            action_ids: vec![ActionId(5000 + i), ActionId(5001 + i)],
            outcome_utility_q32: 65536 * (i as i64),
        });
    }

    let schema_signatures = vec![
        Arc::from("entity:service_node:v1"),
        Arc::from("relation:calls_downstream:v1"),
        Arc::from("experience:latency_mitigation:v1"),
    ];

    let learning_records = vec![ExportedLearningRecord {
        record_id: 7001,
        subject_id: 1001,
        kind: Arc::from("causal_hypothesis"),
        epistemic_status: EpistemicStatus::Provisional,
        provenance_id: 9001,
        payload_digest: [7u8; 32],
    }];

    CanonicalExportArchive {
        format_version: CANONICAL_EXPORT_VERSION,
        snapshot_lsn: 10_000,
        entities,
        relations,
        experiences,
        learning_records,
        schema_signatures,
    }
}
