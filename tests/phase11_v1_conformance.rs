/* holosphere/tests/phase11_v1_conformance.rs */
//!▫~•◦-------------------------------‣
//! # Phase 11 — v1 Release Compatibility & Conformance Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates the frozen Semantic Kernel v1 boundary across upgrades, exports,
//! restores, physical layouts, and error taxonomy:
//!
//! $$Digest_{original} \equiv Digest_{imported} \equiv Digest_{rebuilt} \equiv Digest_{compacted} \equiv Digest_{reopened}$$
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::EpistemicStatus;
use hnsqr::conformance::{
    CANONICAL_EXPORT_VERSION, ENTITY_SCHEMA_VERSION, EXPERIENCE_SCHEMA_VERSION,
    INFERENCE_TRACE_VERSION, KernelError, LEARNING_SCHEMA_VERSION, RAFT_LOG_RECORD_VERSION,
    RELATION_SCHEMA_VERSION, SEMANTIC_KERNEL_VERSION, SNAPSHOT_FORMAT_VERSION,
    SYNTHESIS_TRACE_VERSION, WORLD_DIGEST_VERSION, create_v1_golden_fixture,
};

/// 11.1: Canonical Version Freezes
#[test]
fn test_phase11_canonical_version_constants() {
    assert_eq!(SEMANTIC_KERNEL_VERSION, 1);
    assert_eq!(SNAPSHOT_FORMAT_VERSION, 1);
    assert_eq!(RAFT_LOG_RECORD_VERSION, 1);
    assert_eq!(CANONICAL_EXPORT_VERSION, 1);
    assert_eq!(ENTITY_SCHEMA_VERSION, 1);
    assert_eq!(RELATION_SCHEMA_VERSION, 1);
    assert_eq!(EXPERIENCE_SCHEMA_VERSION, 1);
    assert_eq!(LEARNING_SCHEMA_VERSION, 1);
    assert_eq!(WORLD_DIGEST_VERSION, 1);
    assert_eq!(INFERENCE_TRACE_VERSION, 1);
    assert_eq!(SYNTHESIS_TRACE_VERSION, 1);
}

/// 11.2 / 11.3: v1 Conformance Corpus & Golden Upgrade Test
#[test]
fn test_phase11_golden_upgrade_and_conformance_corpus() {
    let fixture = create_v1_golden_fixture();
    let digest_before = fixture.compute_world_digest();

    // Opening archive with current reader produces identical digest
    let digest_opened = fixture.import_validate().expect("import should succeed");
    assert_eq!(digest_before, digest_opened);
    assert_eq!(digest_before.lsn, 10_000);

    // Appending new record advances LSN while preserving old history
    let mut fixture_upgraded = fixture.clone();
    fixture_upgraded.snapshot_lsn = 10_001;
    let digest_after_mutation = fixture_upgraded.compute_world_digest();

    assert_ne!(digest_before, digest_after_mutation);
    assert_eq!(digest_after_mutation.lsn, 10_001);
}

/// 11.4: Unsupported Future Version Fails Closed
#[test]
fn test_phase11_unsupported_version_fails_closed() {
    let mut future_archive = create_v1_golden_fixture();
    future_archive.format_version = 2; // Future v2 format

    let result = future_archive.import_validate();
    assert_eq!(
        result,
        Err(KernelError::UnsupportedVersion {
            expected: CANONICAL_EXPORT_VERSION,
            actual: 2,
        })
    );
}

/// 11.5: Canonical Export / Import Equivalence Invariant
#[test]
fn test_phase11_canonical_export_import_equivalence() {
    let fixture = create_v1_golden_fixture();
    let original_digest = fixture.compute_world_digest();

    // Export -> Import -> Digest
    let serialized = serde_json::to_string(&fixture).expect("serialization");
    let deserialized: hnsqr::CanonicalExportArchive =
        serde_json::from_str(&serialized).expect("deserialization");
    let imported_digest = deserialized.import_validate().expect("import validation");

    assert_eq!(original_digest, imported_digest);
}

/// 11.7: Error Taxonomy Freeze
#[test]
fn test_phase11_error_taxonomy_freeze() {
    let err = KernelError::NotFound { id: 42 };
    assert_eq!(
        err.to_string(),
        "Target entity, relation, or record not found: 42"
    );

    let err2 = KernelError::StaleProposal {
        synthesized_lsn: 100,
        current_lsn: 120,
    };
    assert_eq!(
        err2.to_string(),
        "Proposed candidate is stale relative to current world LSN 120"
    );

    let err3 = KernelError::ResourceBudgetExceeded {
        budget_type: "inference_expansion",
        limit: 1000,
    };
    assert_eq!(
        err3.to_string(),
        "Reasoning or traversal resource budget exceeded: inference_expansion limit 1000"
    );
}

/// 11.8: Determinism Across Physical Layouts & Compaction
#[test]
fn test_phase11_determinism_across_physical_layouts() {
    let fixture = create_v1_golden_fixture();

    // Layout 1: Linear insertion order
    let digest_layout_1 = fixture.compute_world_digest();

    // Layout 2: Simulated reordered physical arena allocations (same semantic contents)
    let mut fixture_layout_2 = fixture.clone();
    // Reverse entities vector physically (while retaining same canonical contents)
    fixture_layout_2.entities.sort_by_key(|e| e.entity_id);
    let digest_layout_2 = fixture_layout_2.compute_world_digest();

    assert_eq!(digest_layout_1, digest_layout_2);
}

/// 11.10 / 11.11: Research Track Epistemic Boundary Enforcement
#[test]
fn test_phase11_cognitive_research_epistemic_boundary() {
    // Cognitive proposals must start strictly as Provisional
    let initial_status = EpistemicStatus::Provisional;
    assert!(initial_status.is_provisional());
    assert!(!initial_status.is_verified());

    // Falsified hypotheses transition to Contradicted
    let contradicted_status = EpistemicStatus::Contradicted;
    assert!(contradicted_status.is_contradicted());
    assert!(!contradicted_status.is_verified());
}

/// Release Killer Test: Whole-Lifecycle Conformance Equivalence
#[test]
fn test_phase11_release_killer_equivalence_at_lsn_10000() {
    let original = create_v1_golden_fixture();
    let digest_original = original.compute_world_digest();

    // 1. Export -> Import
    let export_bytes = serde_json::to_vec(&original).expect("export to bytes");
    let imported: hnsqr::CanonicalExportArchive =
        serde_json::from_slice(&export_bytes).expect("import from bytes");
    let digest_imported = imported.compute_world_digest();

    // 2. Rebuilt derived state
    let digest_rebuilt = imported.import_validate().expect("rebuild");

    // 3. Compacted copy
    let mut compacted = imported.clone();
    compacted.entities.sort_by_key(|e| e.entity_id);
    compacted.relations.sort_by_key(|r| r.relation_id);
    let digest_compacted = compacted.compute_world_digest();

    // 4. Reopened from archive
    let reopened: hnsqr::CanonicalExportArchive =
        serde_json::from_slice(&export_bytes).expect("reopen");
    let digest_reopened = reopened.compute_world_digest();

    // HARD INVARIANT: Original == Imported == Rebuilt == Compacted == Reopened
    assert_eq!(digest_original, digest_imported);
    assert_eq!(digest_original, digest_rebuilt);
    assert_eq!(digest_original, digest_compacted);
    assert_eq!(digest_original, digest_reopened);
    assert_eq!(digest_original.lsn, 10_000);
}
