/* holosphere/src/entity/mod.rs */
//!▫~•◦-------------------------------‣
//! # Unified Entity Kernel Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Foundational entity universe kernel. All vectors, graph topologies, temporal
//! lineages, epistemic justifications, and provenance chains are native
//! projections of the unified entity universe.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod arena;
pub mod context;
pub mod eligibility;
pub mod epistemic;
pub mod exact;
pub mod header;
pub mod id;
pub mod mutation;
pub mod outcome;
pub mod provenance;
pub mod read;
pub mod relation;
pub mod segment;
pub mod snapshot;
pub mod stats;
pub mod status;
pub mod vector;
pub mod version;

// Re-exports
pub use arena::EntityArena;
pub use context::{ContextRecord, ContextSignature};
pub use eligibility::{
    EligibilityError, EligibilityRepresentation, EligibilitySet, EligibilityView,
};
pub use epistemic::{
    EpistemicTransitionError, LifecycleTransitionError, validate_epistemic_transition,
    validate_lifecycle_transition,
};
pub use exact::{
    CosineMetric, DistanceFunction, EuclideanMetric, ExactEligibilityCostModel,
    ExactEligibilityProof, ExactRetrievalContext, ExactScanOperator, ExactScanPlan,
    ExactVectorMetric, InnerProductMetric, ProjectiveOverlapMetric, ScoredEntity, exact_top_k,
    exact_top_k_scalar, masked_dense_scan, resolve_metric, sparse_gather_scan,
};
pub use header::{
    ENTITY_FLAG_HAS_INFERENCE_SIDECAR, ENTITY_FLAG_HAS_PROPERTIES, ENTITY_FLAG_HAS_PROVENANCE,
    ENTITY_FLAG_HAS_VECTOR, ENTITY_FLAG_HAS_VERSION_HISTORY, ENTITY_FLAG_LIVE, EntityHeader,
};
pub use id::{
    DurableEvidenceRef, EntityId, EntityIndex, NULL_ROW_REF, ProvenanceId, ProvenanceIndex,
    RelationId, RelationTypeId, RoleId, VectorLayout, VectorLayoutId, VectorNormalization,
    VectorScalarType, VersionId, VersionIndex,
};
pub use mutation::{EntityMutation, MutationApplyError};
pub use outcome::{OutcomeMetricDirection, OutcomeMetricSchema, OutcomeObservation};
pub use provenance::{ProvenanceArena, ProvenanceRecord, ProvenanceRow};
pub use read::{EntityReadSnapshot, ResolvedEntityVersion};
pub use relation::{
    DurableRoleBinding, RelationInstance, RelationType, RelationTypeState, SegmentRoleBinding,
};
pub use segment::EntitySegment;
pub use snapshot::{
    SNAPSHOT_V3_MAGIC, SnapshotV3Error, SnapshotV3SectionHeader, decode_snapshot_v3,
    encode_snapshot_v3,
};
pub use stats::DeterministicEvidenceStats;
pub use status::{EpistemicStatus, LifecycleStatus};
pub use vector::VectorArena;
pub use version::{DurableEntityVersion, VersionRelation, VersionRow, VersionTable};

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_entity_header_layout_pod() {
        assert_eq!(std::mem::size_of::<EntityHeader>(), 32);
        assert_eq!(std::mem::align_of::<EntityHeader>(), 32);

        let mut header = EntityHeader::default();
        header.label_fast_mask = 0xDEADBEEFCAFE;
        header.version_row = 12;
        header.provenance_row = 34;
        header.vector_row = 56;
        header.property_row = 78;
        header.label_overflow_row = 90;
        header.set_epistemic(EpistemicStatus::Inferred);
        header.vector_layout_id = 42;

        let bytes = bytemuck::bytes_of(&header);
        assert_eq!(bytes.len(), 32);

        let recovered: &EntityHeader = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.label_fast_mask, 0xDEADBEEFCAFE);
        assert_eq!(recovered.version_row, 12);
        assert_eq!(recovered.provenance_row, 34);
        assert_eq!(recovered.vector_row, 56);
        assert_eq!(recovered.property_row, 78);
        assert_eq!(recovered.label_overflow_row, 90);
        assert_eq!(recovered.epistemic(), EpistemicStatus::Inferred);
        assert_eq!(recovered.vector_layout_id, 42);
    }

    #[test]
    fn test_provenance_row_layout_pod() {
        assert_eq!(std::mem::size_of::<ProvenanceRow>(), 80);
        assert_eq!(std::mem::align_of::<ProvenanceRow>(), 8);

        let mut row = ProvenanceRow::default();
        row.commit_lsn = 10042;
        row.timestamp_ms = 1718000000;
        row.signature_hash[0] = 0xAA;
        row.signature_hash[31] = 0xFF;
        row.source_uri_id = 1;
        row.actor_id = 2;
        row.extraction_method_id = 3;
        row.evidence_start = 10;
        row.evidence_len = 2;
        row.set_confidence_f32(0.85);

        let bytes = bytemuck::bytes_of(&row);
        assert_eq!(bytes.len(), 80);

        let recovered: &ProvenanceRow = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.commit_lsn, 10042);
        assert_eq!(recovered.timestamp_ms, 1718000000);
        assert_eq!(recovered.signature_hash[0], 0xAA);
        assert_eq!(recovered.signature_hash[31], 0xFF);
        assert_eq!(recovered.source_uri_id, 1);
        assert!((recovered.confidence_f32() - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_segment_role_binding_layout_pod() {
        assert_eq!(std::mem::size_of::<SegmentRoleBinding>(), 16);
        assert_eq!(std::mem::align_of::<SegmentRoleBinding>(), 8);

        let binding = SegmentRoleBinding {
            relation_id: 999999,
            entity: 1234,
            role_id: 5,
            flags: 0,
        };

        let bytes = bytemuck::bytes_of(&binding);
        assert_eq!(bytes.len(), 16);

        let recovered: &SegmentRoleBinding = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.relation_id, 999999);
        assert_eq!(recovered.entity, 1234);
        assert_eq!(recovered.role_id, 5);
    }

    #[test]
    fn test_version_row_layout_pod() {
        assert_eq!(std::mem::size_of::<VersionRow>(), 56);
        assert_eq!(std::mem::align_of::<VersionRow>(), 8);

        let row = VersionRow {
            entity_id: 1001,
            version_id: 2002,
            valid_from_lsn: 50,
            valid_until_lsn: 100,
            prev_version_row: 0,
            provenance_row: 10,
            vector_row: 20,
            property_row: 30,
            epistemic_status: EpistemicStatus::Observed as u8,
            lifecycle_status: LifecycleStatus::Superseded as u8,
            relation_kind: VersionRelation::Supersedes as u8,
            reserved: 0,
            confidence_q16: 65536,
        };

        let bytes = bytemuck::bytes_of(&row);
        assert_eq!(bytes.len(), 56);

        let recovered: &VersionRow = bytemuck::from_bytes(bytes);
        assert_eq!(recovered.entity_id, 1001);
        assert_eq!(recovered.version_id, 2002);
        assert_eq!(recovered.valid_from_lsn, 50);
        assert_eq!(recovered.valid_until_lsn, 100);
    }

    #[test]
    fn test_provenance_arena_roundtrip() {
        let arena = ProvenanceArena::new(1);
        let record = ProvenanceRecord {
            source_uri: Arc::from("file:///data/telemetry_wal.log"),
            actor_id: Arc::from("agent_ingest_01"),
            extraction_method: Arc::from("direct_log_parse"),
            commit_lsn: 1000,
            timestamp_ms: 1718123456,
            confidence: 0.95,
            evidence: vec![
                DurableEvidenceRef::Entity(101),
                DurableEvidenceRef::Entity(102),
            ],
            signature_hash: [7u8; 32],
        };

        let (id, row_idx) = arena.append(&record);
        assert_eq!(id, 1);
        assert_eq!(row_idx, 0);

        let resolved = arena.resolve_record(row_idx).expect("must resolve record");
        assert_eq!(
            resolved.source_uri.as_ref(),
            "file:///data/telemetry_wal.log"
        );
        assert_eq!(resolved.actor_id.as_ref(), "agent_ingest_01");
        assert_eq!(resolved.commit_lsn, 1000);
        assert_eq!(
            resolved.evidence,
            vec![
                DurableEvidenceRef::Entity(101),
                DurableEvidenceRef::Entity(102)
            ]
        );
        assert!((resolved.confidence - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_deterministic_stats_promotion() {
        let mut stats = DeterministicEvidenceStats::default();
        let utility_1_0_q32 = 1i64 << 32;

        for _ in 0..5 {
            stats.record_observation(true, utility_1_0_q32);
        }

        assert_eq!(stats.observation_count, 5);
        assert_eq!(stats.successes, 5);
        assert_eq!(stats.failures, 0);
        assert_eq!(stats.utility_sum_q32, 5 * utility_1_0_q32);

        assert!(stats.meets_promotion_threshold(5, 4, 4 * utility_1_0_q32, 0));

        stats.record_contradiction();
        assert!(!stats.meets_promotion_threshold(5, 4, 4 * utility_1_0_q32, 0));
    }

    #[test]
    fn test_phase2_identity_and_compaction() {
        let segment = Arc::new(EntitySegment::new(1, 100));

        let m1 = EntityMutation::Create {
            entity_id: 100,
            header: EntityHeader::default(),
            initial_version_id: 501,
            provenance_id: 1001,
            provenance_record: None,
            epistemic_status: EpistemicStatus::Observed,
        };
        m1.apply(&segment, 10).unwrap();

        let m2 = EntityMutation::Create {
            entity_id: 101,
            header: EntityHeader::default(),
            initial_version_id: 502,
            provenance_id: 1002,
            provenance_record: None,
            epistemic_status: EpistemicStatus::Observed,
        };
        m2.apply(&segment, 20).unwrap();

        let m3 = EntityMutation::Create {
            entity_id: 102,
            header: EntityHeader::default(),
            initial_version_id: 503,
            provenance_id: 1003,
            provenance_record: None,
            epistemic_status: EpistemicStatus::Observed,
        };
        m3.apply(&segment, 30).unwrap();

        assert!(segment.arena.delete(1));

        let compacted = segment.compact(2);
        assert_eq!(compacted.arena.live_count(), 2);
        assert_eq!(compacted.arena.id_to_index(100), Some(0));
        assert_eq!(compacted.arena.id_to_index(102), Some(1));
        assert_eq!(compacted.arena.index_to_id(0), Some(100));
        assert_eq!(compacted.arena.index_to_id(1), Some(102));

        assert!(compacted.arena.id_to_index(101).is_none());
    }

    #[test]
    fn test_phase2_exact_temporal_predicates() {
        let mut row = VersionRow::default();
        row.valid_from_lsn = 100;
        row.valid_until_lsn = 200;

        assert!(!row.visible_at(99));
        assert!(row.visible_at(100));
        assert!(row.visible_at(150));
        assert!(row.visible_at(199));
        assert!(!row.visible_at(200));
        assert!(!row.visible_at(250));

        row.valid_until_lsn = u64::MAX;
        assert!(row.visible_at(100));
        assert!(row.visible_at(5000));
    }

    #[test]
    fn test_phase2_epistemic_adjudication_rules() {
        assert!(
            validate_epistemic_transition(EpistemicStatus::Provisional, EpistemicStatus::Inferred)
                .is_ok()
        );
        assert!(
            validate_epistemic_transition(
                EpistemicStatus::Provisional,
                EpistemicStatus::Contradicted
            )
            .is_ok()
        );
        assert!(
            validate_epistemic_transition(EpistemicStatus::Inferred, EpistemicStatus::Contradicted)
                .is_ok()
        );
        assert!(
            validate_epistemic_transition(EpistemicStatus::Observed, EpistemicStatus::Contradicted)
                .is_ok()
        );
        assert!(
            validate_epistemic_transition(EpistemicStatus::Asserted, EpistemicStatus::Contradicted)
                .is_ok()
        );

        assert!(
            validate_epistemic_transition(EpistemicStatus::Inferred, EpistemicStatus::Observed)
                .is_err()
        );
        assert!(
            validate_epistemic_transition(EpistemicStatus::Provisional, EpistemicStatus::Observed)
                .is_err()
        );
        assert!(
            validate_epistemic_transition(EpistemicStatus::Contradicted, EpistemicStatus::Observed)
                .is_err()
        );
        assert!(
            validate_epistemic_transition(EpistemicStatus::Contradicted, EpistemicStatus::Inferred)
                .is_err()
        );
    }

    #[test]
    fn test_phase2_snapshot_v3_fail_closed_and_roundtrip() {
        let segment = Arc::new(EntitySegment::new(1, 1));
        let prov_rec = ProvenanceRecord {
            source_uri: Arc::from("file:///sensor_01.log"),
            actor_id: Arc::from("sensor_agent"),
            extraction_method: Arc::from("parser"),
            commit_lsn: 100,
            timestamp_ms: 1718000000,
            confidence: 0.99,
            evidence: vec![DurableEvidenceRef::Entity(42)],
            signature_hash: [3u8; 32],
        };

        EntityMutation::Create {
            entity_id: 42,
            header: EntityHeader::default(),
            initial_version_id: 101,
            provenance_id: 201,
            provenance_record: Some(prov_rec),
            epistemic_status: EpistemicStatus::Observed,
        }
        .apply(&segment, 100)
        .unwrap();

        let bytes = encode_snapshot_v3(&segment, 100);
        assert!(&bytes[0..8] == &SNAPSHOT_V3_MAGIC);

        let (lsn, recovered) = decode_snapshot_v3(&bytes).expect("must decode snapshot v3");
        assert_eq!(lsn, 100);

        let snap = Arc::new(recovered).read_snapshot(100);
        let current = snap.current(42).expect("entity 42 must be visible");
        assert_eq!(current.entity_id, 42);
        assert_eq!(current.version_id, 101);
        assert_eq!(current.epistemic_status, EpistemicStatus::Observed);
        assert_eq!(
            current.provenance.unwrap().source_uri.as_ref(),
            "file:///sensor_01.log"
        );

        let mut corrupted = bytes.clone();
        let len = corrupted.len();
        corrupted[len - 5] ^= 0xFF;
        assert!(decode_snapshot_v3(&corrupted).is_err());
    }

    #[test]
    fn test_phase2_snapshot_isolation() {
        let segment = Arc::new(EntitySegment::new(1, 1));

        EntityMutation::Create {
            entity_id: 777,
            header: EntityHeader::default(),
            initial_version_id: 1,
            provenance_id: 1,
            provenance_record: None,
            epistemic_status: EpistemicStatus::Provisional,
        }
        .apply(&segment, 100)
        .unwrap();

        let s1 = segment.read_snapshot(100);

        EntityMutation::TransitionEpistemic {
            entity_id: 777,
            version_id: 2,
            expected: EpistemicStatus::Provisional,
            next: EpistemicStatus::Inferred,
            evidence: vec![],
            provenance_id: 2,
            provenance_record: None,
        }
        .apply(&segment, 200)
        .unwrap();

        let s2 = segment.read_snapshot(200);

        assert_eq!(
            s1.current(777).unwrap().epistemic_status,
            EpistemicStatus::Provisional
        );
        assert_eq!(
            s2.current(777).unwrap().epistemic_status,
            EpistemicStatus::Inferred
        );
    }

    #[test]
    fn test_phase2_apollo_deadline_system_test() {
        let segment = Arc::new(EntitySegment::new(1, 1));
        let apollo_id = 9001;

        let prov_june = ProvenanceRecord {
            source_uri: Arc::from("file:///apollo_spec_v1.pdf"),
            actor_id: Arc::from("mission_lead"),
            extraction_method: Arc::from("spec_ingest"),
            commit_lsn: 100,
            timestamp_ms: 1718000100,
            confidence: 1.0,
            evidence: vec![],
            signature_hash: [1u8; 32],
        };

        EntityMutation::Create {
            entity_id: apollo_id,
            header: EntityHeader {
                property_row: 6,
                ..EntityHeader::default()
            },
            initial_version_id: 1,
            provenance_id: 1,
            provenance_record: Some(prov_june),
            epistemic_status: EpistemicStatus::Observed,
        }
        .apply(&segment, 100)
        .unwrap();

        let prov_aug = ProvenanceRecord {
            source_uri: Arc::from("file:///quarterly_review_q2.pdf"),
            actor_id: Arc::from("program_manager"),
            extraction_method: Arc::from("review_ingest"),
            commit_lsn: 200,
            timestamp_ms: 1718000200,
            confidence: 1.0,
            evidence: vec![DurableEvidenceRef::EntityVersion(apollo_id, 1)],
            signature_hash: [2u8; 32],
        };

        EntityMutation::CreateVersion {
            entity_id: apollo_id,
            version_id: 2,
            provenance_id: 2,
            provenance_record: Some(prov_aug),
            epistemic_status: EpistemicStatus::Observed,
            lifecycle_status: LifecycleStatus::Active,
            relation_kind: VersionRelation::Supersedes,
            property_row: 8,
            vector_row: NULL_ROW_REF,
        }
        .apply(&segment, 200)
        .unwrap();

        let prov_sep = ProvenanceRecord {
            source_uri: Arc::from("file:///vendor_slip_notice.pdf"),
            actor_id: Arc::from("vendor_lead"),
            extraction_method: Arc::from("vendor_notice"),
            commit_lsn: 300,
            timestamp_ms: 1718000300,
            confidence: 0.98,
            evidence: vec![DurableEvidenceRef::EntityVersion(apollo_id, 2)],
            signature_hash: [3u8; 32],
        };

        EntityMutation::CreateVersion {
            entity_id: apollo_id,
            version_id: 3,
            provenance_id: 3,
            provenance_record: Some(prov_sep),
            epistemic_status: EpistemicStatus::Observed,
            lifecycle_status: LifecycleStatus::Active,
            relation_kind: VersionRelation::Supersedes,
            property_row: 9,
            vector_row: NULL_ROW_REF,
        }
        .apply(&segment, 300)
        .unwrap();

        let verify_all = |seg: &Arc<EntitySegment>| {
            let snap = seg.read_snapshot(400);

            let v150 = snap.as_of(apollo_id, 150).expect("must resolve at 150");
            assert_eq!(v150.property_row, 6);
            assert_eq!(v150.version_id, 1);
            assert_eq!(v150.valid_from_lsn, 100);
            assert_eq!(v150.valid_until_lsn, Some(200));

            let v250 = snap.as_of(apollo_id, 250).expect("must resolve at 250");
            assert_eq!(v250.property_row, 8);
            assert_eq!(v250.version_id, 2);
            assert_eq!(v250.valid_from_lsn, 200);
            assert_eq!(v250.valid_until_lsn, Some(300));

            let v350 = snap.as_of(apollo_id, 350).expect("must resolve at 350");
            assert_eq!(v350.property_row, 9);
            assert_eq!(v350.version_id, 3);
            assert_eq!(v350.valid_from_lsn, 300);
            assert_eq!(v350.valid_until_lsn, None);

            let hist = snap.history(apollo_id);
            assert_eq!(hist.len(), 3);
            assert_eq!(hist[0].property_row, 6);
            assert_eq!(hist[1].property_row, 8);
            assert_eq!(hist[2].property_row, 9);

            let current = snap.current(apollo_id).expect("must resolve current");
            assert_eq!(
                current.provenance.unwrap().source_uri.as_ref(),
                "file:///vendor_slip_notice.pdf"
            );
        };

        verify_all(&segment);
        let compacted = segment.compact(2);
        verify_all(&compacted);
    }

    // ─── MANDATORY PHASE 3 TESTS ──────────────────────────────────────────────

    #[test]
    fn test_phase3_exact_operators_and_boundary_conditions() {
        let dim = 8;
        let segment = Arc::new(EntitySegment::with_dimension(1, 1, dim));

        // Ingest 20 entities with known distinct vectors
        for i in 1..=20 {
            let vec: Vec<f32> = (0..dim)
                .map(|d| (i as f32 * 0.1) + (d as f32 * 0.01))
                .collect();
            let vrow = segment.vector_arena.append(&vec).unwrap();

            let header = EntityHeader {
                vector_row: vrow,
                flags: ENTITY_FLAG_LIVE | ENTITY_FLAG_HAS_VECTOR,
                ..EntityHeader::default()
            };

            EntityMutation::Create {
                entity_id: i as u64,
                header,
                initial_version_id: i as u64,
                provenance_id: i as u64,
                provenance_record: None,
                epistemic_status: EpistemicStatus::Observed,
            }
            .apply(&segment, 100)
            .unwrap();
        }

        let snap = segment.read_snapshot(100);
        let query = vec![1.0f32; dim];

        // 1. Boundary: Empty Eligibility
        let empty_e = EligibilitySet::empty(100, 1, 20);
        let (res_empty, proof_empty) =
            exact_top_k(&snap, &query, &empty_e, 5, DistanceFunction::Cosine, None);
        assert!(res_empty.is_empty());
        assert_eq!(proof_empty.scored_count, 0);

        // 2. Boundary: |E| = 1
        let single_e = EligibilitySet::from_sparse(100, 1, 20, vec![5]); // entity_id 6 (index 5)
        let (res_single, _) =
            exact_top_k(&snap, &query, &single_e, 5, DistanceFunction::Cosine, None);
        assert_eq!(res_single.len(), 1);
        assert_eq!(res_single[0].entity_id, 6);

        // 3. Boundary: |E| < k (e.g. |E| = 3, k = 10)
        let small_e = EligibilitySet::from_sparse(100, 1, 20, vec![2, 4, 6]);
        let (res_small, _) =
            exact_top_k(&snap, &query, &small_e, 10, DistanceFunction::Cosine, None);
        assert_eq!(res_small.len(), 3);

        // 4. Operator Equivalence: SparseGather == MaskedDense == ScalarReference
        let subset_e = EligibilitySet::from_sparse(100, 1, 20, (0..15).collect());
        let (scalar_res, _) =
            exact_top_k_scalar(&snap, &query, &subset_e, 5, DistanceFunction::Cosine);
        let (sparse_res, _) = sparse_gather_scan(
            &snap,
            &query,
            &subset_e.to_sparse_indices(),
            5,
            DistanceFunction::Cosine,
        );
        let (dense_res, _) = masked_dense_scan(
            &snap,
            &query,
            &subset_e.to_dense_bitmap(),
            5,
            DistanceFunction::Cosine,
        );

        assert_eq!(scalar_res, sparse_res);
        assert_eq!(sparse_res, dense_res);
    }

    #[test]
    fn test_phase3_randomized_exact_equivalence_property() {
        let dim = 16;
        let population = 100;
        let segment = Arc::new(EntitySegment::with_dimension(1, 1, dim));

        // Generate synthetic corpus
        for i in 0..population {
            let mut vec = vec![0.0f32; dim];
            for d in 0..dim {
                vec[d] = ((i * 17 + d * 31) % 100) as f32 / 100.0;
            }
            let vrow = segment.vector_arena.append(&vec).unwrap();
            let header = EntityHeader {
                vector_row: vrow,
                flags: ENTITY_FLAG_LIVE | ENTITY_FLAG_HAS_VECTOR,
                ..EntityHeader::default()
            };
            EntityMutation::Create {
                entity_id: (i + 1) as u64,
                header,
                initial_version_id: (i + 1) as u64,
                provenance_id: (i + 1) as u64,
                provenance_record: None,
                epistemic_status: EpistemicStatus::Observed,
            }
            .apply(&segment, 50)
            .unwrap();
        }

        let snap = segment.read_snapshot(50);
        let query: Vec<f32> = (0..dim).map(|d| (d as f32 * 0.2).sin()).collect();

        // Test over various randomized subsets and k values
        let subsets: Vec<Vec<EntityIndex>> = vec![
            (0..20).collect(),
            (10..60).filter(|x| x % 2 == 0).collect(),
            (0..population as EntityIndex)
                .filter(|x| x % 3 == 0)
                .collect(),
            (50..population as EntityIndex).collect(),
            (0..population as EntityIndex).collect(), // Full population N
        ];

        for (t, indices) in subsets.into_iter().enumerate() {
            let eligibility = EligibilitySet::from_sparse(50, 1, population, indices);
            let k = 10;

            let (scalar_topk, _) =
                exact_top_k_scalar(&snap, &query, &eligibility, k, DistanceFunction::Cosine);
            let (sparse_topk, _) = sparse_gather_scan(
                &snap,
                &query,
                &eligibility.to_sparse_indices(),
                k,
                DistanceFunction::Cosine,
            );
            let (dense_topk, _) = masked_dense_scan(
                &snap,
                &query,
                &eligibility.to_dense_bitmap(),
                k,
                DistanceFunction::Cosine,
            );

            assert_eq!(scalar_topk.len(), sparse_topk.len());
            assert_eq!(sparse_topk.len(), dense_topk.len());

            for i in 0..scalar_topk.len() {
                assert_eq!(
                    scalar_topk[i].entity_id, sparse_topk[i].entity_id,
                    "Entity ID mismatch between scalar and sparse at rank {} on subset test {}",
                    i, t
                );
                assert_eq!(
                    sparse_topk[i].entity_id, dense_topk[i].entity_id,
                    "Entity ID mismatch between sparse and dense at rank {} on subset test {}",
                    i, t
                );
                assert!(
                    (scalar_topk[i].score - sparse_topk[i].score).abs() < 1e-4,
                    "Score difference between scalar and sparse at rank {}",
                    i
                );
                assert!(
                    (sparse_topk[i].score - dense_topk[i].score).abs() < 1e-4,
                    "Score difference between sparse and dense at rank {}",
                    i
                );
            }
        }
    }
}
