/* holosphere/src/relation/mod.rs */
//!▫~•◦-------------------------------‣
//! # Native Dynamic Hypergraph Relations Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides first-class N-ary hypergraph relation semantics with durable
//! identities, dynamic schemas, immutable provenance, temporal versioning,
//! and inverted incidence query acceleration.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod arena;
pub mod binding;
pub mod header;
pub mod id;
pub mod incidence;
pub mod instance;
pub mod mutation;
pub mod projection;
pub mod query;
pub mod read;
pub mod schema;
pub mod version;

// Re-exports
pub use arena::RelationArena;
pub use binding::SegmentRoleBinding;
pub use header::{
    RELATION_FLAG_HAS_PROVENANCE, RELATION_FLAG_HAS_VERSION_HISTORY, RELATION_FLAG_LIVE,
    RelationHeader,
};
pub use id::{
    DurableRoleBinding, RelationId, RelationIndex, RelationTypeId, RelationVersionId, RoleId,
};
pub use incidence::{IncidenceIndex, IncidenceKey};
pub use instance::{DurableRelationInstance, compute_canonical_fingerprint};
pub use mutation::{RelationMutation, RelationMutationError};
pub use projection::{
    BinaryCsrProjection, BinaryProjectionCache, ProjectedBinaryRelationEdge,
    project_resolved_relation,
};
pub use query::{
    HyperPattern, HyperPatternError, HyperPatternMatch, HyperPatternMember, HyperPatternSemantics,
    RelationQuery,
};
pub use read::{RelationReadSnapshot, RelationSegment, ResolvedRelationVersion};
pub use schema::{
    BinaryProjectionSpec, ProjectionDirection, RelationType, RelationTypeState, RoleSchema,
    SchemaScope, SchemaValidationError,
};
pub use version::{RelationVersionRow, RelationVersionTable};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::header::EntityHeader;
    use crate::entity::id::DurableEvidenceRef;
    use crate::entity::mutation::EntityMutation;
    use crate::entity::provenance::ProvenanceRecord;
    use crate::entity::segment::EntitySegment;
    use crate::entity::status::EpistemicStatus;
    use std::sync::Arc;

    #[test]
    fn test_phase4_schema_validation_and_admission_rules() {
        let roles = vec![
            RoleSchema {
                role_id: 1,
                name: Arc::from("Work"),
                min_count: 1,
                max_count: 1,
                required: true,
            },
            RoleSchema {
                role_id: 2,
                name: Arc::from("Author"),
                min_count: 1,
                max_count: 10,
                required: true,
            },
        ];

        let schema = RelationType {
            id: 10,
            name: Arc::from("AUTHORED_BY"),
            schema_version: 1,
            state: RelationTypeState::Admitted,
            roles,
            binary_projection: Some(BinaryProjectionSpec {
                source_role: 1,
                target_role: 2,
                direction: ProjectionDirection::Directed,
            }),
            provenance_id: 1,
            structural_fingerprint: 0,
        };

        // 1. Valid 2-ary binding
        let valid_bindings = vec![
            DurableRoleBinding {
                entity_id: 100,
                role_id: 1,
            },
            DurableRoleBinding {
                entity_id: 200,
                role_id: 2,
            },
        ];
        assert!(schema.validate_bindings(&valid_bindings).is_ok());

        // 2. Missing required role (Author)
        let missing_author = vec![DurableRoleBinding {
            entity_id: 100,
            role_id: 1,
        }];
        assert!(schema.validate_bindings(&missing_author).is_err());

        // 3. Unknown role ID 99
        let unknown_role = vec![
            DurableRoleBinding {
                entity_id: 100,
                role_id: 1,
            },
            DurableRoleBinding {
                entity_id: 200,
                role_id: 2,
            },
            DurableRoleBinding {
                entity_id: 300,
                role_id: 99,
            },
        ];
        assert!(schema.validate_bindings(&unknown_role).is_err());

        // 4. Over-cardinality (Work max is 1)
        let over_work = vec![
            DurableRoleBinding {
                entity_id: 100,
                role_id: 1,
            },
            DurableRoleBinding {
                entity_id: 101,
                role_id: 1,
            },
            DurableRoleBinding {
                entity_id: 200,
                role_id: 2,
            },
        ];
        assert!(schema.validate_bindings(&over_work).is_err());
    }

    #[test]
    fn test_phase4_canonicalization_and_fingerprinting() {
        let type_id = 50;
        let schema_ver = 1;

        let b1 = vec![
            DurableRoleBinding {
                entity_id: 10,
                role_id: 1,
            },
            DurableRoleBinding {
                entity_id: 20,
                role_id: 2,
            },
        ];
        let b2 = vec![
            DurableRoleBinding {
                entity_id: 20,
                role_id: 2,
            },
            DurableRoleBinding {
                entity_id: 10,
                role_id: 1,
            },
        ];

        // Fingerprint is invariant to input role order
        let fp1 = compute_canonical_fingerprint(type_id, schema_ver, &b1);
        let fp2 = compute_canonical_fingerprint(type_id, schema_ver, &b2);
        assert_eq!(fp1, fp2);

        // Role reversal produces different fingerprint (no implicit symmetry)
        let b_reversed = vec![
            DurableRoleBinding {
                entity_id: 20,
                role_id: 1,
            },
            DurableRoleBinding {
                entity_id: 10,
                role_id: 2,
            },
        ];
        let fp_rev = compute_canonical_fingerprint(type_id, schema_ver, &b_reversed);
        assert_ne!(fp1, fp_rev);
    }

    #[test]
    fn test_phase4_incidence_posting_intersection() {
        let incidence = IncidenceIndex::new();
        let type_id = 1;

        // Relation 101: Role 1 = Ent 10, Role 2 = Ent 20
        incidence.insert(type_id, 1, 10, 101);
        incidence.insert(type_id, 2, 20, 101);

        // Relation 102: Role 1 = Ent 10, Role 2 = Ent 30
        incidence.insert(type_id, 1, 10, 102);
        incidence.insert(type_id, 2, 30, 102);

        // Relation 103: Role 1 = Ent 15, Role 2 = Ent 20
        incidence.insert(type_id, 1, 15, 103);
        incidence.insert(type_id, 2, 20, 103);

        // Lookup single role: Role 1 = Ent 10 -> [101, 102]
        let p_role1 = incidence.lookup(type_id, 1, 10);
        assert_eq!(p_role1, vec![101, 102]);

        // Lookup single role: Role 2 = Ent 20 -> [101, 103]
        let p_role2 = incidence.lookup(type_id, 2, 20);
        assert_eq!(p_role2, vec![101, 103]);

        // Intersect: Role 1 = 10 AND Role 2 = 20 -> [101]
        let intersected = IncidenceIndex::intersect(&[p_role1, p_role2]);
        assert_eq!(intersected, vec![101]);
    }

    #[test]
    fn test_phase4_caused_outage_temporal_lineage_and_compaction() {
        let ent_seg = Arc::new(EntitySegment::new(1, 1));
        let rel_seg = Arc::new(RelationSegment::new(1, 1));

        // 1. Create participating entities (DatabaseA=1001, DiskSaturation=2001, ProdEuWest=3001)
        for &id in &[1001u64, 2001, 3001] {
            EntityMutation::Create {
                entity_id: id,
                header: EntityHeader::default(),
                initial_version_id: id,
                provenance_id: id,
                provenance_record: None,
                epistemic_status: EpistemicStatus::Observed,
            }
            .apply(&ent_seg, 50)
            .unwrap();
        }

        // 2. Register 3-ary Relation Schema: CAUSED_OUTAGE
        let outage_type = RelationType {
            id: 77,
            name: Arc::from("CAUSED_OUTAGE"),
            schema_version: 1,
            state: RelationTypeState::Admitted,
            roles: vec![
                RoleSchema {
                    role_id: 1,
                    name: Arc::from("Component"),
                    min_count: 1,
                    max_count: 1,
                    required: true,
                },
                RoleSchema {
                    role_id: 2,
                    name: Arc::from("Trigger"),
                    min_count: 1,
                    max_count: 1,
                    required: true,
                },
                RoleSchema {
                    role_id: 3,
                    name: Arc::from("Environment"),
                    min_count: 1,
                    max_count: 1,
                    required: true,
                },
            ],
            binary_projection: None,
            provenance_id: 1,
            structural_fingerprint: 0,
        };
        rel_seg.register_type(outage_type);

        let outage_id = 882;
        let bindings = vec![
            DurableRoleBinding {
                entity_id: 1001,
                role_id: 1,
            },
            DurableRoleBinding {
                entity_id: 2001,
                role_id: 2,
            },
            DurableRoleBinding {
                entity_id: 3001,
                role_id: 3,
            },
        ];

        // LSN 100: Create relation as Provisional
        let prov1 = ProvenanceRecord {
            source_uri: Arc::from("file:///telemetry_anomaly.log"),
            actor_id: Arc::from("anomaly_detector"),
            extraction_method: Arc::from("stat_correlation"),
            commit_lsn: 100,
            timestamp_ms: 1718000100,
            confidence: 0.6,
            evidence: vec![],
            signature_hash: [1u8; 32],
        };

        RelationMutation::CreateRelation {
            relation_id: outage_id,
            relation_type_id: 77,
            bindings: bindings.clone(),
            provenance_id: 1,
            provenance_record: Some(prov1),
            epistemic_status: EpistemicStatus::Provisional,
        }
        .apply(&rel_seg, &ent_seg, 100)
        .unwrap();

        // LSN 200: Transition to Inferred
        let prov2 = ProvenanceRecord {
            source_uri: Arc::from("file:///incident_triage.pdf"),
            actor_id: Arc::from("sre_lead"),
            extraction_method: Arc::from("root_cause_analysis"),
            commit_lsn: 200,
            timestamp_ms: 1718000200,
            confidence: 0.9,
            evidence: vec![DurableEvidenceRef::Entity(1001)],
            signature_hash: [2u8; 32],
        };

        RelationMutation::TransitionEpistemic {
            relation_id: outage_id,
            version_id: 2,
            expected: EpistemicStatus::Provisional,
            next: EpistemicStatus::Inferred,
            evidence: vec![DurableEvidenceRef::Entity(1001)],
            provenance_id: 2,
            provenance_record: Some(prov2),
        }
        .apply(&rel_seg, &ent_seg, 200)
        .unwrap();

        // LSN 300: Transition to Contradicted (e.g. post-mortem found alternative cause)
        let prov3 = ProvenanceRecord {
            source_uri: Arc::from("file:///final_postmortem.pdf"),
            actor_id: Arc::from("principal_investigator"),
            extraction_method: Arc::from("postmortem"),
            commit_lsn: 300,
            timestamp_ms: 1718000300,
            confidence: 1.0,
            evidence: vec![],
            signature_hash: [3u8; 32],
        };

        RelationMutation::TransitionEpistemic {
            relation_id: outage_id,
            version_id: 3,
            expected: EpistemicStatus::Inferred,
            next: EpistemicStatus::Contradicted,
            evidence: vec![],
            provenance_id: 3,
            provenance_record: Some(prov3),
        }
        .apply(&rel_seg, &ent_seg, 300)
        .unwrap();

        // Verification closure
        let verify_all = |r_seg: &Arc<RelationSegment>, e_seg: &Arc<EntitySegment>| {
            let r_snap = r_seg.read_snapshot(400);
            let e_snap = e_seg.read_snapshot(400);

            // Point in time temporal assertions
            let as_of_150 = r_snap
                .as_of(outage_id, 150, &e_snap)
                .expect("must resolve at 150");
            assert_eq!(as_of_150.epistemic_status, EpistemicStatus::Provisional);
            assert_eq!(as_of_150.valid_from_lsn, 100);
            assert_eq!(as_of_150.valid_until_lsn, Some(200));
            assert_eq!(as_of_150.bindings.len(), 3);

            let as_of_250 = r_snap
                .as_of(outage_id, 250, &e_snap)
                .expect("must resolve at 250");
            assert_eq!(as_of_250.epistemic_status, EpistemicStatus::Inferred);
            assert_eq!(as_of_250.valid_from_lsn, 200);
            assert_eq!(as_of_250.valid_until_lsn, Some(300));

            let as_of_350 = r_snap
                .as_of(outage_id, 350, &e_snap)
                .expect("must resolve at 350");
            assert_eq!(as_of_350.epistemic_status, EpistemicStatus::Contradicted);
            assert_eq!(as_of_350.valid_from_lsn, 300);
            assert_eq!(as_of_350.valid_until_lsn, None);

            // Pattern query matching
            let query = RelationQuery::new()
                .with_type(77)
                .with_role(1, 1001) // Component = DatabaseA
                .with_role(2, 2001) // Trigger = DiskSaturation
                .with_as_of(250);

            let matches = query.execute(&r_snap, &e_snap);
            assert_eq!(matches.len(), 1);
            assert_eq!(matches[0].relation_id, outage_id);
            assert_eq!(matches[0].epistemic_status, EpistemicStatus::Inferred);
        };

        verify_all(&rel_seg, &ent_seg);

        // Compact both entity and relation segments
        let compacted_ent = ent_seg.compact(2);
        let compacted_rel = rel_seg.compact(2, &ent_seg, &compacted_ent);

        verify_all(&compacted_rel, &compacted_ent);
    }
}
