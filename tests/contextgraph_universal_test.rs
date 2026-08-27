/* holosphere/tests/contextgraph_universal_test.rs */
//!▫~•◦-------------------------------‣
//! # HoloSphere ContextGraph Comprehensive Certification Suite
//!▫~•◦-------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use hnsqr::contextgraph::{
    ContextBudget, ContextCompiler, ContextGraphStore, ContextQueryEngine,
    GraphFingerprinter, HtmlVisualizerView, JsonExportView, MarkdownReportView,
    Namespace, QueryPlan, QueryPlanner, ContextQueryRequest, SourceAdapter, SourceInput,
    adapters::code_rust::RustSourceAdapter, adapters::fs::FilesystemSourceAdapter,
    adapters::markdown::MarkdownSourceAdapter, schema::*,
};
use hnsqr::transport::model_gateway::{EvidenceClass, VerificationState};

#[test]
fn test_gate_1_universal_entity_and_relation_model() {
    let ns = Namespace::new("workspace:holosphere");
    let fingerprint = [42u8; 32];

    let fn_entity_id = EntityId::compute(
        &ns,
        &EntityKind::code_function(),
        "search",
        Some("file:///src/vector/index.rs#L40-L60"),
        &fingerprint,
    );

    let doc_claim_id = EntityId::compute(
        &ns,
        &EntityKind::document_claim(),
        "Bounded Rivero proof guarantee",
        Some("file:///docs/proof.md#L10-L15"),
        &fingerprint,
    );

    let relation = Relation::binary(
        &doc_claim_id,
        &fn_entity_id,
        RelationKind::supports(),
        RelationOrigin::Extracted,
    );

    assert_eq!(relation.kind.as_str(), "supports");
    assert_eq!(relation.origin, RelationOrigin::Extracted);
    assert_eq!(relation.confidence, 1.0);
    assert_eq!(relation.participants.len(), 2);
    assert_eq!(relation.participants[0].role, "source");
    assert_eq!(relation.participants[1].role, "target");
}

#[test]
fn test_gate_2_and_4_rust_ast_and_rationale_extraction() {
    let source_code = r#"
/// High performance vector index search
// SAFETY: caller guarantees vector slice is aligned to 32 bytes
// WHY: SIMD dual-accumulator instruction throughput
pub fn search_internal(&self, query: &[f32], k: usize) -> Vec<usize> {
    self.score_candidates(query, k)
}

fn score_candidates(&self, query: &[f32], k: usize) -> Vec<usize> {
    vec![0]
}
"#;

    let adapter = RustSourceAdapter::new();
    let input = SourceInput::from_text(source_code, "file:///src/test_index.rs", "rust");
    let ns = Namespace::new("test_ns");

    let batch = adapter.extract(&input, &ns).expect("Rust extraction should succeed");

    assert!(batch.entities.len() >= 3, "Should extract file, search_internal, score_candidates, rationale");

    let rationale_entities: Vec<_> = batch
        .entities
        .iter()
        .filter(|e| e.kind == EntityKind::code_rationale())
        .collect();
    assert_eq!(rationale_entities.len(), 2, "Should extract SAFETY and WHY rationale notes");

    let justifies_rel = batch
        .relations
        .iter()
        .find(|r| r.kind == RelationKind::justifies());
    assert!(justifies_rel.is_some(), "Should emit justifies relation for SAFETY note");
}

#[test]
fn test_gate_3_atomic_contextgraph_store_publication() {
    let store = ContextGraphStore::new();
    let ns = Namespace::new("test_store");

    let ent1 = Entity {
        id: EntityId("ent_1".to_string()),
        kind: EntityKind::code_function(),
        label: "HNSQRIndex::search".to_string(),
        namespace: ns.clone(),
        locator: None,
        attributes: BTreeMap::new(),
        provenance: Vec::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
        fingerprint: [1u8; 32],
        valid_from_lsn: 0,
    };

    let ent2 = Entity {
        id: EntityId("ent_2".to_string()),
        kind: EntityKind::code_function(),
        label: "score_candidates".to_string(),
        namespace: ns.clone(),
        locator: None,
        attributes: BTreeMap::new(),
        provenance: Vec::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
        fingerprint: [2u8; 32],
        valid_from_lsn: 0,
    };

    let rel = Relation::call(&ent1.id, &ent2.id, RelationOrigin::Extracted);

    let delta = ContextGraphDelta {
        namespace: ns,
        insert_entities: vec![ent1, ent2],
        delete_entities: Vec::new(),
        insert_relations: vec![rel],
        delete_relations: Vec::new(),
        touched_locators: Vec::new(),
    };

    let lsn1 = store.commit_delta(delta);
    assert_eq!(lsn1, 1);

    let state = store.snapshot();
    assert_eq!(state.entities.len(), 2);
    assert_eq!(state.relations.len(), 1);
    assert_ne!(state.canonical_fingerprint, [0u8; 32]);
}

#[test]
fn test_gate_6_deterministic_canonical_graph_fingerprint() {
    let ns = Namespace::new("ns");
    let ent_a = Entity {
        id: EntityId("ent_a".to_string()),
        kind: EntityKind::code_function(),
        label: "A".to_string(),
        namespace: ns.clone(),
        locator: None,
        attributes: BTreeMap::new(),
        provenance: Vec::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
        fingerprint: [1u8; 32],
        valid_from_lsn: 0,
    };
    let ent_b = Entity {
        id: EntityId("ent_b".to_string()),
        kind: EntityKind::code_function(),
        label: "B".to_string(),
        namespace: ns,
        locator: None,
        attributes: BTreeMap::new(),
        provenance: Vec::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
        fingerprint: [2u8; 32],
        valid_from_lsn: 0,
    };
    let rel_ab = Relation::call(&ent_a.id, &ent_b.id, RelationOrigin::Extracted);

    // Feed in reverse order
    let fp1 = GraphFingerprinter::compute_fingerprint(&[ent_b.clone(), ent_a.clone()], &[rel_ab.clone()]);
    let fp2 = GraphFingerprinter::compute_fingerprint(&[ent_a, ent_b], &[rel_ab]);

    assert_eq!(fp1, fp2, "Fingerprint MUST be bit-exact regardless of evaluation or array order");
}

#[test]
fn test_gate_7_and_8_query_planner_and_budget_governed_navigation() {
    let store = ContextGraphStore::new();
    let ns = Namespace::new("test_queries");

    let e_gateway = Entity {
        id: EntityId("ent_gw".to_string()),
        kind: EntityKind::system_service(),
        label: "ModelToolService".to_string(),
        namespace: ns.clone(),
        locator: None,
        attributes: BTreeMap::new(),
        provenance: Vec::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
        fingerprint: [10u8; 32],
        valid_from_lsn: 0,
    };

    let e_router = Entity {
        id: EntityId("ent_router".to_string()),
        kind: EntityKind::code_struct(),
        label: "GatewayRouter".to_string(),
        namespace: ns.clone(),
        locator: None,
        attributes: BTreeMap::new(),
        provenance: Vec::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
        fingerprint: [11u8; 32],
        valid_from_lsn: 0,
    };

    let e_index = Entity {
        id: EntityId("ent_index".to_string()),
        kind: EntityKind::code_struct(),
        label: "HNSQRIndex".to_string(),
        namespace: ns.clone(),
        locator: None,
        attributes: BTreeMap::new(),
        provenance: Vec::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
        fingerprint: [12u8; 32],
        valid_from_lsn: 0,
    };

    let rel1 = Relation::call(&e_gateway.id, &e_router.id, RelationOrigin::Extracted);
    let rel2 = Relation::call(&e_router.id, &e_index.id, RelationOrigin::Extracted);

    store.commit_delta(ContextGraphDelta {
        namespace: ns,
        insert_entities: vec![e_gateway.clone(), e_router.clone(), e_index.clone()],
        delete_entities: Vec::new(),
        insert_relations: vec![rel1, rel2],
        delete_relations: Vec::new(),
        touched_locators: Vec::new(),
    });

    let state = store.snapshot();
    let budget = ContextBudget::default();

    // 1. Search
    let search_res = ContextQueryEngine::search(&state, "GatewayRouter", None, &budget);
    assert_eq!(search_res.entities.len(), 1);
    assert_eq!(search_res.entities[0].id, e_router.id);

    // 2. Explore
    let explore_res = ContextQueryEngine::explore(&state, &e_router.id, &budget).expect("Explore should find entity");
    assert_eq!(explore_res.relations.len(), 2, "Router has 1 incoming call and 1 outgoing call");

    // 3. Path
    let path_res = ContextQueryEngine::path(&state, &e_gateway.id, &e_index.id, &budget).expect("Path should be found");
    assert_eq!(path_res.relations.len(), 2, "Path from Gateway -> Router -> Index should be 2 hops");

    // 4. Impact
    let impact_res = ContextQueryEngine::impact(&state, &e_index.id, &budget);
    assert!(impact_res.entities.len() >= 2, "Impact of changing HNSQRIndex includes Router and Gateway");

    // 5. Query Planner
    let plan = QueryPlanner::plan(&ContextQueryRequest {
        from_entity: Some("ent_gw".to_string()),
        to_entity: Some("ent_index".to_string()),
        ..Default::default()
    });
    assert_eq!(plan, QueryPlan::PathSearch);
}

#[test]
fn test_gate_10_end_to_end_workspace_self_compilation() {
    let fs_adapter = FilesystemSourceAdapter::new();
    let sources = fs_adapter.crawl_directory("src/contextgraph").expect("Directory crawl should succeed");
    assert!(!sources.is_empty(), "Should discover source files in src/contextgraph");

    let compiler = ContextCompiler::default();
    let ns = Namespace::new("workspace:holosphere_contextgraph");
    let output = compiler.compile(&ns, &sources).expect("Compilation must succeed without errors");

    assert!(output.entities.len() >= 10, "Should extract dozens of entities from contextgraph sources");
    assert!(output.relations.len() >= 5, "Should resolve and extract structural relations");

    let store = ContextGraphStore::new();
    let lsn = store.commit_delta(output.into_delta());
    assert_eq!(lsn, 1);

    let state = store.snapshot();
    let markdown = MarkdownReportView::generate(&state);
    assert!(markdown.contains("HoloSphere ContextGraph Report"));
    assert!(markdown.contains("Universal Metrics"));

    let html = HtmlVisualizerView::generate_html(&state);
    assert!(html.contains("<!DOCTYPE html>"));
    assert!(html.contains("HoloSphere ContextGraph"));
}

#[test]
fn test_graph_diff_and_upsert_invariants() {
    let store = ContextGraphStore::new();
    let ns = Namespace::new("test_diff_ns");

    let ent1 = Entity {
        id: EntityId("ent_alpha".to_string()),
        kind: EntityKind::code_function(),
        label: "Alpha".to_string(),
        namespace: ns.clone(),
        locator: None,
        attributes: BTreeMap::new(),
        provenance: Vec::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
        fingerprint: [100u8; 32],
        valid_from_lsn: 0,
    };

    let delta1 = ContextGraphDelta {
        namespace: ns.clone(),
        insert_entities: vec![ent1.clone()],
        delete_entities: Vec::new(),
        insert_relations: Vec::new(),
        delete_relations: Vec::new(),
        touched_locators: Vec::new(),
    };
    store.commit_delta(delta1);
    let state_v1 = store.snapshot();

    // Modify ent1 and add ent2
    let mut ent1_modified = ent1.clone();
    ent1_modified.label = "AlphaModified".to_string();
    ent1_modified.fingerprint = [101u8; 32];

    let ent2 = Entity {
        id: EntityId("ent_beta".to_string()),
        kind: EntityKind::code_struct(),
        label: "Beta".to_string(),
        namespace: ns.clone(),
        locator: None,
        attributes: BTreeMap::new(),
        provenance: Vec::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
        fingerprint: [200u8; 32],
        valid_from_lsn: 0,
    };

    let rel = Relation::call(&ent1.id, &ent2.id, RelationOrigin::Extracted);

    let delta2 = ContextGraphDelta {
        namespace: ns,
        insert_entities: vec![ent1_modified, ent2],
        delete_entities: Vec::new(),
        insert_relations: vec![rel],
        delete_relations: Vec::new(),
        touched_locators: Vec::new(),
    };
    store.commit_delta(delta2);
    let state_v2 = store.snapshot();

    let diff = ContextQueryEngine::diff(&state_v1, &state_v2);
    assert_eq!(diff.from_lsn, 1);
    assert_eq!(diff.to_lsn, 2);
    assert_eq!(diff.added_entities.len(), 1);
    assert_eq!(diff.added_entities[0].id, EntityId("ent_beta".to_string()));
    assert_eq!(diff.modified_entities.len(), 1);
    assert_eq!(diff.modified_entities[0], EntityId("ent_alpha".to_string()));
    assert_eq!(diff.added_relations.len(), 1);

    // Verify index cleanliness on update: "Alpha" should not return "ent_alpha" anymore, only "AlphaModified"
    let old_lookup = store.lookup_by_label("Alpha");
    assert!(old_lookup.is_empty(), "Old label 'Alpha' must be cleaned up on upsert");
    let new_lookup = store.lookup_by_label("AlphaModified");
    assert_eq!(new_lookup.len(), 1);
}

