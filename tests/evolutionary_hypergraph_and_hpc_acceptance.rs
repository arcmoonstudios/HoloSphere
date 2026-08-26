/* tests/evolutionary_hypergraph_and_hpc_acceptance.rs */
use hnsqr::cluster::topology::{ClusterHeartbeat, TopologyManager};
use hnsqr::cluster::world_digest::WorldStateDigest;
use hnsqr::ecosystem::kv_cache::AttentionSinkKvCache;
use hnsqr::entity::id::ProvenanceId;
use hnsqr::entity::stats::OnlineStatsAccumulator;
use hnsqr::entity::status::EpistemicStatus;
use hnsqr::graph::query::ast::{
    Direction, GraphPattern, QueryAst, ReturnClause, ReturnItem, WhereClause,
};
use hnsqr::graph::query::logical::LogicalPlan;
use hnsqr::graph::query::symbols::SymbolTable;
use hnsqr::graph::storage::edge_delta::EdgeRecord;
use hnsqr::learning::adjudication::policy::{AdjudicationDecisionCode, AdjudicationPolicy};
use hnsqr::learning::adjudication::transition::evaluate_adjudication_with_causal;
use hnsqr::learning::evidence::accumulator::EvidenceSummary;
use hnsqr::learning::inference::rune_evo::reasoning::blade::Cl24EntityBasis;
use hnsqr::metadata::geo::{GeoPoint, GeoPolygon};
use hnsqr::metadata::index::{FilterExpr, MetadataInvertedIndex};
use hnsqr::planning::planner::{ExecutionPlan, RetrievalContract, UniversalPlanner};
use hnsqr::relation::incidence::{IncidenceIndex, IncidenceTraverser};
use hnsqr::relation::instance::DurableRelationInstance;

#[test]
fn test_ekh_graph_edge_to_durable_relation_unification() {
    let edge = EdgeRecord::new(42, 100, 200, 1.0, 0);

    let prov: ProvenanceId = 999;
    let relation = DurableRelationInstance::from_graph_edge(777, &edge, prov);

    assert_eq!(relation.id, 777);
    assert_eq!(relation.type_id, 42);
    assert_eq!(relation.provenance_id, prov);
    assert_eq!(relation.epistemic_status, EpistemicStatus::Observed);
    assert_eq!(relation.bindings.len(), 2);
    assert_eq!(relation.bindings[0].role_id, 1);
    assert_eq!(relation.bindings[0].entity_id, 100);
    assert_eq!(relation.bindings[1].role_id, 2);
    assert_eq!(relation.bindings[1].entity_id, 200);

    // Multivector Cl(24) basis conversion of the unified relation
    let basis = Cl24EntityBasis::new(vec![100, 200, 300]).unwrap();
    let blade = basis.blade_for_relation(&relation, 1.0).unwrap();
    assert_eq!(blade.grade(), 2); // 2-blade in Cl(24)
}

#[test]
fn test_ekh_incidence_traverser_multi_role_intersection() {
    let index = IncidenceIndex::new();

    // Type 1: Hyperedge (Role 1: Entity 10, Role 2: Entity 20, Role 3: Entity 30) -> Rel 100
    index.insert(1, 1, 10, 100);
    index.insert(1, 2, 20, 100);
    index.insert(1, 3, 30, 100);

    // Type 1: Hyperedge (Role 1: Entity 10, Role 2: Entity 20, Role 3: Entity 99) -> Rel 101
    index.insert(1, 1, 10, 101);
    index.insert(1, 2, 20, 101);
    index.insert(1, 3, 99, 101);

    let traverser = IncidenceTraverser::new(&index);

    // Query intersection for (Role 1: 10, Role 2: 20) -> [100, 101]
    let res = traverser.query_role_intersection(1, &[(1, 10), (2, 20)]);
    assert_eq!(res, vec![100, 101]);

    // Query intersection for (Role 1: 10, Role 2: 20, Role 3: 30) -> [100]
    let res2 = traverser.query_role_intersection(1, &[(1, 10), (2, 20), (3, 30)]);
    assert_eq!(res2, vec![100]);
}

#[test]
fn test_hyper_match_query_planning() {
    let mut symbols = SymbolTable::default();
    symbols.intern("a");
    symbols.intern("b");
    symbols.intern("c");
    symbols.intern("target");

    let ast = QueryAst {
        vector_match: None,
        patterns: vec![GraphPattern::HyperMatch {
            antecedent_aliases: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            rel_alias: Some("h".to_string()),
            rel_type: Some(5),
            consequent_aliases: vec!["target".to_string()],
            direction: Direction::Outgoing,
        }],
        where_clause: WhereClause::default(),
        return_clause: ReturnClause {
            items: vec![ReturnItem::Alias("target".to_string())],
            limit: Some(10),
        },
        mutations: Vec::new(),
        unwind: None,
        subqueries: Vec::new(),
    };

    let plan = hnsqr::graph::query::QueryPlanner::build_logical_plan(&ast, &symbols);
    assert!(matches!(plan, LogicalPlan::Limit { .. }));
}

#[test]
fn test_multivector_maxsim_planner_routing() {
    let contract = RetrievalContract::MultiVectorMaxSim {
        token_dim: 128,
        top_k_centroids: 32,
    };
    let plan = UniversalPlanner::plan(10_000, 128, None, contract, false);
    assert_eq!(plan, ExecutionPlan::MultiVectorMaxSim);
}

#[test]
fn test_topology_anti_entropy_divergence_trigger() {
    let manager = TopologyManager::new(4);
    let digest1 = WorldStateDigest::compute(100, [1; 32], [2; 32], [3; 32], [4; 32], [5; 32]);
    let digest2 = WorldStateDigest::compute(100, [1; 32], [9; 32], [3; 32], [4; 32], [5; 32]);

    manager.update_local_digest(digest1);

    // Matching heartbeat does not trigger anti-entropy
    let matching_hb = ClusterHeartbeat {
        node_id: "node_2".to_string(),
        term: 1,
        world_digest: Some(digest1),
    };
    assert!(manager.handle_heartbeat(matching_hb));
    assert_eq!(
        manager
            .anti_entropy_triggers
            .load(std::sync::atomic::Ordering::Relaxed),
        0
    );

    // Divergent heartbeat triggers anti-entropy
    let divergent_hb = ClusterHeartbeat {
        node_id: "node_3".to_string(),
        term: 1,
        world_digest: Some(digest2),
    };
    assert!(!manager.handle_heartbeat(divergent_hb));
    assert_eq!(
        manager
            .anti_entropy_triggers
            .load(std::sync::atomic::Ordering::Relaxed),
        1
    );
}

#[test]
fn test_attention_sink_kv_cache_preservation() {
    let mut cache = AttentionSinkKvCache::new(2, 4); // 2 sink tokens, 4 recent tokens

    // Push 10 token vectors
    let tokens: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 8]).collect();
    cache.push_tokens(&tokens);

    assert_eq!(cache.token_count(), 6);
    // Initial sink tokens (0, 1) preserved
    assert_eq!(cache.cached_tokens[0], vec![0.0; 8]);
    assert_eq!(cache.cached_tokens[1], vec![1.0; 8]);
    // Most recent 4 tokens (6, 7, 8, 9) preserved
    assert_eq!(cache.cached_tokens[2], vec![6.0; 8]);
    assert_eq!(cache.cached_tokens[5], vec![9.0; 8]);
}

#[test]
fn test_causal_counterfactual_adjudication_gating() {
    let policy = AdjudicationPolicy::default();
    let summary = EvidenceSummary {
        observation_count: 50,
        support_count: 45,
        contradiction_count: 0,
        utility_sum_q32: 100 << 32,
        ..Default::default()
    };

    // When counterfactual invariant holds -> promotes to Inferred
    let (status_pass, code_pass, _) =
        evaluate_adjudication_with_causal(&summary, EpistemicStatus::Provisional, &policy, true);
    assert_eq!(status_pass, EpistemicStatus::Inferred);
    assert_eq!(code_pass, AdjudicationDecisionCode::SupportThresholdReached);

    // When counterfactual invariant fails -> retained as Provisional
    let (status_fail, code_fail, _) =
        evaluate_adjudication_with_causal(&summary, EpistemicStatus::Provisional, &policy, false);
    assert_eq!(status_fail, EpistemicStatus::Provisional);
    assert_eq!(code_fail, AdjudicationDecisionCode::ContextDependent);
}

#[test]
fn test_welford_online_stats_accumulator_convergence() {
    let mut acc = OnlineStatsAccumulator::new();
    let samples = [10.0, 12.0, 23.0, 23.0, 16.0, 23.0, 21.0, 16.0];
    for &s in &samples {
        acc.update(s);
    }

    assert_eq!(acc.count, 8);
    assert!((acc.mean - 18.0).abs() < 1e-5);
    let expected_variance = 192.0 / 7.0;
    assert!((acc.variance() - expected_variance).abs() < 1e-5);
    assert!((acc.std_dev() - expected_variance.sqrt()).abs() < 1e-5);
}

#[test]
fn test_geospatial_polygon_filtering_primitives() {
    let ring = vec![
        GeoPoint::new(40.70, -74.02),
        GeoPoint::new(40.70, -73.95),
        GeoPoint::new(40.80, -73.95),
        GeoPoint::new(40.80, -74.02),
    ];
    let poly = GeoPolygon::new(ring).unwrap();

    let inside_point = GeoPoint::new(40.75, -73.98);
    let outside_point = GeoPoint::new(40.85, -73.98);

    assert!(poly.contains_point(&inside_point));
    assert!(!poly.contains_point(&outside_point));

    let index = MetadataInvertedIndex::new();
    index.insert_metadata(0, &serde_json::json!({ "location": "40.75,-73.98" }));
    index.insert_metadata(1, &serde_json::json!({ "location": "40.85,-73.98" }));

    let filter_within = FilterExpr::geo_within("location", poly);
    let mask_within = index.evaluate_filter(&filter_within, 2);
    assert!(mask_within.contains(0));
    assert!(!mask_within.contains(1));

    let center = GeoPoint::new(40.75, -73.98);
    let filter_radius = FilterExpr::geo_radius("location", center, 2.0); // 2 km radius
    let mask_radius = index.evaluate_filter(&filter_radius, 2);
    assert!(mask_radius.contains(0));
    assert!(!mask_radius.contains(1));
}

#[test]
fn test_ai_framework_vector_store_adapter_integration() {
    use hnsqr::ecosystem::{
        FrameworkDocument, HNSQRVectorStore, LangChainAdapter, LlamaIndexAdapter,
    };
    use hnsqr::{HNSQRConfig, HNSQRIndex, VectorEmbedding};
    use num_complex::Complex32;
    use std::sync::Arc;

    let index = Arc::new(HNSQRIndex::new(HNSQRConfig::default(), 4));
    let v0 = VectorEmbedding::from_complex(vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
    ])
    .into_normalized();

    let doc0 = FrameworkDocument {
        id: "doc_alpha".to_string(),
        text: "Quantum and Graph DB".to_string(),
        metadata: std::collections::HashMap::new(),
        score: None,
    };

    let inserted = index.add_documents(vec![doc0], vec![v0.clone()]).unwrap();
    assert_eq!(inserted, 1);

    let langchain = LangChainAdapter::new(index.clone());
    let results = langchain.similarity_search(&v0, 1).unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "doc_alpha");

    let llamaindex = LlamaIndexAdapter::new(index);
    let l_results = llamaindex.query(&v0, 1).unwrap();
    assert_eq!(l_results.len(), 1);
    assert_eq!(l_results[0].id, "doc_alpha");
}

#[tokio::test]
async fn test_distributed_coordinator_dr_replication_hook() {
    use hnsqr::VectorEmbedding;
    use hnsqr::cluster::coordinator::DistributedCoordinator;
    use hnsqr::cluster::disaster_recovery::DisasterRecoveryCoordinator;
    use num_complex::Complex32;
    use std::sync::Arc;

    let coordinator = DistributedCoordinator::new(4, 2, 100);
    let dr = Arc::new(DisasterRecoveryCoordinator::new("us-east-1", "us-west-2"));
    coordinator.set_dr_coordinator(dr.clone());

    let v = VectorEmbedding::from_complex(vec![
        Complex32::new(1.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
        Complex32::new(0.0, 0.0),
    ])
    .into_normalized();

    let receipt = coordinator.insert_fenced("node_1", v, None).await.unwrap();
    assert!(receipt.applied_index >= 1);

    let sla = dr.compute_dr_sla();
    assert_eq!(sla.primary_lsn, receipt.applied_index);
    assert_eq!(sla.primary_region, "us-east-1");
    assert_eq!(sla.secondary_region, "us-west-2");
}

#[test]
fn test_kubernetes_operator_self_healing_plans() {
    use hnsqr::kubernetes::operator::{
        HNSQRClusterSpec, KubernetesOperator, OperatorLifecyclePhase,
    };

    let spec = HNSQRClusterSpec::default();

    let (phase_cert, actions_cert) = KubernetesOperator::plan_tls_cert_rotation(&spec);
    assert_eq!(phase_cert, OperatorLifecyclePhase::RotatingCertificates);
    assert!(!actions_cert.is_empty());

    let (phase_disk, actions_disk) =
        KubernetesOperator::plan_degraded_disk_replacement("prod", "hnsqr-0");
    assert_eq!(phase_disk, OperatorLifecyclePhase::ReplacingDegradedDisk);
    assert!(!actions_disk.is_empty());

    let (phase_node, actions_node) =
        KubernetesOperator::plan_failed_node_replacement("prod", "node-ip-10-0-1-5");
    assert_eq!(phase_node, OperatorLifecyclePhase::ReplacingFailedNode);
    assert!(!actions_node.is_empty());
}

#[test]
fn test_gpu_tensor_accelerator_complex_gemm() {
    use hnsqr::VectorEmbedding;
    use hnsqr::vector::gpu_tensor::{GpuDeviceConfig, GpuTensorAccelerator};
    use num_complex::Complex32;

    let accelerator = GpuTensorAccelerator::new(GpuDeviceConfig::default());

    let v1 =
        VectorEmbedding::from_complex(vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)])
            .into_normalized();

    let v2 =
        VectorEmbedding::from_complex(vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, -1.0)])
            .into_normalized();

    let gemm_out = accelerator.batched_complex_gemm(&[v1.clone()], &[v1.clone(), v2.clone()]);
    assert_eq!(gemm_out.len(), 1);
    assert_eq!(gemm_out[0].len(), 2);
    // <v1, v1> is self-inner-product (1.0)
    assert!((gemm_out[0][0] - 1.0).abs() < 1e-4);
}

#[test]
fn test_resp_server_graph_and_search_commands() {
    use hnsqr::ecosystem::MemoryKvStore;
    use hnsqr::graph::mutation::GraphMutationApplier;
    use hnsqr::graph::storage::generation::GraphGeneration;
    use hnsqr::transport::resp::{RespFrame, RespServer};
    use hnsqr::{LabelCatalog, RelTypeCatalog};
    use parking_lot::RwLock;
    use std::sync::Arc;

    let kv = Arc::new(MemoryKvStore::new());
    let index = Arc::new(hnsqr::HNSQRIndex::new(hnsqr::HNSQRConfig::default(), 32));
    let graph_gen = Arc::new(RwLock::new(GraphGeneration::new_mutable(1)));
    let label_cat = Arc::new(LabelCatalog::default());
    let rel_cat = Arc::new(RelTypeCatalog::default());
    let graph = Arc::new(GraphMutationApplier::new(graph_gen, label_cat, rel_cat));
    let server = RespServer::with_index_and_graph(kv, Some(index), Some(graph));

    let graph_res = server.handle_command(&[
        "GRAPH.QUERY".to_string(),
        "social_graph".to_string(),
        "MATCH (a)->(b) RETURN b".to_string(),
    ]);
    assert!(matches!(graph_res, RespFrame::Array(_)));

    let search_res = server.handle_command(&[
        "FT.SEARCH".to_string(),
        "idx_docs".to_string(),
        "quantum retrieval".to_string(),
    ]);
    assert!(matches!(search_res, RespFrame::Array(_)));
}

#[test]
fn test_security_audit_siem_export() {
    use hnsqr::security::audit::{AuditAction, AuditLogger};
    use hnsqr::security::siem::SiemFormat;

    let logger = AuditLogger::new();
    let _ = logger
        .append(
            "admin@hnsqr.io",
            AuditAction::CertificateRotation {
                cert_id: "cert_2026_01".to_string(),
            },
        )
        .unwrap();

    let syslog_events = logger.export_siem_events(SiemFormat::Rfc5424Syslog);
    assert_eq!(syslog_events.len(), 1);
    assert!(syslog_events[0].contains("hnsqr-engine"));
    assert!(syslog_events[0].contains("admin@hnsqr.io"));

    let json_events = logger.export_siem_events(SiemFormat::StructuredJson);
    assert_eq!(json_events.len(), 1);
    assert!(json_events[0].contains("cert_2026_01"));
}
