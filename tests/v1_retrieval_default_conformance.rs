/* holosphere/tests/v1_retrieval_default_conformance.rs */
//!▫~•◦-------------------------------‣
//! # Semantic Kernel v1 Retrieval Default Conformance Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates the frozen Semantic Kernel v1 retrieval default contract:
//!   - Default SearchPlan is unconditionally `SearchPlan::Exact`.
//!   - Default RetrievalContract is unconditionally `RetrievalContract::Exact`.
//!   - Default HNSQRConfig has `search_plan: SearchPlan::Exact`.
//!   - Model Gateway / REST API defaults omitted retrieval contract to `exact`.
//!   - Legacy `certified_exact: true` maps explicitly to Certified, while omitted maps to Exact.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use hnsqr::planning::RetrievalContract;
use hnsqr::{
    AccessRole, AuthRegistry, GatewayRouter, HNSQRConfig, ModelGatewayAuth, ModelKnowledgeStore,
    ModelToolService, SearchPlan, create_model_api_router,
};
use http_body_util::BodyExt;
use tower::ServiceExt;

#[test]
fn test_v1_retrieval_default_invariants_rust_core() {
    // 1. SearchPlan::default() must be Exact
    assert_eq!(SearchPlan::default(), SearchPlan::Exact);

    // 2. RetrievalContract::default() must be Exact
    assert_eq!(RetrievalContract::default(), RetrievalContract::Exact);

    // 3. HNSQRConfig::default().search_plan must be SearchPlan::Exact
    let config = HNSQRConfig::default();
    assert_eq!(config.search_plan, SearchPlan::Exact);
}

#[tokio::test]
async fn test_v1_retrieval_default_omitted_contract_in_rest_gateway() {
    let registry = Arc::new(AuthRegistry::new());
    registry.register_key("test-key", "tenant-alpha", AccessRole::ReadWrite, 100);
    let service = Arc::new(ModelToolService::new(
        Arc::new(GatewayRouter::new("test-gateway", false)),
        Arc::new(ModelKnowledgeStore::in_memory()),
        Arc::new(ModelGatewayAuth::new(registry, false)),
    ));
    let app = create_model_api_router(service);

    // Ingest one item
    let remember_body = serde_json::json!({
        "idempotency_key": "k-1",
        "id": "item-1",
        "kind": "fact",
        "content": "quantum phase interference in hyperdimensional space",
        "provenance": [{"source_id": "test", "content_hash": "sha256:abcd"}]
    });
    let rem_req = Request::builder()
        .method("POST")
        .uri("/v1/knowledge/remember")
        .header("content-type", "application/json")
        .header("authorization", "Bearer test-key")
        .body(Body::from(remember_body.to_string()))
        .unwrap();
    let rem_resp = app.clone().oneshot(rem_req).await.unwrap();
    assert_eq!(rem_resp.status(), StatusCode::OK);

    // 1. Omitted retrieval_contract / certified_exact -> must default to "exact"
    let search_default = serde_json::json!({
        "query_text": "quantum phase interference",
        "k": 5
    });
    let search_req = Request::builder()
        .method("POST")
        .uri("/v1/knowledge/search")
        .header("content-type", "application/json")
        .header("authorization", "Bearer test-key")
        .body(Body::from(search_default.to_string()))
        .unwrap();
    let search_resp = app.clone().oneshot(search_req).await.unwrap();
    assert_eq!(search_resp.status(), StatusCode::OK);
    let search_bytes = search_resp.into_body().collect().await.unwrap().to_bytes();
    let search_json: serde_json::Value = serde_json::from_slice(&search_bytes).unwrap();
    assert_eq!(search_json["retrieval_contract"], "exact");
    assert_eq!(search_json["results"][0]["id"], "item-1");

    // 2. Explicit certified_exact: true -> maps to "certified"
    let search_cert = serde_json::json!({
        "query_text": "quantum phase interference",
        "k": 5,
        "certified_exact": true
    });
    let search_cert_req = Request::builder()
        .method("POST")
        .uri("/v1/knowledge/search")
        .header("content-type", "application/json")
        .header("authorization", "Bearer test-key")
        .body(Body::from(search_cert.to_string()))
        .unwrap();
    let search_cert_resp = app.clone().oneshot(search_cert_req).await.unwrap();
    assert_eq!(search_cert_resp.status(), StatusCode::OK);
    let cert_bytes = search_cert_resp.into_body().collect().await.unwrap().to_bytes();
    let cert_json: serde_json::Value = serde_json::from_slice(&cert_bytes).unwrap();
    assert_eq!(cert_json["retrieval_contract"], "certified");

    // 3. Explicit retrieval_contract: "exact" -> maps to "exact"
    let search_explicit_exact = serde_json::json!({
        "query_text": "quantum phase interference",
        "k": 5,
        "retrieval_contract": "exact"
    });
    let search_exact_req = Request::builder()
        .method("POST")
        .uri("/v1/knowledge/search")
        .header("content-type", "application/json")
        .header("authorization", "Bearer test-key")
        .body(Body::from(search_explicit_exact.to_string()))
        .unwrap();
    let search_exact_resp = app.oneshot(search_exact_req).await.unwrap();
    assert_eq!(search_exact_resp.status(), StatusCode::OK);
    let exact_bytes = search_exact_resp.into_body().collect().await.unwrap().to_bytes();
    let exact_json: serde_json::Value = serde_json::from_slice(&exact_bytes).unwrap();
    assert_eq!(exact_json["retrieval_contract"], "exact");
}
