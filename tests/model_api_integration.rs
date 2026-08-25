use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use hnsqr::{
    AccessRole, AuthRegistry, GatewayRouter, ModelGatewayAuth, ModelKnowledgeStore,
    ModelToolService, create_mcp_router, create_model_api_router,
};
use http_body_util::BodyExt;
use tower::ServiceExt;

fn application() -> Router {
    let registry = Arc::new(AuthRegistry::new());
    registry.register_key("read-alpha", "alpha", AccessRole::ReadOnly, 100);
    registry.register_key("write-alpha", "alpha", AccessRole::ReadWrite, 100);
    registry.register_key("read-beta", "beta", AccessRole::ReadOnly, 100);
    let service = Arc::new(ModelToolService::new(
        Arc::new(GatewayRouter::new("unused", false)),
        Arc::new(ModelKnowledgeStore::in_memory()),
        Arc::new(ModelGatewayAuth::new(registry, false)),
    ));
    create_model_api_router(Arc::clone(&service)).merge(create_mcp_router(service))
}

async fn json(response: axum::response::Response) -> serde_json::Value {
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    serde_json::from_slice(&bytes).unwrap()
}

fn post(path: &str, token: Option<&str>, body: serde_json::Value) -> Request<Body> {
    let mut builder = Request::builder()
        .method("POST")
        .uri(path)
        .header("content-type", "application/json");
    if let Some(token) = token {
        builder = builder.header("authorization", format!("Bearer {token}"));
    }
    builder.body(Body::from(body.to_string())).unwrap()
}

#[tokio::test]
async fn mcp_is_fail_closed_and_enforces_write_roles() {
    let app = application();
    let initialize = serde_json::json!({
        "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
    });
    let unauthorized = app
        .clone()
        .oneshot(post("/mcp", None, initialize.clone()))
        .await
        .unwrap();
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);

    let initialized = app
        .clone()
        .oneshot(post("/mcp", Some("read-alpha"), initialize))
        .await
        .unwrap();
    assert_eq!(initialized.status(), StatusCode::OK);
    assert_eq!(
        json(initialized).await["result"]["protocolVersion"],
        "2025-06-18"
    );

    let remember = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "holosphere.remember",
            "arguments": {
                "idempotency_key": "key-1",
                "id": "fact-1",
                "kind": "fact",
                "content": "thermal load is reduced by liquid cooling",
                "provenance": [{"source_id": "test", "content_hash": "sha256:test"}]
            }
        }
    });
    let denied = app
        .clone()
        .oneshot(post("/mcp", Some("read-alpha"), remember.clone()))
        .await
        .unwrap();
    assert_eq!(json(denied).await["error"]["code"], -32001);

    let accepted = app
        .oneshot(post("/mcp", Some("write-alpha"), remember))
        .await
        .unwrap();
    let accepted = json(accepted).await;
    assert_eq!(
        accepted["result"]["structuredContent"]["results"]["id"],
        "fact-1"
    );
}

#[tokio::test]
async fn rest_search_returns_evidence_without_cross_tenant_leakage() {
    let app = application();
    let remember = serde_json::json!({
        "idempotency_key": "key-1",
        "id": "fact-1",
        "kind": "fact",
        "content": "thermal load is reduced by liquid cooling",
        "provenance": [{"source_id": "test", "content_hash": "sha256:test"}]
    });
    let inserted = app
        .clone()
        .oneshot(post(
            "/v1/knowledge/remember",
            Some("write-alpha"),
            remember,
        ))
        .await
        .unwrap();
    assert_eq!(inserted.status(), StatusCode::OK);

    let search = serde_json::json!({"query_text": "thermal cooling", "k": 5});
    let alpha = app
        .clone()
        .oneshot(post(
            "/v1/knowledge/search",
            Some("read-alpha"),
            search.clone(),
        ))
        .await
        .unwrap();
    let alpha = json(alpha).await;
    assert_eq!(alpha["results"][0]["id"], "fact-1");
    assert_eq!(alpha["content_is_untrusted"], true);

    let beta = app
        .oneshot(post("/v1/knowledge/search", Some("read-beta"), search))
        .await
        .unwrap();
    assert!(json(beta).await["results"].as_array().unwrap().is_empty());
}
