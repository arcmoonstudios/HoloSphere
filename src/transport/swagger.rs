/* hnsqr/src/transport/swagger.rs */
//!▫~•◦-------------------------------‣
//! # OpenAPI 3.1 & Interactive Swagger UI Documentation Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Serves OpenAPI 3.1 JSON schemas and interactive Swagger UI / ReDoc client
//! on `/docs`, `/swagger`, and `/openapi.json`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use axum::response::Html;

/// Embedded interactive Swagger UI HTML.
pub const SWAGGER_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>HoloSphere Interactive API Docs</title>
  <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  <style>
    body { margin: 0; background: #0b0f19; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    .topbar { display: none !important; }
    .swagger-ui .info .title { color: #38bdf8 !important; }
  </style>
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    window.onload = () => {
      window.ui = SwaggerUIBundle({
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        deepLinking: true,
        presets: [
          SwaggerUIBundle.presets.apis,
          SwaggerUIBundle.SwaggerUIStandalonePreset
        ],
        layout: "BaseLayout"
      });
    };
  </script>
</body>
</html>"#;

/// OpenAPI 3.1 JSON Specification generator.
pub struct OpenApiSpecGenerator;

impl OpenApiSpecGenerator {
    /// Generates canonical OpenAPI 3.1 specification for all HoloSphere REST endpoints.
    pub fn generate_spec_json() -> String {
        serde_json::json!({
            "openapi": "3.1.0",
            "info": {
                "title": "HoloSphere Universal Data & Retrieval Engine API",
                "version": "0.1.0",
                "description": "100.000% Certified Proof Search, Native Graph-RAG, Relational SQL ACID, 4D Hypercube Tensor Slicing, and Vectorized OLAP."
            },
            "paths": {
                "/v1/collections/{name}/search": {
                    "post": {
                        "summary": "Execute Certified Dense Vector Search",
                        "parameters": [
                            {
                                "name": "name",
                                "in": "path",
                                "required": true,
                                "schema": { "type": "string" }
                            }
                        ],
                        "requestBody": {
                            "required": true,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "vector": { "type": "array", "items": { "type": "number" } },
                                            "query_text": { "type": "string" },
                                            "k": { "type": "integer", "default": 10 },
                                            "certified_exact": { "type": "boolean", "default": true }
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {
                            "200": { "description": "Exact Top-K Search Results with Proof Upper Bounds" }
                        }
                    }
                },
                "/v1/collections/{name}/insert": {
                    "post": {
                        "summary": "Insert or Upsert Vector Document",
                        "responses": { "200": { "description": "Mutation Receipt with Raft LSN" } }
                    }
                },
                "/v1/graph/query": {
                    "post": {
                        "summary": "Execute Cypher/GQL Graph Query with VECTOR MATCH",
                        "responses": { "200": { "description": "Tabular Graph Results" } }
                    }
                },
                "/v1/sql/execute": {
                    "post": {
                        "summary": "Execute Relational SQL Query with ACID Isolation",
                        "responses": { "200": { "description": "Tabular SQL Rows" } }
                    }
                },
                "/v1/hypercube/slice": {
                    "post": {
                        "summary": "Slice N-Dimensional Hypercube Tensor Subvolume",
                        "responses": { "200": { "description": "Voxel Coordinates and Cell Values" } }
                    }
                }
            }
        }).to_string()
    }
}

/// Handler returning the interactive Swagger UI HTML.
pub async fn swagger_handler() -> Html<&'static str> {
    Html(SWAGGER_HTML)
}

/// Handler returning OpenAPI 3.1 JSON spec.
pub async fn openapi_spec_handler() -> String {
    OpenApiSpecGenerator::generate_spec_json()
}
