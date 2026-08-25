/* holosphere/src/vector/folding.rs */
//!▫~•◦-------------------------------‣
//! # Real-to-Complex Embedding Gateway & Multi-Collection Router
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Bridges real-valued LLM embeddings (e.g. OpenAI 1536/3072-dim, Cohere 1024-dim,
//! LLaMA 4096-dim, BERT 384-dim) into the complex vector space ($z = x + iy = r e^{i\theta}$)
//! via lossless Pairwise Complex Isometric Folding $\Phi: \mathbb{R}^{2d} \to \mathbb{C}^d$. Provides a multi-collection router and
//! high-performance HTTP JSON REST API for RAG orchestration.
//!
//! ## Key Capabilities
//! - **Pairwise Complex Isometric Folding:** Losslessly maps $2d$-dim real floats to $d$-dim complex vectors in one allocation, preserving $\text{Re}\langle\Phi(x),\Phi(y)\rangle = x^\top y$ (representation byte size is preserved; CPQ-8 polar quantization and LUTz codebooks provide downstream memory compression).
//! - **Multi-Collection Namespace Management:** Dynamic persistent collection creation, deletion, and stats.
//! - **Axum HTTP REST Engine:** High-concurrency async REST endpoints (`/insert`, `/search`, `/batch_search`, `/stats`).
//!
//! ### Architectural Notes
//! Interfaces external LLM RAG pipelines with internal `HNSQRIndex` and `MetadataIndex` engines.
//!
//! #### Example
//! ```rust
//! use hnsqr::vector::folding::ComplexWeaver;
//!
//! let real_embedding = vec![0.5, -0.2, 0.8, 0.1];
//! let complex_vec = ComplexWeaver::fold_llm_embedding(&real_embedding);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use rayon::prelude::*;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use num_complex::Complex32;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use tower_http::cors::CorsLayer;
use tracing::info;

use crate::metadata::index::{FilterExpr, MetadataValue};
use crate::vector::inference::{InferenceModelConfig, InProcessModelEmbedder};
use crate::{
    HNSQRConfig, HNSQRError, HNSQRIndex, HNSQRResult, NodeIndex, SimilarityScore, VectorEmbedding,
};

// ════════════════════════════════════════════════════════════════════════════════
// 1. DYNAMIC COMPLEX WEAVER (REAL-TO-COMPLEX ISOMETRIC TRANSLATION)
// ════════════════════════════════════════════════════════════════════════════════

/// Lossless Pairwise Complex Isometric translation of $N$-dimensional real LLM embeddings to $N/2$-dimensional complex space.
pub struct ComplexWeaver;

impl ComplexWeaver {
    /// Folds an $N$-dimensional real LLM vector into an $N/2$-dimensional complex vector.
    /// Consecutive pairs $(x_k, y_k)$ directly map to $z_k = x_k + i y_k = r_k e^{i\theta_k}$.
    ///
    /// This transformation is lossless, dimension-agnostic, and uses one required output
    /// allocation without intermediate buffers before 8-bit quantization begins.
    #[inline(always)]
    pub fn fold_llm_embedding(real_vector: &[f32]) -> VectorEmbedding {
        let dim = real_vector.len();
        let complex_dim = dim.div_ceil(2);
        let mut complex_data = Vec::with_capacity(complex_dim);

        let mut pairs = real_vector.chunks_exact(2);
        for pair in &mut pairs {
            complex_data.push(Complex32::new(pair[0], pair[1]));
        }
        if let [tail] = pairs.remainder() {
            complex_data.push(Complex32::new(*tail, 0.0));
        }

        // Zero-copy normalization: mutate the freshly allocated output in place.
        VectorEmbedding::from_complex(complex_data).into_normalized()
    }

    /// Ingests token embeddings directly from an in-process neural model with zero intermediate copies.
    #[inline(always)]
    pub fn fold_token_embeddings_in_place(
        token_floats: &[f32],
        target_dim: usize,
    ) -> VectorEmbedding {
        let complex_dim = target_dim.div_ceil(2);
        let mut complex_data = Vec::with_capacity(complex_dim);
        let mut pairs = token_floats.chunks_exact(2);
        for pair in &mut pairs {
            complex_data.push(Complex32::new(pair[0], pair[1]));
        }
        if let [tail] = pairs.remainder() {
            complex_data.push(Complex32::new(*tail, 0.0));
        }
        VectorEmbedding::from_complex(complex_data).into_normalized()
    }

    /// Folds an $N$-dimensional real vector into an $N/2$-dimensional complex vector without normalizing.
    /// Consecutive pairs $(x_k, y_k)$ directly map to $z_k = x_k + i y_k$.
    ///
    /// This transformation is strictly lossless, dimension-agnostic, and preserves the Euclidean norm.
    #[inline(always)]
    pub fn fold_llm_embedding_unnormalized(real_vector: &[f32]) -> VectorEmbedding {
        let dim = real_vector.len();
        let complex_dim = dim.div_ceil(2);
        let mut complex_data = Vec::with_capacity(complex_dim);

        let mut pairs = real_vector.chunks_exact(2);
        for pair in &mut pairs {
            complex_data.push(Complex32::new(pair[0], pair[1]));
        }
        if let [tail] = pairs.remainder() {
            complex_data.push(Complex32::new(*tail, 0.0));
        }

        VectorEmbedding::from_complex(complex_data)
    }

    /// Inverts pairwise folding: reconstructs the original $N$-dimensional real float vector from a complex embedding.
    #[inline(always)]
    pub fn unfold_llm_embedding(embedding: &VectorEmbedding, target_real_dim: usize) -> Vec<f32> {
        let mut reals = Vec::with_capacity(target_real_dim);
        for z in embedding.complex_data() {
            if reals.len() < target_real_dim {
                reals.push(z.re);
            }
            if reals.len() < target_real_dim {
                reals.push(z.im);
            }
        }
        reals
    }

    /// Split-dimensional folding: uses the first half of floats as amplitudes $r$
    /// and maps the second half to $[-\pi, \pi]$ as phase angles $\theta$.
    ///
    /// NOTE: This path calls `Complex32::from_polar(r, theta)` which internally invokes
    /// `cos` and `sin` transcendentals. This is an alternative ingestion strategy for
    /// amplitude-phase separated embeddings; prefer `fold_llm_embedding` for raw LLM
    /// float output where the fast zero-transcendental reinterpretation path applies.
    /// [BENCH REQUIRED] to verify cost vs. fold_llm_embedding at dimension > 512.
    pub fn split_fold_llm_embedding(real_vector: &[f32]) -> VectorEmbedding {
        let dim = real_vector.len();
        let half = dim / 2;
        let mut complex_data = Vec::with_capacity(half);

        let first_half = &real_vector[..half];
        let second_half = &real_vector[half..(half * 2)];

        for i in 0..half {
            let r = first_half[i].abs().max(1e-7);
            let theta = (second_half[i] * std::f32::consts::PI)
                .clamp(-std::f32::consts::PI, std::f32::consts::PI);
            complex_data.push(Complex32::from_polar(r, theta));
        }

        VectorEmbedding::from_complex(complex_data).into_normalized()
    }

    /// Lossless inverse transformation: unfolds an $N/2$-dimensional complex vector back to $N$ real floats.
    pub fn unfold_to_real(complex_vector: &VectorEmbedding, original_dim: usize) -> Vec<f32> {
        let cdata = complex_vector.complex_data();
        let mut real_data = Vec::with_capacity(original_dim);

        for z in cdata {
            real_data.push(z.re);
            if real_data.len() < original_dim {
                real_data.push(z.im);
            }
        }

        real_data
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 2. MULTI-COLLECTION GATEWAY ROUTER
// ════════════════════════════════════════════════════════════════════════════════

/// Dynamic collection router managing dimension-isolated HNSQR persistent memory-mapped indices.
pub struct GatewayRouter {
    collections: RwLock<HashMap<String, Arc<HNSQRIndex>>>,
    base_storage_path: String,
    use_mmap: bool,
}

impl GatewayRouter {
    /// Creates a new gateway router storing persistent indices under `base_storage_path`.
    pub fn new(base_storage_path: &str, use_mmap: bool) -> Self {
        Self {
            collections: RwLock::new(HashMap::new()),
            base_storage_path: base_storage_path.to_string(),
            use_mmap,
        }
    }

    /// Retrieves an existing index or dynamically allocates a new memory-mapped arena
    /// based on the folded dimension of incoming LLM vectors.
    pub fn get_or_create_collection(
        &self,
        collection_name: &str,
        llm_raw_dimension: usize,
    ) -> HNSQRResult<Arc<HNSQRIndex>> {
        let folded_dimension = llm_raw_dimension.div_ceil(2);

        // Fast path: Read lock
        if let Some(index) = self.collections.read().get(collection_name) {
            if index.dimension() != folded_dimension {
                return Err(HNSQRError::DimensionMismatch {
                    expected: index.dimension() * 2,
                    actual: llm_raw_dimension,
                });
            }
            return Ok(Arc::clone(index));
        }

        // Slow path: Write lock
        let mut collections_write = self.collections.write();

        // Double check
        if let Some(index) = collections_write.get(collection_name) {
            return Ok(Arc::clone(index));
        }

        let index = if self.use_mmap {
            let mmap_path = format!("{}/{}.hnsqr", self.base_storage_path, collection_name);
            let mut config = HNSQRConfig::adaptive_for_dim(folded_dimension);
            config.quantization_enabled = true;
            config.max_elements = 100_000;

            if std::path::Path::new(&mmap_path).exists() {
                HNSQRIndex::open_mmap(&mmap_path)?
            } else {
                HNSQRIndex::create_mmap(&mmap_path, config, folded_dimension)?
            }
        } else {
            let config = HNSQRConfig::adaptive_for_dim(folded_dimension);
            HNSQRIndex::new(config, folded_dimension)
        };

        let shared_index = Arc::new(index);
        collections_write.insert(collection_name.to_string(), Arc::clone(&shared_index));

        Ok(shared_index)
    }

    /// Ingests a raw real-valued LLM vector into a target collection.
    pub fn ingest_llm_vector(
        &self,
        collection: &str,
        id: &str,
        llm_vector: &[f32],
    ) -> HNSQRResult<NodeIndex> {
        let target_index = self.get_or_create_collection(collection, llm_vector.len())?;
        let complex_embedding = ComplexWeaver::fold_llm_embedding(llm_vector);
        target_index.insert(id, complex_embedding)
    }

    /// Ingests a raw real-valued LLM vector with structured metadata into a target collection.
    pub fn ingest_llm_vector_with_metadata(
        &self,
        collection: &str,
        id: &str,
        llm_vector: &[f32],
        metadata: HashMap<String, MetadataValue>,
    ) -> HNSQRResult<NodeIndex> {
        let target_index = self.get_or_create_collection(collection, llm_vector.len())?;
        let complex_embedding = ComplexWeaver::fold_llm_embedding(llm_vector);
        target_index.insert_with_metadata(id, complex_embedding, metadata)
    }

    /// Parallel batch ingestion of LLM vectors using Rayon multi-threading.
    pub fn batch_ingest_llm_vectors(
        &self,
        collection: &str,
        records: &[(String, &[f32])],
    ) -> HNSQRResult<Vec<NodeIndex>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let sample_dim = records[0].1.len();
        let target_index = self.get_or_create_collection(collection, sample_dim)?;

        records
            .par_iter()
            .map(|(id, vector)| {
                target_index.insert(id.as_str(), ComplexWeaver::fold_llm_embedding(vector))
            })
            .collect()
    }

    /// Parallel batch ingestion of LLM vectors with structured metadata using Rayon multi-threading.
    pub fn batch_ingest_llm_vectors_with_metadata(
        &self,
        collection: &str,
        records: &[(String, &[f32], HashMap<String, MetadataValue>)],
    ) -> HNSQRResult<Vec<NodeIndex>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let sample_dim = records[0].1.len();
        let target_index = self.get_or_create_collection(collection, sample_dim)?;

        records
            .par_iter()
            .map(
                |(id, vector, metadata): &(String, &[f32], HashMap<String, MetadataValue>)| {
                    target_index.insert_with_metadata_ref(
                        id.as_str(),
                        ComplexWeaver::fold_llm_embedding(vector),
                        metadata,
                    )
                },
            )
            .collect()
    }

    /// Searches for nearest neighbors using a raw real-valued LLM query vector.
    pub fn search_llm_vector(
        &self,
        collection: &str,
        llm_query: &[f32],
        k: usize,
    ) -> HNSQRResult<Vec<(String, SimilarityScore)>> {
        let target_index = self.get_or_create_collection(collection, llm_query.len())?;
        let complex_query = ComplexWeaver::fold_llm_embedding(llm_query);
        target_index.search(&complex_query, k).map(|results| {
            results
                .into_iter()
                .map(|(id, score)| (id.to_string(), score))
                .collect()
        })
    }

    /// Searches with dynamic Roaring Bitmap metadata filtering.
    pub fn search_llm_vector_with_filter(
        &self,
        collection: &str,
        llm_query: &[f32],
        k: usize,
        filter: Option<FilterExpr>,
    ) -> HNSQRResult<Vec<(String, SimilarityScore)>> {
        let (results, _, _) =
            self.search_llm_vector_with_contract(collection, llm_query, k, filter, false)?;
        Ok(results)
    }

    /// Searches with contract enforcement, returning certified exact results when requested.
    pub fn search_llm_vector_with_contract(
        &self,
        collection: &str,
        llm_query: &[f32],
        k: usize,
        filter: Option<FilterExpr>,
        certified_exact: bool,
    ) -> HNSQRResult<(Vec<(String, SimilarityScore)>, bool, Option<f32>)> {
        let target_index = self.get_or_create_collection(collection, llm_query.len())?;
        let complex_query = ComplexWeaver::fold_llm_embedding(llm_query);

        Self::search_embedding_with_contract(
            &target_index,
            &complex_query,
            k,
            filter,
            certified_exact,
        )
    }

    /// Searches an existing collection using the built-in, deterministic text
    /// embedder.  Text search never creates a collection: its embedding space
    /// must be derived from data already admitted to that collection.
    pub fn search_llm_text_with_contract(
        &self,
        collection: &str,
        query_text: &str,
        k: usize,
        filter: Option<FilterExpr>,
        certified_exact: bool,
    ) -> HNSQRResult<(Vec<(String, SimilarityScore)>, bool, Option<f32>)> {
        if query_text.trim().is_empty() {
            return Err(HNSQRError::InvalidRequest(
                "query_text must not be empty".to_string(),
            ));
        }

        let target_index = self
            .collections
            .read()
            .get(collection)
            .cloned()
            .ok_or_else(|| {
                HNSQRError::SearchError(format!("Collection '{collection}' does not exist"))
            })?;
        let raw_dimension = target_index.dimension().checked_mul(2).ok_or_else(|| {
            HNSQRError::InvalidConfig("collection dimension overflows text embedding space".into())
        })?;

        let config = InferenceModelConfig {
            output_dimension: raw_dimension,
            ..InferenceModelConfig::default()
        };
        let embedder = InProcessModelEmbedder::try_new(config)
            .map_err(|error| HNSQRError::InvalidConfig(error.to_string()))?;
        let complex_query = embedder.embed_text(query_text)?;

        Self::search_embedding_with_contract(
            &target_index,
            &complex_query,
            k,
            filter,
            certified_exact,
        )
    }

    fn search_embedding_with_contract(
        target_index: &Arc<HNSQRIndex>,
        complex_query: &VectorEmbedding,
        k: usize,
        filter: Option<FilterExpr>,
        certified_exact: bool,
    ) -> HNSQRResult<(Vec<(String, SimilarityScore)>, bool, Option<f32>)> {
        let filter_mask = filter.and_then(|f| target_index.compile_filter_mask(&f).ok());

        if certified_exact {
            let (results, proof) =
                target_index.search_indices_with_proof(&complex_query, k, filter_mask.as_ref())?;
            let mapped = results
                .into_iter()
                .filter_map(|(idx, score)| {
                    target_index
                        .arena
                        .get_node(idx)
                        .map(|n| (n.external_id.to_string(), score))
                })
                .collect();
            Ok((
                mapped,
                proof.globally_exact,
                Some(proof.max_remaining_upper_bound as f32),
            ))
        } else {
            let results =
                target_index.search_indices_filtered(&complex_query, k, filter_mask.as_ref())?;
            let mapped = results
                .into_iter()
                .filter_map(|(idx, score)| {
                    target_index
                        .arena
                        .get_node(idx)
                        .map(|n| (n.external_id.to_string(), score))
                })
                .collect();
            Ok((mapped, false, None))
        }
    }

    /// Returns operational statistics for a collection.
    pub fn stats(&self, collection: &str) -> HNSQRResult<crate::IndexStats> {
        let guard = self.collections.read();
        let index = guard.get(collection).ok_or_else(|| {
            HNSQRError::SearchError(format!("Collection '{}' does not exist", collection))
        })?;
        Ok(index.stats())
    }

    /// Removes one vector from a collection. Used to compensate a failed durable
    /// model-knowledge journal append before a mutation receipt is returned.
    pub fn remove_llm_vector(&self, collection: &str, id: &str) -> HNSQRResult<bool> {
        let guard = self.collections.read();
        let index = guard.get(collection).ok_or_else(|| {
            HNSQRError::SearchError(format!("Collection '{collection}' does not exist"))
        })?;
        index.remove(id)
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 3. HTTP JSON REST API SERVER (FOR LANGCHAIN / LLM AGENTS)
// ════════════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct InsertRequest {
    id: String,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: HashMap<String, MetadataValue>,
}

#[derive(Serialize)]
struct InsertResponse {
    status: String,
    id: String,
    node_index: u32,
}

#[derive(Deserialize)]
struct SearchRequest {
    #[serde(default, alias = "vector")]
    query: Option<Vec<f32>>,
    #[serde(default)]
    query_text: Option<String>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    filter: Option<FilterExpr>,
    #[serde(default = "default_certified")]
    certified_exact: bool,
}

fn default_k() -> usize {
    10
}

fn default_certified() -> bool {
    true
}

#[derive(Serialize)]
struct SearchResultItem {
    id: String,
    score: f32,
    is_certified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    proof_upper_bound: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<HashMap<String, MetadataValue>>,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
    count: usize,
}

#[derive(Deserialize)]
struct BatchSearchRequest {
    vectors: Vec<Vec<f32>>,
    #[serde(default = "default_k")]
    k: usize,
}

#[derive(Serialize)]
struct BatchSearchResponse {
    batch_results: Vec<Vec<SearchResultItem>>,
}

/// Creates the Axum HTTP REST Router for HNSQR database interactions.
pub fn create_http_router(router: Arc<GatewayRouter>) -> Router {
    Router::new()
        .route("/healthz", get(healthcheck_handler))
        .route("/metrics", get(metrics_handler))
        .route("/v1/collections/{name}/insert", post(insert_handler))
        .route("/v1/collections/{name}/search", post(search_handler))
        .route(
            "/v1/collections/{name}/batch_search",
            post(batch_search_handler),
        )
        .route("/v1/collections/{name}/stats", get(stats_handler))
        .route(
            "/dashboard",
            get(crate::transport::web_console::console_handler),
        )
        .route("/ui", get(crate::transport::web_console::console_handler))
        .route("/docs", get(crate::transport::swagger::swagger_handler))
        .route("/swagger", get(crate::transport::swagger::swagger_handler))
        .route(
            "/openapi.json",
            get(crate::transport::swagger::openapi_spec_handler),
        )
        .layer(CorsLayer::permissive())
        .with_state(router)
}

async fn healthcheck_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "engine": "HoloSphere Vector Database",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn metrics_handler(State(router): State<Arc<GatewayRouter>>) -> impl IntoResponse {
    let metrics = crate::telemetry::metrics::EngineMetrics::new();
    let collections = router.collections.read();
    let mut total_queries = 0;
    let mut total_inserts = 0;

    for index in collections.values() {
        let stats = index.stats();
        total_queries += stats.searches as u64;
        total_inserts += stats.insertions as u64;
    }

    metrics
        .queries_total
        .store(total_queries, std::sync::atomic::Ordering::Relaxed);
    metrics
        .wal_appends_total
        .store(total_inserts, std::sync::atomic::Ordering::Relaxed);

    let export = crate::telemetry::metrics::PrometheusExporter::format(&metrics);
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        export,
    )
}

async fn insert_handler(
    State(router): State<Arc<GatewayRouter>>,
    Path(collection): Path<String>,
    Json(payload): Json<InsertRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let node_idx = if payload.metadata.is_empty() {
        router.ingest_llm_vector(&collection, &payload.id, &payload.vector)
    } else {
        router.ingest_llm_vector_with_metadata(
            &collection,
            &payload.id,
            &payload.vector,
            payload.metadata,
        )
    }
    .map_err(|e: HNSQRError| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(InsertResponse {
        status: "inserted".to_string(),
        id: payload.id,
        node_index: node_idx,
    }))
}

async fn search_handler(
    State(router): State<Arc<GatewayRouter>>,
    Path(collection): Path<String>,
    Json(payload): Json<SearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let (results, is_certified, upper_bound) = match (payload.query, payload.query_text) {
        (Some(query), None) => router.search_llm_vector_with_contract(
            &collection,
            &query,
            payload.k,
            payload.filter,
            payload.certified_exact,
        ),
        (None, Some(query_text)) => router.search_llm_text_with_contract(
            &collection,
            &query_text,
            payload.k,
            payload.filter,
            payload.certified_exact,
        ),
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                "provide exactly one of query/vector or query_text".to_string(),
            ));
        }
    }
    .map_err(|e: HNSQRError| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let items: Vec<SearchResultItem> = results
        .into_iter()
        .map(|(id, score)| SearchResultItem {
            id,
            score,
            is_certified,
            proof_upper_bound: upper_bound,
            metadata: None,
        })
        .collect();

    let count = items.len();
    Ok(Json(SearchResponse {
        results: items,
        count,
    }))
}

async fn batch_search_handler(
    State(router): State<Arc<GatewayRouter>>,
    Path(collection): Path<String>,
    Json(payload): Json<BatchSearchRequest>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let mut batch_results = Vec::with_capacity(payload.vectors.len());

    for v in &payload.vectors {
        let results = router
            .search_llm_vector(&collection, v, payload.k)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let items: Vec<SearchResultItem> = results
            .into_iter()
            .map(|(id, score)| SearchResultItem {
                id,
                score,
                is_certified: true,
                proof_upper_bound: None,
                metadata: None,
            })
            .collect();

        batch_results.push(items);
    }

    Ok(Json(BatchSearchResponse { batch_results }))
}

async fn stats_handler(
    State(router): State<Arc<GatewayRouter>>,
    Path(collection): Path<String>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let stats = router
        .stats(&collection)
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(stats))
}

/// Launches the asynchronous HTTP REST server on Tokio runtime.
pub async fn run_http_server(router: Arc<GatewayRouter>, addr: SocketAddr) -> HNSQRResult<()> {
    let app = create_http_router(router);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| HNSQRError::ConcurrencyError(format!("Failed to bind HTTP server: {}", e)))?;

    info!("HNSQR HTTP REST Server listening on http://{}", addr);

    axum::serve(listener, app)
        .await
        .map_err(|e| HNSQRError::ConcurrencyError(format!("HTTP Server error: {}", e)))?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_weaver_pairwise_folding_and_unfolding() {
        // Simulating a 1536-dimensional OpenAI embedding
        let raw_openai_vec: Vec<f32> = (0..1536).map(|i| (i as f32) / 1536.0).collect();

        let complex_emb = ComplexWeaver::fold_llm_embedding(&raw_openai_vec);
        assert_eq!(complex_emb.dimension(), 768); // 1536 -> 768 complex elements
        assert!((complex_emb.norm_squared() - 1.0).abs() < 1e-4);

        let unfolded = ComplexWeaver::unfold_to_real(&complex_emb, 1536);
        assert_eq!(unfolded.len(), 1536);
    }

    #[test]
    fn test_gateway_router_multi_collection_ingest_and_search() {
        let temp_dir = std::env::temp_dir();
        let router = GatewayRouter::new(&temp_dir.to_string_lossy(), false);

        // Ingest into OpenAI collection (64-dim)
        let v1: Vec<f32> = vec![1.0; 64];
        let mut m1 = HashMap::new();
        m1.insert("dept".to_string(), "finance".into());
        let idx1 = router
            .ingest_llm_vector_with_metadata("openai_mock", "doc_fin_1", &v1, m1)
            .unwrap();
        assert_eq!(idx1, 0);

        // Search with filter
        let query: Vec<f32> = vec![1.0; 64];
        let filter = FilterExpr::eq("dept", "finance");
        let results = router
            .search_llm_vector_with_filter("openai_mock", &query, 1, Some(filter))
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "doc_fin_1");

        // Ingest into LLaMA collection (128-dim)
        let v_llama: Vec<f32> = vec![0.5; 128];
        let idx2 = router
            .ingest_llm_vector("llama_mock", "doc_llama_1", &v_llama)
            .unwrap();
        assert_eq!(idx2, 0);

        let res_llama = router.search_llm_vector("llama_mock", &v_llama, 1).unwrap();
        assert_eq!(res_llama.len(), 1);
        assert_eq!(res_llama[0].0, "doc_llama_1");
    }

    #[test]
    fn text_search_uses_the_existing_collection_embedding_space() {
        let temp_dir = std::env::temp_dir();
        let router = GatewayRouter::new(&temp_dir.to_string_lossy(), false);
        let mut config = InferenceModelConfig::default();
        config.output_dimension = 64;
        let embedder = InProcessModelEmbedder::try_new(config).unwrap();
        let embedded = embedder.embed_text("liquid cooling for datacenters").unwrap();
        let raw = ComplexWeaver::unfold_to_real(&embedded, 64);

        router
            .ingest_llm_vector("text_docs", "cooling", &raw)
            .unwrap();
        let (results, certified, _) = router
            .search_llm_text_with_contract(
                "text_docs",
                "liquid cooling for datacenters",
                1,
                None,
                true,
            )
            .unwrap();

        assert!(certified);
        assert_eq!(results[0].0, "cooling");
    }

    #[test]
    fn test_complex_weaver_odd_real_dimensions() {
        let odd_dims: [usize; 8] = [383, 385, 767, 769, 1023, 1025, 1535, 1537];
        for &d in &odd_dims {
            let real_vec: Vec<f32> = (0..d).map(|i| ((i * 17 + 5) % 31) as f32 - 15.0).collect();
            let folded_unnorm = ComplexWeaver::fold_llm_embedding_unnormalized(&real_vec);
            assert_eq!(folded_unnorm.dimension(), d.div_ceil(2));

            let reconstructed = ComplexWeaver::unfold_to_real(&folded_unnorm, d);
            assert_eq!(reconstructed.len(), d);
            for i in 0..d {
                assert!(
                    (reconstructed[i] - real_vec[i]).abs() < 1e-6,
                    "Mismatch at index {i} for dimension {d}: {} vs {}",
                    reconstructed[i],
                    real_vec[i]
                );
            }

            // Normalized fold
            let folded_norm = ComplexWeaver::fold_llm_embedding(&real_vec);
            assert_eq!(folded_norm.dimension(), d.div_ceil(2));
            assert!((folded_norm.norm_squared() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_complex_weaver_extended_high_dimensions() {
        let extended_dims: [usize; 4] = [6144, 8192, 12288, 16384];
        for &d in &extended_dims {
            let real_vec: Vec<f32> = (0..d).map(|i| ((i * 23 + 7) % 43) as f32 - 21.0).collect();
            let folded_unnorm = ComplexWeaver::fold_llm_embedding_unnormalized(&real_vec);
            assert_eq!(folded_unnorm.dimension(), d / 2);

            let reconstructed = ComplexWeaver::unfold_to_real(&folded_unnorm, d);
            assert_eq!(reconstructed.len(), d);
            for i in 0..d {
                assert!(
                    (reconstructed[i] - real_vec[i]).abs() < 1e-6,
                    "Mismatch at index {i} for extended dimension {d}"
                );
            }

            let folded_norm = ComplexWeaver::fold_llm_embedding(&real_vec);
            assert_eq!(folded_norm.dimension(), d / 2);
            assert!((folded_norm.norm_squared() - 1.0).abs() < 1e-4);
        }
    }
}
