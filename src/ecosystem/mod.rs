/* holosphere/src/ecosystem/mod.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR AI Ecosystem & Framework Integration Layer
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides standard adapters and typed protocol bindings for LangChain,
//! LlamaIndex, and Haystack over the unified HNSQR service core.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::metadata::index::MetadataValue;
use crate::{HNSQRResult, SimilarityScore, VectorEmbedding};

pub mod agent_memory;
pub mod kv_cache;
pub mod sdks;

pub use agent_memory::{
    AutonomousMemoryConsolidator, EpisodicFact, FactCategory, UserPersonaProfile,
};
pub use kv_cache::{KvValue, MemoryKvStore};
pub use sdks::{ClientSearchResult, HNSQRClientConfig, HNSQRClientRouter};

/// Universal document abstraction for AI frameworks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameworkDocument {
    pub id: String,
    pub text: String,
    pub metadata: HashMap<String, MetadataValue>,
    pub score: Option<SimilarityScore>,
}

/// Unified AI Vector Store interface implemented across LangChain, LlamaIndex, and Haystack.
pub trait HNSQRVectorStore: Send + Sync {
    /// Ingests a batch of documents with dense embeddings and metadata.
    fn add_documents(
        &self,
        docs: Vec<FrameworkDocument>,
        vectors: Vec<VectorEmbedding>,
    ) -> HNSQRResult<usize>;

    /// Executes similarity search with optional Certified exactness certification.
    fn similarity_search(
        &self,
        query: &VectorEmbedding,
        k: usize,
        certified_exact: bool,
    ) -> HNSQRResult<Vec<FrameworkDocument>>;
}

impl HNSQRVectorStore for crate::HNSQRIndex {
    fn add_documents(
        &self,
        docs: Vec<FrameworkDocument>,
        vectors: Vec<VectorEmbedding>,
    ) -> HNSQRResult<usize> {
        let count = docs.len().min(vectors.len());
        for (doc, vec) in docs.into_iter().zip(vectors).take(count) {
            self.insert_with_metadata(doc.id, vec, doc.metadata)?;
        }
        Ok(count)
    }

    fn similarity_search(
        &self,
        query: &VectorEmbedding,
        k: usize,
        certified_exact: bool,
    ) -> HNSQRResult<Vec<FrameworkDocument>> {
        let contract = if certified_exact {
            crate::planning::planner::RetrievalContract::Certified
        } else {
            crate::planning::planner::RetrievalContract::Exact
        };
        let results = self.search_indices_with_contract(query, k, None, contract)?;
        Ok(results
            .into_iter()
            .map(|(idx, score)| {
                let node = self.get_node_by_index(idx).ok();
                FrameworkDocument {
                    id: node
                        .as_ref()
                        .map_or_else(|| idx.to_string(), |n| n.external_id.to_string()),
                    text: String::new(),
                    metadata: HashMap::new(),
                    score: Some(score),
                }
            })
            .collect())
    }
}

/// LangChain integration adapter.
pub struct LangChainAdapter<S: HNSQRVectorStore> {
    pub store: Arc<S>,
}

impl<S: HNSQRVectorStore> LangChainAdapter<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
    }

    pub fn similarity_search(
        &self,
        query: &VectorEmbedding,
        k: usize,
    ) -> HNSQRResult<Vec<FrameworkDocument>> {
        self.store.similarity_search(query, k, true)
    }
}

/// LlamaIndex integration adapter.
pub struct LlamaIndexAdapter<S: HNSQRVectorStore> {
    pub store: Arc<S>,
}

impl<S: HNSQRVectorStore> LlamaIndexAdapter<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
    }

    pub fn query(
        &self,
        query: &VectorEmbedding,
        similarity_top_k: usize,
    ) -> HNSQRResult<Vec<FrameworkDocument>> {
        self.store.similarity_search(query, similarity_top_k, true)
    }
}

/// Haystack integration adapter.
pub struct HaystackAdapter<S: HNSQRVectorStore> {
    pub store: Arc<S>,
}

impl<S: HNSQRVectorStore> HaystackAdapter<S> {
    pub fn new(store: Arc<S>) -> Self {
        Self { store }
    }

    pub fn run_retrieval(
        &self,
        query: &VectorEmbedding,
        top_k: usize,
    ) -> HNSQRResult<Vec<FrameworkDocument>> {
        self.store.similarity_search(query, top_k, true)
    }
}
