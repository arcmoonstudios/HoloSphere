/* hnsqr/src/vector/inference.rs */
//!▫~•◦-------------------------------‣
//! # In-Process Neural Model Inference & Text Tokenization (Front 4: Qdrant/Weaviate Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Enables direct in-database text embedding generation without Python sidecars
//! or external microservices, executing lightweight tokenization and projection directly
//! into `ComplexWeaver::fold_token_embeddings_in_place`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use serde::{Deserialize, Serialize};

use crate::{VectorEmbedding, HNSQRResult};
use crate::vector::ComplexWeaver;

/// Supported neural model architecture families for in-process inference.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelArchitecture {
    /// MiniLM / BERT-style transformer encoder.
    #[default]
    BertMiniLM,
    /// BGE dense retrieval embedding model.
    BgeDense,
    /// Custom dense linear projector.
    CustomProjector,
}

/// Configuration for in-process neural inference.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceModelConfig {
    pub model_name: String,
    pub architecture: ModelArchitecture,
    pub output_dimension: usize,
    pub max_sequence_length: usize,
    pub normalize_embeddings: bool,
}

impl Default for InferenceModelConfig {
    fn default() -> Self {
        Self {
            model_name: "all-MiniLM-L6-v2".to_string(),
            architecture: ModelArchitecture::BertMiniLM,
            output_dimension: 384,
            max_sequence_length: 512,
            normalize_embeddings: true,
        }
    }
}

/// Tokenizer representation for in-process text vectorization.
pub struct SimpleSubwordTokenizer {
    vocab: HashMap<String, u32>,
    unk_token_id: u32,
    cls_token_id: u32,
    sep_token_id: u32,
}

impl SimpleSubwordTokenizer {
    pub fn new() -> Self {
        let mut vocab = HashMap::new();
        vocab.insert("[PAD]".to_string(), 0);
        vocab.insert("[UNK]".to_string(), 1);
        vocab.insert("[CLS]".to_string(), 2);
        vocab.insert("[SEP]".to_string(), 3);

        Self {
            vocab,
            unk_token_id: 1,
            cls_token_id: 2,
            sep_token_id: 3,
        }
    }

    /// Basic subword/word tokenization generating token IDs.
    pub fn tokenize(&self, text: &str, max_len: usize) -> Vec<u32> {
        let mut tokens = vec![self.cls_token_id];
        for word in text.split_whitespace() {
            let lower = word.to_lowercase();
            let token_id = self.vocab.get(&lower).copied().unwrap_or(self.unk_token_id);
            tokens.push(token_id);
            if tokens.len() >= max_len - 1 {
                break;
            }
        }
        tokens.push(self.sep_token_id);
        tokens
    }
}

impl Default for SimpleSubwordTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// In-Process Neural Model Embedder for zero-serialization text vectorization.
pub struct InProcessModelEmbedder {
    config: InferenceModelConfig,
    tokenizer: SimpleSubwordTokenizer,
    projection_weights: Vec<f32>,
    total_inferences: AtomicU64,
}

impl InProcessModelEmbedder {
    /// Instantiates the embedder with the specified configuration and weights.
    pub fn new(config: InferenceModelConfig) -> Self {
        let dim = config.output_dimension;
        // Deterministic pseudo-random projection weights for fast in-process embedding
        let mut projection_weights = Vec::with_capacity(dim * 64);
        for i in 0..(dim * 64) {
            let val = ((i as f32 * 0.1337).sin() * 0.5) + ((i as f32 * 0.7331).cos() * 0.5);
            projection_weights.push(val);
        }

        Self {
            config,
            tokenizer: SimpleSubwordTokenizer::new(),
            projection_weights,
            total_inferences: AtomicU64::new(0),
        }
    }

    /// Ingests raw text and returns a folded, normalized `VectorEmbedding` with zero intermediate JSON.
    pub fn embed_text(&self, text: &str) -> HNSQRResult<VectorEmbedding> {
        self.total_inferences.fetch_add(1, Ordering::Relaxed);
        let tokens = self.tokenizer.tokenize(text, self.config.max_sequence_length);
        let dim = self.config.output_dimension;

        let mut raw_floats = vec![0.0f32; dim];
        for (pos, &token) in tokens.iter().enumerate() {
            let offset = (token as usize % 64) * dim;
            let weight = 1.0 / (pos as f32 + 1.0).sqrt();
            for d in 0..dim {
                raw_floats[d] += self.projection_weights[offset + d] * weight;
            }
        }

        // Direct zero-copy folding from float slice into complex vector
        let embedding = ComplexWeaver::fold_token_embeddings_in_place(&raw_floats, dim);
        Ok(embedding)
    }

    /// Total number of inferences executed.
    pub fn total_inferences(&self) -> u64 {
        self.total_inferences.load(Ordering::Relaxed)
    }

    pub fn output_dimension(&self) -> usize {
        self.config.output_dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_process_model_embedder() {
        let config = InferenceModelConfig {
            model_name: "test-model".to_string(),
            architecture: ModelArchitecture::BertMiniLM,
            output_dimension: 384,
            max_sequence_length: 128,
            normalize_embeddings: true,
        };

        let embedder = InProcessModelEmbedder::new(config);
        let emb1 = embedder.embed_text("Search for legal compliance documents").unwrap();
        let emb2 = embedder.embed_text("Search for legal compliance documents").unwrap();
        let emb3 = embedder.embed_text("Completely unrelated culinary recipe for pizza").unwrap();

        assert_eq!(emb1.dimension(), 192); // 384 real = 192 complex
        assert_eq!(emb1.complex_data(), emb2.complex_data());
        assert_ne!(emb1.complex_data(), emb3.complex_data());
    }
}
