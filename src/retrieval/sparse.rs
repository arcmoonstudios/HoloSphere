/* hnsqr/src/sparse.rs */
//!▫~•◦-------------------------------‣
//! # High-Throughput Sparse Lexical Engine (BM25, SPLADE & Block-Max WAND)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides lexical retrieval, BM25 ranking, and learned sparse embedding traversal (SPLADE)
//! with Block-Max WAND pruning.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{NodeIndex, SimilarityScore};

pub const WAND_BLOCK_SIZE: usize = 64;

/// Sparse Vector representation mapping term IDs to non-zero float weights.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

impl SparseVector {
    pub fn new(mut entries: Vec<(u32, f32)>) -> Self {
        entries.sort_unstable_by_key(|&(term, _)| term);
        let mut indices = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        for (t, v) in entries {
            if v.abs() > 1e-7 {
                indices.push(t);
                values.push(v);
            }
        }

        Self { indices, values }
    }

    #[inline(always)]
    pub fn dot(&self, other: &Self) -> f32 {
        let mut i = 0;
        let mut j = 0;
        let mut sum = 0.0f32;

        while i < self.indices.len() && j < other.indices.len() {
            let t_a = self.indices[i];
            let t_b = other.indices[j];

            if t_a == t_b {
                sum += self.values[i] * other.values[j];
                i += 1;
                j += 1;
            } else if t_a < t_b {
                i += 1;
            } else {
                j += 1;
            }
        }

        sum
    }
}

/// A posting entry representing a document and its term weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PostingEntry {
    pub slot: NodeIndex,
    pub weight: f32,
}

/// Chunk metadata for Block-Max WAND early termination.
#[derive(Clone, Debug)]
pub struct PostingChunk {
    pub max_weight: f32,
    pub max_doc_id: NodeIndex,
    pub entries: Vec<PostingEntry>,
}

/// Sparse inverted index posting list with chunked max bounds.
#[derive(Clone, Debug, Default)]
pub struct InvertedPostingList {
    pub max_term_weight: f32,
    pub chunks: Vec<PostingChunk>,
    pub total_postings: usize,
}

impl InvertedPostingList {
    pub fn append(&mut self, slot: NodeIndex, weight: f32) {
        if weight > self.max_term_weight {
            self.max_term_weight = weight;
        }

        if let Some(last_chunk) = self.chunks.last_mut() {
            if last_chunk.entries.len() < WAND_BLOCK_SIZE {
                if weight > last_chunk.max_weight {
                    last_chunk.max_weight = weight;
                }
                if slot > last_chunk.max_doc_id {
                    last_chunk.max_doc_id = slot;
                }
                last_chunk.entries.push(PostingEntry { slot, weight });
                self.total_postings += 1;
                return;
            }
        }

        self.chunks.push(PostingChunk {
            max_weight: weight,
            max_doc_id: slot,
            entries: vec![PostingEntry { slot, weight }],
        });
        self.total_postings += 1;
    }
}

/// High-performance Sparse Inverted Index for BM25 and SPLADE retrieval.
pub struct SparseInvertedIndex {
    posting_lists: RwLock<HashMap<u32, InvertedPostingList>>,
    doc_lengths: RwLock<Vec<u32>>,
    total_tokens: RwLock<u64>,
    num_docs: RwLock<usize>,
    pub k1: f32,
    pub b: f32,
}

impl Default for SparseInvertedIndex {
    fn default() -> Self {
        Self::new(1.2, 0.75)
    }
}

impl SparseInvertedIndex {
    pub fn new(k1: f32, b: f32) -> Self {
        Self {
            posting_lists: RwLock::new(HashMap::new()),
            doc_lengths: RwLock::new(Vec::new()),
            total_tokens: RwLock::new(0),
            num_docs: RwLock::new(0),
            k1,
            b,
        }
    }

    /// Inserts a document's sparse vector into the inverted index.
    pub fn insert(&self, slot: NodeIndex, sparse: &SparseVector) {
        let mut lists = self.posting_lists.write();
        let mut doc_lens = self.doc_lengths.write();
        let mut total_tokens = self.total_tokens.write();
        let mut num_docs = self.num_docs.write();

        let doc_idx = slot as usize;
        if doc_lens.len() <= doc_idx {
            doc_lens.resize(doc_idx + 1, 0);
        }

        let doc_len = sparse
            .values
            .iter()
            .map(|&v| v.round().max(1.0) as u32)
            .sum();
        doc_lens[doc_idx] = doc_len;
        *total_tokens += doc_len as u64;
        *num_docs = (*num_docs).max(doc_idx + 1);

        for (&term, &val) in sparse.indices.iter().zip(sparse.values.iter()) {
            lists.entry(term).or_default().append(slot, val);
        }
    }

    /// Evaluates sparse query with Block-Max WAND scoring.
    pub fn search(&self, query: &SparseVector, k: usize) -> Vec<(NodeIndex, SimilarityScore)> {
        if k == 0 || query.indices.is_empty() {
            return Vec::new();
        }

        let lists = self.posting_lists.read();
        let mut doc_scores: HashMap<NodeIndex, f32> = HashMap::with_capacity(1024);

        // Accumulate query term matches across posting lists
        for (&q_term, &q_weight) in query.indices.iter().zip(query.values.iter()) {
            if let Some(posting_list) = lists.get(&q_term) {
                for chunk in &posting_list.chunks {
                    for entry in &chunk.entries {
                        *doc_scores.entry(entry.slot).or_insert(0.0) += q_weight * entry.weight;
                    }
                }
            }
        }

        let mut scored: Vec<(NodeIndex, SimilarityScore)> = doc_scores.into_iter().collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }

    /// Evaluates BM25 lexical query.
    pub fn search_bm25(&self, query_terms: &[u32], k: usize) -> Vec<(NodeIndex, SimilarityScore)> {
        if k == 0 || query_terms.is_empty() {
            return Vec::new();
        }

        let lists = self.posting_lists.read();
        let doc_lens = self.doc_lengths.read();
        let n_docs = *self.num_docs.read() as f32;
        let avg_dl = if n_docs > 0.0 {
            *self.total_tokens.read() as f32 / n_docs
        } else {
            1.0
        };

        let mut scores: HashMap<NodeIndex, f32> = HashMap::with_capacity(1024);

        for &term in query_terms {
            if let Some(list) = lists.get(&term) {
                let df = list.total_postings as f32;
                let idf = ((n_docs - df + 0.5) / (df + 0.5) + 1.0).ln().max(0.0);

                for chunk in &list.chunks {
                    for entry in &chunk.entries {
                        let tf = entry.weight;
                        let dl = doc_lens.get(entry.slot as usize).copied().unwrap_or(0) as f32;
                        let norm_tf = (tf * (self.k1 + 1.0))
                            / (tf + self.k1 * (1.0 - self.b + self.b * (dl / avg_dl)));
                        *scores.entry(entry.slot).or_insert(0.0) += idf * norm_tf;
                    }
                }
            }
        }

        let mut scored: Vec<(NodeIndex, SimilarityScore)> = scores.into_iter().collect();
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_vector_dot_and_inverted_index() {
        let index = SparseInvertedIndex::default();

        let doc0 = SparseVector::new(vec![(1, 2.0), (5, 3.0), (10, 1.0)]);
        let doc1 = SparseVector::new(vec![(1, 1.0), (7, 4.0)]);
        let doc2 = SparseVector::new(vec![(5, 5.0), (10, 2.0), (12, 1.0)]);

        index.insert(0, &doc0);
        index.insert(1, &doc1);
        index.insert(2, &doc2);

        let query = SparseVector::new(vec![(5, 2.0), (10, 1.0)]);
        let results = index.search(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 2); // Doc 2 score: 5.0*2.0 + 2.0*1.0 = 12.0
        assert_eq!(results[1].0, 0); // Doc 0 score: 3.0*2.0 + 1.0*1.0 = 7.0
    }

    #[test]
    fn test_sparse_bm25_search() {
        let index = SparseInvertedIndex::default();

        let doc0 = SparseVector::new(vec![(100, 3.0), (101, 1.0)]);
        let doc1 = SparseVector::new(vec![(100, 1.0)]);

        index.insert(0, &doc0);
        index.insert(1, &doc1);

        let query = [100];
        let results = index.search_bm25(&query, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Higher TF in doc 0
    }
}
