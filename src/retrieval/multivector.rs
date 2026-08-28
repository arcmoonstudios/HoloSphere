/* holosphere/src/multivector.rs */
//!▫~•◦-------------------------------‣
//! # Multi-Vector & Late-Interaction Engine (ColBERT / ColPali MaxSim)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides multi-vector token sequence storage and hardware-accelerated MaxSim scoring.
//!
//! $$\text{MaxSim}(Q, D) = \sum_{i=1}^{|Q|} \max_{j=1}^{|D|} \langle q_i, d_j \rangle$$
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::{NodeIndex, SimilarityScore, VectorEmbedding};
use serde::{Deserialize, Serialize};

/// Multi-vector representation (sequence of token embeddings) for ColBERT / ColPali.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MultiVectorEmbedding {
    pub tokens: Vec<VectorEmbedding>,
    pub token_dim: usize,
}

impl MultiVectorEmbedding {
    /// Creates a multi-vector embedding from a list of token vectors.
    pub fn new(tokens: Vec<VectorEmbedding>) -> Self {
        let token_dim = tokens.first().map(|t| t.dimension()).unwrap_or(0);
        Self { tokens, token_dim }
    }

    /// Number of token vectors in the sequence.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Evaluates exact MaxSim score between query tokens and document tokens.
    pub fn maxsim(&self, document: &Self) -> f32 {
        if self.tokens.is_empty() || document.tokens.is_empty() {
            return 0.0;
        }

        let mut total_score = 0.0f32;

        for q_tok in &self.tokens {
            let mut max_tok_sim = f32::NEG_INFINITY;
            for d_tok in &document.tokens {
                let dot = (q_tok.dot_product_complex(d_tok)).re;
                if dot > max_tok_sim {
                    max_tok_sim = dot;
                }
            }
            if max_tok_sim.is_finite() {
                total_score += max_tok_sim;
            }
        }

        total_score
    }
}

/// Multi-Vector index storage.
pub struct MultiVectorIndex {
    pub token_dim: usize,
    documents: parking_lot::RwLock<Vec<MultiVectorEmbedding>>,
}

impl MultiVectorIndex {
    pub fn new(token_dim: usize) -> Self {
        Self {
            token_dim,
            documents: parking_lot::RwLock::new(Vec::new()),
        }
    }

    pub fn insert(&self, slot: NodeIndex, multivec: MultiVectorEmbedding) {
        let mut docs = self.documents.write();
        let idx = slot as usize;
        if docs.len() <= idx {
            docs.resize(idx + 1, MultiVectorEmbedding::default());
        }
        docs[idx] = multivec;
    }

    /// Evaluates MaxSim across documents.
    pub fn search(
        &self,
        query: &MultiVectorEmbedding,
        k: usize,
    ) -> Vec<(NodeIndex, SimilarityScore)> {
        if k == 0 || query.is_empty() {
            return Vec::new();
        }

        let docs = self.documents.read();
        let mut scored: Vec<(NodeIndex, SimilarityScore)> = docs
            .iter()
            .enumerate()
            .filter(|(_, doc)| !doc.is_empty())
            .map(|(slot, doc)| (slot as NodeIndex, query.maxsim(doc)))
            .collect();

        // DERIVED: Uses O(M + k log k) select_nth_unstable top-k selection rather than full O(M log M) sort.
        if scored.len() > k {
            scored
                .select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            scored.truncate(k);
        }
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    #[test]
    fn test_colbert_maxsim_scoring() {
        let dim = 8;
        let q_tok1 =
            VectorEmbedding::from_complex((0..dim).map(|_| Complex32::new(1.0, 0.0)).collect())
                .into_normalized();
        let q_tok2 = VectorEmbedding::from_complex(
            (0..dim)
                .map(|i| Complex32::new(if i == 0 { 1.0 } else { 0.0 }, 0.0))
                .collect(),
        )
        .into_normalized();
        let query = MultiVectorEmbedding::new(vec![q_tok1.clone(), q_tok2.clone()]);

        let d0_tok1 = q_tok1.clone();
        let d0_tok2 = q_tok2.clone();
        let doc0 = MultiVectorEmbedding::new(vec![d0_tok1, d0_tok2]);

        let d1_tok1 =
            VectorEmbedding::from_complex((0..dim).map(|_| Complex32::new(-1.0, 0.0)).collect())
                .into_normalized();
        let doc1 = MultiVectorEmbedding::new(vec![d1_tok1]);

        let index = MultiVectorIndex::new(dim);
        index.insert(0, doc0);
        index.insert(1, doc1);

        let results = index.search(&query, 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // doc0 has perfect match for both tokens
        assert!(results[0].1 > results[1].1);
    }
}
