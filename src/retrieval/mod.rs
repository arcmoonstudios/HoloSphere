/* hnsqr/src/retrieval/mod.rs */
//!▫~•◦-------------------------------‣
//! # Multi-Modal & Hybrid Retrieval Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod hybrid;
pub mod linguistic;
pub mod multivector;
pub mod sparse;

pub use hybrid::{HybridFusionEngine, HybridFusionMethod, ModalityRankings, RRF_DEFAULT_K};
pub use linguistic::{
    FuzzyLevenshteinAutomaton, LanguageMode, MorphologicalStemmer, PhoneticMatcher,
};
pub use multivector::{MultiVectorEmbedding, MultiVectorIndex};
pub use sparse::{
    InvertedPostingList, PostingChunk, PostingEntry, SparseInvertedIndex, SparseVector,
};
