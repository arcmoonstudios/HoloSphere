/* holosphere/src/retrieval/mod.rs */
//!▫~•◦-------------------------------‣
//! # Multi-Modal & Hybrid Retrieval Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod hybrid;
pub mod linguistic;
pub mod multivector;
pub mod performance_trial;
pub mod sparse;
pub mod top_k;

pub use hybrid::{HybridFusionEngine, HybridFusionMethod, ModalityRankings, RRF_DEFAULT_K};
pub use linguistic::{
    FuzzyLevenshteinAutomaton, LanguageMode, MorphologicalStemmer, PhoneticMatcher,
};
pub use multivector::{MultiVectorEmbedding, MultiVectorIndex};
pub use performance_trial::{
    AdmissionGateStatus, BenchmarkRecord, BenchmarkRunIdentity, CertifiedEvidence,
    HnswBuildDescriptor, HnswSearchDescriptor, RetrievalTrial, TrialValidationError,
    evaluate_admission_gates,
};
pub use sparse::{
    InvertedPostingList, PostingChunk, PostingEntry, SparseInvertedIndex, SparseVector,
};
pub use top_k::{BoundedTopKCollector, TopKScore};
