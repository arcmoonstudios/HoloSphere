/* hnsqr/src/proof/mod.rs */
//!▫~•◦-------------------------------‣
//! # Corpus-Covering Semantic Proof Engine (Gate B0/B1)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a mathematically rigorous, corpus-covering hierarchical proof structure
//! that formally guarantees $100.000\%$ exact Top-$K$ retrieval across multi-segment
//! memory architectures.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod bounds;
pub mod lutz;
pub mod search;
pub mod tree;

pub use bounds::{
    PROOF_BLOCK_COMPLEX_DIM, ProofCentroidCode, ProofQuery, evaluate_node_upper_bound_f64,
};
pub use lutz::{
    LutzCertifier, LutzCode, LutzGlobalCertified, LutzQueryTable, SemanticRerankPlan,
};
pub use search::{
    DenseExactProof, Finalist, GlobalExactProofSearch, ProofFrontierEntry, SegmentProofView,
    TopKAccumulator,
};
pub use tree::{PROOF_LEAF_TARGET, ProofNode, SemanticProofTree};
