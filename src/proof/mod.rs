/* holosphere/src/proof/mod.rs */
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

pub mod benchmark_snapshot;
pub mod bounds;
pub mod search;
pub mod tree;

pub use benchmark_snapshot::{
    PROOF_BENCHMARK_ARTIFACT_VERSION, ProofBenchmarkArtifact, proof_benchmark_artifact_filename,
};
pub use bounds::{
    PROOF_BLOCK_COMPLEX_DIM, ProofCentroidCode, ProofQuery, evaluate_node_upper_bound_f64,
};
pub use search::{
    DenseExactProof, Finalist, GlobalExactProofSearch, GlobalPacProofSearch, ProofFrontierEntry,
    SegmentProofView, TopKAccumulator,
};
pub use tree::{ManifoldGeometryProfile, PROOF_LEAF_TARGET, ProofNode, SemanticProofTree};
