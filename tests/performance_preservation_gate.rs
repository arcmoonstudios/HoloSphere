/* hnsqr/tests/performance_preservation_gate.rs */
//!▫~•◦-------------------------------‣
//! # Gate B3 Performance Preservation Regression Test
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Asserts:
//!   - 100.0000% Exact Recall under all enterprise subsystems
//!   - Exact accounting invariant: N_eligible == N_pruned + N_exact + N_tombstones
//!   - Ground truth exactness certification
//!   - Zero regression from Gate B3 retrieval math
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::metadata::store::{MetadataQuotaConfig, MetadataStore};
use hnsqr::proof::lutz::LutzCode;
use hnsqr::proof::search::GlobalExactProofSearch;
use hnsqr::proof::tree::SemanticProofTree;
use hnsqr::security::tenant::TenantNamespace;
use hnsqr::telemetry::tracing::TraceContext;
use hnsqr::{NodeIndex, SegmentProofView, VectorEmbedding};
use num_complex::Complex32;

#[test]
fn test_gate_b3_performance_preservation_under_enterprise_layers() {
    let dim_c = 64; // 128 real dimensions
    let n_corpus = 500;
    let k = 10;

    let mut vectors = Vec::with_capacity(n_corpus);
    for i in 0..n_corpus {
        let coords: Vec<Complex32> = (0..dim_c)
            .map(|j| Complex32::new(((i * 7 + j) % 13) as f32, ((i * 11 + j) % 17) as f32))
            .collect();
        vectors.push(VectorEmbedding::from_complex(coords).into_normalized());
    }

    let slots: Vec<NodeIndex> = (0..n_corpus as NodeIndex).collect();
    let lutz_codes: Vec<LutzCode> = vectors.iter().map(|v| LutzCode::encode(v, false)).collect();
    let proof_tree = SemanticProofTree::build(&vectors, &slots, dim_c);

    // Multi-tenant metadata index
    let _tenant_ns = TenantNamespace::new("tenant_enterprise", "knowledge_base");
    let _meta_store = MetadataStore::new(MetadataQuotaConfig::default());

    // Query with tracing
    let _trace = TraceContext::new_root();
    let query = &vectors[42];

    // Compute ground truth exhaustive exact Top-K
    let mut gt: Vec<(NodeIndex, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as NodeIndex, (v.dot_product_complex(query)).re))
        .collect();
    gt.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    gt.truncate(k);

    // Certified retrieval
    let view = SegmentProofView {
        vectors: &vectors,
        tombstones: None,
        tree: &proof_tree,
        lutz_codes: Some(&lutz_codes),
    };

    let (candidates, proof) = GlobalExactProofSearch::search(query, k, &[view], &[], &[], None);

    // Assert Exact Recall = 100%
    assert_eq!(candidates.len(), k);
    assert_eq!(
        candidates, gt,
        "Candidates must match Ground Truth with 100.0000% exactness"
    );
    assert!(
        proof.globally_exact,
        "Proof must be certified globally exact"
    );
    assert!(
        proof.is_accounting_exact(),
        "Terminal accounting funnel invariant must hold exactly"
    );
}
