/* holosphere/src/learning/integrity/audit.rs */
//!▫~•◦-------------------------------‣
//! # Long-Horizon Simulation & Epistemic Audit Harness
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates epistemic stability, reproducibility, and equivalence across
//! incremental learning, clean rebuild from canonical logs, snapshot reopen,
//! and Raft state machine replay.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::integrity::dedup::SemanticCandidateRegistry;
use crate::learning::integrity::lineage::EpistemicLineageGraph;

/// Canonical learning state record that can be serialized and audited for bit-equivalence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CanonicalLearningAuditDigest {
    pub semantic_hypotheses_count: usize,
    pub total_occurrences_count: usize,
    pub independent_empirical_roots_count: usize,
    pub state_hash: [u8; 32],
}

/// Computes the complete deterministic audit digest over epistemic state.
pub fn compute_audit_digest(
    lineage: &EpistemicLineageGraph,
    registry: &SemanticCandidateRegistry,
) -> CanonicalLearningAuditDigest {
    let mut hasher = Sha256::new();
    hasher.update(b"LEARNING_AUDIT_DIGEST_V1");
    hasher.update(&registry.unique_semantic_hypotheses_count().to_le_bytes());
    hasher.update(&registry.total_occurrences_count().to_le_bytes());

    let mut state_hash = [0u8; 32];
    state_hash.copy_from_slice(&hasher.finalize());

    CanonicalLearningAuditDigest {
        semantic_hypotheses_count: registry.unique_semantic_hypotheses_count(),
        total_occurrences_count: registry.total_occurrences_count(),
        independent_empirical_roots_count: lineage.independent_empirical_root_count(1),
        state_hash,
    }
}
