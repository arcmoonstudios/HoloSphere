/* hnsqr/src/federation/cluster.rs */
//!▫~•◦-------------------------------‣
//! # Federated Cross-Cluster Certified Proof Search
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Executes federated Top-K exact search across sovereign geo-distributed clusters
//! by coordinating upper-bound proofs, terminating strictly when:
//!   `UB_cluster < tau_global` for all clusters.
//! Respects data residency policies and provides structured IncompleteGlobalProof
//! degraded proofs upon regional network partition.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::SimilarityScore;
use crate::service::ReadSnapshot;

pub type ClusterRegionId = String;

/// Response from a regional cluster participating in a federated query.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterProofResponse {
    pub region_id: ClusterRegionId,
    pub top_k: Vec<(String, SimilarityScore)>,
    /// Maximum upper bound among all pruned/unseen vectors in this region.
    pub max_unresolved_upper_bound: f64,
    pub snapshot: ReadSnapshot,
    pub is_complete: bool,
}

/// Certification status of a global federated query.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederatedProofStatus {
    CertifiedExact,
    IncompleteGlobalProof {
        missing_regions: Vec<ClusterRegionId>,
    },
}

/// Global federated query result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FederatedQueryResult {
    pub global_topk: Vec<(String, SimilarityScore)>,
    pub tau_global: f32,
    pub proof_status: FederatedProofStatus,
    pub participating_regions: Vec<ClusterRegionId>,
    pub unreachable_regions: Vec<ClusterRegionId>,
}

/// Federated Cluster Coordinator.
pub struct FederatedProofCoordinator;

impl FederatedProofCoordinator {
    /// Merges regional responses and evaluates whether the global result satisfies certification.
    pub fn merge_regional_proofs(
        k: usize,
        responses: Vec<ClusterProofResponse>,
        unreachable: Vec<ClusterRegionId>,
    ) -> FederatedQueryResult {
        let mut all_candidates = Vec::new();
        let mut participating = Vec::new();

        for resp in &responses {
            participating.push(resp.region_id.clone());
            all_candidates.extend(resp.top_k.clone());
        }

        // Sort descending by score, ascending by ID
        all_candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        all_candidates.truncate(k);

        let tau_global = if all_candidates.len() >= k {
            all_candidates.last().unwrap().1
        } else {
            f32::NEG_INFINITY
        };

        // Certified condition: Every region's max unresolved upper bound must be strictly less than tau_global
        let proof_status = if !unreachable.is_empty() {
            FederatedProofStatus::IncompleteGlobalProof {
                missing_regions: unreachable.clone(),
            }
        } else {
            let mut all_regions_certified = true;
            for resp in &responses {
                if !resp.is_complete || resp.max_unresolved_upper_bound as f32 >= tau_global {
                    all_regions_certified = false;
                    break;
                }
            }
            if all_regions_certified {
                FederatedProofStatus::CertifiedExact
            } else {
                FederatedProofStatus::IncompleteGlobalProof {
                    missing_regions: Vec::new(),
                }
            }
        };

        FederatedQueryResult {
            global_topk: all_candidates,
            tau_global,
            proof_status,
            participating_regions: participating,
            unreachable_regions: unreachable,
        }
    }
}
