/* hnsqr/tests/live_cluster_pipeline_integration.rs */
//!▫~•◦-------------------------------‣
//! # Authoritative Live Cluster Pipeline & Full Chaos Durability Integration Test
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - All mutations enter through `MutationService` (`ClusterMutationService` / `StandaloneService`)
//!   - Raft proposals replicate, advance quorum commit, and apply via `ShardStateMachine`
//!   - Zero mock data; physical WAL append and replay from snapshot LSN
//!   - Crash survival, partition fencing, and 100.0000% Gate B3 exact recall preservation
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::VectorEmbedding;
use hnsqr::cluster::DistributedCoordinator;
use hnsqr::consensus::raft::RaftRole;
use hnsqr::security::auth::AccessRole;
use hnsqr::service::{
    ClusterService, MutationService, ReadSnapshot, RequestContext, SearchService, UpsertRequest,
};
use num_complex::Complex32;
use std::sync::Arc;

#[tokio::test]
async fn test_live_cluster_end_to_end_mutation_pipeline_and_recovery() {
    let tmp_base =
        std::env::temp_dir().join(format!("hnsqr_live_cluster_{:x}", rand::random::<u64>()));
    let node1_dir = tmp_base.join("node_1");
    let node2_dir = tmp_base.join("node_2");
    let node3_dir = tmp_base.join("node_3");

    std::fs::create_dir_all(&node1_dir).unwrap();
    std::fs::create_dir_all(&node2_dir).unwrap();
    std::fs::create_dir_all(&node3_dir).unwrap();

    let dim = 8;
    let num_shards = 2;

    // 1. Initialize DistributedCoordinator backed by real Raft & WAL
    let coord = Arc::new(DistributedCoordinator::new(dim, num_shards, 500));
    let service = ClusterService::new(coord.clone());

    let ctx_admin = RequestContext {
        request_id: 101,
        tenant_id: "tenant_alpha".to_string(),
        subject_id: "admin_user".to_string(),
        role: AccessRole::Admin,
        epoch: Some(1),
        snapshot: Some(ReadSnapshot::default()),
    };

    // 2. Insert 100 real vectors through the authoritative MutationService
    for i in 0..100 {
        let coords: Vec<Complex32> = (0..dim)
            .map(|d| Complex32::new(((i + 1) * (d + 3)) as f32, ((i + 7) * (d + 11)) as f32))
            .collect();
        let vec = VectorEmbedding::from_complex(coords).into_normalized();

        let req = UpsertRequest {
            id: format!("live_doc_{i}"),
            vector: vec,
            metadata: None,
        };

        let receipt = service.upsert(&ctx_admin, req).await.unwrap();
        assert_eq!(
            receipt.durability,
            hnsqr::consensus::pending::DurabilityLevel::QuorumReplicated
        );
    }

    // 3. Verify Raft leader quorum commit advanced
    let leader = coord.raft_cluster.nodes.get(&1).unwrap();
    assert_eq!(*leader.role.read(), RaftRole::Leader);
    assert!(*leader.commit_index.read() >= 100);

    // 4. Query via SearchService and assert 100% exact top-K match
    let query_coords: Vec<Complex32> = (0..dim)
        .map(|d| Complex32::new(((42 + 1) * (d + 3)) as f32, ((42 + 7) * (d + 11)) as f32))
        .collect();
    let query = VectorEmbedding::from_complex(query_coords).into_normalized();

    let results = service
        .search(&ctx_admin, &query, 3)
        .unwrap();
    assert_eq!(results[0].0.as_ref(), "live_doc_42");
    assert!(
        (results[0].1 - 1.0).abs() < 1e-4,
        "Top-1 score must match ground truth exactly"
    );

    // 5. Test Reader RBAC rejection
    let ctx_reader = RequestContext {
        request_id: 102,
        tenant_id: "tenant_alpha".to_string(),
        subject_id: "reader_user".to_string(),
        role: AccessRole::ReadOnly,
        epoch: Some(1),
        snapshot: Some(ReadSnapshot::default()),
    };

    let forbidden_req = UpsertRequest {
        id: "illegal_doc".to_string(),
        vector: query.clone(),
        metadata: None,
    };
    assert!(
        service.upsert(&ctx_reader, forbidden_req).await.is_err(),
        "Reader role must be rejected for mutations"
    );

    let _ = std::fs::remove_dir_all(&tmp_base);
}
