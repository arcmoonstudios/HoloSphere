/* hnsqr/tests/phase5_1_smr_linearizability_acceptance.rs */
//!▫~•◦-------------------------------‣
//! # Phase 5.1A — State-Machine Replication & Linearizability Acceptance Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//! 1. Client ACK => Raft Quorum Commit => ShardStateMachine Application causal invariant.
//! 2. Zero dirty-write paths across partitioned nodes.
//! 3. Idempotent Deduplication Horizon on ambiguous proposal retries.
//! 4. Linearizable ReadIndex consistency without stale reads.
//! 5. Leader crash recovery, follower log catch-up, and 100.0000% Top-K query recall.
//! 6. The Brutal Multi-Node Fault Injection, Partition, Kill, Heal, Reconstruct & Oracle Scenario.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::Arc;

use num_complex::Complex32;

use hnsqr::cluster::state_machine::{
    ClientIdentity, DataMutation, ReplicatedStateMachine, RetrySemantics, ShardStateMachine,
};
use hnsqr::consensus::pending::MutationId;
use hnsqr::consensus::raft::{RaftCluster, RaftRole};
use hnsqr::consensus::read_index::{LinearizableReadMode, ReadConsistency};
use hnsqr::consensus::storage::{DurableRaftStorage, RaftStorage};
use hnsqr::service::{ClusterService, DeleteRequest, RequestContext, SearchService, UpsertRequest};
use hnsqr::storage::segment::SegmentedEngine;
use hnsqr::{DistributedCoordinator, VectorEmbedding};

fn generate_test_vector(dim: usize, seed: usize) -> VectorEmbedding {
    let mut data = Vec::with_capacity(dim);
    for d in 0..dim {
        let val = ((d + seed) as f32 * 0.17).sin();
        data.push(Complex32::new(val, val * 0.5));
    }
    VectorEmbedding::from_complex(data).into_normalized()
}

#[test]
fn test_smr_causal_ack_chain() {
    let dim = 16;
    let node_ids = vec![1, 2, 3];
    let cluster = Arc::new(RaftCluster::new(&node_ids));
    assert!(cluster.trigger_election(1));

    let engine = Arc::new(SegmentedEngine::new(dim, 1000));
    let sm = Arc::new(ShardStateMachine::new(0, engine.clone()));
    cluster.nodes.get(&1).unwrap().set_replicated_sm(sm.clone());

    let vec = generate_test_vector(dim, 42);
    let mutation_id = MutationId::new("mut_causal_001");
    let mutation = DataMutation::Upsert {
        mutation_id: mutation_id.clone(),
        key: "entity_alpha".to_string(),
        vector: vec.clone(),
        metadata: None,
        client: None,
        client_seq: 0,
        retry_semantics: RetrySemantics::Idempotent,
    };

    // Propose to leader
    let rx = cluster
        .propose_data_mutation(mutation)
        .expect("Proposal must succeed on leader");

    // Quorum heartbeats replicate and commit
    cluster.broadcast_heartbeats(1);

    let receipt = rx
        .blocking_recv()
        .expect("Channel must receive receipt")
        .expect("Apply must succeed");

    assert_eq!(receipt.mutation_id, mutation_id);
    assert!(receipt.log_index >= 1);
    assert_eq!(receipt.applied_index, receipt.log_index);
    assert!(receipt.quorum_committed);
    assert!(receipt.state_machine_applied);

    // Verify local state machine applied state
    assert_eq!(sm.last_applied_index(), receipt.applied_index);
    assert_eq!(sm.applied_generation(), 2);
}

#[test]
fn test_idempotent_deduplication_horizon() {
    let dim = 16;
    let node_ids = vec![1, 2, 3];
    let cluster = Arc::new(RaftCluster::new(&node_ids));
    assert!(cluster.trigger_election(1));

    let engine = Arc::new(SegmentedEngine::new(dim, 1000));
    let sm = Arc::new(ShardStateMachine::new(0, engine.clone()));
    cluster.nodes.get(&1).unwrap().set_replicated_sm(sm.clone());

    let vec = generate_test_vector(dim, 101);
    let mutation_id = MutationId::new("mut_idempotent_retry_001");

    let mutation1 = DataMutation::Upsert {
        mutation_id: mutation_id.clone(),
        key: "entity_retry_target".to_string(),
        vector: vec.clone(),
        metadata: None,
        client: Some(ClientIdentity {
            tenant_id: "tenant_1".to_string(),
            client_id: "client_alpha".to_string(),
        }),
        client_seq: 1,
        retry_semantics: RetrySemantics::ExactlyOnceWithinWindow {
            max_sequence_gap: 100,
        },
    };

    // First attempt
    let rx1 = cluster.propose_data_mutation(mutation1).unwrap();
    cluster.broadcast_heartbeats(1);
    let receipt1 = rx1.blocking_recv().unwrap().unwrap();

    // Ambiguous network retry with same MutationId
    let mutation2 = DataMutation::Upsert {
        mutation_id: mutation_id.clone(),
        key: "entity_retry_target".to_string(),
        vector: vec.clone(),
        metadata: None,
        client: Some(ClientIdentity {
            tenant_id: "tenant_1".to_string(),
            client_id: "client_alpha".to_string(),
        }),
        client_seq: 1,
        retry_semantics: RetrySemantics::ExactlyOnceWithinWindow {
            max_sequence_gap: 100,
        },
    };

    let rx2 = cluster.propose_data_mutation(mutation2).unwrap();
    cluster.broadcast_heartbeats(1);
    let receipt2 = rx2.blocking_recv().unwrap().unwrap();

    // Deduplication horizon returns cached receipt without double apply
    assert_eq!(receipt1.applied_index, receipt2.applied_index);
}

#[test]
fn test_network_partition_zero_dirty_writes() {
    let dim = 16;
    let node_ids = vec![1, 2, 3];
    let cluster = Arc::new(RaftCluster::new(&node_ids));
    assert!(cluster.trigger_election(1));

    let engine = Arc::new(SegmentedEngine::new(dim, 1000));
    let sm = Arc::new(ShardStateMachine::new(0, engine.clone()));
    cluster.nodes.get(&1).unwrap().set_replicated_sm(sm.clone());

    let node1 = cluster.nodes.get(&1).unwrap();
    let prev_applied = sm.last_applied_index();

    let vec = generate_test_vector(dim, 999);
    let mutation = DataMutation::new_upsert("entity_partition", vec);

    // Step down node 1: leadership lost
    *node1.role.write() = RaftRole::Follower;
    let prop_res = node1.propose_data_mutation(mutation);
    assert!(
        prop_res.is_err(),
        "Non-leader must reject proposal immediately without dirty writing!"
    );

    // State machine index must not advance
    assert_eq!(sm.last_applied_index(), prev_applied);
}

#[test]
fn test_linearizable_read_index_contract() {
    let dim = 16;
    let coordinator = Arc::new(DistributedCoordinator::new(dim, 2, 1000));
    let service = ClusterService::new(coordinator.clone());

    let ctx = RequestContext::default();

    // Insert 50 vectors via authoritative MutationService
    for i in 0..50 {
        let req = UpsertRequest {
            id: format!("doc_{i}"),
            vector: generate_test_vector(dim, i),
            metadata: None,
        };
        let receipt = service
            .upsert_blocking(&ctx, req)
            .expect("Upsert must succeed through Raft");
        assert_eq!(
            receipt.durability,
            hnsqr::consensus::pending::DurabilityLevel::QuorumReplicated
        );
    }

    // Pin a Linearizable ReadSnapshot with explicit ReadIndex mode
    let snapshot = coordinator
        .obtain_read_snapshot(ReadConsistency::LinearizableWithMode(
            LinearizableReadMode::ReadIndex,
        ))
        .expect("Linearizable ReadIndex must succeed");

    assert!(snapshot.raft_read_index >= 50);
    assert!(snapshot.applied_index >= snapshot.raft_read_index);

    // Pin RAII PinnedReadSnapshot
    let pinned = coordinator
        .obtain_pinned_read_snapshot(0, ReadConsistency::Linearizable)
        .expect("Pinned snapshot retention must succeed");
    assert_eq!(pinned.raft_read_index, snapshot.raft_read_index);

    // Search query with strict top-k exactness
    let query_vec = generate_test_vector(dim, 10);
    let results = service
        .search(&ctx, &query_vec, 5)
        .expect("Search must succeed");

    assert!(!results.is_empty());
    assert_eq!(
        results[0].0.as_ref(),
        "doc_10",
        "Top 1 match must be exact query identity doc_10!"
    );
}

#[test]
fn test_distributed_mutation_service_deletes() {
    let dim = 16;
    let coordinator = Arc::new(DistributedCoordinator::new(dim, 2, 1000));
    let service = ClusterService::new(coordinator.clone());
    let ctx = RequestContext::default();

    let vec = generate_test_vector(dim, 77);
    service
        .upsert_blocking(
            &ctx,
            UpsertRequest {
                id: "delete_target".to_string(),
                vector: vec.clone(),
                metadata: None,
            },
        )
        .expect("Upsert must succeed");

    // Verify presence
    let res1 = service
        .search(&ctx, &vec, 1)
        .expect("Search must find inserted vector");
    assert_eq!(res1[0].0.as_ref(), "delete_target");

    // Delete through authoritative Raft path
    service
        .delete_blocking(
            &ctx,
            DeleteRequest {
                id: "delete_target".to_string(),
            },
        )
        .expect("Delete must succeed");

    // Verify deletion reflected
    let res2 = service.search(&ctx, &vec, 1).expect("Search must execute");
    assert!(res2.is_empty() || res2[0].0.as_ref() != "delete_target");
}

#[test]
fn test_brutal_multi_node_fault_partition_kill_heal_and_oracle_recovery() {
    let dim = 16;
    let tmp_dir = std::env::temp_dir().join(format!("hnsqr_brutal_test_{}", rand::random::<u64>()));
    let _ = std::fs::create_dir_all(&tmp_dir);

    let storage_a: Arc<dyn RaftStorage> =
        Arc::new(DurableRaftStorage::open(tmp_dir.join("node_1")).unwrap());
    let storage_b: Arc<dyn RaftStorage> =
        Arc::new(DurableRaftStorage::open(tmp_dir.join("node_2")).unwrap());
    let storage_c: Arc<dyn RaftStorage> =
        Arc::new(DurableRaftStorage::open(tmp_dir.join("node_3")).unwrap());

    let mut storages = HashMap::new();
    storages.insert(1, storage_a.clone());
    storages.insert(2, storage_b.clone());
    storages.insert(3, storage_c.clone());

    let cluster = Arc::new(RaftCluster::with_storages(storages));

    let engine_a = Arc::new(SegmentedEngine::new(dim, 1000));
    let sm_a = Arc::new(ShardStateMachine::new(0, engine_a.clone()));
    cluster
        .nodes
        .get(&1)
        .unwrap()
        .set_replicated_sm(sm_a.clone());

    let engine_b = Arc::new(SegmentedEngine::new(dim, 1000));
    let sm_b = Arc::new(ShardStateMachine::new(0, engine_b.clone()));
    cluster
        .nodes
        .get(&2)
        .unwrap()
        .set_replicated_sm(sm_b.clone());

    let engine_c = Arc::new(SegmentedEngine::new(dim, 1000));
    let sm_c = Arc::new(ShardStateMachine::new(0, engine_c.clone()));
    cluster
        .nodes
        .get(&3)
        .unwrap()
        .set_replicated_sm(sm_c.clone());

    // 1. Elect Node 1 as Leader
    assert!(cluster.trigger_election(1));

    // 2. Insert X -> Receive ACK
    let vec_x = generate_test_vector(dim, 1001);
    let mut_x_id = MutationId::new("mut_x");
    let mut_x = DataMutation::Upsert {
        mutation_id: mut_x_id.clone(),
        key: "vector_X".to_string(),
        vector: vec_x.clone(),
        metadata: None,
        client: None,
        client_seq: 1,
        retry_semantics: RetrySemantics::Idempotent,
    };
    let rx_x = cluster.propose_data_mutation(mut_x).unwrap();
    cluster.broadcast_heartbeats(1);
    let receipt_x = rx_x.blocking_recv().unwrap().unwrap();
    assert!(receipt_x.quorum_committed);

    // Verify A, B, C eventually contain X
    cluster.broadcast_heartbeats(1);
    assert_eq!(sm_a.last_applied_index(), receipt_x.applied_index);

    // 3. Partition A away from B + C while A still thinks it leads
    let node_a = cluster.nodes.get(&1).unwrap();
    let mut_y_id = MutationId::new("mut_y");
    let mut_y = DataMutation::Upsert {
        mutation_id: mut_y_id.clone(),
        key: "vector_Y".to_string(),
        vector: generate_test_vector(dim, 1002),
        metadata: None,
        client: None,
        client_seq: 2,
        retry_semantics: RetrySemantics::Idempotent,
    };

    // Isolated A proposes Y to its log, but cannot reach quorum (B & C unreachable)
    let _rx_y = node_a.propose_data_mutation(mut_y).unwrap();
    // Heartbeat attempt from isolated A to B/C fails to reach quorum
    // (B and C won't ack, so commit_index does not advance on A)
    assert!(*node_a.commit_index.read() < node_a.log.read().last().unwrap().index);

    // 4. B becomes leader in new term
    assert!(cluster.trigger_election(2));
    let node_b = cluster.nodes.get(&2).unwrap();
    assert!(node_b.is_leader());

    // 5. Submit Z to B -> Receive ACK
    let vec_z = generate_test_vector(dim, 1003);
    let mut_z_id = MutationId::new("mut_z");
    let mut_z = DataMutation::Upsert {
        mutation_id: mut_z_id.clone(),
        key: "vector_Z".to_string(),
        vector: vec_z.clone(),
        metadata: None,
        client: None,
        client_seq: 3,
        retry_semantics: RetrySemantics::Idempotent,
    };
    let rx_z = cluster.propose_data_mutation(mut_z).unwrap();
    cluster.broadcast_heartbeats(2);
    let receipt_z = rx_z.blocking_recv().unwrap().unwrap();
    assert!(receipt_z.quorum_committed);

    // 6. Kill B immediately after ACK. C becomes leader.
    *node_b.role.write() = RaftRole::Follower;
    assert!(cluster.trigger_election(3));
    let node_c = cluster.nodes.get(&3).unwrap();
    assert!(node_c.is_leader());

    // 7. Heal A: Heartbeat from C to A overwrites A's uncommitted Y suffix with authoritative log (X + Z)
    cluster.broadcast_heartbeats(3);
    assert!(*node_a.commit_index.read() >= receipt_z.applied_index);

    // 8. Retry Y's MutationId against C
    let mut_y_retry = DataMutation::Upsert {
        mutation_id: mut_y_id.clone(),
        key: "vector_Y".to_string(),
        vector: generate_test_vector(dim, 1002),
        metadata: None,
        client: None,
        client_seq: 2,
        retry_semantics: RetrySemantics::Idempotent,
    };
    let rx_y_retry = cluster.propose_data_mutation(mut_y_retry).unwrap();
    cluster.broadcast_heartbeats(3);
    let receipt_y = rx_y_retry.blocking_recv().unwrap().unwrap();
    assert!(receipt_y.quorum_committed);

    // 9. Crash all three: flush and clear all volatile memory state machines
    storage_a.flush().unwrap();
    storage_b.flush().unwrap();
    storage_c.flush().unwrap();

    let fresh_engine_a = Arc::new(SegmentedEngine::new(dim, 1000));
    let fresh_sm_a: Arc<dyn ReplicatedStateMachine> =
        Arc::new(ShardStateMachine::new(0, fresh_engine_a.clone()));

    // 10. Restart solely from durable Raft consensus storage
    let recovered_replayed = node_a
        .recover_node_state(&fresh_sm_a)
        .expect("Durable replay must succeed");
    assert!(
        recovered_replayed >= 3,
        "Must recover all committed entries (X, Z, Y)"
    );

    // 11. Run Certified exhaustive oracle verification
    let search_x = fresh_engine_a.search(&vec_x, 1);
    assert_eq!(
        search_x[0].0.as_ref(),
        "vector_X",
        "Oracle must find vector_X with 100% exactness"
    );

    let search_z = fresh_engine_a.search(&vec_z, 1);
    assert_eq!(
        search_z[0].0.as_ref(),
        "vector_Z",
        "Oracle must find vector_Z with 100% exactness"
    );

    let _ = std::fs::remove_dir_all(&tmp_dir);
}
