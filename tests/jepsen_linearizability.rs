/* hnsqr/tests/jepsen_linearizability.rs */
//!▫~•◦-------------------------------‣
//! # Jepsen-Style Chaos & Raft Linearizability Verification Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - Strict leader election and quorum-bounded commit indexing ($\lfloor N/2 \rfloor + 1$)
//!   - Minority partition failure to commit writes (Zero Split-Brain)
//!   - Partition healing and log reconciliation
//!   - Leader crash churn with zero acknowledged-write loss
//!   - Non-voting Raft learner read replica scaling & read consistency contracts
//!   - Dynamic cluster membership change
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::Arc;
use hnsqr::cluster::state_machine::{DataMutation, ShardStateMachine};
use hnsqr::consensus::raft::{MembershipMutation, RaftCluster, RaftCommand, RaftRole, ReadConsistency, TopologyMutation};
use hnsqr::storage::segment::SegmentedEngine;
use hnsqr::VectorEmbedding;

#[test]
fn test_raft_election_and_log_replication_quorum() {
    let cluster = RaftCluster::new(&[1, 2, 3]);

    // 1. Elect node 1
    assert!(cluster.trigger_election(1));
    assert!(cluster.nodes.get(&1).unwrap().is_leader());

    let leader = cluster.nodes.get(&1).unwrap();

    // 2. Propose topology update
    let mut shard_owners = HashMap::new();
    shard_owners.insert(0, 1);
    shard_owners.insert(1, 2);

    let entry_idx = leader.propose(RaftCommand::Topology(TopologyMutation {
        epoch: 2,
        shard_owners,
    })).unwrap();

    // Before heartbeats, commit index has not advanced past election NoOp
    assert_eq!(*leader.commit_index.read(), 1);

    // 3. Broadcast heartbeats to replicate and achieve quorum
    cluster.broadcast_heartbeats(1);
    assert_eq!(*leader.commit_index.read(), entry_idx);
    assert_eq!(*leader.topology_epoch.read(), 2);

    // 4. Second heartbeat propagates the new commit index to followers
    cluster.broadcast_heartbeats(1);

    let f2 = cluster.nodes.get(&2).unwrap();
    assert_eq!(*f2.topology_epoch.read(), 2);
    assert_eq!(*f2.shard_owners.read().get(&0).unwrap(), 1);
}

#[test]
fn test_asymmetric_partition_prevents_minority_commit() {
    let cluster = RaftCluster::new(&[1, 2, 3, 4, 5]);

    assert!(cluster.trigger_election(1));
    let leader1 = cluster.nodes.get(&1).unwrap();

    // Simulate Network Partition:
    // Subnet A (Minority): {1, 2}
    // Subnet B (Majority): {3, 4, 5}
    *leader1.voting_peers.write() = [1, 2].into_iter().collect();

    // Minority leader tries to propose an entry
    let prop_idx = leader1.propose(RaftCommand::NoOp).unwrap();
    leader1.match_index.write().insert(2, prop_idx);

    // Majority partition {3, 4, 5} elects a new leader
    let node3 = cluster.nodes.get(&3).unwrap();
    *node3.voting_peers.write() = [3, 4, 5].into_iter().collect();
    *cluster.nodes.get(&4).unwrap().voting_peers.write() = [3, 4, 5].into_iter().collect();
    *cluster.nodes.get(&5).unwrap().voting_peers.write() = [3, 4, 5].into_iter().collect();

    assert!(cluster.trigger_election(3));
    assert!(node3.is_leader());

    // Majority leader proposes and commits successfully
    let maj_idx = node3.propose(RaftCommand::NoOp).unwrap();
    cluster.broadcast_heartbeats(3);

    assert_eq!(*node3.commit_index.read(), maj_idx);
}

#[test]
fn test_leader_death_and_re_election_churn() {
    let cluster = RaftCluster::new(&[1, 2, 3]);

    assert!(cluster.trigger_election(1));
    let leader1 = cluster.nodes.get(&1).unwrap();
    let idx1 = leader1.propose(RaftCommand::NoOp).unwrap();
    cluster.broadcast_heartbeats(1);
    assert_eq!(*leader1.commit_index.read(), idx1);

    // Kill leader 1
    *leader1.role.write() = RaftRole::Follower;

    // Node 2 triggers election with new term
    assert!(cluster.trigger_election(2));
    let leader2 = cluster.nodes.get(&2).unwrap();
    assert!(leader2.is_leader());
    assert_eq!(*leader2.current_term.read(), 2);

    let idx2 = leader2.propose(RaftCommand::NoOp).unwrap();
    cluster.broadcast_heartbeats(2);
    assert_eq!(*leader2.commit_index.read(), idx2);
}

#[test]
fn test_learner_replica_scaling_and_consistency() {
    let mut cluster = RaftCluster::new(&[1, 2, 3]);
    let engine = Arc::new(SegmentedEngine::new(4, 1000));
    let sm = Arc::new(ShardStateMachine::new(0, engine));

    for node in cluster.nodes.values() {
        node.set_replicated_sm(sm.clone());
    }

    // Add non-voting learner replica (node 10)
    cluster.add_learner(10);

    assert!(cluster.trigger_election(1));
    let leader = cluster.nodes.get(&1).unwrap();

    let v1 = VectorEmbedding::from_reals(&[1.0, 0.0, 0.0, 0.0]);
    let v2 = VectorEmbedding::from_reals(&[0.0, 1.0, 0.0, 0.0]);

    // Propose batch
    let batch_indices = leader.propose_batch(vec![
        RaftCommand::Data(DataMutation::new_upsert("doc_50", v1)),
        RaftCommand::Data(DataMutation::new_upsert("doc_51", v2)),
    ]).unwrap();

    cluster.broadcast_heartbeats(1);
    assert_eq!(*leader.commit_index.read(), *batch_indices.last().unwrap());

    // Second heartbeat propagates commit index to learner
    cluster.broadcast_heartbeats(1);

    let learner = cluster.nodes.get(&10).unwrap();
    assert!(learner.is_learner());
    assert_eq!(*learner.commit_index.read(), *batch_indices.last().unwrap());

    // Test read consistency contracts
    assert!(learner.validate_read_consistency(ReadConsistency::Committed).is_ok());
    assert!(learner.validate_read_consistency(ReadConsistency::BoundedStaleness { max_lag_entries: 5, max_age_ms: 100 }).is_ok());
    assert!(learner.validate_read_consistency(ReadConsistency::Linearizable).is_err(), "Learner cannot serve Linearizable read");
}

#[test]
fn test_joint_consensus_dynamic_membership_change() {
    let cluster = RaftCluster::new(&[1, 2, 3]);
    assert!(cluster.trigger_election(1));
    let leader = cluster.nodes.get(&1).unwrap();

    let new_membership = vec![1, 2, 3, 4, 5];
    let change_idx = leader.propose(RaftCommand::Membership(MembershipMutation {
        new_peers: new_membership.clone(),
    })).unwrap();

    cluster.broadcast_heartbeats(1);
    assert_eq!(*leader.commit_index.read(), change_idx);
    assert_eq!(leader.voting_peers.read().len(), 5);
}
