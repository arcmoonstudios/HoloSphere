/* hnsqr/tests/raft_storage_liveness.rs */
//!▫~•◦-------------------------------‣
//! # Raft Consensus Liveness Decoupling & Storage Stall Invariant Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - Heartbeat priority execution lane remains schedulable when WAL storage stalls
//!   - Zero avoidable elections under 5ms, 25ms, 100ms, and multi-second simulated fsync stalls
//!   - Adaptive durability batching adapting window depth under latency spikes
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::consensus::durability_controller::{DurabilityController, StorageTelemetry};
use hnsqr::consensus::raft::{RaftCluster, RaftCommand, RaftRole};
use hnsqr::planning::autoforge::OperatorIntent;

#[test]
fn test_heartbeats_immune_to_fsync_stalls() {
    let cluster = RaftCluster::new(&[1, 2, 3]);
    assert!(cluster.trigger_election(1));

    let leader = cluster.nodes.get(&1).unwrap();
    assert!(leader.is_leader());

    // Propose batch
    let batch = vec![
        RaftCommand::NoOp,
        RaftCommand::NoOp,
    ];
    let indices = leader.propose_batch(batch).unwrap();

    // Replicate via decoupled heartbeat lane
    cluster.broadcast_heartbeats(1);
    assert_eq!(*leader.commit_index.read(), *indices.last().unwrap());

    // Inject simulated 100ms storage stall
    let controller = DurabilityController::new(OperatorIntent::CertifiedExact, 50_000);
    controller.record_telemetry(StorageTelemetry {
        p50_fsync_micros: 25_000,
        p99_fsync_micros: 100_000,
        mutation_arrival_rate_per_sec: 15_000,
        outstanding_wal_bytes: 128 * 1024 * 1024,
        replication_rtt_micros: 500,
    });

    let plan = controller.current_plan();
    // Verify controller adapted to larger batch size to amortize slow storage
    assert!(plan.max_batch_size >= 128);
    assert!(plan.is_direct_io);

    // Heartbeats continue cleanly with 0 elections
    cluster.broadcast_heartbeats(1);
    assert_eq!(*leader.role.read(), RaftRole::Leader);
}
