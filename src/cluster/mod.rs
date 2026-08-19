/* hnsqr/src/cluster/mod.rs */
//!▫~•◦-------------------------------‣
//! # Distributed Consensus, Epoch Fencing & Online Shard Migration Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod control_plane;
pub mod coordinator;
pub mod disaster_recovery;
pub mod federation;
pub mod migration;
pub mod ring;
pub mod serverless;
pub mod state_machine;
pub mod stream_ingest;
pub mod topology;

pub use control_plane::{
    ClusterId, ControlPlaneReconciliationPlan, DBaaSControlPlane, DesiredClusterState,
    ObservedClusterState, OrganizationId, TenantUsageReport, UsageBillingMeter, VpcPeeringConfig,
};
pub use coordinator::{DistributedCoordinator, LocalShard};
pub use disaster_recovery::{DisasterRecoveryCoordinator, DisasterRecoverySla};
pub use federation::{
    CrossRegionReplicator, FederatedMutationEvent, FederatedRegionManager, GeoRoutingTable,
    RegionEndpointStatus, VectorClockTimestamp,
};
pub use migration::{MigrationPhase, MigrationTask};
pub use ring::{ConsistentHashRing, ShardId};
pub use serverless::{EphemeralWorker, ServerlessQueryRouter, WorkerState};
pub use state_machine::{
    ApplyReceipt, ClientIdentity, DataMutation, DeduplicationHorizon, ReplicatedStateMachine,
    RetrySemantics, ShardStateMachine, UniversalSnapshot, UniversalSnapshotHandle,
};
pub use stream_ingest::{AsyncLogStreamIngestor, StreamIngestStats};
pub use topology::{ClusterTopology, NodeAddress, ShardReplica, ShardRole};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorEmbedding;
    use num_complex::Complex32;

    #[test]
    fn test_consistent_hash_ring_migration_bound() {
        let num_initial_shards = 5;
        let mut ring = ConsistentHashRing::new(128);
        for s in 0..num_initial_shards {
            ring.add_shard(s);
        }

        let n_keys = 10_000;
        let keys: Vec<String> = (0..n_keys).map(|i| format!("key_entity_{i}")).collect();

        let initial_placements: Vec<ShardId> = keys
            .iter()
            .map(|k| ring.shard_for_key(k).expect("Must map key"))
            .collect();

        // Add 6th shard (5 -> 6 shards)
        ring.add_shard(5);

        let mut remapped = 0usize;
        for (i, k) in keys.iter().enumerate() {
            let new_shard = ring.shard_for_key(k).expect("Must map key");
            if new_shard != initial_placements[i] {
                assert_eq!(new_shard, 5, "Keys must only remap to the new shard!");
                remapped += 1;
            }
        }

        let remap_ratio = remapped as f64 / n_keys as f64;
        let theoretical_1_over_n = 1.0 / 6.0; // ~16.67%
        assert!(
            (remap_ratio - theoretical_1_over_n).abs() < 0.05,
            "Remap ratio was {:.2}%, expected ~{:.2}%",
            remap_ratio * 100.0,
            theoretical_1_over_n * 100.0
        );
    }

    #[test]
    fn test_distributed_epoch_fenced_writes() {
        let dim = 8;
        let num_shards = 4;
        let coordinator = DistributedCoordinator::new(dim, num_shards, 10);
        let cur_epoch = coordinator.epoch();

        let v = VectorEmbedding::from_complex(
            (0..dim)
                .map(|d| Complex32::new(d as f32, d as f32))
                .collect(),
        )
        .into_normalized();

        // Valid epoch write
        assert!(coordinator.insert_fenced_blocking("doc_1", v.clone(), Some(cur_epoch)).is_ok());

        // Stale epoch write rejected
        let stale_result = coordinator.insert_fenced_blocking("doc_2", v, Some(cur_epoch - 1));
        assert!(stale_result.is_err(), "Stale epoch writes must be rejected!");
    }

    #[test]
    fn test_online_migration_protocol_flow() {
        let dim = 8;
        let num_shards = 4;
        let coordinator = DistributedCoordinator::new(dim, num_shards, 10);
        let initial_epoch = coordinator.epoch();

        // Execute migration
        assert!(coordinator.execute_migration(101, 0, 1).is_ok());
        assert_eq!(coordinator.epoch(), initial_epoch + 1);
    }
}
