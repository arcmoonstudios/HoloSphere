/* hnsqr/tests/end_to_end_durability_and_consensus.rs */
//!▫~•◦-------------------------------‣
//! # End-to-End WAL Durability & Raft Distributed Consensus Ingestion Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - Real WAL logging on live `HNSQRIndex::insert` and `HNSQRIndex::remove`
//!   - Crash recovery via `HNSQRIndex::recover_from_wal` restoring 100% of data and exact recall
//!   - `SegmentedEngine` WAL integration and crash replay
//!   - `DistributedCoordinator::insert_fenced` proposing real `WalMutation` payloads through Raft
//!     leader consensus and applying them to shard state machines
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;
use hnsqr::cluster::DistributedCoordinator;
use hnsqr::consensus::raft::{RaftCommand, RaftRole};
use hnsqr::proof::lutz::SemanticRerankPlan;
use hnsqr::storage::segment::SegmentedEngine;
use hnsqr::storage::wal::{DurabilityPolicy, WalManager};
use hnsqr::{HNSQRConfig, HNSQRIndex, VectorEmbedding};
use num_complex::Complex32;

#[test]
fn test_single_node_wal_crash_recovery_preserves_exact_search() {
    let tmp = std::env::temp_dir().join(format!("hnsqr_wal_e2e_{:x}", rand::random::<u64>()));
    std::fs::create_dir_all(&tmp).unwrap();

    let dim = 16;
    let config = HNSQRConfig {
        max_elements: 1000,
        ..Default::default()
    };

    // 1. Start live index with WAL attached
    {
        let index = HNSQRIndex::new(config.clone(), dim)
            .with_wal(&tmp, DurabilityPolicy::WalSync)
            .unwrap();

        // Insert 20 vectors
        for i in 0..20 {
            let coords: Vec<Complex32> = (0..dim)
                .map(|d| Complex32::new((i * 7 + d) as f32, (i * 11 + d) as f32))
                .collect();
            let vec = VectorEmbedding::from_complex(coords).into_normalized();
            index.insert(format!("doc_{i}"), vec).unwrap();
        }

        // Delete doc_5
        assert!(index.remove("doc_5").unwrap());

        // Process "crashes" here by dropping `index` from RAM without taking a snapshot!
    }

    // 2. Restart fresh process and recover from WAL
    {
        let new_index = HNSQRIndex::new(config, dim)
            .with_wal(&tmp, DurabilityPolicy::WalSync)
            .unwrap();

        let replayed = new_index.recover_from_wal().unwrap();
        assert_eq!(replayed, 21, "Must replay 20 inserts + 1 delete from WAL");

        // Verify doc_5 is deleted
        assert!(new_index.get_node("doc_5").is_err());

        // Verify doc_10 exists and search returns exact top-1 match
        let query_coords: Vec<Complex32> = (0..dim)
            .map(|d| Complex32::new((10 * 7 + d) as f32, (10 * 11 + d) as f32))
            .collect();
        let query = VectorEmbedding::from_complex(query_coords).into_normalized();

        let results = new_index.search(&query, 3).unwrap();
        assert_eq!(results[0].0.as_ref(), "doc_10");
        assert!((results[0].1 - 1.0).abs() < 1e-4);
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_segmented_engine_wal_persistence_and_replay() {
    let tmp = std::env::temp_dir().join(format!("hnsqr_seg_wal_{:x}", rand::random::<u64>()));
    std::fs::create_dir_all(&tmp).unwrap();

    let dim = 8;
    let wal = Arc::new(WalManager::open(&tmp).unwrap());

    // 1. Active writes to SegmentedEngine
    {
        let engine = SegmentedEngine::new(dim, 10).with_wal(wal.clone(), DurabilityPolicy::WalSync);

        for i in 0..15 {
            let coords: Vec<Complex32> = (0..dim).map(|d| Complex32::new((i + d) as f32, 0.0)).collect();
            let vec = VectorEmbedding::from_complex(coords).into_normalized();
            engine.insert(format!("seg_{i}"), vec).unwrap();
        }
    }

    // 2. Recover into new engine from WAL
    {
        let new_engine = SegmentedEngine::new(dim, 10).with_wal(wal, DurabilityPolicy::WalSync);
        let replayed = new_engine.recover_from_wal().unwrap();
        assert_eq!(replayed, 15);

        let query_coords: Vec<Complex32> = (0..dim).map(|d| Complex32::new((3 + d) as f32, 0.0)).collect();
        let query = VectorEmbedding::from_complex(query_coords).into_normalized();

        let topk = new_engine.search(&query, 1, SemanticRerankPlan::ExactSimd);
        assert_eq!(topk[0].0.as_ref(), "seg_3");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn test_distributed_coordinator_raft_replicated_ingest() {
    let dim = 8;
    let coord = DistributedCoordinator::new(dim, 4, 100);

    let leader = coord.raft_cluster.nodes.get(&1).unwrap();
    assert_eq!(*leader.role.read(), RaftRole::Leader);

    // Ingest vectors via fenced coordinator
    for i in 0..10 {
        let coords: Vec<Complex32> = (0..dim)
            .map(|d| Complex32::new((i * 3 + d) as f32, (i * 5 + d) as f32))
            .collect();
        let vec = VectorEmbedding::from_complex(coords).into_normalized();
        coord.insert_fenced(format!("cluster_doc_{i}"), vec, Some(1)).unwrap();
    }

    // Verify Raft leader proposed and committed mutations containing vector data
    let log = leader.log.read();
    assert!(log.len() >= 10);
    let has_reified_mutations = log.iter().any(|entry| {
        matches!(&entry.command,
            RaftCommand::Data(hnsqr::cluster::state_machine::DataMutation::Upsert { .. })
        )
    });
    assert!(has_reified_mutations, "Raft log entries must contain actual vector data payloads");

    // Query cluster
    let query_coords: Vec<Complex32> = (0..dim)
        .map(|d| Complex32::new((7 * 3 + d) as f32, (7 * 5 + d) as f32))
        .collect();
    let query = VectorEmbedding::from_complex(query_coords).into_normalized();

    let search_res = coord.search(&query, 1, SemanticRerankPlan::ExactSimd);
    assert_eq!(search_res[0].0.as_ref(), "cluster_doc_7");
}
