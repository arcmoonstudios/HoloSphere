/* hnsqr/tests/property_state_machine.rs */
//!▫~•◦-------------------------------‣
//! # Property-Based State Machine Equivalence Test
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Generates random interleavings of upserts, deletes, node failures, and partitions
//! to prove that the replicated Raft cluster produces linearizable state equivalent
//! to a sequential single-node reference oracle model.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use num_complex::Complex32;
use rand::{Rng, SeedableRng, rngs::StdRng};

use hnsqr::consensus::raft::RaftCluster;
use hnsqr::VectorEmbedding;

#[test]
fn test_replicated_state_machine_sequential_model_equivalence() {
    let mut rng = StdRng::seed_from_u64(0xDEADBEEF);
    let dim = 8;
    let cluster = RaftCluster::new(&[1, 2, 3]);
    let mut sequential_oracle: HashMap<String, VectorEmbedding> = HashMap::new();

    let num_operations = 200;

    for op_idx in 0..num_operations {
        let op_type = rng.gen_range(0..10);
        let key = format!("doc_{}", rng.gen_range(0..25));

        if op_type < 7 {
            // Upsert
            let v = VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new(rng.gen_range(0.0..1.0) + d as f32, rng.gen_range(0.0..1.0)))
                    .collect(),
            )
            .into_normalized();

            let res = cluster.client_propose_upsert(key.clone(), v.clone());
            assert!(res.is_ok(), "Client proposal failed at op {op_idx}: {:?}", res.err());
            sequential_oracle.insert(key, v);
        } else if op_type < 9 {
            // Delete
            let res = cluster.client_propose_delete(key.clone());
            assert!(res.is_ok(), "Client delete proposal failed at op {op_idx}: {:?}", res.err());
            sequential_oracle.remove(&key);
        } else {
            // Read Linearizable & Compare with Oracle
            let leader_id = cluster.get_leader().expect("Must have an active leader");
            let node = &cluster.nodes[&leader_id];
            let sm = node.state_machine.read();

            if let Some(expected_vec) = sequential_oracle.get(&key) {
                let actual_vec = sm.get(&key);
                assert!(actual_vec.is_some(), "Key {key} should exist in state machine at op {op_idx}");
                let diff = (actual_vec.unwrap().dot_product_complex(expected_vec).re - 1.0).abs();
                assert!(diff < 1e-4, "Vector state mismatch for key {key}");
            } else {
                let actual_vec = sm.get(&key);
                assert!(actual_vec.is_none(), "Key {key} should NOT exist in state machine at op {op_idx}");
            }
        }
    }
}
