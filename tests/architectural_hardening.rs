//! Architectural Hardening & Production Integrity Suite
//!
//! Verifies the 5 architectural hardening pillars:
//!   1. Consistent Hashing Ring: Bounds remapping to ~1/N during cluster scale-out.
//!   2. Dual-Accumulator SIMD: Validates complex dot product across scalar and vector kernels.
//!   3. Zero-OOM Streamed Compaction: Two-phase pre-allocated compaction with tombstone deduplication.
//!   4. AST Filter Bitmask Compilation: Verifies RoaringBitmap filter execution without dynamic closures.
//!   5. Adversarial Degenerate Manifolds & NaN/Inf Fuzzing: Verifies stability on extreme/corrupted LLM inputs.

use num_complex::Complex32;
use hnsqr::cluster::ConsistentHashRing;
use hnsqr::metadata::index::{FilterExpr, MetadataInvertedIndex};
use hnsqr::proof::lutz::{LutzCode, SemanticRerankPlan};
use hnsqr::proof::{GlobalExactProofSearch, SegmentProofView, SemanticProofTree};
use hnsqr::storage::segment::SegmentedEngine;
use hnsqr::{dot_product_complex_simd, NodeIndex, VectorEmbedding};

// ─────────────────────────────────────────────────────────────────────────────
// 1. CONSISTENT HASHING SCALE-OUT BOUNDS
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_consistent_hashing_scale_out_migration_bound() {
    let initial_shards = 5;
    let mut ring = ConsistentHashRing::new(128);
    for s in 0..initial_shards {
        ring.add_shard(s);
    }

    let n_keys = 20_000;
    let keys: Vec<String> = (0..n_keys).map(|i| format!("tenant_record_{i}")).collect();

    let initial_map: Vec<u32> = keys
        .iter()
        .map(|k| ring.shard_for_key(k).expect("Must map key"))
        .collect();

    // Scale out: Add 6th shard (5 -> 6 shards)
    ring.add_shard(5);

    let mut remapped = 0usize;
    for (i, k) in keys.iter().enumerate() {
        let new_shard = ring.shard_for_key(k).expect("Must map key");
        if new_shard != initial_map[i] {
            // Remapped keys must ONLY move to the new shard 5
            assert_eq!(
                new_shard, 5,
                "Key {k} remapped from {} to unexpected shard {new_shard}",
                initial_map[i]
            );
            remapped += 1;
        }
    }

    let remap_ratio = remapped as f64 / n_keys as f64;
    let theoretical_1_over_n = 1.0 / 6.0; // 16.67%
    // Remap ratio must be tightly bounded around 1/N (12% to 22%), NOT 83% like naive modulo!
    assert!(
        (remap_ratio - theoretical_1_over_n).abs() < 0.04,
        "Remap ratio was {:.2}%, expected ~{:.2}%",
        remap_ratio * 100.0,
        theoretical_1_over_n * 100.0
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. SIMD KERNEL EXACTNESS & ALIGNMENT
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_simd_complex_dot_product_unaligned_and_varied_lengths() {
    let lengths = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 256, 384, 768, 1536];

    for &len in &lengths {
        let a: Vec<Complex32> = (0..len)
            .map(|i| Complex32::new((i as f32 * 0.1).sin(), (i as f32 * 0.2).cos()))
            .collect();
        let b: Vec<Complex32> = (0..len)
            .map(|i| Complex32::new((i as f32 * 0.3).cos(), (i as f32 * 0.4).sin()))
            .collect();

        // Exact scalar calculation
        let mut expected_re = 0.0f32;
        let mut expected_im = 0.0f32;
        for i in 0..len {
            expected_re += a[i].re * b[i].re + a[i].im * b[i].im;
            expected_im += a[i].re * b[i].im - a[i].im * b[i].re;
        }

        let simd_res = dot_product_complex_simd(&a, &b);

        assert!(
            (simd_res.re - expected_re).abs() < 1e-4,
            "Real part mismatch for len {len}: simd={}, expected={}",
            simd_res.re, expected_re
        );
        assert!(
            (simd_res.im - expected_im).abs() < 1e-4,
            "Imag part mismatch for len {len}: simd={}, expected={}",
            simd_res.im, expected_im
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. ZERO-OOM STREAMED COMPACTION & DEDUPLICATION
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_streamed_compaction_memory_and_deduplication() {
    let dim = 16;
    let engine = SegmentedEngine::new(dim, 20);

    // Insert 100 vectors across multiple segment freezes
    for i in 0..100 {
        let v = VectorEmbedding::from_complex(
            (0..dim)
                .map(|d| Complex32::new((i * 3 + d) as f32, (i * 5 + d) as f32))
                .collect(),
        )
        .into_normalized();
        engine.insert(format!("doc_{i}"), v).unwrap();
    }

    // Overwrite docs 0..20 with newer versions
    for i in 0..20 {
        let v = VectorEmbedding::from_complex(
            (0..dim)
                .map(|d| Complex32::new((i * 11 + d) as f32, (i * 13 + d) as f32))
                .collect(),
        )
        .into_normalized();
        engine.insert(format!("doc_{i}"), v).unwrap();
    }

    // Delete docs 80..100
    for i in 80..100 {
        engine.delete(&format!("doc_{i}"));
    }

    // Trigger streamed compaction
    let purged = engine.compact().unwrap();
    assert!(purged >= 40, "Compaction must purge superseded and tombstoned vectors");

    // Search must find doc_0 with its new vector, and not find deleted doc_85
    let query_new_0 = VectorEmbedding::from_complex(
        (0..dim)
            .map(|d| Complex32::new(d as f32, d as f32))
            .collect(),
    )
    .into_normalized();

    let res = engine.search(&query_new_0, 5, SemanticRerankPlan::ExactSimd);
    assert_eq!(res[0].0.as_ref(), "doc_0");
    assert!((res[0].1 - 1.0).abs() < 1e-4);

    let query_deleted = VectorEmbedding::from_complex(
        (0..dim)
            .map(|d| Complex32::new((85 * 3 + d) as f32, (85 * 5 + d) as f32))
            .collect(),
    )
    .into_normalized();

    let res_del = engine.search(&query_deleted, 5, SemanticRerankPlan::ExactSimd);
    for (id, _) in res_del {
        assert_ne!(id.as_ref(), "doc_85", "Deleted documents must never appear in search");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. AST FILTER BITMASK COMPILATION (NO POINTER CLOSURES)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_ast_filter_compilation_and_execution() {
    let index = MetadataInvertedIndex::new();

    // Index 1,000 documents with structured JSON fields
    for i in 0..1000u32 {
        let tenant = if i % 2 == 0 { "tenant_alpha" } else { "tenant_beta" };
        let category = if i % 3 == 0 { "finance" } else if i % 3 == 1 { "tech" } else { "health" };
        let rating = (i % 10) as i64;

        let meta = serde_json::json!({
            "tenant": tenant,
            "category": category,
            "rating": rating
        });
        index.insert_metadata(i, &meta);
    }

    // AST: (tenant == "tenant_alpha" AND category == "tech")
    let expr = FilterExpr::and(vec![
        FilterExpr::eq("tenant", "tenant_alpha"),
        FilterExpr::eq("category", "tech"),
    ]);

    let mask = index.evaluate_filter(&expr, 1000);

    // Verify bitmask correctness
    for i in 0..1000u32 {
        let expected = (i % 2 == 0) && (i % 3 == 1);
        assert_eq!(
            mask.contains(i),
            expected,
            "Filter compilation mismatch at slot {i}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. ADVERSARIAL MANIFOLDS, NAN/INF & DEGENERATE VECTOR FUZZING
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_adversarial_degenerate_manifolds_and_non_finite_fuzzing() {
    let dim = 64;

    // 1. All-Zero Vector
    let zero_vec = VectorEmbedding::from_complex(vec![Complex32::new(0.0, 0.0); dim]);
    let norm_zero = zero_vec.into_normalized();
    assert_eq!(norm_zero.norm(), 0.0);
    let code_zero = LutzCode::encode(&norm_zero, true);
    assert_eq!(code_zero.max_scale_l0, 0.0);

    // 2. Corrupted Vector with NaNs and Infs
    let mut corrupted_data = vec![Complex32::new(1.0, 2.0); dim];
    corrupted_data[0] = Complex32::new(f32::NAN, 1.0);
    corrupted_data[1] = Complex32::new(f32::INFINITY, -f32::INFINITY);
    corrupted_data[2] = Complex32::new(-f32::NAN, 0.0);

    let corrupted_vec = VectorEmbedding::from_complex(corrupted_data);
    let sanitized_norm = corrupted_vec.into_normalized();
    assert!(sanitized_norm.norm().is_finite());
    for &z in sanitized_norm.complex_data() {
        assert!(z.re.is_finite());
        assert!(z.im.is_finite());
    }

    // 3. Subnormal Floats
    let subnormal_vec = VectorEmbedding::from_complex(
        (0..dim)
            .map(|_| Complex32::new(1e-40, 1e-42))
            .collect(),
    )
    .into_normalized();
    assert!(subnormal_vec.norm().is_finite());

    // 4. Rank-1 Single-Coordinate Spike
    let mut spike_data = vec![Complex32::new(0.0, 0.0); dim];
    spike_data[17] = Complex32::new(100.0, 0.0);
    let spike_vec = VectorEmbedding::from_complex(spike_data).into_normalized();
    assert!((spike_vec.norm() - 1.0).abs() < 1e-5);
    assert!((spike_vec.complex_data()[17].re - 1.0).abs() < 1e-5);

    // 5. Build Proof Tree with Degenerate Set
    let corpus = vec![norm_zero, sanitized_norm, subnormal_vec, spike_vec];
    let slots: Vec<NodeIndex> = (0..4).collect();
    let tree = SemanticProofTree::build(&corpus, &slots, dim);
    assert_eq!(tree.total_vectors(), 4);

    let query = corpus[3].clone();
    let seg_view = SegmentProofView {
        tree: &tree,
        vectors: &corpus,
        lutz_codes: None,
        tombstones: None,
    };
    let (res, proof) = GlobalExactProofSearch::search(
        &query,
        2,
        &[seg_view],
        &[],
        &[],
        None,
    );

    assert_eq!(res.len(), 2);
    assert_eq!(res[0].0, 3, "Spike vector must match query at rank 0");
    assert!(proof.globally_exact);
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. MODULE TOPOLOGY INVARIANT: ONLY lib.rs DIRECTLY IN src/
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn src_root_contains_only_lib_rs() {
    let mut illegal = Vec::new();

    for entry in std::fs::read_dir("src").unwrap() {
        let path = entry.unwrap().path();
        if path.is_file()
            && path.extension().is_some_and(|ext| ext == "rs")
            && path.file_name().is_some_and(|name| name != "lib.rs")
        {
            illegal.push(path);
        }
    }

    assert!(
        illegal.is_empty(),
        "Root-level Rust modules are strictly forbidden: {illegal:#?}"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. CONSENSUS SAFETY & ZERO SWALLOWED STORAGE ERRORS INVARIANT
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_no_swallowed_storage_errors_in_consensus_module() {
    let consensus_dir = std::path::Path::new("src/consensus");
    for entry in std::fs::read_dir(consensus_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|ext| ext == "rs") {
            let content = std::fs::read_to_string(&path).unwrap();
            for (line_idx, line) in content.lines().enumerate() {
                if (line.contains("let _ = self.storage") || line.contains("let _ = storage."))
                    && !line.contains("save_progress")
                    && !line.contains("append_entries(&default_entries)")
                {
                    panic!(
                        "Forbidden ignored storage persistence on line {} of {}: '{}'",
                        line_idx + 1,
                        path.display(),
                        line.trim()
                    );
                }
            }
        }
    }
}

#[test]
fn test_uncommitted_persisted_suffix_not_promoted_during_recovery() {
    use std::sync::Arc;
    use hnsqr::cluster::state_machine::{DataMutation, ShardStateMachine};
    use hnsqr::consensus::raft::RaftNode;
    use hnsqr::consensus::storage::DurableRaftStorage;

    let tmp_dir = std::env::temp_dir().join(format!("hnsqr_uncommitted_rec_{}", rand::random::<u64>()));
    std::fs::create_dir_all(&tmp_dir).unwrap();

    let storage = Arc::new(DurableRaftStorage::open(&tmp_dir).unwrap());
    let node = RaftNode::with_storage(1, vec![1, 2, 3], storage.clone());
    *node.role.write() = hnsqr::consensus::raft::RaftRole::Leader;
    *node.current_term.write() = 1;

    let dim = 4;
    let engine = Arc::new(SegmentedEngine::new(dim, 100));
    let sm = Arc::new(ShardStateMachine::new(0, engine.clone()));

    // Leader proposes mutation at index 1 without quorum
    let vec = VectorEmbedding::from_reals(&[1.0, 0.0, 0.0, 0.0]);
    let cmd = hnsqr::consensus::raft::RaftCommand::Data(DataMutation::new_upsert("doc_42", vec.clone()));
    let idx = node.propose(cmd).unwrap();
    assert_eq!(idx, 1);

    // HardState / log are persisted on disk, but commit_index is 0 (no quorum!)
    assert_eq!(*node.commit_index.read(), 0);

    // Node restarts fresh from storage
    let fresh_engine = Arc::new(SegmentedEngine::new(dim, 100));
    let fresh_sm: Arc<dyn hnsqr::cluster::state_machine::ReplicatedStateMachine> =
        Arc::new(ShardStateMachine::new(0, fresh_engine.clone()));
    let recovered = node.recover_node_state(&fresh_sm).unwrap();

    // MUST NOT apply uncommitted entry 1
    assert_eq!(recovered, 0, "Uncommitted log entries must NEVER be applied during recovery!");
    assert_eq!(*node.commit_index.read(), 0);
    assert_eq!(*node.last_applied.read(), 0);

    let search_res = fresh_engine.search(&vec, 1, SemanticRerankPlan::ExactSimd);
    assert!(search_res.is_empty(), "Uncommitted entry must not exist in state machine!");

    let _ = std::fs::remove_dir_all(&tmp_dir);
}

#[test]
fn test_follower_storage_failure_prevents_quorum_commit() {
    use std::sync::Arc;
    use hnsqr::consensus::raft::{AppendEntriesArgs, RaftCluster, RaftCommand, RaftLogEntry};
    use hnsqr::consensus::storage::{MemoryRaftStorage, RaftStorage};

    let storage_a = Arc::new(MemoryRaftStorage::new());
    let storage_b = Arc::new(MemoryRaftStorage::new());
    let storage_c = Arc::new(MemoryRaftStorage::new());

    let mut storages = std::collections::HashMap::new();
    storages.insert(1, storage_a as Arc<dyn RaftStorage>);
    storages.insert(2, storage_b.clone() as Arc<dyn RaftStorage>);
    storages.insert(3, storage_c as Arc<dyn RaftStorage>);

    let cluster = RaftCluster::with_storages(storages);
    assert!(cluster.trigger_election(1));
    let leader = cluster.nodes.get(&1).unwrap();

    // Inject disk failure on follower B
    storage_b.fail_before_log_persist.store(true, std::sync::atomic::Ordering::SeqCst);

    let follower_b = cluster.nodes.get(&2).unwrap();
    let reply = follower_b.handle_append_entries(&AppendEntriesArgs {
        term: 1,
        leader_id: 1,
        prev_log_index: 0,
        prev_log_term: 0,
        entries: vec![RaftLogEntry {
            term: 1,
            index: 1,
            command: RaftCommand::NoOp,
        }],
        leader_commit: 0,
        is_heartbeat: false,
    });

    // Follower B MUST return success=false on storage failure!
    assert!(!reply.success, "Follower with storage failure must reply success=false!");
}

#[test]
fn test_upsert_metadata_preservation_and_patching() {
    use std::collections::HashMap;
    use std::sync::Arc;
    use hnsqr::cluster::state_machine::{DataMutation, ShardStateMachine, ReplicatedStateMachine};
    use hnsqr::metadata::index::MetadataValue;

    let dim = 4;
    let engine = Arc::new(SegmentedEngine::new(dim, 100));
    let sm = ShardStateMachine::new(0, engine.clone());

    let mut meta = HashMap::new();
    meta.insert("author".to_string(), MetadataValue::String("Lord Xyn".to_string()));
    meta.insert("rating".to_string(), MetadataValue::Integer(100));

    let vec = VectorEmbedding::from_reals(&[1.0, 2.0, 3.0, 4.0]);
    let mutation = DataMutation::new_upsert_with_metadata("item_1", vec, Some(meta));
    sm.apply(1, &mutation).unwrap();

    let stored_meta = engine.get_metadata("item_1").expect("Metadata must be stored");
    assert_eq!(stored_meta.get("author").unwrap(), &MetadataValue::String("Lord Xyn".to_string()));
    assert_eq!(stored_meta.get("rating").unwrap(), &MetadataValue::Integer(100));

    // Patch metadata
    let mut patch = HashMap::new();
    patch.insert("rating".to_string(), MetadataValue::Integer(200));
    patch.insert("tag".to_string(), MetadataValue::String("certified".to_string()));

    let patch_mutation = DataMutation::MetadataPatch {
        mutation_id: hnsqr::consensus::pending::MutationId::generate(),
        key: "item_1".to_string(),
        metadata: patch,
    };
    sm.apply(2, &patch_mutation).unwrap();

    let updated_meta = engine.get_metadata("item_1").unwrap();
    assert_eq!(updated_meta.get("rating").unwrap(), &MetadataValue::Integer(200));
    assert_eq!(updated_meta.get("tag").unwrap(), &MetadataValue::String("certified".to_string()));
    assert_eq!(updated_meta.get("author").unwrap(), &MetadataValue::String("Lord Xyn".to_string()));
}

#[test]
fn test_atomic_batch_rollback_on_prevalidation_failure() {
    use std::sync::Arc;
    use hnsqr::cluster::state_machine::{DataMutation, ShardStateMachine, ReplicatedStateMachine};

    let dim = 4;
    let engine = Arc::new(SegmentedEngine::new(dim, 100));
    let sm = ShardStateMachine::new(0, engine.clone());

    let v_valid = VectorEmbedding::from_reals(&[1.0, 1.0, 1.0, 1.0]);
    let v_invalid_dim = VectorEmbedding::from_reals(&[1.0, 1.0]); // Dim mismatch!

    let batch = DataMutation::Batch {
        mutation_id: hnsqr::consensus::pending::MutationId::generate(),
        mutations: vec![
            DataMutation::new_upsert("batch_1", v_valid.clone()),
            DataMutation::new_upsert("batch_2", v_invalid_dim),
        ],
    };

    let res = sm.apply(1, &batch);
    assert!(res.is_err(), "Batch with mismatched dimension must fail prevalidation");

    // Verify batch_1 was NOT inserted (Zero partial visibility)
    let search = engine.search(&v_valid, 10, SemanticRerankPlan::ExactSimd);
    assert!(search.is_empty(), "Failed batch must leave zero partial state!");
}

#[test]
fn test_deduplication_horizon_sequence_window_gap_rejection() {
    use std::sync::Arc;
    use hnsqr::cluster::state_machine::{ClientIdentity, DataMutation, RetrySemantics, ShardStateMachine, ReplicatedStateMachine};

    let dim = 4;
    let engine = Arc::new(SegmentedEngine::new(dim, 100));
    let sm = ShardStateMachine::new(0, engine.clone());

    let client = ClientIdentity {
        tenant_id: "tenant_abc".to_string(),
        client_id: "client_1".to_string(),
    };

    let v = VectorEmbedding::from_reals(&[1.0, 0.0, 0.0, 0.0]);

    // First sequence = 1
    let m1 = DataMutation::Upsert {
        mutation_id: hnsqr::consensus::pending::MutationId::generate(),
        key: "k1".to_string(),
        vector: v.clone(),
        metadata: None,
        client: Some(client.clone()),
        client_seq: 1,
        retry_semantics: RetrySemantics::ExactlyOnceWithinWindow { max_sequence_gap: 10 },
    };
    sm.apply(1, &m1).unwrap();

    // Sequence gap jump = 50 (> max_sequence_gap 10)
    let m_gap = DataMutation::Upsert {
        mutation_id: hnsqr::consensus::pending::MutationId::generate(),
        key: "k2".to_string(),
        vector: v.clone(),
        metadata: None,
        client: Some(client.clone()),
        client_seq: 50,
        retry_semantics: RetrySemantics::ExactlyOnceWithinWindow { max_sequence_gap: 10 },
    };
    let res_gap = sm.apply(2, &m_gap);
    assert!(res_gap.is_err(), "Sequence jump exceeding max_sequence_gap must be rejected");

    // Stale sequence = 0 (< last_seq 1)
    let m_stale = DataMutation::Upsert {
        mutation_id: hnsqr::consensus::pending::MutationId::generate(),
        key: "k3".to_string(),
        vector: v.clone(),
        metadata: None,
        client: Some(client),
        client_seq: 0,
        retry_semantics: RetrySemantics::ExactlyOnceWithinWindow { max_sequence_gap: 10 },
    };
    let res_stale = sm.apply(3, &m_stale);
    assert!(res_stale.is_err(), "Stale sequence must be rejected");
}
