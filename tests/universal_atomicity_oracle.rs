/* hnsqr/tests/universal_atomicity_oracle.rs */
//!▫~•◦-------------------------------‣
//! # Universal Multi-Paradigm Atomicity & State Invariant Oracle
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Torture test verifying the Sacred Invariants of HoloSphere:
//! 1. All-or-nothing atomicity across Vector, Graph, Relational SQL, Agent Memory, and Hypercube.
//! 2. $H(S_{\text{before}}) \equiv H(S_{\text{after failed batch}})$ for failure at every child position.
//! 3. Pinned snapshot integrity: physical memory & version retention.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

use hnsqr::cluster::state_machine::{DataMutation, ReplicatedStateMachine, ShardStateMachine};
use hnsqr::consensus::pending::MutationId;
use hnsqr::ecosystem::agent_memory::{AutonomousMemoryConsolidator, EpisodicFact, FactCategory};
use hnsqr::graph::catalog::labels::LabelCatalog;
use hnsqr::graph::catalog::relationships::RelTypeCatalog;
use hnsqr::graph::mutation::apply::GraphMutationApplier;
use hnsqr::graph::mutation::command::GraphMutation;
use hnsqr::graph::storage::generation::GraphGeneration;
use hnsqr::storage::relational_acid::{
    ColumnDefinition, RelationalRow, RelationalSqlEngine, SqlType, SqlValue, TableSchema,
};
use hnsqr::storage::segment::SegmentedEngine;
use hnsqr::vector::hypercube::HypercubeTensorSpace;
use hnsqr::VectorEmbedding;

/// Calculates an exact structural fingerprint across all 5 converged paradigm backends.
fn compute_state_fingerprint(
    engine: &SegmentedEngine,
    graph_gen: &Arc<RwLock<GraphGeneration>>,
    sql: &RelationalSqlEngine,
    memory: &AutonomousMemoryConsolidator,
    hypercube: &HypercubeTensorSpace,
) -> (usize, usize, usize, usize, usize, Option<f32>) {
    let active_vectors = engine.stats().iter().map(|s| s.live_vectors).sum::<usize>();
    let (node_count, edge_count) = {
        let g = graph_gen.read();
        let nodes = g.nodes.node_count();
        let edges = g.edge_delta.as_ref().map(|d| d.len()).unwrap_or(0);
        (nodes, edges)
    };
    let sql_rows = sql.execute_select("accounts", None, None).unwrap_or_default().len();
    let memory_facts = memory
        .get_profile("usr_root")
        .map(|p| p.consolidated_facts.len())
        .unwrap_or(0);
    let voxel_val = hypercube.get_voxel(&[0, 0, 0, 0]);

    (active_vectors, node_count, edge_count, sql_rows, memory_facts, voxel_val)
}

#[test]
fn test_universal_all_or_nothing_atomicity_oracle() {
    let engine = Arc::new(SegmentedEngine::new(8, 1000));
    
    // 1. Initialize Graph Backend
    let graph_generation = Arc::new(RwLock::new(GraphGeneration::new_mutable(1)));
    let label_catalog = Arc::new(LabelCatalog::default());
    let rel_catalog = Arc::new(RelTypeCatalog::default());
    let graph_applier = Arc::new(GraphMutationApplier::new(
        graph_generation.clone(),
        label_catalog.clone(),
        rel_catalog.clone(),
    ));

    // 2. Initialize SQL Backend
    let sql = Arc::new(RelationalSqlEngine::new());
    let table_schema = TableSchema {
        name: "accounts".into(),
        primary_key_column: "acc_id".into(),
        columns: vec![
            ColumnDefinition {
                name: "acc_id".into(),
                data_type: SqlType::Text,
                is_primary_key: true,
                is_nullable: false,
                foreign_key_target: None,
            },
            ColumnDefinition {
                name: "balance".into(),
                data_type: SqlType::Float,
                is_primary_key: false,
                is_nullable: false,
                foreign_key_target: None,
            },
        ],
    };
    sql.create_table(table_schema).unwrap();

    // 3. Initialize Memory and Tensor Backends
    let memory = Arc::new(AutonomousMemoryConsolidator::new());
    let hypercube = Arc::new(HypercubeTensorSpace::new(vec![4, 4, 4, 4]));

    let sm = ShardStateMachine::with_all_paradigms(
        1,
        engine.clone(),
        Some(graph_applier.clone()),
        Some(sql.clone()),
        Some(memory.clone()),
        Some(hypercube.clone()),
    );

    // 4. Commit Initial Baseline State across ALL 5 Paradigms
    let mut initial_row = HashMap::new();
    initial_row.insert("acc_id".into(), SqlValue::Text("acc_init".into()));
    initial_row.insert("balance".into(), SqlValue::Float(100.0));

    let initial_fact = EpisodicFact {
        fact_id: "fact_init".into(),
        subject: "usr_root".into(),
        predicate: "role".into(),
        object: "admin".into(),
        category: FactCategory::UserPreference,
        confidence: 1.0,
        emotional_salience: 0.5,
        recall_count: 1,
        last_accessed_secs: 10,
        created_at_secs: 10,
    };

    let baseline_batch = DataMutation::Batch {
        mutation_id: MutationId::generate(),
        mutations: vec![
            // Paradigm 1: Vector Upsert
            DataMutation::new_upsert("acc_init", VectorEmbedding::from_reals(&[0.5; 8]).into_normalized()),
            // Paradigm 2: Graph Node Creation
            DataMutation::new_graph(GraphMutation::CreateNode {
                external_id: "node_init".into(),
                labels: vec![1],
                properties: HashMap::new(),
                vector_slot: Some(0),
            }),
            // Paradigm 3: Relational SQL Row Insert
            DataMutation::new_sql_insert("accounts", RelationalRow { values: initial_row }),
            // Paradigm 4: Agent Memory Fact Append
            DataMutation::new_agent_memory("usr_root", initial_fact),
            // Paradigm 5: Hypercube Tensor Voxel Set
            DataMutation::new_hypercube_voxel(vec![0, 0, 0, 0], 1.0),
        ],
    };

    let base_receipt = sm.apply(100, &baseline_batch).expect("Baseline commit must succeed");
    assert_eq!(base_receipt.applied_index, 100);

    let baseline_fingerprint = compute_state_fingerprint(&engine, &graph_generation, &sql, &memory, &hypercube);
    assert_eq!(baseline_fingerprint, (1, 1, 0, 1, 1, Some(1.0)));

    let baseline_snapshot = sm.pin_universal_snapshot();
    let baseline_physical_snap = sm.pin_physical_snapshot();
    assert_eq!(baseline_physical_snap.generation, baseline_snapshot.generation);

    // 5. Failure Case A: Child 1 (Vector) has Invalid Dimension (99 != 8)
    let mut valid_row = HashMap::new();
    valid_row.insert("acc_id".into(), SqlValue::Text("acc_failed_a".into()));
    valid_row.insert("balance".into(), SqlValue::Float(500.0));

    let batch_fail_child_1 = DataMutation::Batch {
        mutation_id: MutationId::generate(),
        mutations: vec![
            // Child 1 FAILS: 99D real vector on 8D engine
            DataMutation::new_upsert("acc_failed_a", VectorEmbedding::from_reals(&[0.1; 99])),
            DataMutation::new_sql_insert("accounts", RelationalRow { values: valid_row.clone() }),
            DataMutation::new_hypercube_voxel(vec![1, 1, 1, 1], 99.0),
        ],
    };

    assert!(sm.apply(101, &batch_fail_child_1).is_err());
    assert_eq!(sm.last_applied_index(), 100);
    assert_eq!(sm.applied_generation(), baseline_snapshot.generation);
    assert_eq!(compute_state_fingerprint(&engine, &graph_generation, &sql, &memory, &hypercube), baseline_fingerprint);

    // 6. Failure Case B: Child 2 (SQL) References Non-Existent Table
    let batch_fail_child_2 = DataMutation::Batch {
        mutation_id: MutationId::generate(),
        mutations: vec![
            DataMutation::new_upsert("acc_failed_b", VectorEmbedding::from_reals(&[0.1; 8]).into_normalized()),
            // Child 2 FAILS: "non_existent_table"
            DataMutation::new_sql_insert("non_existent_table", RelationalRow { values: valid_row.clone() }),
            DataMutation::new_hypercube_voxel(vec![1, 1, 1, 1], 99.0),
        ],
    };

    assert!(sm.apply(102, &batch_fail_child_2).is_err());
    assert_eq!(sm.last_applied_index(), 100);
    assert_eq!(sm.applied_generation(), baseline_snapshot.generation);
    assert_eq!(compute_state_fingerprint(&engine, &graph_generation, &sql, &memory, &hypercube), baseline_fingerprint);

    // 7. Failure Case C: Child 2 (SQL) Row Lacks Mandatory Primary Key
    let mut invalid_pk_row = HashMap::new();
    invalid_pk_row.insert("balance".into(), SqlValue::Float(999.0)); // Missing "acc_id"

    let batch_fail_missing_pk = DataMutation::Batch {
        mutation_id: MutationId::generate(),
        mutations: vec![
            DataMutation::new_upsert("acc_failed_c", VectorEmbedding::from_reals(&[0.1; 8]).into_normalized()),
            // Child 2 FAILS: Missing Primary Key
            DataMutation::new_sql_insert("accounts", RelationalRow { values: invalid_pk_row }),
            DataMutation::new_hypercube_voxel(vec![1, 1, 1, 1], 99.0),
        ],
    };

    assert!(sm.apply(103, &batch_fail_missing_pk).is_err());
    assert_eq!(sm.last_applied_index(), 100);
    assert_eq!(sm.applied_generation(), baseline_snapshot.generation);
    assert_eq!(compute_state_fingerprint(&engine, &graph_generation, &sql, &memory, &hypercube), baseline_fingerprint);

    // 8. Failure Case D: Child 3 (AgentMemory) has Invalid Emotional Salience (1.99 > 1.0)
    let invalid_fact = EpisodicFact {
        fact_id: "fact_err".into(),
        subject: "usr_root".into(),
        predicate: "quota".into(),
        object: "unlimited".into(),
        category: FactCategory::UserPreference,
        confidence: 0.9,
        emotional_salience: 1.99, // INVALID (> 1.0)
        recall_count: 1,
        last_accessed_secs: 10,
        created_at_secs: 10,
    };

    let batch_fail_child_3 = DataMutation::Batch {
        mutation_id: MutationId::generate(),
        mutations: vec![
            DataMutation::new_upsert("acc_failed_d", VectorEmbedding::from_reals(&[0.1; 8]).into_normalized()),
            DataMutation::new_sql_insert("accounts", RelationalRow { values: valid_row.clone() }),
            // Child 3 FAILS: Invalid salience
            DataMutation::new_agent_memory("usr_root", invalid_fact),
            DataMutation::new_hypercube_voxel(vec![1, 1, 1, 1], 99.0),
        ],
    };

    assert!(sm.apply(104, &batch_fail_child_3).is_err());
    assert_eq!(sm.last_applied_index(), 100);
    assert_eq!(sm.applied_generation(), baseline_snapshot.generation);
    assert_eq!(compute_state_fingerprint(&engine, &graph_generation, &sql, &memory, &hypercube), baseline_fingerprint);

    // 9. Failure Case E: Child 4 (Hypercube) has Invalid Coordinate Dimension (5D != 4D)
    let batch_fail_child_4 = DataMutation::Batch {
        mutation_id: MutationId::generate(),
        mutations: vec![
            DataMutation::new_upsert("acc_failed_e", VectorEmbedding::from_reals(&[0.1; 8]).into_normalized()),
            DataMutation::new_sql_insert("accounts", RelationalRow { values: valid_row.clone() }),
            // Child 4 FAILS: 5 coordinates passed to 4D tensor space
            DataMutation::new_hypercube_voxel(vec![1, 2, 3, 4, 5], 99.0),
        ],
    };

    assert!(sm.apply(105, &batch_fail_child_4).is_err());
    assert_eq!(sm.last_applied_index(), 100);
    assert_eq!(sm.applied_generation(), baseline_snapshot.generation);
    assert_eq!(compute_state_fingerprint(&engine, &graph_generation, &sql, &memory, &hypercube), baseline_fingerprint);

    // 10. Verify Pinnacle Invariant: State remains bit-for-bit identical to baseline
    let post_failure_snapshot = sm.pin_universal_snapshot();
    assert_eq!(baseline_snapshot, post_failure_snapshot);
    assert_eq!(
        compute_state_fingerprint(&engine, &graph_generation, &sql, &memory, &hypercube),
        (1, 1, 0, 1, 1, Some(1.0))
    );
}
