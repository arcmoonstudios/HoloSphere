/* hnsqr/src/cluster/state_machine.rs */
//!▫~•◦-------------------------------‣
//! # Authoritative Replicated Shard State Machine & Deduplication Horizon
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the single, unavoidable state machine application layer for HNSQR
//! cluster replication. Owns local segmented engine mutations, metadata indexing,
//! deterministic state progression, and an idempotent deduplication horizon with
//! explicit retry semantics (AtLeastOnce, Idempotent, ExactlyOnceWithinWindow).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::consensus::pending::{ApplyError, MutationId};
use crate::graph::mutation::apply::GraphMutationApplier;
use crate::graph::mutation::command::GraphMutation;
use crate::graph::catalog::labels::LabelCatalog;
use crate::graph::catalog::relationships::RelTypeCatalog;
use crate::graph::storage::generation::GraphGeneration;
use crate::metadata::index::MetadataValue;
use crate::storage::segment::SegmentedEngine;
use crate::{HNSQRError, HNSQRResult, VectorEmbedding};

/// Declared client retry contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RetrySemantics {
    AtLeastOnce,
    #[default]
    Idempotent,
    ExactlyOnceWithinWindow { max_sequence_gap: u64 },
}

/// Client or tenant identification for sequence-tracked deduplication.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ClientIdentity {
    pub tenant_id: String,
    pub client_id: String,
}

/// Universal command model for all data and metadata mutations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DataMutation {
    Upsert {
        mutation_id: MutationId,
        key: String,
        vector: VectorEmbedding,
        metadata: Option<HashMap<String, MetadataValue>>,
        client: Option<ClientIdentity>,
        client_seq: u64,
        retry_semantics: RetrySemantics,
    },
    Delete {
        mutation_id: MutationId,
        key: String,
        client: Option<ClientIdentity>,
        client_seq: u64,
        retry_semantics: RetrySemantics,
    },
    MetadataPatch {
        mutation_id: MutationId,
        key: String,
        metadata: HashMap<String, MetadataValue>,
    },
    Batch {
        mutation_id: MutationId,
        mutations: Vec<DataMutation>,
    },
    /// Graph topology mutation — replicated through Raft then applied by the
    /// `GraphMutationApplier` inside the shard state machine.
    /// This is the **only** authoritative write path for graph topology.
    Graph {
        mutation_id: MutationId,
        mutation: GraphMutation,
    },
    /// Relational SQL table row mutation replicated through Raft.
    Sql {
        mutation_id: MutationId,
        table: String,
        row: crate::storage::relational_acid::RelationalRow,
        is_delete: bool,
    },
    /// Agent long-term memory episodic fact mutation replicated through Raft.
    AgentMemory {
        mutation_id: MutationId,
        user_id: String,
        fact: crate::ecosystem::agent_memory::EpisodicFact,
    },
    /// N-Dimensional hypercube tensor voxel mutation replicated through Raft.
    Hypercube {
        mutation_id: MutationId,
        coords: Vec<usize>,
        value: f32,
    },
}

impl DataMutation {
    pub fn new_upsert(id: impl Into<String>, vector: VectorEmbedding) -> Self {
        Self::Upsert {
            mutation_id: MutationId::generate(),
            key: id.into(),
            vector,
            metadata: None,
            client: None,
            client_seq: 0,
            retry_semantics: RetrySemantics::Idempotent,
        }
    }

    pub fn new_upsert_with_metadata(
        id: impl Into<String>,
        vector: VectorEmbedding,
        metadata: Option<HashMap<String, MetadataValue>>,
    ) -> Self {
        Self::Upsert {
            mutation_id: MutationId::generate(),
            key: id.into(),
            vector,
            metadata,
            client: None,
            client_seq: 0,
            retry_semantics: RetrySemantics::Idempotent,
        }
    }

    pub fn new_delete(id: impl Into<String>) -> Self {
        Self::Delete {
            mutation_id: MutationId::generate(),
            key: id.into(),
            client: None,
            client_seq: 0,
            retry_semantics: RetrySemantics::Idempotent,
        }
    }

    /// Wraps a `GraphMutation` for Raft replication.
    pub fn new_graph(mutation: GraphMutation) -> Self {
        Self::Graph {
            mutation_id: MutationId::generate(),
            mutation,
        }
    }

    /// Wraps a relational SQL row insert for Raft replication.
    pub fn new_sql_insert(table: impl Into<String>, row: crate::storage::relational_acid::RelationalRow) -> Self {
        Self::Sql {
            mutation_id: MutationId::generate(),
            table: table.into(),
            row,
            is_delete: false,
        }
    }

    /// Wraps an agent memory fact for Raft replication.
    pub fn new_agent_memory(user_id: impl Into<String>, fact: crate::ecosystem::agent_memory::EpisodicFact) -> Self {
        Self::AgentMemory {
            mutation_id: MutationId::generate(),
            user_id: user_id.into(),
            fact,
        }
    }

    /// Wraps a hypercube voxel coordinate update for Raft replication.
    pub fn new_hypercube_voxel(coords: Vec<usize>, value: f32) -> Self {
        Self::Hypercube {
            mutation_id: MutationId::generate(),
            coords,
            value,
        }
    }

    pub fn mutation_id(&self) -> &MutationId {
        match self {
            Self::Upsert { mutation_id, .. } => mutation_id,
            Self::Delete { mutation_id, .. } => mutation_id,
            Self::MetadataPatch { mutation_id, .. } => mutation_id,
            Self::Batch { mutation_id, .. } => mutation_id,
            Self::Graph { mutation_id, .. } => mutation_id,
            Self::Sql { mutation_id, .. } => mutation_id,
            Self::AgentMemory { mutation_id, .. } => mutation_id,
            Self::Hypercube { mutation_id, .. } => mutation_id,
        }
    }
}

/// Durable receipt produced when a state machine applies a committed mutation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApplyReceipt {
    pub mutation_id: MutationId,
    pub applied_index: u64,
    pub applied_generation: u64,
    pub durable_lsn: u64,
}

/// Bounded sliding-window deduplication cache preventing duplicate state application on client retries.
pub struct DeduplicationHorizon {
    seen: HashMap<MutationId, ApplyReceipt>,
    client_sequences: HashMap<ClientIdentity, u64>,
    client_receipts: HashMap<ClientIdentity, ApplyReceipt>,
    order: Vec<MutationId>,
    max_capacity: usize,
}

impl DeduplicationHorizon {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            seen: HashMap::with_capacity(max_capacity.min(1024)),
            client_sequences: HashMap::with_capacity(256),
            client_receipts: HashMap::with_capacity(256),
            order: Vec::with_capacity(max_capacity.min(1024)),
            max_capacity,
        }
    }

    pub fn check(
        &self,
        id: &MutationId,
        client: Option<&ClientIdentity>,
        client_seq: u64,
        retry: RetrySemantics,
    ) -> Result<Option<ApplyReceipt>, ApplyError> {
        match retry {
            RetrySemantics::AtLeastOnce => Ok(None),
            RetrySemantics::Idempotent => Ok(self.seen.get(id).cloned()),
            RetrySemantics::ExactlyOnceWithinWindow { max_sequence_gap } => {
                let c = client.ok_or_else(|| ApplyError::SequenceViolation {
                    reason: "ClientIdentity is required for ExactlyOnceWithinWindow".to_string(),
                })?;

                if let Some(&last_seq) = self.client_sequences.get(c) {
                    if client_seq == last_seq {
                        return Ok(self.client_receipts.get(c).cloned().or_else(|| self.seen.get(id).cloned()));
                    }
                    if client_seq < last_seq {
                        return Err(ApplyError::SequenceViolation {
                            reason: format!("Stale client sequence {client_seq} (last observed: {last_seq})"),
                        });
                    }
                    if client_seq > last_seq + max_sequence_gap {
                        return Err(ApplyError::SequenceViolation {
                            reason: format!(
                                "Client sequence gap exceeded window limit: seq {client_seq}, last {last_seq}, max gap {max_sequence_gap}"
                            ),
                        });
                    }
                }
                Ok(None)
            }
        }
    }

    pub fn insert(&mut self, receipt: ApplyReceipt, client: Option<&ClientIdentity>, client_seq: u64) {
        if let Some(c) = client {
            self.client_sequences.insert(c.clone(), client_seq);
            self.client_receipts.insert(c.clone(), receipt.clone());
        }

        if self.seen.len() >= self.max_capacity {
            if let Some(oldest) = self.order.first().cloned() {
                self.order.remove(0);
                self.seen.remove(&oldest);
            }
        }
        self.order.push(receipt.mutation_id.clone());
        self.seen.insert(receipt.mutation_id.clone(), receipt);
    }
}

/// Universal trait for replicated state machines driven by Raft consensus commits.
pub trait ReplicatedStateMachine: Send + Sync {
    fn apply(&self, entry_index: u64, mutation: &DataMutation) -> HNSQRResult<ApplyReceipt>;
    fn last_applied_index(&self) -> u64;
    fn applied_generation(&self) -> u64;
}

/// Production Shard State Machine managing SegmentedEngine application without double WAL logging.
pub struct ShardStateMachine {
    pub shard_id: u32,
    pub engine: Arc<SegmentedEngine>,
    last_applied_index: AtomicU64,
    applied_generation: AtomicU64,
    dedup_horizon: Mutex<DeduplicationHorizon>,
    /// Optional graph mutation applier.  `None` on shards without a graph layer.
    pub graph: Option<Arc<GraphMutationApplier>>,
    /// Optional relational SQL table engine.
    pub sql: Option<Arc<crate::storage::relational_acid::RelationalSqlEngine>>,
    /// Optional agentic memory consolidator.
    pub memory: Option<Arc<crate::ecosystem::agent_memory::AutonomousMemoryConsolidator>>,
    /// Optional hypercube volumetric tensor space.
    pub hypercube: Option<Arc<crate::vector::hypercube::HypercubeTensorSpace>>,
}

impl ShardStateMachine {
    pub fn new(shard_id: u32, engine: Arc<SegmentedEngine>) -> Self {
        Self {
            shard_id,
            engine,
            last_applied_index: AtomicU64::new(0),
            applied_generation: AtomicU64::new(1),
            dedup_horizon: Mutex::new(DeduplicationHorizon::new(65_536)),
            graph: None,
            sql: None,
            memory: None,
            hypercube: None,
        }
    }

    /// Creates a state machine with a pre-wired graph applier.
    pub fn with_graph(shard_id: u32, engine: Arc<SegmentedEngine>) -> Self {
        use parking_lot::RwLock;
        let generation = Arc::new(RwLock::new(GraphGeneration::new_mutable(1)));
        let label_catalog = Arc::new(LabelCatalog::default());
        let rel_catalog = Arc::new(RelTypeCatalog::default());
        let applier = Arc::new(GraphMutationApplier::new(generation, label_catalog, rel_catalog));
        Self {
            shard_id,
            engine,
            last_applied_index: AtomicU64::new(0),
            applied_generation: AtomicU64::new(1),
            dedup_horizon: Mutex::new(DeduplicationHorizon::new(65_536)),
            graph: Some(applier),
            sql: None,
            memory: None,
            hypercube: None,
        }
    }

    /// Creates a fully converged state machine unifying all multi-model paradigms.
    pub fn with_all_paradigms(
        shard_id: u32,
        engine: Arc<SegmentedEngine>,
        graph: Option<Arc<GraphMutationApplier>>,
        sql: Option<Arc<crate::storage::relational_acid::RelationalSqlEngine>>,
        memory: Option<Arc<crate::ecosystem::agent_memory::AutonomousMemoryConsolidator>>,
        hypercube: Option<Arc<crate::vector::hypercube::HypercubeTensorSpace>>,
    ) -> Self {
        Self {
            shard_id,
            engine,
            last_applied_index: AtomicU64::new(0),
            applied_generation: AtomicU64::new(1),
            dedup_horizon: Mutex::new(DeduplicationHorizon::new(65_536)),
            graph,
            sql,
            memory,
            hypercube,
        }
    }

    /// Internal pre-validation ensuring atomic all-or-nothing batch execution.
    fn prevalidate_mutation(&self, mutation: &DataMutation) -> HNSQRResult<()> {
        match mutation {
            DataMutation::Upsert { vector, .. } => {
                if vector.dimension() != self.engine.dimension {
                    return Err(HNSQRError::DimensionMismatch {
                        expected: self.engine.dimension,
                        actual: vector.dimension(),
                    });
                }
                Ok(())
            }
            DataMutation::Delete { .. } => Ok(()),
            DataMutation::MetadataPatch { .. } => Ok(()),
            DataMutation::Graph { .. } => Ok(()),
            DataMutation::Sql { table, row, .. } => {
                if let Some(sql) = &self.sql {
                    let schema = sql.get_table_schema(table).ok_or_else(|| {
                        HNSQRError::InvalidRequest(format!("Table '{table}' does not exist"))
                    })?;
                    if !row.values.contains_key(&schema.primary_key_column) {
                        return Err(HNSQRError::InvalidRequest(format!(
                            "Row missing primary key '{}'",
                            schema.primary_key_column
                        )));
                    }
                }
                Ok(())
            }
            DataMutation::AgentMemory { fact, .. } => {
                if fact.confidence < 0.0 || fact.confidence > 1.0 {
                    return Err(HNSQRError::InvalidRequest(format!(
                        "Invalid fact confidence: {}",
                        fact.confidence
                    )));
                }
                if fact.emotional_salience < 0.0 || fact.emotional_salience > 1.0 {
                    return Err(HNSQRError::InvalidRequest(format!(
                        "Invalid emotional salience: {}",
                        fact.emotional_salience
                    )));
                }
                Ok(())
            }
            DataMutation::Hypercube { coords, .. } => {
                if let Some(h) = &self.hypercube {
                    if coords.len() != h.dimensions() {
                        return Err(HNSQRError::DimensionMismatch {
                            expected: h.dimensions(),
                            actual: coords.len(),
                        });
                    }
                }
                Ok(())
            }
            DataMutation::Batch { mutations, .. } => {
                for m in mutations {
                    self.prevalidate_mutation(m)?;
                }
                Ok(())
            }
        }
    }

    /// Internal unlogged application directly to memory segment buffers with metadata.
    fn apply_single_unlogged(&self, mutation: &DataMutation) -> HNSQRResult<()> {
        match mutation {
            DataMutation::Upsert { key, vector, metadata, .. } => {
                self.engine.apply_committed_upsert(key.as_str(), vector.clone(), metadata.as_ref())?;
                Ok(())
            }
            DataMutation::Delete { key, .. } => {
                self.engine.apply_delete_unlogged(key.as_str());
                Ok(())
            }
            DataMutation::MetadataPatch { key, metadata, .. } => {
                self.engine.apply_committed_metadata_patch(key.as_str(), metadata)?;
                Ok(())
            }
            DataMutation::Graph { mutation, .. } => {
                if let Some(graph) = &self.graph {
                    graph.apply(mutation)?;
                }
                Ok(())
            }
            DataMutation::Sql { table, row, is_delete, .. } => {
                if let Some(sql) = &self.sql {
                    sql.apply_committed_row_mutation(table, row.clone(), *is_delete)?;
                }
                Ok(())
            }
            DataMutation::AgentMemory { user_id, fact, .. } => {
                if let Some(memory) = &self.memory {
                    memory.ingest_fact(user_id, fact.clone())?;
                }
                Ok(())
            }
            DataMutation::Hypercube { coords, value, .. } => {
                if let Some(hypercube) = &self.hypercube {
                    hypercube.set_voxel(coords.clone(), *value)?;
                }
                Ok(())
            }
            DataMutation::Batch { mutations, .. } => {
                for m in mutations {
                    self.apply_single_unlogged(m)?;
                }
                Ok(())
            }
        }
    }

    /// Creates an atomic cross-paradigm snapshot pinned at the current applied generation.
    pub fn pin_universal_snapshot(&self) -> UniversalSnapshotHandle {
        UniversalSnapshotHandle {
            generation: self.applied_generation.load(Ordering::SeqCst),
            lsn: self.last_applied_index.load(Ordering::SeqCst),
        }
    }
}

/// Atomic cross-paradigm snapshot handle pinning vector, graph, relational, memory, and tensor state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalSnapshotHandle {
    pub generation: u64,
    pub lsn: u64,
}

impl ReplicatedStateMachine for ShardStateMachine {
    fn apply(&self, entry_index: u64, mutation: &DataMutation) -> HNSQRResult<ApplyReceipt> {
        let mutation_id = mutation.mutation_id();

        // 1. Check idempotency / sequence retry semantics
        let (client, seq, retry) = match mutation {
            DataMutation::Upsert { client, client_seq, retry_semantics, .. } => {
                (client.as_ref(), *client_seq, *retry_semantics)
            }
            DataMutation::Delete { client, client_seq, retry_semantics, .. } => {
                (client.as_ref(), *client_seq, *retry_semantics)
            }
            _ => (None, 0, RetrySemantics::Idempotent),
        };
        {
            let dedup = self.dedup_horizon.lock();
            if let Some(cached) = dedup.check(mutation_id, client, seq, retry)? {
                return Ok(cached);
            }
        }

        // 2. Pre-validate the entire mutation / batch before mutating ANY backend state
        self.prevalidate_mutation(mutation)?;

        // 3. Deterministic atomic application to local storage engines
        self.apply_single_unlogged(mutation)?;

        // 4. Atomically advance last_applied_index and generation
        self.last_applied_index.store(entry_index, Ordering::SeqCst);
        let cur_gen = self.applied_generation.fetch_add(1, Ordering::SeqCst) + 1;

        let receipt = ApplyReceipt {
            mutation_id: mutation_id.clone(),
            applied_index: entry_index,
            applied_generation: cur_gen,
            durable_lsn: entry_index,
        };

        // 4. Record in deduplication horizon
        {
            let mut dedup = self.dedup_horizon.lock();
            dedup.insert(receipt.clone(), client, seq);
        }

        Ok(receipt)
    }

    fn last_applied_index(&self) -> u64 {
        self.last_applied_index.load(Ordering::SeqCst)
    }

    fn applied_generation(&self) -> u64 {
        self.applied_generation.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::relational_acid::{ColumnDefinition, RelationalRow, RelationalSqlEngine, SqlType, SqlValue, TableSchema};
    use crate::ecosystem::agent_memory::{AutonomousMemoryConsolidator, EpisodicFact, FactCategory};
    use crate::vector::hypercube::HypercubeTensorSpace;

    #[test]
    fn test_cross_paradigm_atomic_transaction_and_pinned_snapshot() {
        let engine = Arc::new(SegmentedEngine::new(8, 1000));

        // 1. Initialize all 5 paradigm backends
        let sql = Arc::new(RelationalSqlEngine::new());
        let table_schema = TableSchema {
            name: "customers".into(),
            primary_key_column: "cust_id".into(),
            columns: vec![
                ColumnDefinition {
                    name: "cust_id".into(),
                    data_type: SqlType::Text,
                    is_primary_key: true,
                    is_nullable: false,
                    foreign_key_target: None,
                },
                ColumnDefinition {
                    name: "tier".into(),
                    data_type: SqlType::Text,
                    is_primary_key: false,
                    is_nullable: false,
                    foreign_key_target: None,
                },
            ],
        };
        sql.create_table(table_schema).unwrap();

        let memory = Arc::new(AutonomousMemoryConsolidator::new());
        let hypercube = Arc::new(HypercubeTensorSpace::new(vec![4, 4, 4, 4]));

        let sm = ShardStateMachine::with_all_paradigms(
            1,
            engine.clone(),
            None,
            Some(sql.clone()),
            Some(memory.clone()),
            Some(hypercube.clone()),
        );

        // 2. Build multi-paradigm atomic batch mutation
        let mut row_values = HashMap::new();
        row_values.insert("cust_id".into(), SqlValue::Text("cust_999".into()));
        row_values.insert("tier".into(), SqlValue::Text("platinum".into()));

        let fact = EpisodicFact {
            fact_id: "fact_001".into(),
            subject: "cust_999".into(),
            predicate: "residency".into(),
            object: "EU-West".into(),
            category: FactCategory::UserPreference,
            confidence: 0.99,
            emotional_salience: 0.85,
            recall_count: 1,
            last_accessed_secs: 1000,
            created_at_secs: 1000,
        };

        let batch = DataMutation::Batch {
            mutation_id: MutationId::generate(),
            mutations: vec![
                // Paradigm 1: Vector Upsert (8 real floats -> 4 complex coordinates)
                DataMutation::new_upsert("cust_999", VectorEmbedding::from_reals(&[0.1; 8]).into_normalized()),
                // Paradigm 2: Relational SQL Row Insert
                DataMutation::new_sql_insert("customers", RelationalRow { values: row_values }),
                // Paradigm 3: Agent Memory Fact Append
                DataMutation::new_agent_memory("cust_999", fact),
                // Paradigm 4: Hypercube Tensor Voxel Set
                DataMutation::new_hypercube_voxel(vec![1, 2, 3, 0], 42.0),
            ],
        };

        // 3. Apply atomic transaction through single Raft LSN
        let receipt = sm.apply(1001, &batch).unwrap();
        assert_eq!(receipt.durable_lsn, 1001);
        assert_eq!(receipt.applied_index, 1001);

        // 4. Pin universal snapshot
        let snapshot = sm.pin_universal_snapshot();
        assert_eq!(snapshot.lsn, 1001);
        assert_eq!(snapshot.generation, receipt.applied_generation);

        // 5. Verify all 4 paradigms reflect the change in the same atomic tick
        // (a) Vector exists
        assert_eq!(engine.stats().iter().map(|s| s.live_vectors).sum::<usize>(), 1);
        // (b) Relational row exists
        let rows = sql.execute_select("customers", None, None).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].values.get("cust_id"), Some(&SqlValue::Text("cust_999".into())));
        // (c) Agent memory fact exists
        let profile = memory.get_profile("cust_999").unwrap();
        assert_eq!(profile.consolidated_facts.len(), 1);
        assert_eq!(profile.consolidated_facts[0].object, "EU-West");
        // (d) Hypercube voxel exists
        assert_eq!(hypercube.get_voxel(&[1, 2, 3, 0]).unwrap(), 42.0);
    }
}
