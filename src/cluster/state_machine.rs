/* holosphere/src/cluster/state_machine.rs */
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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::consensus::pending::{ApplyError, MutationId};
use crate::graph::catalog::labels::LabelCatalog;
use crate::graph::catalog::relationships::RelTypeCatalog;
use crate::graph::mutation::apply::GraphMutationApplier;
use crate::graph::mutation::command::GraphMutation;
use crate::graph::storage::generation::GraphGeneration;
use crate::learning::discovery::{
    DiscoveryStateMutation, DiscoveryStateSnapshot, ReplicatedDiscoveryAction,
    materialize_relation_type,
};
use crate::learning::{DeclarativeOperator, LearningSegment, OperatorLifecycle};
use crate::metadata::index::MetadataValue;
use crate::relation::{RelationSegment, RelationType};
use crate::storage::segment::SegmentedEngine;
use crate::{HNSQRError, HNSQRResult, VectorEmbedding};

/// Declared client retry contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum RetrySemantics {
    AtLeastOnce,
    #[default]
    Idempotent,
    ExactlyOnceWithinWindow {
        max_sequence_gap: u64,
    },
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
    /// Fault-tolerant batch where independent domain mutations are staged and applied
    /// with domain error isolation, preventing errors in one domain from aborting others.
    FaultTolerantBatch {
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
    /// Governed discovered-operator lifecycle transition replicated through Raft.
    DiscoveryOperator {
        mutation_id: MutationId,
        operator: DeclarativeOperator,
        expected_previous: Option<OperatorLifecycle>,
    },
    /// Schema, mapping, evaluation, experiment, safety-kernel, and audit mutation.
    DiscoveryState {
        mutation_id: MutationId,
        mutation: DiscoveryStateMutation,
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
    pub fn new_sql_insert(
        table: impl Into<String>,
        row: crate::storage::relational_acid::RelationalRow,
    ) -> Self {
        Self::Sql {
            mutation_id: MutationId::generate(),
            table: table.into(),
            row,
            is_delete: false,
        }
    }

    /// Wraps an agent memory fact for Raft replication.
    pub fn new_agent_memory(
        user_id: impl Into<String>,
        fact: crate::ecosystem::agent_memory::EpisodicFact,
    ) -> Self {
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

    pub fn new_discovery_operator_transition(
        operator: DeclarativeOperator,
        expected_previous: Option<OperatorLifecycle>,
    ) -> Self {
        Self::DiscoveryOperator {
            mutation_id: MutationId::generate(),
            operator,
            expected_previous,
        }
    }

    pub fn new_discovery_state_mutation(mutation: DiscoveryStateMutation) -> Self {
        Self::DiscoveryState {
            mutation_id: MutationId::generate(),
            mutation,
        }
    }

    /// Materializes one ordered continuous-discovery action as a Raft command.
    /// Audit actions bind the caller-observed replicated audit head and therefore
    /// fail closed if another writer advanced the chain first.
    pub fn new_discovery_action(
        action: ReplicatedDiscoveryAction,
        expected_audit_head: [u8; 32],
    ) -> Self {
        match action {
            ReplicatedDiscoveryAction::Operator(plan) => {
                Self::new_discovery_operator_transition(plan.operator, plan.expected_previous)
            }
            ReplicatedDiscoveryAction::State(mutation) => {
                Self::new_discovery_state_mutation(mutation)
            }
            ReplicatedDiscoveryAction::Audit(action) => {
                Self::new_discovery_state_mutation(DiscoveryStateMutation::AppendAudit {
                    action,
                    expected_previous_hash: expected_audit_head,
                })
            }
        }
    }

    pub fn new_batch(mutations: Vec<DataMutation>) -> Self {
        Self::Batch {
            mutation_id: MutationId::generate(),
            mutations,
        }
    }

    /// Wraps multiple mutations in a fault-tolerant batch with failure isolation.
    pub fn new_fault_tolerant_batch(mutations: Vec<DataMutation>) -> Self {
        Self::FaultTolerantBatch {
            mutation_id: MutationId::generate(),
            mutations,
        }
    }

    pub fn mutation_id(&self) -> &MutationId {
        match self {
            Self::Upsert { mutation_id, .. } => mutation_id,
            Self::Delete { mutation_id, .. } => mutation_id,
            Self::MetadataPatch { mutation_id, .. } => mutation_id,
            Self::Batch { mutation_id, .. } => mutation_id,
            Self::FaultTolerantBatch { mutation_id, .. } => mutation_id,
            Self::Graph { mutation_id, .. } => mutation_id,
            Self::Sql { mutation_id, .. } => mutation_id,
            Self::AgentMemory { mutation_id, .. } => mutation_id,
            Self::Hypercube { mutation_id, .. } => mutation_id,
            Self::DiscoveryOperator { mutation_id, .. } => mutation_id,
            Self::DiscoveryState { mutation_id, .. } => mutation_id,
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
                        return Ok(self
                            .client_receipts
                            .get(c)
                            .cloned()
                            .or_else(|| self.seen.get(id).cloned()));
                    }
                    if client_seq < last_seq {
                        return Err(ApplyError::SequenceViolation {
                            reason: format!(
                                "Stale client sequence {client_seq} (last observed: {last_seq})"
                            ),
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

    pub fn insert(
        &mut self,
        receipt: ApplyReceipt,
        client: Option<&ClientIdentity>,
        client_seq: u64,
    ) {
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
    /// Optional governed discovery catalog, mutated only at committed LSNs.
    pub learning: Option<Arc<LearningSegment>>,
    /// Canonical N-ary relation catalog receiving governed evolved schemas.
    pub relations: Option<Arc<RelationSegment>>,
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
            learning: None,
            relations: None,
        }
    }

    /// Creates a state machine with a pre-wired graph applier.
    pub fn with_graph(shard_id: u32, engine: Arc<SegmentedEngine>) -> Self {
        use parking_lot::RwLock;
        let generation = Arc::new(RwLock::new(GraphGeneration::new_mutable(1)));
        let label_catalog = Arc::new(LabelCatalog::default());
        let rel_catalog = Arc::new(RelTypeCatalog::default());
        let applier = Arc::new(GraphMutationApplier::new(
            generation,
            label_catalog,
            rel_catalog,
        ));
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
            learning: None,
            relations: None,
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
            learning: None,
            relations: None,
        }
    }

    /// Wires the governed discovery catalog into the authoritative Raft apply path.
    pub fn with_learning_discovery(mut self, learning: Arc<LearningSegment>) -> Self {
        self.learning = Some(learning);
        self
    }

    /// Wires the canonical hypergraph catalog to the governed schema apply path.
    pub fn with_evolved_relation_catalog(mut self, relations: Arc<RelationSegment>) -> Self {
        self.relations = Some(relations);
        self
    }
}

/// Concrete staged delta prepared for atomic publication across all 5 paradigms.
#[derive(Clone, Debug)]
pub enum PreparedDelta {
    VectorUpsert {
        key: String,
        vector: VectorEmbedding,
        metadata: Option<HashMap<String, MetadataValue>>,
    },
    VectorDelete {
        key: String,
    },
    VectorMetadataPatch {
        key: String,
        metadata: HashMap<String, MetadataValue>,
    },
    Graph(GraphMutation),
    Sql {
        table: String,
        pk: String,
        row: crate::storage::relational_acid::RelationalRow,
        is_delete: bool,
    },
    AgentMemory {
        user_id: String,
        fact: crate::ecosystem::agent_memory::EpisodicFact,
    },
    Hypercube {
        coords: Vec<usize>,
        value: f32,
    },
    DiscoveryOperator {
        operator: DeclarativeOperator,
        expected_previous: Option<OperatorLifecycle>,
        commit_lsn: u64,
    },
    DiscoveryState {
        mutation: DiscoveryStateMutation,
        commit_lsn: u64,
        evolved_relation_type: Option<RelationType>,
    },
}

impl ShardStateMachine {
    /// Internal staged pre-validation ensuring atomic all-or-nothing batch execution.
    fn prepare_mutation(
        &self,
        mutation: &DataMutation,
        deltas: &mut Vec<PreparedDelta>,
        commit_lsn: u64,
    ) -> HNSQRResult<()> {
        match mutation {
            DataMutation::Upsert {
                key,
                vector,
                metadata,
                ..
            } => {
                if vector.dimension() != self.engine.dimension {
                    return Err(HNSQRError::DimensionMismatch {
                        expected: self.engine.dimension,
                        actual: vector.dimension(),
                    });
                }
                deltas.push(PreparedDelta::VectorUpsert {
                    key: key.clone(),
                    vector: vector.clone(),
                    metadata: metadata.clone(),
                });
                Ok(())
            }
            DataMutation::Delete { key, .. } => {
                deltas.push(PreparedDelta::VectorDelete { key: key.clone() });
                Ok(())
            }
            DataMutation::MetadataPatch { key, metadata, .. } => {
                deltas.push(PreparedDelta::VectorMetadataPatch {
                    key: key.clone(),
                    metadata: metadata.clone(),
                });
                Ok(())
            }
            DataMutation::Graph { mutation, .. } => {
                let graph = self.graph.as_ref().ok_or_else(|| {
                    HNSQRError::InvalidRequest(
                        "Graph backend not enabled on this shard".to_string(),
                    )
                })?;
                graph.prevalidate(mutation)?;
                deltas.push(PreparedDelta::Graph(mutation.clone()));
                Ok(())
            }
            DataMutation::Sql {
                table,
                row,
                is_delete,
                ..
            } => {
                let sql = self.sql.as_ref().ok_or_else(|| {
                    HNSQRError::InvalidRequest("SQL backend not enabled on this shard".to_string())
                })?;
                let schema = sql.get_table_schema(table).ok_or_else(|| {
                    HNSQRError::InvalidRequest(format!("Table '{table}' does not exist"))
                })?;
                let pk_val = row.values.get(&schema.primary_key_column).ok_or_else(|| {
                    HNSQRError::InvalidRequest(format!(
                        "Row missing primary key '{}'",
                        schema.primary_key_column
                    ))
                })?;
                let pk_str = match pk_val {
                    crate::storage::relational_acid::SqlValue::Text(s) => s.clone(),
                    crate::storage::relational_acid::SqlValue::Integer(i) => i.to_string(),
                    _ => {
                        return Err(HNSQRError::InvalidRequest(
                            "Primary key must be Text or Integer".into(),
                        ));
                    }
                };
                if !*is_delete {
                    for col in &schema.columns {
                        if !col.is_nullable && !row.values.contains_key(&col.name) {
                            return Err(HNSQRError::InvalidRequest(format!(
                                "Non-nullable column '{}' missing in row",
                                col.name
                            )));
                        }
                    }
                }
                deltas.push(PreparedDelta::Sql {
                    table: table.clone(),
                    pk: pk_str,
                    row: row.clone(),
                    is_delete: *is_delete,
                });
                Ok(())
            }
            DataMutation::AgentMemory { user_id, fact, .. } => {
                let _memory = self.memory.as_ref().ok_or_else(|| {
                    HNSQRError::InvalidRequest(
                        "Agent memory backend not enabled on this shard".to_string(),
                    )
                })?;
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
                deltas.push(PreparedDelta::AgentMemory {
                    user_id: user_id.clone(),
                    fact: fact.clone(),
                });
                Ok(())
            }
            DataMutation::Hypercube { coords, value, .. } => {
                let h = self.hypercube.as_ref().ok_or_else(|| {
                    HNSQRError::InvalidRequest(
                        "Hypercube backend not enabled on this shard".to_string(),
                    )
                })?;
                if coords.len() != h.dimensions() {
                    return Err(HNSQRError::DimensionMismatch {
                        expected: h.dimensions(),
                        actual: coords.len(),
                    });
                }
                let shape = h.shape();
                for (d, (&coord, &max_dim)) in coords.iter().zip(shape.iter()).enumerate() {
                    if coord >= max_dim {
                        return Err(HNSQRError::InvalidRequest(format!(
                            "Coordinate index out of bounds on dimension {d}: {coord} >= {max_dim}"
                        )));
                    }
                }
                deltas.push(PreparedDelta::Hypercube {
                    coords: coords.clone(),
                    value: *value,
                });
                Ok(())
            }
            DataMutation::DiscoveryOperator {
                operator,
                expected_previous,
                ..
            } => {
                if deltas.iter().any(|delta| {
                    matches!(
                        delta,
                        PreparedDelta::DiscoveryOperator {
                            operator: staged,
                            ..
                        } if staged.id == operator.id
                    )
                }) {
                    return Err(HNSQRError::InvalidRequest(
                        "Only one lifecycle transition per discovered operator is allowed in a committed LSN"
                            .to_string(),
                    ));
                }
                let learning = self.learning.as_ref().ok_or_else(|| {
                    HNSQRError::InvalidRequest(
                        "Learning discovery backend not enabled on this shard".to_string(),
                    )
                })?;
                learning
                    .governed_discovery
                    .verify_kernel_installed()
                    .map_err(|error| HNSQRError::InvalidRequest(error.to_string()))?;
                learning
                    .discovery
                    .prevalidate(operator, *expected_previous, commit_lsn)
                    .map_err(|error| HNSQRError::InvalidRequest(error.to_string()))?;
                deltas.push(PreparedDelta::DiscoveryOperator {
                    operator: operator.clone(),
                    expected_previous: *expected_previous,
                    commit_lsn,
                });
                Ok(())
            }
            DataMutation::DiscoveryState { mutation, .. } => {
                if deltas.iter().any(|delta| {
                    matches!(
                        delta,
                        PreparedDelta::DiscoveryState { mutation: staged, .. }
                            if staged.conflict_key() == mutation.conflict_key()
                    )
                }) {
                    return Err(HNSQRError::InvalidRequest(
                        "Only one mutation per governed discovery key is allowed in a committed LSN"
                            .to_string(),
                    ));
                }
                let learning = self.learning.as_ref().ok_or_else(|| {
                    HNSQRError::InvalidRequest(
                        "Learning discovery backend not enabled on this shard".to_string(),
                    )
                })?;
                learning
                    .governed_discovery
                    .prevalidate(mutation, commit_lsn)
                    .map_err(|error| HNSQRError::InvalidRequest(error.to_string()))?;
                let evolved_relation_type = match mutation {
                    DiscoveryStateMutation::UpsertSchema { record, .. } => {
                        let provenance_id = record
                            .proposal
                            .empirical_roots
                            .iter()
                            .next()
                            .map_or(0, |root| root.0);
                        materialize_relation_type(&record.proposal, provenance_id)
                    }
                    _ => None,
                };
                if let (Some(relations), Some(rtype)) =
                    (&self.relations, evolved_relation_type.as_ref())
                {
                    relations
                        .prevalidate_evolved_type(rtype)
                        .map_err(HNSQRError::InvalidRequest)?;
                }
                deltas.push(PreparedDelta::DiscoveryState {
                    mutation: mutation.clone(),
                    commit_lsn,
                    evolved_relation_type,
                });
                Ok(())
            }
            DataMutation::Batch { mutations, .. } => {
                for m in mutations {
                    self.prepare_mutation(m, deltas, commit_lsn)?;
                }
                Ok(())
            }
            DataMutation::FaultTolerantBatch { mutations, .. } => {
                for m in mutations {
                    let _ = self.prepare_mutation(m, deltas, commit_lsn);
                }
                Ok(())
            }
        }
    }

    /// Atomically publishes all prepared deltas to physical engines.
    fn publish_deltas(&self, deltas: Vec<PreparedDelta>) -> HNSQRResult<()> {
        for delta in deltas {
            match delta {
                PreparedDelta::VectorUpsert {
                    key,
                    vector,
                    metadata,
                } => {
                    self.engine
                        .apply_committed_upsert(key.as_str(), vector, metadata.as_ref())?;
                }
                PreparedDelta::VectorDelete { key } => {
                    self.engine.apply_delete_unlogged(key.as_str());
                }
                PreparedDelta::VectorMetadataPatch { key, metadata } => {
                    self.engine
                        .apply_committed_metadata_patch(key.as_str(), &metadata)?;
                }
                PreparedDelta::Graph(m) => {
                    if let Some(graph) = &self.graph {
                        graph.apply(&m)?;
                    }
                }
                PreparedDelta::Sql {
                    table,
                    row,
                    is_delete,
                    ..
                } => {
                    if let Some(sql) = &self.sql {
                        sql.apply_committed_row_mutation(&table, row, is_delete)?;
                    }
                }
                PreparedDelta::AgentMemory { user_id, fact } => {
                    if let Some(memory) = &self.memory {
                        memory.ingest_fact(&user_id, fact)?;
                    }
                }
                PreparedDelta::Hypercube { coords, value } => {
                    if let Some(hypercube) = &self.hypercube {
                        hypercube.set_voxel(coords, value)?;
                    }
                }
                PreparedDelta::DiscoveryOperator {
                    operator,
                    expected_previous,
                    commit_lsn,
                } => {
                    if let Some(learning) = &self.learning {
                        learning
                            .discovery
                            .apply(operator, expected_previous, commit_lsn)
                            .map_err(|error| HNSQRError::InvalidRequest(error.to_string()))?;
                    }
                }
                PreparedDelta::DiscoveryState {
                    mutation,
                    commit_lsn,
                    evolved_relation_type,
                } => {
                    if let Some(learning) = &self.learning {
                        learning
                            .governed_discovery
                            .apply(mutation, commit_lsn)
                            .map_err(|error| HNSQRError::InvalidRequest(error.to_string()))?;
                    }
                    if let (Some(relations), Some(rtype)) = (&self.relations, evolved_relation_type)
                    {
                        relations.synchronize_evolved_type(rtype);
                    }
                }
            }
        }
        Ok(())
    }

    /// Creates an atomic cross-paradigm snapshot metadata handle pinned at the current applied generation.
    pub fn pin_universal_snapshot(&self) -> UniversalSnapshotHandle {
        UniversalSnapshotHandle {
            generation: self.applied_generation.load(Ordering::SeqCst),
            lsn: self.last_applied_index.load(Ordering::SeqCst),
        }
    }

    /// Retains physical backing state and immutable snapshots across all 5 converged paradigms.
    /// Holding this snapshot guarantees that background mutations/vacuum/compaction/GC cannot
    /// mutate or reclaim underlying data out from under an active query session.
    pub fn pin_physical_snapshot(&self) -> UniversalSnapshot {
        let lsn = self.last_applied_index.load(Ordering::SeqCst);
        let cur_gen = self.applied_generation.load(Ordering::SeqCst);
        UniversalSnapshot {
            generation: cur_gen,
            lsn,
            vector_snapshot: Some(self.engine.snapshot()),
            graph_snapshot: self.graph.as_ref().map(|g| g.snapshot(lsn)),
            sql_snapshot: self.sql.as_ref().map(|s| s.snapshot()),
            memory_snapshot: self.memory.as_ref().map(|m| m.snapshot()),
            hypercube_snapshot: self.hypercube.as_ref().map(|h| h.snapshot()),
            immutable_segments: self.engine.immutable_segments_snapshot(),
            active_segment: self.engine.active_mutable_segment(),
            graph_generation: self.graph.as_ref().map(|g| g.generation()),
            sql: self.sql.clone(),
            memory: self.memory.clone(),
            hypercube: self.hypercube.clone(),
            discovered_operators: self
                .learning
                .as_ref()
                .map(|learning| learning.discovery.snapshot_at(lsn))
                .unwrap_or_default(),
            governed_discovery: self
                .learning
                .as_ref()
                .map(|learning| learning.governed_discovery.snapshot_at(lsn)),
            evolved_relation_types: self
                .relations
                .as_ref()
                .map(|relations| relations.types.read().clone())
                .unwrap_or_default(),
        }
    }
}

/// Atomic cross-paradigm snapshot handle pinning vector, graph, relational, memory, and tensor state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniversalSnapshotHandle {
    pub generation: u64,
    pub lsn: u64,
}

/// Physical pinned snapshot retaining active generation and immutable snapshot handles across all 5 converged paradigms.
pub struct UniversalSnapshot {
    pub generation: u64,
    pub lsn: u64,
    /// Immutable point-in-time vector engine snapshot at LSN k.
    pub vector_snapshot: Option<crate::storage::segment::ImmutableVectorSnapshot>,
    /// Immutable point-in-time graph topology snapshot at LSN k.
    pub graph_snapshot: Option<crate::graph::storage::snapshot::ImmutableGraphSnapshot>,
    /// Immutable point-in-time relational MVCC snapshot at LSN k.
    pub sql_snapshot: Option<crate::storage::relational_acid::RelationalSqlSnapshot>,
    /// Immutable point-in-time agent memory persona snapshot at LSN k.
    pub memory_snapshot: Option<crate::ecosystem::agent_memory::AutonomousMemorySnapshot>,
    /// Immutable point-in-time hypercube tensor space snapshot at LSN k.
    pub hypercube_snapshot: Option<crate::vector::hypercube::HypercubeSnapshot>,

    pub immutable_segments: Vec<Arc<crate::storage::segment::ImmutableSegment>>,
    pub active_segment: Arc<crate::storage::segment::MutableSegment>,
    pub graph_generation:
        Option<Arc<parking_lot::RwLock<crate::graph::storage::generation::GraphGeneration>>>,
    pub sql: Option<Arc<crate::storage::relational_acid::RelationalSqlEngine>>,
    pub memory: Option<Arc<crate::ecosystem::agent_memory::AutonomousMemoryConsolidator>>,
    pub hypercube: Option<Arc<crate::vector::hypercube::HypercubeTensorSpace>>,
    /// Governed operator versions visible at this snapshot's committed LSN.
    pub discovered_operators: Vec<DeclarativeOperator>,
    pub governed_discovery: Option<DiscoveryStateSnapshot>,
    /// Canonical evolved N-ary schemas visible with the same pinned LSN.
    pub evolved_relation_types: Vec<RelationType>,
}

impl ReplicatedStateMachine for ShardStateMachine {
    fn apply(&self, entry_index: u64, mutation: &DataMutation) -> HNSQRResult<ApplyReceipt> {
        let mutation_id = mutation.mutation_id();

        // 1. Check idempotency / sequence retry semantics
        let (client, seq, retry) = match mutation {
            DataMutation::Upsert {
                client,
                client_seq,
                retry_semantics,
                ..
            } => (client.as_ref(), *client_seq, *retry_semantics),
            DataMutation::Delete {
                client,
                client_seq,
                retry_semantics,
                ..
            } => (client.as_ref(), *client_seq, *retry_semantics),
            _ => (None, 0, RetrySemantics::Idempotent),
        };
        {
            let dedup = self.dedup_horizon.lock();
            if let Some(cached) = dedup.check(mutation_id, client, seq, retry)? {
                return Ok(cached);
            }
        }

        // 2. PREPARE: Stage and validate all deltas before touching any backend
        let mut prepared_deltas = Vec::new();
        self.prepare_mutation(mutation, &mut prepared_deltas, entry_index)?;

        // 3. ATOMIC PUBLISH: Apply all prepared deltas to physical engines
        self.publish_deltas(prepared_deltas)?;

        // 4. Atomically advance last_applied_index and generation
        self.last_applied_index.store(entry_index, Ordering::SeqCst);
        let cur_gen = self.applied_generation.fetch_add(1, Ordering::SeqCst) + 1;

        let receipt = ApplyReceipt {
            mutation_id: mutation_id.clone(),
            applied_index: entry_index,
            applied_generation: cur_gen,
            durable_lsn: entry_index,
        };

        // 5. Record in deduplication horizon
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
    use std::collections::BTreeSet;

    use crate::ecosystem::agent_memory::{
        AutonomousMemoryConsolidator, EpisodicFact, FactCategory,
    };
    use crate::learning::{
        DiscoveredMotif, DomainId, FeatureId, GovernanceAuthority, ImmutableSafetyKernel, MotifId,
        ResolutionId,
    };
    use crate::storage::relational_acid::{
        ColumnDefinition, RelationalRow, RelationalSqlEngine, SqlType, SqlValue, TableSchema,
    };
    use crate::vector::hypercube::HypercubeTensorSpace;

    #[test]
    fn test_discovery_operator_lifecycle_is_replicated_and_snapshot_visible() {
        let learning = Arc::new(LearningSegment::new(1));
        let state = ShardStateMachine::new(1, Arc::new(SegmentedEngine::new(8, 1000)))
            .with_learning_discovery(Arc::clone(&learning));
        state
            .apply(
                1,
                &DataMutation::new_discovery_state_mutation(
                    DiscoveryStateMutation::InstallSafetyKernel {
                        kernel: ImmutableSafetyKernel::v1(256),
                    },
                ),
            )
            .unwrap();
        let motif = DiscoveredMotif {
            id: MotifId([7; 32]),
            conditions: vec![FeatureId(1), FeatureId(2)],
            resolution: ResolutionId(9),
            successes: 6,
            contradictions: 0,
            supporting_domains: BTreeSet::from([DomainId(1), DomainId(2)]),
            empirical_roots: BTreeSet::new(),
            precision_q32: 1i64 << 32,
        };
        let mut operator = DeclarativeOperator::from_motif(&motif);
        operator.lifecycle = OperatorLifecycle::Provisional;

        let duplicate_transition = DataMutation::new_batch(vec![
            DataMutation::new_discovery_operator_transition(operator.clone(), None),
            DataMutation::new_discovery_operator_transition(operator.clone(), None),
        ]);
        assert!(state.apply(9, &duplicate_transition).is_err());
        assert_eq!(state.last_applied_index(), 1);
        assert!(learning.discovery.snapshot().is_empty());

        state
            .apply(
                10,
                &DataMutation::new_discovery_operator_transition(operator.clone(), None),
            )
            .unwrap();

        operator.lifecycle = OperatorLifecycle::FalsificationTesting;
        state
            .apply(
                11,
                &DataMutation::new_discovery_operator_transition(
                    operator.clone(),
                    Some(OperatorLifecycle::Provisional),
                ),
            )
            .unwrap();

        operator.lifecycle = OperatorLifecycle::Shadow;
        state
            .apply(
                12,
                &DataMutation::new_discovery_operator_transition(
                    operator.clone(),
                    Some(OperatorLifecycle::FalsificationTesting),
                ),
            )
            .unwrap();

        operator.lifecycle = OperatorLifecycle::ShadowValidated;
        state
            .apply(
                13,
                &DataMutation::new_discovery_operator_transition(
                    operator.clone(),
                    Some(OperatorLifecycle::Shadow),
                ),
            )
            .unwrap();

        operator.lifecycle = OperatorLifecycle::Admitted;
        operator.admission_authority = Some(GovernanceAuthority::ReplicatedPolicy {
            policy_id: 42,
            version: 1,
        });
        state
            .apply(
                14,
                &DataMutation::new_discovery_operator_transition(
                    operator,
                    Some(OperatorLifecycle::ShadowValidated),
                ),
            )
            .unwrap();

        let snapshot = state.pin_physical_snapshot();
        assert_eq!(snapshot.lsn, 14);
        assert_eq!(snapshot.discovered_operators.len(), 1);
        assert_eq!(
            snapshot.discovered_operators[0].lifecycle,
            OperatorLifecycle::Admitted
        );
        assert_eq!(
            learning.discovery.snapshot_at(12)[0].lifecycle,
            OperatorLifecycle::Shadow
        );
    }

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
                DataMutation::new_upsert(
                    "cust_999",
                    VectorEmbedding::from_reals(&[0.1; 8]).into_normalized(),
                ),
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

        // 4. Pin universal snapshot and physical MVCC snapshot
        let snapshot = sm.pin_universal_snapshot();
        assert_eq!(snapshot.lsn, 1001);
        assert_eq!(snapshot.generation, receipt.applied_generation);

        let phys_snap = sm.pin_physical_snapshot();
        assert_eq!(phys_snap.lsn, 1001);
        let snap_sql = phys_snap.sql_snapshot.as_ref().unwrap();
        assert_eq!(
            snap_sql
                .execute_select("customers", None, None)
                .unwrap()
                .len(),
            1
        );
        let snap_hypercube = phys_snap.hypercube_snapshot.as_ref().unwrap();
        assert_eq!(snap_hypercube.get_voxel(&[1, 2, 3, 0]).unwrap(), 42.0);
        let snap_memory = phys_snap.memory_snapshot.as_ref().unwrap();
        assert_eq!(
            snap_memory
                .get_profile("cust_999")
                .unwrap()
                .consolidated_facts[0]
                .object,
            "EU-West"
        );

        // 5. Verify live state
        assert_eq!(
            engine.stats().iter().map(|s| s.live_vectors).sum::<usize>(),
            1
        );
        let rows = sql.execute_select("customers", None, None).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(
            rows[0].values.get("cust_id"),
            Some(&SqlValue::Text("cust_999".into()))
        );
        let profile = memory.get_profile("cust_999").unwrap();
        assert_eq!(profile.consolidated_facts.len(), 1);
        assert_eq!(profile.consolidated_facts[0].object, "EU-West");
        assert_eq!(hypercube.get_voxel(&[1, 2, 3, 0]).unwrap(), 42.0);

        // 6. Mutate live state and ensure pinned snapshot at LSN 1001 remains immutable
        hypercube.set_voxel(vec![1, 2, 3, 0], 999.0).unwrap();
        assert_eq!(hypercube.get_voxel(&[1, 2, 3, 0]).unwrap(), 999.0);
        assert_eq!(snap_hypercube.get_voxel(&[1, 2, 3, 0]).unwrap(), 42.0);

        // Verify vector snapshot isolation
        let snap_vec = phys_snap.vector_snapshot.as_ref().unwrap();
        assert_eq!(
            snap_vec
                .search(
                    &VectorEmbedding::from_reals(&[0.1; 8]).into_normalized(),
                    1,
                    crate::proof::lutz::SemanticRerankPlan::ExactSimd
                )
                .len(),
            1
        );
    }

    #[test]
    fn test_batch_atomicity_discards_all_mutations_on_apply_time_failure() {
        let engine = Arc::new(SegmentedEngine::new(8, 1000));
        let sql = Arc::new(RelationalSqlEngine::new());
        let table_schema = TableSchema {
            name: "accounts".into(),
            primary_key_column: "acc_id".into(),
            columns: vec![ColumnDefinition {
                name: "acc_id".into(),
                data_type: SqlType::Text,
                is_primary_key: true,
                is_nullable: false,
                foreign_key_target: None,
            }],
        };
        sql.create_table(table_schema).unwrap();

        let sm = ShardStateMachine::with_all_paradigms(
            1,
            engine.clone(),
            None,
            Some(sql.clone()),
            None,
            None,
        );

        // Construct a batch where vector upsert is completely valid,
        // but SQL row has Float primary key (which passes table existence check but fails PK type validation)
        let mut row_values = HashMap::new();
        row_values.insert("acc_id".into(), SqlValue::Float(12.5));

        let batch = DataMutation::Batch {
            mutation_id: MutationId::generate(),
            mutations: vec![
                DataMutation::new_upsert(
                    "should_not_exist",
                    VectorEmbedding::from_reals(&[0.5; 8]).into_normalized(),
                ),
                DataMutation::new_sql_insert("accounts", RelationalRow { values: row_values }),
            ],
        };

        // Apply MUST fail
        let res = sm.apply(50, &batch);
        assert!(res.is_err(), "Batch with invalid SQL PK must return Err");

        // Staged state verification: Apply(Batch)=Err => S_after = S_before
        assert_eq!(
            sm.last_applied_index(),
            0,
            "LSN must not advance on batch failure"
        );
        assert_eq!(
            engine.stats().iter().map(|s| s.live_vectors).sum::<usize>(),
            0,
            "Vector engine must have 0 vectors"
        );
        let results = engine.search(
            &VectorEmbedding::from_reals(&[0.5; 8]).into_normalized(),
            1,
            crate::proof::lutz::SemanticRerankPlan::ExactSimd,
        );
        assert!(
            results.is_empty(),
            "Uncommitted vector must not exist in engine"
        );
    }

    #[test]
    fn test_all_five_paradigm_immutable_snapshot_isolation() {
        use crate::graph::catalog::labels::LabelCatalog;
        use crate::graph::catalog::relationships::RelTypeCatalog;
        use crate::graph::storage::generation::GraphGeneration;
        use parking_lot::RwLock;

        let engine = Arc::new(SegmentedEngine::new(8, 1000));
        let graph_gen = Arc::new(RwLock::new(GraphGeneration::new_mutable(1)));
        let label_cat = Arc::new(LabelCatalog::default());
        let rel_cat = Arc::new(RelTypeCatalog::default());
        let graph_applier = Arc::new(GraphMutationApplier::new(graph_gen, label_cat, rel_cat));

        let sql = Arc::new(RelationalSqlEngine::new());
        sql.create_table(TableSchema {
            name: "items".into(),
            primary_key_column: "item_id".into(),
            columns: vec![ColumnDefinition {
                name: "item_id".into(),
                data_type: SqlType::Text,
                is_primary_key: true,
                is_nullable: false,
                foreign_key_target: None,
            }],
        })
        .unwrap();

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

        // Apply initial state across all 5 paradigms
        let mut row_v = HashMap::new();
        row_v.insert("item_id".into(), SqlValue::Text("it_1".into()));

        let initial_batch = DataMutation::Batch {
            mutation_id: MutationId::generate(),
            mutations: vec![
                DataMutation::new_upsert(
                    "v_1",
                    VectorEmbedding::from_reals(&[0.1; 8]).into_normalized(),
                ),
                DataMutation::new_graph(GraphMutation::CreateNode {
                    external_id: "node_1".into(),
                    labels: vec![1],
                    properties: HashMap::new(),
                    vector_slot: None,
                }),
                DataMutation::new_sql_insert("items", RelationalRow { values: row_v }),
                DataMutation::new_agent_memory(
                    "user_1",
                    EpisodicFact {
                        fact_id: "f1".into(),
                        subject: "user_1".into(),
                        predicate: "pref".into(),
                        object: "dark_mode".into(),
                        category: FactCategory::UserPreference,
                        confidence: 0.9,
                        emotional_salience: 0.5,
                        recall_count: 1,
                        last_accessed_secs: 100,
                        created_at_secs: 100,
                    },
                ),
                DataMutation::new_hypercube_voxel(vec![0, 0, 0, 0], 10.0),
            ],
        };
        sm.apply(10, &initial_batch).unwrap();

        // Pin Universal Snapshot S_k at LSN 10
        let snap = sm.pin_physical_snapshot();
        assert_eq!(snap.lsn, 10);

        // Mutate all 5 live backends after snapshot
        sm.apply(
            11,
            &DataMutation::new_upsert(
                "v_2",
                VectorEmbedding::from_reals(&[0.9; 8]).into_normalized(),
            ),
        )
        .unwrap();
        sm.apply(
            12,
            &DataMutation::new_graph(GraphMutation::CreateNode {
                external_id: "node_2".into(),
                labels: vec![2],
                properties: HashMap::new(),
                vector_slot: None,
            }),
        )
        .unwrap();
        let mut row_v2 = HashMap::new();
        row_v2.insert("item_id".into(), SqlValue::Text("it_2".into()));
        sm.apply(
            13,
            &DataMutation::new_sql_insert("items", RelationalRow { values: row_v2 }),
        )
        .unwrap();
        sm.apply(
            14,
            &DataMutation::new_hypercube_voxel(vec![0, 0, 0, 0], 999.0),
        )
        .unwrap();

        // Verify Snapshot S_k is completely frozen and untouched:
        let v_snap = snap.vector_snapshot.as_ref().unwrap();
        let v_res = v_snap.search(
            &VectorEmbedding::from_reals(&[0.9; 8]).into_normalized(),
            10,
            crate::proof::lutz::SemanticRerankPlan::ExactSimd,
        );
        assert!(
            !v_res.iter().any(|(k, _)| k.as_ref() == "v_2"),
            "Snapshot must NOT see v_2"
        );

        let g_snap = snap.graph_snapshot.as_ref().unwrap();
        assert_eq!(
            g_snap.node_count(),
            1,
            "Graph snapshot must have exactly 1 node"
        );
        assert!(
            g_snap.get_node_index("node_2").is_none(),
            "Graph snapshot must NOT see node_2"
        );

        let sql_snap = snap.sql_snapshot.as_ref().unwrap();
        assert_eq!(
            sql_snap.execute_select("items", None, None).unwrap().len(),
            1,
            "SQL snapshot must have 1 row"
        );

        let cube_snap = snap.hypercube_snapshot.as_ref().unwrap();
        assert_eq!(
            cube_snap.get_voxel(&[0, 0, 0, 0]).unwrap(),
            10.0,
            "Hypercube snapshot must have old voxel value"
        );
    }

    #[test]
    fn test_prevalidation_rejects_missing_backend_and_bounds_violation() {
        let engine = Arc::new(SegmentedEngine::new(8, 1000));
        let hypercube = Arc::new(HypercubeTensorSpace::new(vec![4, 4, 4, 4]));

        // State machine without SQL or Graph backends
        let sm = ShardStateMachine::with_all_paradigms(
            1,
            engine.clone(),
            None,
            None,
            None,
            Some(hypercube),
        );

        // (a) Hypercube out-of-bounds coordinate should fail prevalidation
        let bad_hypercube = DataMutation::new_hypercube_voxel(vec![10, 0, 0, 0], 1.0);
        assert!(sm.apply(100, &bad_hypercube).is_err());

        // (b) Absent SQL backend should fail prevalidation
        let mut row = HashMap::new();
        row.insert("id".into(), SqlValue::Text("1".into()));
        let bad_sql = DataMutation::new_sql_insert("test", RelationalRow { values: row });
        assert!(sm.apply(101, &bad_sql).is_err());
    }

    #[test]
    fn test_fault_tolerant_batch_isolates_domain_failures() {
        let engine = Arc::new(SegmentedEngine::new(8, 1000));
        let hypercube = Arc::new(HypercubeTensorSpace::new(vec![4, 4, 4, 4]));

        let sm = ShardStateMachine::with_all_paradigms(
            1,
            engine.clone(),
            None,
            None,
            None,
            Some(hypercube),
        );

        // Batch with valid vector upsert, valid hypercube voxel, and invalid missing SQL backend
        let valid_vec = VectorEmbedding::new(vec![0.1; 8]);
        let mut row = HashMap::new();
        row.insert("id".into(), SqlValue::Text("1".into()));
        let bad_sql = DataMutation::new_sql_insert("test", RelationalRow { values: row });

        let ft_batch = DataMutation::new_fault_tolerant_batch(vec![
            DataMutation::new_upsert("isolated_doc", valid_vec.clone()),
            bad_sql,
            DataMutation::new_hypercube_voxel(vec![1, 1, 1, 1], 99.0),
        ]);

        let receipt = sm
            .apply(200, &ft_batch)
            .expect("FaultTolerantBatch must succeed");
        assert_eq!(receipt.applied_index, 200);

        // Verify valid vector was inserted
        assert!(
            !engine
                .search(
                    &valid_vec,
                    1,
                    crate::proof::lutz::SemanticRerankPlan::ExactSimd
                )
                .is_empty()
        );
        // Verify valid hypercube voxel was updated
        let snap = sm.pin_physical_snapshot();
        let cube_snap = snap.hypercube_snapshot.unwrap();
        assert_eq!(cube_snap.get_voxel(&[1, 1, 1, 1]).unwrap(), 99.0);
    }
}
