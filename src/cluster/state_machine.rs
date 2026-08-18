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

    pub fn mutation_id(&self) -> &MutationId {
        match self {
            Self::Upsert { mutation_id, .. } => mutation_id,
            Self::Delete { mutation_id, .. } => mutation_id,
            Self::MetadataPatch { mutation_id, .. } => mutation_id,
            Self::Batch { mutation_id, .. } => mutation_id,
            Self::Graph { mutation_id, .. } => mutation_id,
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
                // If no graph applier is present, graph mutations are silently
                // accepted (no-op) so vector-only shards stay compatible.
                Ok(())
            }
            DataMutation::Batch { mutations, .. } => {
                self.prevalidate_mutation(mutation)?;
                for m in mutations {
                    self.apply_single_unlogged(m)?;
                }
                Ok(())
            }
        }
    }
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

        // 2. Deterministic atomic application to local storage engine
        self.apply_single_unlogged(mutation)?;

        // 3. Atomically advance last_applied_index and generation
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
