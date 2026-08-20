/* hnsqr/src/service/mod.rs */
//!▫~•◦-------------------------------‣
//! # Production Request Context, Certified Read Snapshots & Public RPC Services
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the public API surface for standalone and distributed HNSQR deployments,
//! unifying request context tracking, strict linearizable read snapshot pinning,
//! authoritative consensus mutation routing, and RBAC security gates.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::cluster::ring::ShardId;
use crate::cluster::DistributedCoordinator;
use crate::consensus::pending::{DurabilityLevel, MutationId};
use crate::consensus::read_index::ReadConsistency;
use crate::metadata::index::MetadataValue;
use crate::proof::lutz::SemanticRerankPlan;
use crate::security::auth::AccessRole;
use crate::{HNSQRError, HNSQRIndex, HNSQRResult, SimilarityScore, VectorEmbedding};

/// Explicit immutable snapshot descriptor pinned to every Certified query execution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReadSnapshot {
    pub topology_epoch: u64,
    pub raft_read_index: u64,
    pub applied_index: u64,
    pub immutable_generation: u64,
    pub mutable_lsn: u64,
}

impl Default for ReadSnapshot {
    fn default() -> Self {
        Self {
            topology_epoch: 1,
            raft_read_index: 0,
            applied_index: 0,
            immutable_generation: 1,
            mutable_lsn: 0,
        }
    }
}

/// RAII-pinned snapshot retaining underlying immutable segments and active mutable view against compaction reclamation.
#[derive(Clone)]
pub struct PinnedReadSnapshot {
    pub topology_epoch: u64,
    pub raft_read_index: u64,
    pub applied_index: u64,
    pub mutable_lsn: u64,
    pub immutable_segments: Arc<[Arc<crate::storage::segment::ImmutableSegment>]>,
    pub active_segment: Arc<crate::storage::segment::MutableSegment>,
    /// Per-shard RAII pins held across all hosted shards — prevents background compaction
    /// from reclaiming segments on shard ID ≥ 1 while a scatter-gather search is in flight.
    pub all_shard_snapshots: HashMap<
        crate::cluster::ring::ShardId,
        (
            Arc<[Arc<crate::storage::segment::ImmutableSegment>]>,
            Arc<crate::storage::segment::MutableSegment>,
        ),
    >,
}

/// Context accompanying every client or internal request.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RequestContext {
    pub request_id: u64,
    pub tenant_id: String,
    pub subject_id: String,
    pub role: AccessRole,
    pub epoch: Option<u64>,
    pub snapshot: Option<ReadSnapshot>,
}

impl Default for RequestContext {
    fn default() -> Self {
        Self {
            request_id: 1,
            tenant_id: "default".to_string(),
            subject_id: "system".to_string(),
            role: AccessRole::Admin,
            epoch: None,
            snapshot: Some(ReadSnapshot::default()),
        }
    }
}

/// Request payload for an Upsert mutation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpsertRequest {
    pub id: String,
    pub vector: VectorEmbedding,
    pub metadata: Option<HashMap<String, MetadataValue>>,
}

/// Request payload for a Delete mutation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeleteRequest {
    pub id: String,
}

/// Durable receipt returned upon successful mutation commit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationReceipt {
    pub id: String,
    pub shard_id: ShardId,
    pub mutation_id: MutationId,
    pub term: u64,
    pub log_index: u64,
    pub applied_index: u64,
    pub applied_generation: u64,
    pub topology_epoch: u64,
    pub durability: DurabilityLevel,
}

/// Authoritative trait for mutation operations.
#[async_trait::async_trait]
pub trait MutationService: Send + Sync {
    async fn upsert(&self, ctx: &RequestContext, req: UpsertRequest) -> HNSQRResult<MutationReceipt>;
    async fn delete(&self, ctx: &RequestContext, req: DeleteRequest) -> HNSQRResult<MutationReceipt>;
}

/// Authoritative trait for search retrieval operations.
pub trait SearchService: Send + Sync {
    fn search(
        &self,
        ctx: &RequestContext,
        query: &VectorEmbedding,
        k: usize,
        rerank_plan: SemanticRerankPlan,
    ) -> HNSQRResult<Vec<(Arc<str>, SimilarityScore)>>;

    fn graph_query(
        &self,
        ctx: &RequestContext,
        query: &str,
    ) -> HNSQRResult<crate::graph::query::executor::QueryResult>;
}

/// Combined HNSQR production service contract.
pub trait HNSQRService: MutationService + SearchService + Send + Sync {}

// ────────────────────────────────────────────────────────────────────────
// 1. Standalone Single-Node Production Service
// ────────────────────────────────────────────────────────────────────────

pub struct StandaloneService {
    index: Arc<HNSQRIndex>,
    slo_manager: Option<Arc<crate::telemetry::slo::SloManager>>,
}

impl StandaloneService {
    pub fn new(index: Arc<HNSQRIndex>) -> Self {
        Self { index, slo_manager: None }
    }

    /// Attaches a `SloManager` for real-time error-budget burn-rate tracking.
    pub fn with_slo(mut self, slo: Arc<crate::telemetry::slo::SloManager>) -> Self {
        self.slo_manager = Some(slo);
        self
    }

    pub fn index(&self) -> &Arc<HNSQRIndex> {
        &self.index
    }

    pub fn upsert_blocking(&self, ctx: &RequestContext, req: UpsertRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            if let Some(slo) = &self.slo_manager {
                slo.record_query_event(false);
            }
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Upsert".to_string()));
        }

        let res = if let Some(meta) = req.metadata {
            self.index.insert_with_metadata(req.id.as_str(), req.vector, meta)
        } else {
            self.index.insert(req.id.as_str(), req.vector)
        };

        if let Some(slo) = &self.slo_manager {
            slo.record_query_event(res.is_ok());
        }
        res?;

        Ok(MutationReceipt {
            id: req.id,
            shard_id: 0,
            mutation_id: MutationId::new("standalone_upsert"),
            term: 1,
            log_index: 1,
            applied_index: 1,
            applied_generation: 1,
            topology_epoch: 1,
            durability: DurabilityLevel::MemoryOnly,
        })
    }

    pub fn delete_blocking(&self, ctx: &RequestContext, req: DeleteRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            if let Some(slo) = &self.slo_manager {
                slo.record_query_event(false);
            }
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Delete".to_string()));
        }

        let res = self.index.remove(&req.id);
        if let Some(slo) = &self.slo_manager {
            slo.record_query_event(res.is_ok());
        }
        res?;

        Ok(MutationReceipt {
            id: req.id,
            shard_id: 0,
            mutation_id: MutationId::new("standalone_delete"),
            term: 1,
            log_index: 1,
            applied_index: 1,
            applied_generation: 1,
            topology_epoch: 1,
            durability: DurabilityLevel::MemoryOnly,
        })
    }
}

#[async_trait::async_trait]
impl MutationService for StandaloneService {
    async fn upsert(&self, ctx: &RequestContext, req: UpsertRequest) -> HNSQRResult<MutationReceipt> {
        self.upsert_blocking(ctx, req)
    }

    async fn delete(&self, ctx: &RequestContext, req: DeleteRequest) -> HNSQRResult<MutationReceipt> {
        self.delete_blocking(ctx, req)
    }
}

impl SearchService for StandaloneService {
    fn search(
        &self,
        _ctx: &RequestContext,
        query: &VectorEmbedding,
        k: usize,
        _rerank_plan: SemanticRerankPlan,
    ) -> HNSQRResult<Vec<(Arc<str>, SimilarityScore)>> {
        let res = self.index.search(query, k);
        if let Some(slo) = &self.slo_manager {
            slo.record_query_event(res.is_ok());
        }
        res
    }

    fn graph_query(
        &self,
        _ctx: &RequestContext,
        _query: &str,
    ) -> HNSQRResult<crate::graph::query::executor::QueryResult> {
        Err(HNSQRError::UnsupportedFeature(
            "Graph queries require ClusterService".to_string(),
        ))
    }
}

impl HNSQRService for StandaloneService {}

// ────────────────────────────────────────────────────────────────────────
// 2. Clustered Multi-Node Distributed Service
// ────────────────────────────────────────────────────────────────────────

pub struct ClusterService {
    coordinator: Arc<DistributedCoordinator>,
    slo_manager: Option<Arc<crate::telemetry::slo::SloManager>>,
}

impl ClusterService {
    pub fn new(coordinator: Arc<DistributedCoordinator>) -> Self {
        Self { coordinator, slo_manager: None }
    }

    pub fn with_slo(mut self, slo: Arc<crate::telemetry::slo::SloManager>) -> Self {
        self.slo_manager = Some(slo);
        self
    }

    pub fn coordinator(&self) -> &Arc<DistributedCoordinator> {
        &self.coordinator
    }

    pub fn upsert_blocking(&self, ctx: &RequestContext, req: UpsertRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            if let Some(slo) = &self.slo_manager {
                slo.record_query_event(false);
            }
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Upsert".to_string()));
        }

        let commit_receipt = match self.coordinator.insert_fenced_blocking(req.id.clone(), req.vector, ctx.epoch) {
            Ok(receipt) => {
                if let Some(slo) = &self.slo_manager {
                    slo.record_query_event(true);
                }
                receipt
            }
            Err(e) => {
                if let Some(slo) = &self.slo_manager {
                    slo.record_query_event(false);
                }
                return Err(e);
            }
        };
        let shard_id = self.coordinator.shard_for_key(&req.id);

        Ok(MutationReceipt {
            id: req.id,
            shard_id,
            mutation_id: commit_receipt.mutation_id,
            term: commit_receipt.term,
            log_index: commit_receipt.log_index,
            applied_index: commit_receipt.applied_index,
            applied_generation: 1,
            topology_epoch: commit_receipt.topology_epoch,
            durability: commit_receipt.durability,
        })
    }

    pub fn delete_blocking(&self, ctx: &RequestContext, req: DeleteRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            if let Some(slo) = &self.slo_manager {
                slo.record_query_event(false);
            }
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Delete".to_string()));
        }

        let commit_receipt = match self.coordinator.delete_blocking(&req.id) {
            Ok(receipt) => {
                if let Some(slo) = &self.slo_manager {
                    slo.record_query_event(true);
                }
                receipt
            }
            Err(e) => {
                if let Some(slo) = &self.slo_manager {
                    slo.record_query_event(false);
                }
                return Err(e);
            }
        };
        let shard_id = self.coordinator.shard_for_key(&req.id);

        Ok(MutationReceipt {
            id: req.id,
            shard_id,
            mutation_id: commit_receipt.mutation_id,
            term: commit_receipt.term,
            log_index: commit_receipt.log_index,
            applied_index: commit_receipt.applied_index,
            applied_generation: 1,
            topology_epoch: commit_receipt.topology_epoch,
            durability: commit_receipt.durability,
        })
    }
}

#[async_trait::async_trait]
impl MutationService for ClusterService {
    async fn upsert(&self, ctx: &RequestContext, req: UpsertRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            if let Some(slo) = &self.slo_manager {
                slo.record_query_event(false);
            }
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Upsert".to_string()));
        }

        let commit_receipt = match self.coordinator.insert_fenced(req.id.clone(), req.vector, ctx.epoch).await {
            Ok(receipt) => {
                if let Some(slo) = &self.slo_manager {
                    slo.record_query_event(true);
                }
                receipt
            }
            Err(e) => {
                if let Some(slo) = &self.slo_manager {
                    slo.record_query_event(false);
                }
                return Err(e);
            }
        };
        let shard_id = self.coordinator.shard_for_key(&req.id);

        Ok(MutationReceipt {
            id: req.id,
            shard_id,
            mutation_id: commit_receipt.mutation_id,
            term: commit_receipt.term,
            log_index: commit_receipt.log_index,
            applied_index: commit_receipt.applied_index,
            applied_generation: 1,
            topology_epoch: commit_receipt.topology_epoch,
            durability: commit_receipt.durability,
        })
    }

    async fn delete(&self, ctx: &RequestContext, req: DeleteRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            if let Some(slo) = &self.slo_manager {
                slo.record_query_event(false);
            }
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Delete".to_string()));
        }

        let commit_receipt = match self.coordinator.delete(&req.id).await {
            Ok(receipt) => {
                if let Some(slo) = &self.slo_manager {
                    slo.record_query_event(true);
                }
                receipt
            }
            Err(e) => {
                if let Some(slo) = &self.slo_manager {
                    slo.record_query_event(false);
                }
                return Err(e);
            }
        };
        let shard_id = self.coordinator.shard_for_key(&req.id);

        Ok(MutationReceipt {
            id: req.id,
            shard_id,
            mutation_id: commit_receipt.mutation_id,
            term: commit_receipt.term,
            log_index: commit_receipt.log_index,
            applied_index: commit_receipt.applied_index,
            applied_generation: 1,
            topology_epoch: commit_receipt.topology_epoch,
            durability: commit_receipt.durability,
        })
    }
}

impl SearchService for ClusterService {
    fn search(
        &self,
        _ctx: &RequestContext,
        query: &VectorEmbedding,
        k: usize,
        rerank_plan: SemanticRerankPlan,
    ) -> HNSQRResult<Vec<(Arc<str>, SimilarityScore)>> {
        let pinned = self.coordinator.obtain_cluster_pinned_snapshot(ReadConsistency::Linearizable)?;
        let res = self.coordinator.search_pinned(&pinned, query, k, rerank_plan);
        if let Some(slo) = &self.slo_manager {
            slo.record_query_event(true);
        }
        Ok(res)
    }

    fn graph_query(&self, _ctx: &RequestContext, query: &str) -> HNSQRResult<crate::graph::query::executor::QueryResult> {
        let shards = self.coordinator.local_shards_snapshot();
        let shard = shards.first().ok_or_else(|| HNSQRError::Internal("No shards available".to_string()))?;

        let graph_applier = shard.state_machine.graph.as_ref().ok_or_else(|| {
            HNSQRError::Internal("Graph engine not enabled on this shard".to_string())
        })?;

        let label_catalog = graph_applier.label_catalog();
        let rel_catalog = graph_applier.rel_catalog();

        let compiled = crate::graph::query::planner::QueryPlanner::compile(query, &label_catalog, &rel_catalog, None)
            .map_err(|e| HNSQRError::Internal(e.to_string()))?;

        let gen_lock = graph_applier.generation();
        let gen_id = gen_lock.read().generation;
        let read_gen = Arc::new(crate::graph::storage::generation::GraphReadGeneration::new(
            gen_lock, gen_id,
        ));
        let mut exec_ctx = crate::graph::query::executor::ExecutionContext::new(read_gen);

        if !compiled.ast.mutations.is_empty() {
            let mutations = crate::graph::query::executor::ExecutionContext::compile_mutations(&compiled.ast.mutations);
            for m in mutations {
                let _rx = self.coordinator.raft_cluster.propose_data_mutation(crate::cluster::state_machine::DataMutation::new_graph(m))?;
            }
        }

        let mut result = exec_ctx.execute_with_segmented_engine(&compiled.plan, &shard.state_machine.engine, &std::collections::HashMap::new())?;
        result.column_names = compiled.column_names;
        Ok(result)
    }
}

impl HNSQRService for ClusterService {}
