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
pub trait MutationService: Send + Sync {
    fn upsert(&self, ctx: &RequestContext, req: UpsertRequest) -> HNSQRResult<MutationReceipt>;
    fn delete(&self, ctx: &RequestContext, req: DeleteRequest) -> HNSQRResult<MutationReceipt>;
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
}

/// Combined HNSQR production service contract.
pub trait HNSQRService: MutationService + SearchService + Send + Sync {}

// ────────────────────────────────────────────────────────────────────────
// 1. Standalone Single-Node Production Service
// ────────────────────────────────────────────────────────────────────────

pub struct StandaloneService {
    index: Arc<HNSQRIndex>,
}

impl StandaloneService {
    pub fn new(index: Arc<HNSQRIndex>) -> Self {
        Self { index }
    }

    pub fn index(&self) -> &Arc<HNSQRIndex> {
        &self.index
    }
}

impl MutationService for StandaloneService {
    fn upsert(&self, ctx: &RequestContext, req: UpsertRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Upsert".to_string()));
        }

        if let Some(meta) = req.metadata {
            self.index.insert_with_metadata(req.id.as_str(), req.vector, meta)?;
        } else {
            self.index.insert(req.id.as_str(), req.vector)?;
        }

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

    fn delete(&self, ctx: &RequestContext, req: DeleteRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Delete".to_string()));
        }

        self.index.remove(&req.id)?;

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

impl SearchService for StandaloneService {
    fn search(
        &self,
        _ctx: &RequestContext,
        query: &VectorEmbedding,
        k: usize,
        _rerank_plan: SemanticRerankPlan,
    ) -> HNSQRResult<Vec<(Arc<str>, SimilarityScore)>> {
        self.index.search(query, k)
    }
}

impl HNSQRService for StandaloneService {}

// ────────────────────────────────────────────────────────────────────────
// 2. Clustered Multi-Node Distributed Service
// ────────────────────────────────────────────────────────────────────────

pub struct ClusterService {
    coordinator: Arc<DistributedCoordinator>,
}

impl ClusterService {
    pub fn new(coordinator: Arc<DistributedCoordinator>) -> Self {
        Self { coordinator }
    }

    pub fn coordinator(&self) -> &Arc<DistributedCoordinator> {
        &self.coordinator
    }
}

impl MutationService for ClusterService {
    fn upsert(&self, ctx: &RequestContext, req: UpsertRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Upsert".to_string()));
        }

        let commit_receipt = self.coordinator.insert_fenced(req.id.clone(), req.vector, ctx.epoch)?;
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

    fn delete(&self, ctx: &RequestContext, req: DeleteRequest) -> HNSQRResult<MutationReceipt> {
        if ctx.role == AccessRole::ReadOnly {
            return Err(HNSQRError::Unauthorized("Role ReadOnly cannot perform Delete".to_string()));
        }

        let commit_receipt = self.coordinator.delete(&req.id)?;
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
        let pinned = self.coordinator.obtain_pinned_read_snapshot(0, ReadConsistency::Linearizable)?;
        Ok(self.coordinator.search_pinned(&pinned, query, k, rerank_plan))
    }
}

impl HNSQRService for ClusterService {}
