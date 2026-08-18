/* hnsqr/src/segment.rs */
//!▫~•◦-------------------------------‣
//! # LSM-Style Mutable Segmented Storage Engine & Online Compaction
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides high-throughput continuous inserts, deletions, and non-blocking background
//! compaction for HNSQR vector collections.
//!
//! ## Architecture
//! - **`MutableSegment`**: Lock-free/low-lock append-only buffer holding live incoming vectors,
//!   tombstones, delta Rivero routing, and fast LUTz codes.
//! - **`ImmutableSegment`**: Sealed, read-only segment with precomputed Rivero topology
//!   and LUTz codebooks.
//! - **`SegmentedEngine`**: Coordinates fan-out queries across active and immutable segments,
//!   merges Top-K finalists, and performs atomic generation swaps during background compaction.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use parking_lot::RwLock as PlRwLock;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

use crate::planning::planner::RetrievalContract;
use crate::proof::lutz::{
    LutzCertifier, LutzCode, LutzGlobalCertified, LutzQueryTable, SemanticRerankPlan,
};
use crate::proof::{GlobalExactProofSearch, SegmentProofView, SemanticProofTree};
use crate::rivero::{RiveroCompiler, RiveroProfile, RiveroTerritoryIndex};
use crate::storage::wal::{DurabilityPolicy, WalManager, WalMutation};
use crate::metadata::index::MetadataValue;
use crate::{HNSQRError, HNSQRResult, NodeId, NodeIndex, SimilarityScore, VectorEmbedding};

pub type SegmentId = u64;

/// Segment lifecycle state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SegmentState {
    #[default]
    ActiveMutable,
    FlushedImmutable,
    Compacted,
}

/// Statistics and telemetry for a single storage segment.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SegmentStats {
    pub segment_id: SegmentId,
    pub state: SegmentState,
    pub live_vectors: usize,
    pub deleted_vectors: usize,
    pub capacity: usize,
    pub memory_bytes: usize,
}

/// A mutable, append-only memory segment for active writes and deletes.
pub struct MutableSegment {
    pub id: SegmentId,
    pub dimension: usize,
    pub max_capacity: usize,
    vectors: PlRwLock<Vec<VectorEmbedding>>,
    lutz_codes: PlRwLock<Vec<LutzCode>>,
    id_to_slot: PlRwLock<HashMap<Arc<str>, u32>>,
    slot_to_id: PlRwLock<Vec<Arc<str>>>,
    tombstones: PlRwLock<RoaringBitmap>,
    territories: RiveroTerritoryIndex,
    compiler: RiveroCompiler,
}

impl MutableSegment {
    /// Creates a new mutable segment with a specified capacity threshold.
    pub fn new(id: SegmentId, dimension: usize, max_capacity: usize) -> Self {
        Self {
            id,
            dimension,
            max_capacity,
            vectors: PlRwLock::new(Vec::with_capacity(max_capacity.min(4096))),
            lutz_codes: PlRwLock::new(Vec::with_capacity(max_capacity.min(4096))),
            id_to_slot: PlRwLock::new(HashMap::with_capacity(max_capacity.min(4096))),
            slot_to_id: PlRwLock::new(Vec::with_capacity(max_capacity.min(4096))),
            tombstones: PlRwLock::new(RoaringBitmap::new()),
            territories: RiveroTerritoryIndex::new(),
            compiler: RiveroCompiler::new(dimension),
        }
    }

    /// Appends a vector to the mutable segment. Returns error if capacity exceeded.
    pub fn insert(&self, id: impl Into<NodeId>, vector: VectorEmbedding) -> HNSQRResult<u32> {
        if vector.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: vector.dimension(),
            });
        }

        let node_id: Arc<str> = id.into();
        let mut id_map = self.id_to_slot.write();
        let mut vectors = self.vectors.write();

        if vectors.len() >= self.max_capacity {
            return Err(HNSQRError::IndexFull(self.max_capacity));
        }

        // If ID already exists, mark old slot as tombstone
        if let Some(&old_slot) = id_map.get(&node_id) {
            self.tombstones.write().insert(old_slot);
        }

        let slot = vectors.len() as u32;
        let lutz_code = LutzCode::encode(&vector, true);
        let address = self.compiler.compile(vector.complex_data());

        self.territories.insert(&address, slot);
        self.lutz_codes.write().push(lutz_code);
        self.slot_to_id.write().push(node_id.clone());
        id_map.insert(node_id, slot);
        vectors.push(vector);

        Ok(slot)
    }

    /// Marks a node ID as deleted via tombstone.
    pub fn delete(&self, id: &str) -> bool {
        let id_map = self.id_to_slot.read();
        if let Some(&slot) = id_map.get(id) {
            self.tombstones.write().insert(slot);
            true
        } else {
            false
        }
    }

    /// Returns whether this mutable segment has reached capacity and should be frozen.
    pub fn is_full(&self) -> bool {
        self.vectors.read().len() >= self.max_capacity
    }

    /// Returns segment statistics.
    pub fn stats(&self) -> SegmentStats {
        let total = self.vectors.read().len();
        let deleted = self.tombstones.read().len() as usize;
        let live = total.saturating_sub(deleted);
        let mem = total * (self.dimension * 8 + std::mem::size_of::<LutzCode>() + 64);

        SegmentStats {
            segment_id: self.id,
            state: SegmentState::ActiveMutable,
            live_vectors: live,
            deleted_vectors: deleted,
            capacity: self.max_capacity,
            memory_bytes: mem,
        }
    }

    /// Searches within this mutable segment.
    pub fn search(
        &self,
        query: &VectorEmbedding,
        k: usize,
        rerank_plan: SemanticRerankPlan,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if k == 0 {
            return Vec::new();
        }

        let vectors = self.vectors.read();
        let tombstones = self.tombstones.read();
        let slot_to_id = self.slot_to_id.read();
        let lutz_codes = self.lutz_codes.read();

        let total = vectors.len();
        if total == 0 {
            return Vec::new();
        }

        let q_addr = self.compiler.compile(query.complex_data());
        let config = RiveroProfile::Strict.config();
        let candidate_slots: Vec<NodeIndex> =
            self.territories
                .with_candidates_config(&q_addr, &config, |cands, _| {
                    cands
                        .iter()
                        .copied()
                        .filter(|&slot| (slot as usize) < total && !tombstones.contains(slot))
                        .collect()
                });

        if candidate_slots.is_empty() {
            return Vec::new();
        }

        let concrete_plan = rerank_plan.resolve(&candidate_slots, self.dimension * 8, false);

        let top_slots: Vec<(NodeIndex, SimilarityScore)> = match concrete_plan {
            SemanticRerankPlan::LutzFastScan => {
                let query_lut = LutzQueryTable::build(query);
                let (certified, _) = LutzCertifier::certify(
                    &query_lut,
                    &candidate_slots,
                    |slot| lutz_codes.get(slot as usize),
                    |slot| (query.dot_product_complex(&vectors[slot as usize])).re,
                    k,
                );
                certified
            }
            SemanticRerankPlan::ExactSimd | SemanticRerankPlan::Auto => {
                let mut scored: Vec<(NodeIndex, SimilarityScore)> = candidate_slots
                    .into_iter()
                    .map(|slot| {
                        (
                            slot,
                            (query.dot_product_complex(&vectors[slot as usize])).re,
                        )
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                scored.truncate(k);
                scored
            }
        };

        top_slots
            .into_iter()
            .map(|(slot, score)| (slot_to_id[slot as usize].clone(), score))
            .collect()
    }

    /// Searches within this mutable segment enforcing a declared retrieval contract.
    pub fn search_with_contract(
        &self,
        query: &VectorEmbedding,
        k: usize,
        contract: RetrievalContract,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if k == 0 {
            return Vec::new();
        }

        let vectors = self.vectors.read();
        let tombstones = self.tombstones.read();
        let slot_to_id = self.slot_to_id.read();
        let lutz_codes = self.lutz_codes.read();
        let total = vectors.len();
        if total == 0 {
            return Vec::new();
        }

        match contract {
            RetrievalContract::Exact => {
                let mut scored: Vec<(NodeIndex, SimilarityScore)> = (0..total as NodeIndex)
                    .filter(|&slot| !tombstones.contains(slot))
                    .map(|slot| {
                        (
                            slot,
                            (query.dot_product_complex(&vectors[slot as usize])).re,
                        )
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                scored.truncate(k);
                scored
                    .into_iter()
                    .map(|(s, score)| (slot_to_id[s as usize].clone(), score))
                    .collect()
            }
            RetrievalContract::Certified => {
                let q_addr = self.compiler.compile(query.complex_data());
                let config = RiveroProfile::Strict.config();
                let seed_cands: Vec<(NodeIndex, SimilarityScore)> = self
                    .territories
                    .with_candidates_config(&q_addr, &config, |cands, _| {
                        let mut s: Vec<_> = cands
                            .iter()
                            .copied()
                            .filter(|&slot| (slot as usize) < total && !tombstones.contains(slot))
                            .map(|slot| {
                                (
                                    slot,
                                    (query.dot_product_complex(&vectors[slot as usize])).re,
                                )
                            })
                            .collect();
                        s.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                        s
                    });

                let query_lut = LutzQueryTable::build(query);
                let (certified, _) = LutzGlobalCertified::certify_global(
                    &query_lut,
                    k,
                    &seed_cands,
                    total,
                    None,
                    |slot| !tombstones.contains(slot),
                    |slot| lutz_codes.get(slot as usize),
                    |slot| (query.dot_product_complex(&vectors[slot as usize])).re,
                );

                certified
                    .into_iter()
                    .map(|(s, score)| (slot_to_id[s as usize].clone(), score))
                    .collect()
            }
            _ => self.search(query, k, SemanticRerankPlan::Auto),
        }
    }
}

/// An immutable, optimized segment.
pub struct ImmutableSegment {
    pub id: SegmentId,
    pub dimension: usize,
    vectors: Vec<VectorEmbedding>,
    lutz_codes: Vec<LutzCode>,
    id_to_slot: HashMap<Arc<str>, u32>,
    slot_to_id: Vec<Arc<str>>,
    tombstones: PlRwLock<RoaringBitmap>,
    territories: RiveroTerritoryIndex,
    compiler: RiveroCompiler,
    pub proof_tree: Arc<SemanticProofTree>,
}

impl ImmutableSegment {
    /// Freezes a mutable segment into an immutable segment.
    pub fn freeze(mutable: &MutableSegment) -> Self {
        let vectors = mutable.vectors.read().clone();
        let lutz_codes = mutable.lutz_codes.read().clone();
        let id_to_slot = mutable.id_to_slot.read().clone();
        let slot_to_id = mutable.slot_to_id.read().clone();
        let tombstones = mutable.tombstones.read().clone();

        let compiler = RiveroCompiler::new(mutable.dimension);
        let territories = RiveroTerritoryIndex::new();

        let live_slots: Vec<u32> = (0..vectors.len() as u32)
            .filter(|&slot| !tombstones.contains(slot))
            .collect();

        // Re-index into clean territory structures
        for &slot in &live_slots {
            let v = &vectors[slot as usize];
            let addr = compiler.compile(v.complex_data());
            territories.insert(&addr, slot);
        }

        let proof_tree = Arc::new(SemanticProofTree::build(
            &vectors,
            &live_slots,
            mutable.dimension,
        ));

        Self {
            id: mutable.id,
            dimension: mutable.dimension,
            vectors,
            lutz_codes,
            id_to_slot,
            slot_to_id,
            tombstones: PlRwLock::new(tombstones),
            territories,
            compiler,
            proof_tree,
        }
    }

    /// Marks a node ID as deleted in this immutable segment.
    pub fn delete(&self, id: &str) -> bool {
        if let Some(&slot) = self.id_to_slot.get(id) {
            self.tombstones.write().insert(slot);
            true
        } else {
            false
        }
    }

    /// Returns segment statistics.
    pub fn stats(&self) -> SegmentStats {
        let total = self.vectors.len();
        let deleted = self.tombstones.read().len() as usize;
        let live = total.saturating_sub(deleted);
        let mem = total * (self.dimension * 8 + std::mem::size_of::<LutzCode>() + 64);

        SegmentStats {
            segment_id: self.id,
            state: SegmentState::FlushedImmutable,
            live_vectors: live,
            deleted_vectors: deleted,
            capacity: total,
            memory_bytes: mem,
        }
    }

    /// Searches within this immutable segment.
    pub fn search(
        &self,
        query: &VectorEmbedding,
        k: usize,
        rerank_plan: SemanticRerankPlan,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if k == 0 || self.vectors.is_empty() {
            return Vec::new();
        }

        let tombstones = self.tombstones.read();
        let q_addr = self.compiler.compile(query.complex_data());
        let config = RiveroProfile::Strict.config();

        let candidate_slots: Vec<NodeIndex> =
            self.territories
                .with_candidates_config(&q_addr, &config, |cands, _| {
                    cands
                        .iter()
                        .copied()
                        .filter(|&slot| !tombstones.contains(slot))
                        .collect()
                });

        if candidate_slots.is_empty() {
            return Vec::new();
        }

        let concrete_plan = rerank_plan.resolve(&candidate_slots, self.dimension * 8, false);

        let top_slots: Vec<(NodeIndex, SimilarityScore)> = match concrete_plan {
            SemanticRerankPlan::LutzFastScan => {
                let query_lut = LutzQueryTable::build(query);
                let (certified, _) = LutzCertifier::certify(
                    &query_lut,
                    &candidate_slots,
                    |slot| self.lutz_codes.get(slot as usize),
                    |slot| (query.dot_product_complex(&self.vectors[slot as usize])).re,
                    k,
                );
                certified
            }
            SemanticRerankPlan::ExactSimd | SemanticRerankPlan::Auto => {
                let mut scored: Vec<(NodeIndex, SimilarityScore)> = candidate_slots
                    .into_iter()
                    .map(|slot| {
                        (
                            slot,
                            (query.dot_product_complex(&self.vectors[slot as usize])).re,
                        )
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                scored.truncate(k);
                scored
            }
        };

        top_slots
            .into_iter()
            .map(|(slot, score)| (self.slot_to_id[slot as usize].clone(), score))
            .collect()
    }

    /// Searches within this immutable segment enforcing a declared retrieval contract.
    pub fn search_with_contract(
        &self,
        query: &VectorEmbedding,
        k: usize,
        contract: RetrievalContract,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if k == 0 || self.vectors.is_empty() {
            return Vec::new();
        }

        let tombstones = self.tombstones.read();
        let total = self.vectors.len();

        match contract {
            RetrievalContract::Exact => {
                let mut scored: Vec<(NodeIndex, SimilarityScore)> = (0..total as NodeIndex)
                    .filter(|&slot| !tombstones.contains(slot))
                    .map(|slot| {
                        (
                            slot,
                            (query.dot_product_complex(&self.vectors[slot as usize])).re,
                        )
                    })
                    .collect();
                scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                scored.truncate(k);
                scored
                    .into_iter()
                    .map(|(s, score)| (self.slot_to_id[s as usize].clone(), score))
                    .collect()
            }
            RetrievalContract::Certified => {
                let q_norm = query.clone().into_normalized();
                let q_addr = self.compiler.compile(q_norm.complex_data());
                let config = RiveroProfile::Strict.config();
                let mut seed_cands: Vec<NodeIndex> = Vec::new();
                self.territories
                    .with_candidates_config(&q_addr, &config, |cands, _| {
                        seed_cands.extend(cands.iter().copied().filter(|&s| !tombstones.contains(s)));
                    });

                let seg_view = SegmentProofView {
                    tree: &self.proof_tree,
                    vectors: &self.vectors,
                    lutz_codes: Some(&self.lutz_codes),
                    tombstones: Some(&tombstones),
                };

                let (certified, _) = GlobalExactProofSearch::search(
                    &q_norm,
                    k,
                    &[seg_view],
                    &[],
                    &seed_cands,
                    None,
                );

                certified
                    .into_iter()
                    .map(|(s, score)| (self.slot_to_id[s as usize].clone(), score))
                    .collect()
            }
            _ => self.search(query, k, SemanticRerankPlan::Auto),
        }
    }
}

/// Segmented Storage Engine coordinating active writes, frozen segments, and background compaction.
pub struct SegmentedEngine {
    pub dimension: usize,
    pub max_mutable_capacity: usize,
    next_segment_id: AtomicU64,
    generation: AtomicU64,
    active_mutable: RwLock<Arc<MutableSegment>>,
    immutable_segments: RwLock<Vec<Arc<ImmutableSegment>>>,
    metadata_store: RwLock<HashMap<Arc<str>, HashMap<String, MetadataValue>>>,
    pub wal: RwLock<Option<Arc<WalManager>>>,
    pub wal_durability: RwLock<DurabilityPolicy>,
}

impl SegmentedEngine {
    /// Creates a new segmented engine.
    pub fn new(dimension: usize, max_mutable_capacity: usize) -> Self {
        let initial_mutable = Arc::new(MutableSegment::new(1, dimension, max_mutable_capacity));
        Self {
            dimension,
            max_mutable_capacity,
            next_segment_id: AtomicU64::new(2),
            generation: AtomicU64::new(1),
            active_mutable: RwLock::new(initial_mutable),
            immutable_segments: RwLock::new(Vec::new()),
            metadata_store: RwLock::new(HashMap::new()),
            wal: RwLock::new(None),
            wal_durability: RwLock::new(DurabilityPolicy::WalSync),
        }
    }

    /// Returns the current monotonic segment storage generation.
    pub fn active_generation(&self) -> u64 {
        self.generation.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Configures the engine with a dedicated Write-Ahead Log manager.
    pub fn with_wal(self, wal: Arc<WalManager>, durability: DurabilityPolicy) -> Self {
        *self.wal.write().unwrap() = Some(wal);
        *self.wal_durability.write().unwrap() = durability;
        self
    }

    /// Sets the WAL manager at runtime.
    pub fn set_wal(&self, wal: Arc<WalManager>, durability: DurabilityPolicy) {
        *self.wal.write().unwrap() = Some(wal);
        *self.wal_durability.write().unwrap() = durability;
    }

    /// Applies a committed upsert with full metadata preservation.
    pub fn apply_committed_upsert(
        &self,
        id: impl Into<NodeId>,
        vector: VectorEmbedding,
        metadata: Option<&HashMap<String, MetadataValue>>,
    ) -> HNSQRResult<()> {
        let node_id: Arc<str> = id.into();
        self.apply_upsert_unlogged(node_id.clone(), vector)?;
        if let Some(meta) = metadata {
            self.metadata_store.write().unwrap().insert(node_id, meta.clone());
        }
        Ok(())
    }

    /// Applies a committed metadata patch.
    pub fn apply_committed_metadata_patch(
        &self,
        id: &str,
        patch: &HashMap<String, MetadataValue>,
    ) -> HNSQRResult<()> {
        let mut store = self.metadata_store.write().unwrap();
        let entry: &mut HashMap<String, MetadataValue> = store.entry(Arc::from(id)).or_insert_with(HashMap::new);
        for (k, v) in patch {
            entry.insert(k.clone(), v.clone());
        }
        Ok(())
    }

    /// Retrieves stored metadata for a given vector ID.
    pub fn get_metadata(&self, id: &str) -> Option<HashMap<String, MetadataValue>> {
        self.metadata_store.read().unwrap().get(id).cloned()
    }

    /// Applies a replicated or replayed WAL mutation directly to the memory segments.
    pub fn apply_wal_mutation(&self, mutation: &WalMutation) -> HNSQRResult<()> {
        match mutation {
            WalMutation::Upsert { external_id, vector, metadata } => {
                self.apply_committed_upsert(external_id.as_str(), vector.clone(), metadata.as_ref())
            }
            WalMutation::Delete { external_id } => {
                self.apply_delete_unlogged(external_id);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Low-level unlogged upsert primitive for replicated state-machine application.
    pub(crate) fn apply_upsert_unlogged(&self, id: impl Into<NodeId>, vector: VectorEmbedding) -> HNSQRResult<()> {
        let node_id: Arc<str> = id.into();
        self.delete_internal(&node_id);
        let active = { self.active_mutable.read().unwrap().clone() };
        match active.insert(node_id.clone(), vector.clone()) {
            Ok(_) => Ok(()),
            Err(HNSQRError::IndexFull(_)) => {
                self.rotate_active()?;
                let new_active = { self.active_mutable.read().unwrap().clone() };
                new_active.insert(node_id, vector)?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Low-level unlogged delete primitive for replicated state-machine application.
    pub(crate) fn apply_delete_unlogged(&self, id: &str) -> bool {
        self.metadata_store.write().unwrap().remove(id);
        self.delete_internal(id)
    }

    /// Snapshot of current immutable segments for RAII read pinning.
    pub fn immutable_segments_snapshot(&self) -> Vec<Arc<ImmutableSegment>> {
        self.immutable_segments.read().unwrap().clone()
    }

    /// Active mutable segment reference for RAII read pinning.
    pub fn active_mutable_segment(&self) -> Arc<MutableSegment> {
        self.active_mutable.read().unwrap().clone()
    }

    /// Performs search pinned to a specific snapshot of immutables and active segment view.
    pub fn search_pinned(
        &self,
        immutables: &[Arc<ImmutableSegment>],
        active: &MutableSegment,
        query: &VectorEmbedding,
        k: usize,
        rerank_plan: SemanticRerankPlan,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        let mut all_results = Vec::new();
        for imm in immutables {
            all_results.extend(imm.search(query, k, rerank_plan));
        }
        all_results.extend(active.search(query, k, rerank_plan));
        all_results.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        all_results.truncate(k);
        all_results
    }

    /// Inserts a vector. Durably logs to WAL first if configured, then updates in-memory segments.
    pub fn insert(&self, id: impl Into<NodeId>, vector: VectorEmbedding) -> HNSQRResult<()> {
        let node_id: Arc<str> = id.into();

        // 1. Append to WAL if configured
        if let Some(wal) = self.wal.read().unwrap().as_ref() {
            let durability = *self.wal_durability.read().unwrap();
            let mutation = WalMutation::Upsert {
                external_id: node_id.to_string(),
                vector: vector.clone(),
                metadata: None,
            };
            wal.append(&mutation, durability)?;
        }

        // 2. Mark any existing instance across all segments as deleted
        self.delete_internal(&node_id);

        // 3. Try inserting into active mutable
        let active = { self.active_mutable.read().unwrap().clone() };

        match active.insert(node_id.clone(), vector.clone()) {
            Ok(_) => Ok(()),
            Err(HNSQRError::IndexFull(_)) => {
                // Freeze active segment and rotate
                self.rotate_active()?;
                let new_active = { self.active_mutable.read().unwrap().clone() };
                new_active.insert(node_id, vector)?;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    /// Deletes a vector by ID across all active and immutable segments. Logs to WAL if enabled.
    pub fn delete(&self, id: &str) -> bool {
        if let Some(wal) = self.wal.read().unwrap().as_ref() {
            let durability = *self.wal_durability.read().unwrap();
            let mutation = WalMutation::Delete {
                external_id: id.to_string(),
            };
            let _ = wal.append(&mutation, durability);
        }

        self.delete_internal(id)
    }

    fn delete_internal(&self, id: &str) -> bool {
        let mut found = false;

        // Delete from active mutable
        if self.active_mutable.read().unwrap().delete(id) {
            found = true;
        }

        // Delete from all immutable segments
        let immutables = self.immutable_segments.read().unwrap().clone();
        for seg in immutables {
            if seg.delete(id) {
                found = true;
            }
        }

        found
    }

    /// Replays uncommitted WAL records from disk to restore state after crash.
    pub fn recover_from_wal(&self) -> HNSQRResult<usize> {
        let wal_opt = self.wal.read().unwrap().clone();
        if let Some(wal) = wal_opt {
            let replayed_count = std::sync::atomic::AtomicUsize::new(0);
            wal.replay(0, |_, mutation| {
                self.apply_wal_mutation(&mutation)?;
                replayed_count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })?;
            Ok(replayed_count.load(Ordering::Relaxed))
        } else {
            Ok(0)
        }
    }

    /// Rotates the current active mutable segment into an immutable frozen segment.
    pub fn rotate_active(&self) -> HNSQRResult<()> {
        let mut active_guard = self.active_mutable.write().unwrap();
        let mut immut_guard = self.immutable_segments.write().unwrap();

        let frozen = Arc::new(ImmutableSegment::freeze(&active_guard));
        immut_guard.push(frozen);

        let next_id = self.next_segment_id.fetch_add(1, Ordering::SeqCst);
        *active_guard = Arc::new(MutableSegment::new(
            next_id,
            self.dimension,
            self.max_mutable_capacity,
        ));

        Ok(())
    }

    /// Global search across all active and immutable segments with Top-K score merging.
    pub fn search(
        &self,
        query: &VectorEmbedding,
        k: usize,
        rerank_plan: SemanticRerankPlan,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if k == 0 {
            return Vec::new();
        }

        let active = self.active_mutable.read().unwrap().clone();
        let immutables = self.immutable_segments.read().unwrap().clone();

        // Search active segment
        let mut all_results = active.search(query, k, rerank_plan);

        // Search immutable segments
        for seg in immutables {
            let seg_res = seg.search(query, k, rerank_plan);
            all_results.extend(seg_res);
        }

        // Merge and deduplicate by score descending
        all_results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        // Remove duplicate IDs (keeping the highest score)
        let mut unique_results = Vec::with_capacity(k);
        let mut seen = std::collections::HashSet::with_capacity(k * 2);

        for (id, score) in all_results {
            if seen.insert(id.clone()) {
                unique_results.push((id, score));
                if unique_results.len() >= k {
                    break;
                }
            }
        }

        unique_results
    }

    /// Global search across all active and immutable segments enforcing a declared contract.
    pub fn search_with_contract(
        &self,
        query: &VectorEmbedding,
        k: usize,
        contract: RetrievalContract,
    ) -> Vec<(Arc<str>, SimilarityScore)> {
        if k == 0 {
            return Vec::new();
        }

        let active = self.active_mutable.read().unwrap().clone();
        let immutables = self.immutable_segments.read().unwrap().clone();

        let mut all_results = active.search_with_contract(query, k, contract);
        for seg in immutables {
            let seg_res = seg.search_with_contract(query, k, contract);
            all_results.extend(seg_res);
        }

        all_results.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        let mut unique_results = Vec::with_capacity(k);
        let mut seen = std::collections::HashSet::with_capacity(k * 2);

        for (id, score) in all_results {
            if seen.insert(id.clone()) {
                unique_results.push((id, score));
                if unique_results.len() >= k {
                    break;
                }
            }
        }

        unique_results
    }

    /// Compaction: Merges all immutable segments, purges tombstones, and rebuilds optimized structures.
    pub fn compact(&self) -> HNSQRResult<usize> {
        let immutables = {
            let guard = self.immutable_segments.read().unwrap();
            guard.clone()
        };

        if immutables.len() < 2 {
            return Ok(0); // Nothing to compact
        }

        // Phase 1: Pre-calculate exact surviving live count and latest IDs without vector clones
        let mut seen_ids = std::collections::HashSet::new();
        let mut live_count = 0usize;
        let mut total_purged = 0usize;

        // Iterate newest to oldest to give precedence to latest versions
        for seg in immutables.iter().rev() {
            let tombstones = seg.tombstones.read();
            for slot in 0..seg.vectors.len() {
                if !tombstones.contains(slot as u32) {
                    let id = &seg.slot_to_id[slot];
                    if seen_ids.insert(id.clone()) {
                        live_count += 1;
                    } else {
                        total_purged += 1;
                    }
                } else {
                    total_purged += 1;
                }
            }
        }

        // Phase 2: Exactly pre-sized allocation avoiding dynamic reallocations
        let mut merged_vectors = Vec::with_capacity(live_count);
        let mut merged_id_to_slot = HashMap::with_capacity(live_count);
        let mut merged_slot_to_id = Vec::with_capacity(live_count);
        let mut lutz_codes = Vec::with_capacity(live_count);

        let compiler = RiveroCompiler::new(self.dimension);
        let territories = RiveroTerritoryIndex::new();
        let mut inserted_ids = std::collections::HashSet::with_capacity(live_count);

        // Streaming transfer: extract vectors, encode LUTz and populate Rivero in a single pass
        for seg in immutables.iter().rev() {
            let tombstones = seg.tombstones.read();
            for (slot, v) in seg.vectors.iter().enumerate() {
                if !tombstones.contains(slot as u32) {
                    let id = &seg.slot_to_id[slot];
                    if inserted_ids.insert(id.clone()) {
                        let new_slot = merged_vectors.len() as u32;
                        merged_id_to_slot.insert(id.clone(), new_slot);
                        merged_slot_to_id.push(id.clone());

                        let addr = compiler.compile(v.complex_data());
                        territories.insert(&addr, new_slot);
                        lutz_codes.push(LutzCode::encode(v, true));
                        merged_vectors.push(v.clone());
                    }
                }
            }
        }

        let live_slots: Vec<u32> = (0..merged_vectors.len() as u32).collect();
        let proof_tree = Arc::new(SemanticProofTree::build(
            &merged_vectors,
            &live_slots,
            self.dimension,
        ));

        let new_compact_id = self.next_segment_id.fetch_add(1, Ordering::SeqCst);
        let compacted_segment = Arc::new(ImmutableSegment {
            id: new_compact_id,
            dimension: self.dimension,
            vectors: merged_vectors,
            lutz_codes,
            id_to_slot: merged_id_to_slot,
            slot_to_id: merged_slot_to_id,
            tombstones: PlRwLock::new(RoaringBitmap::new()),
            territories,
            compiler,
            proof_tree,
        });

        // Atomic swap of immutable segments list
        {
            let mut guard = self.immutable_segments.write().unwrap();
            *guard = vec![compacted_segment];
        }

        Ok(total_purged)
    }

    /// Aggregates stats across all segments.
    pub fn stats(&self) -> Vec<SegmentStats> {
        let mut list = Vec::new();
        list.push(self.active_mutable.read().unwrap().stats());
        for seg in self.immutable_segments.read().unwrap().iter() {
            list.push(seg.stats());
        }
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    #[test]
    fn test_segment_insert_delete_and_search() {
        let dim = 16;
        let engine = SegmentedEngine::new(dim, 10);

        for i in 0..25 {
            let v = VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 11 + d) as f32, (i * 7 + d * 3) as f32))
                    .collect(),
            )
            .into_normalized();
            engine.insert(format!("doc_{i}"), v).unwrap();
        }

        let stats = engine.stats();
        assert!(stats.len() >= 3, "Expected multiple rotated segments");

        let query = VectorEmbedding::from_complex(
            (0..dim)
                .map(|d| Complex32::new((5 * 11 + d) as f32, (5 * 7 + d * 3) as f32))
                .collect(),
        )
        .into_normalized();

        let topk = engine.search(&query, 5, SemanticRerankPlan::ExactSimd);
        assert_eq!(topk.len(), 5);
        assert_eq!(topk[0].0.as_ref(), "doc_5");

        // Delete doc_5
        assert!(engine.delete("doc_5"));
        let topk_after = engine.search(&query, 5, SemanticRerankPlan::ExactSimd);
        assert_ne!(
            topk_after[0].0.as_ref(),
            "doc_5",
            "Deleted doc_5 must not appear"
        );
    }

    #[test]
    fn test_segment_compaction_purges_tombstones() {
        let dim = 8;
        let engine = SegmentedEngine::new(dim, 5);

        for i in 0..15 {
            let v = VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i + d) as f32, 0.0))
                    .collect(),
            )
            .into_normalized();
            engine.insert(format!("item_{i}"), v).unwrap();
        }

        // Delete multiple items
        engine.delete("item_0");
        engine.delete("item_1");
        engine.delete("item_5");

        let stats_before = engine.stats();
        let total_deleted_before: usize = stats_before.iter().map(|s| s.deleted_vectors).sum();
        assert!(total_deleted_before >= 3);

        // Run compaction
        let purged = engine.compact().unwrap();
        assert!(
            purged >= 3,
            "Compaction should purge all deleted tombstones"
        );

        let stats_after = engine.stats();
        assert_eq!(stats_after.len(), 2); // 1 active + 1 compacted immutable
        assert_eq!(stats_after[1].deleted_vectors, 0);
    }
}
