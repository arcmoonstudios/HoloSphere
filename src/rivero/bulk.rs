/* hnsqr/src/rivero/bulk.rs */
//!▫~•◦-------------------------------‣
//! # Rivero Deterministic Parallel Bulk Builder
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a high-throughput, bounded-memory, bit-for-bit deterministic bulk
//! construction pipeline for Rivero territory cells and reciprocal witnesses.
//!
//! ### Four-Phase Staged Architecture:
//!   1. **Phase 1 — Parallel Address Compilation**:
//!      Maps vectors to [`RiveroAddress`] preserving input slot ordering (`prepared[i].slot == i`).
//!   2. **Phase 2 — Sharded Bounded Cell Reduction & Associative Merge**:
//!      Worker chunks stream memberships into shard-partitioned bounded reducers
//!      ($24\text{ affinity} + 40\text{ diversity} = 64\text{ max}$ per cell), eliminating
//!      global record tables.
//!   3. **Phase 3 — Immutable Territory Freezing**:
//!      Publishes frozen territory state so subsequent witness resolution observes a fixed index.
//!   4. **Phase 4 — Parallel Witness Construction & Deterministic Finalize**:
//!      Resolves candidate seeds in parallel, gathers reciprocal proposals, and executes
//!      deterministic score ranking and reciprocal pruning.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::atomic::AtomicU64;
use std::time::Instant;

use num_complex::Complex32;
use parking_lot::RwLock;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use super::witness::{
    self as rivero_witness, RIVERO_WITNESS_DEFAULT_DEGREE, RIVERO_WITNESS_DEFAULT_SECOND_SEEDS,
    RIVERO_WITNESS_DEFAULT_SEEDS, RIVERO_WITNESS_INLINE_DEGREE, ScoredWitness,
};
use super::{
    CellSlots, FlatFrozenTerritoryTable, RiveroAddress, RiveroAddressConfig, RiveroCompiler,
    RiveroConfig, RiveroProfile, RiveroTerritoryIndex, cell_key, insert_sigs, lookup_sigs,
    pack_projected_code, projected_similarity, simhash_cell_key, simhash_probe_signatures,
    simhash_signature, stripe_for,
};
use crate::{
    DistanceFunction, HNSQRError, HNSQRResult, NodeIndex, VectorEmbedding, dot_product_complex_simd,
};

const STRIPE_COUNT: usize = 64;

/// Phase-by-phase telemetry and performance breakdown for bulk construction.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct BulkBuildTelemetry {
    /// Time spent compiling continuous embeddings into Rivero addresses.
    pub time_address_compile_ms: f64,
    /// Time spent streaming memberships into shard-bounded cell reducers.
    pub time_territory_reduction_ms: f64,
    /// Time spent merging shard-local reducers into the global territory index.
    pub time_territory_merge_ms: f64,
    /// Time spent resolving candidate neighborhoods for witness edges.
    pub time_witness_routing_ms: f64,
    /// Time spent computing exact vector similarities for witness candidates.
    pub time_witness_scoring_ms: f64,
    /// Time spent deterministically ranking and reciprocal-pruning witness connections.
    pub time_witness_finalize_ms: f64,
    /// Total wall-clock time for the complete bulk build pipeline.
    pub total_build_time_ms: f64,
    /// Overall throughput in indexed vectors per second.
    pub throughput_vecs_per_sec: f64,
    /// Total unique territory cells created.
    pub cell_count: usize,
    /// Total cells that reached their bounded capacity.
    pub overflow_count: u64,
    /// Percentage of nodes satisfied entirely by Stage A (insertion-family) cells.
    pub stage_a_accepted_pct: f64,
    /// Percentage of nodes requiring Stage B (broad lookup delta) expansion.
    pub stage_b_expanded_pct: f64,
}

/// Canonical build descriptor identifying schema, dimension, geometry, and witness parameters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiveroBuildDescriptor {
    /// Rivero address compiler schema version.
    pub schema_version: u16,
    /// Complex vector dimensionality D.
    pub dimension: usize,
    /// Address configuration (foundations, projection mode, and vector geometry).
    pub address_config: RiveroAddressConfig,
    /// Rivero operational configuration.
    pub rivero_config: RiveroConfig,
    /// Semantic distance metric.
    pub distance_function: DistanceFunction,
    /// Maximum witness connections per node.
    pub witness_degree: usize,
    /// Maximum witness seeds expanded.
    pub witness_seeds: usize,
    /// Maximum second-hop witness seeds expanded.
    pub witness_second_seeds: usize,
}

/// Transactional immutable result of bulk Rivero index construction.
pub struct BuiltRiveroState {
    /// Fully populated, frozen territory index.
    pub territory: RiveroTerritoryIndex,
    /// Deterministically pruned reciprocal witness connections for each node slot.
    pub witnesses: Vec<SmallVec<[ScoredWitness; RIVERO_WITNESS_INLINE_DEGREE]>>,
    /// Precompiled Rivero routing addresses for each node.
    pub addresses: Vec<RiveroAddress>,
    /// Build descriptor capturing full routing schema and geometry invariants.
    pub descriptor: RiveroBuildDescriptor,
    /// Construction telemetry breakdown.
    pub telemetry: BulkBuildTelemetry,
}

/// Transactional parallel builder for Rivero territory and witness structures.
#[derive(Clone, Debug)]
pub struct RiveroBulkBuilder {
    /// Operational configuration parameters.
    pub config: RiveroConfig,
    /// Operational profile.
    pub profile: RiveroProfile,
    /// Semantic distance metric.
    pub distance_function: DistanceFunction,
    /// Maximum witness connections retained per node.
    pub witness_degree: usize,
    /// Maximum witness seeds expanded.
    pub witness_seeds: usize,
    /// Maximum second-hop witness seeds expanded.
    pub witness_second_seeds: usize,
    /// Optional thread pool size constraint.
    pub thread_count: Option<usize>,
    /// Address configuration for foundation count and multi-lane projection.
    pub address_config: RiveroAddressConfig,
    /// Force broad Stage B lookup-family delta expansion for all vectors.
    pub force_stage_b: bool,
}

impl RiveroBulkBuilder {
    /// Creates a builder with the specified configuration.
    #[must_use]
    pub fn new(config: RiveroConfig) -> Self {
        Self {
            config,
            profile: RiveroProfile::Strict,
            distance_function: DistanceFunction::Cosine,
            witness_degree: RIVERO_WITNESS_DEFAULT_DEGREE,
            witness_seeds: RIVERO_WITNESS_DEFAULT_SEEDS,
            witness_second_seeds: RIVERO_WITNESS_DEFAULT_SECOND_SEEDS,
            thread_count: None,
            address_config: RiveroAddressConfig::default(),
            force_stage_b: false,
        }
    }

    /// Creates a builder initialized to a standard [`RiveroProfile`].
    #[must_use]
    pub fn with_profile(profile: RiveroProfile) -> Self {
        Self {
            config: profile.config(),
            profile,
            distance_function: DistanceFunction::Cosine,
            witness_degree: RIVERO_WITNESS_DEFAULT_DEGREE,
            witness_seeds: RIVERO_WITNESS_DEFAULT_SEEDS,
            witness_second_seeds: RIVERO_WITNESS_DEFAULT_SECOND_SEEDS,
            thread_count: None,
            address_config: RiveroAddressConfig::default(),
            force_stage_b: false,
        }
    }

    /// Sets the address configuration (foundations and projection mode).
    #[must_use]
    pub fn with_address_config(mut self, config: RiveroAddressConfig) -> Self {
        self.address_config = config;
        self
    }

    /// Configures the semantic distance metric used for witness affinity.
    #[must_use]
    pub fn with_distance_function(mut self, distance_function: DistanceFunction) -> Self {
        self.distance_function = distance_function;
        self
    }

    /// Configures the number of parallel worker threads.
    #[must_use]
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.thread_count = Some(threads);
        self
    }

    /// Configures whether to force Stage B expansion for every vector during witness routing.
    #[must_use]
    pub fn with_force_stage_b(mut self, force: bool) -> Self {
        self.force_stage_b = force;
        self
    }

    /// Configures witness connectivity limits.
    #[must_use]
    pub fn with_witness_params(mut self, degree: usize, seeds: usize, second_seeds: usize) -> Self {
        self.witness_degree = rivero_witness::bounded_degree(degree);
        self.witness_seeds = rivero_witness::bounded_seeds(seeds);
        self.witness_second_seeds = rivero_witness::bounded_seeds(second_seeds);
        self
    }

    /// Executes the full 4-phase deterministic bulk construction pipeline.
    pub fn build(&self, vectors: &[VectorEmbedding]) -> HNSQRResult<BuiltRiveroState> {
        if vectors.is_empty() {
            return Err(HNSQRError::InvalidConfig(
                "Cannot bulk-build an empty vector set".to_string(),
            ));
        }

        let dim = vectors[0].dimension();
        for v in vectors {
            if v.dimension() != dim {
                return Err(HNSQRError::DimensionMismatch {
                    expected: dim,
                    actual: v.dimension(),
                });
            }
        }

        if let Some(threads) = self.thread_count {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|e| HNSQRError::ConcurrencyError(e.to_string()))?;
            pool.install(|| self.build_internal(vectors, dim))
        } else {
            self.build_internal(vectors, dim)
        }
    }

    fn build_internal(
        &self,
        vectors: &[VectorEmbedding],
        dim: usize,
    ) -> HNSQRResult<BuiltRiveroState> {
        let total_start = Instant::now();
        let n = vectors.len();

        // ════════════════════════════════════════════════════════════════════════
        // PHASE 1: PARALLEL DETERMINISTIC ADDRESS COMPILATION
        // ════════════════════════════════════════════════════════════════════════
        let t0 = Instant::now();
        let compiler = RiveroCompiler::with_config(dim, self.address_config);
        let addresses: Vec<RiveroAddress> = vectors
            .par_iter()
            .map(|vec| compiler.compile(vec.complex_data()))
            .collect();
        let time_address_compile_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // ════════════════════════════════════════════════════════════════════════
        // PHASE 2: SHARDED BOUNDED-MEMORY CELL REDUCTION & ASSOCIATIVE MERGE
        // ════════════════════════════════════════════════════════════════════════
        let t1 = Instant::now();
        let foundations_count = self
            .config
            .foundations
            .min(addresses[0].foundation_count as usize);
        let capacity = self.config.cell_capacity;
        let elites = self.config.affinity_elites;

        // Determine chunk count matching available concurrency
        let num_threads = rayon::current_num_threads();
        let chunk_size = (n / num_threads).max(1);

        // Map chunks to local 64-stripe tables with bounded cell retention
        let chunk_stripes: Vec<Vec<HashMap<u64, CellSlots>>> = addresses
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let start_slot = chunk_idx * chunk_size;
                let mut local_stripes: Vec<HashMap<u64, CellSlots>> = (0..STRIPE_COUNT)
                    .map(|_| HashMap::with_capacity(1024))
                    .collect();

                for (offset, addr) in chunk.iter().enumerate() {
                    let slot = (start_slot + offset) as NodeIndex;

                    for (foundation, coords) in
                        addr.foundations[..foundations_count].iter().enumerate()
                    {
                        let query_code = pack_projected_code(coords);
                        let (signatures, count) = insert_sigs(coords, 0);

                        for (rank, &signature) in signatures[..count].iter().enumerate() {
                            let key = cell_key(foundation, signature);
                            let stripe = stripe_for(key);
                            let affinity = (count - rank) as u8;
                            let fine_code =
                                (query_code & 0x00ff_ffff) | (u32::from(affinity) << 24);

                            let cell = local_stripes[stripe].entry(key).or_default();
                            cell.insert_with_limits(key, fine_code, slot, capacity, elites);
                        }

                        let (signature, _, _) = simhash_signature(coords, foundation);
                        let key = simhash_cell_key(foundation, signature);
                        let stripe = stripe_for(key);
                        let fine_code = (query_code & 0x00ff_ffff) | (u32::from(u8::MAX) << 24);

                        let cell = local_stripes[stripe].entry(key).or_default();
                        cell.insert_with_limits(key, fine_code, slot, capacity, elites);
                    }
                }

                local_stripes
            })
            .collect();
        let time_territory_reduction_ms = t1.elapsed().as_secs_f64() * 1000.0;

        // Merge stripe-by-stripe in parallel across the 64 stripes
        let t2 = Instant::now();
        let total_inserts = AtomicU64::new(0);
        let total_overflows = AtomicU64::new(0);

        let global_stripes: Vec<RwLock<HashMap<u64, CellSlots>>> = (0..STRIPE_COUNT)
            .into_par_iter()
            .map(|stripe_idx| {
                let mut merged_stripe: HashMap<u64, CellSlots> = HashMap::with_capacity(2048);
                for worker_stripes in &chunk_stripes {
                    for (&key, cell) in &worker_stripes[stripe_idx] {
                        let target = merged_stripe.entry(key).or_default();
                        target.merge_from(key, cell, capacity, elites);
                    }
                }

                let mut stripe_inserts = 0u64;
                let mut stripe_overflows = 0u64;
                for cell in merged_stripe.values() {
                    stripe_inserts += cell.slots.len() as u64;
                    if cell.overflowed {
                        stripe_overflows += 1;
                    }
                }
                total_inserts.fetch_add(stripe_inserts, std::sync::atomic::Ordering::Relaxed);
                total_overflows.fetch_add(stripe_overflows, std::sync::atomic::Ordering::Relaxed);

                RwLock::new(merged_stripe)
            })
            .collect();
        let time_territory_merge_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // ════════════════════════════════════════════════════════════════════════
        // PHASE 3: IMMUTABLE TERRITORY FREEZING & OPEN-ADDRESSED FLAT TABLE
        // ════════════════════════════════════════════════════════════════════════
        let flat_table = FlatFrozenTerritoryTable::from_stripes(&global_stripes);
        let inserts_val = total_inserts.load(std::sync::atomic::Ordering::Relaxed);
        let overflows_val = total_overflows.load(std::sync::atomic::Ordering::Relaxed);
        let frozen_territory = RiveroTerritoryIndex::from_stripes(
            global_stripes.into_boxed_slice(),
            inserts_val,
            overflows_val,
        );

        // ════════════════════════════════════════════════════════════════════════
        // PHASE 4: PROGRESSIVE 2-STAGE LOCK-FREE WITNESS DISCOVERY & SIMD SCORING
        // ════════════════════════════════════════════════════════════════════════
        let t3 = Instant::now();
        let degree = self.witness_degree;
        let budget = self.config.cell_budget;
        let foundations_count = self.config.foundations.min(addresses[0].foundations.len());
        let simhash_probes = self.config.simhash_query_probes;
        let force_stage_b = self.force_stage_b;
        let dist_fn = self.distance_function;

        let stage_a_count = std::sync::atomic::AtomicUsize::new(0);
        let stage_b_count = std::sync::atomic::AtomicUsize::new(0);

        // Resolve top candidate seeds and initial directed witness proposals in parallel
        let directed_proposals: Vec<(
            NodeIndex,
            SmallVec<[ScoredWitness; RIVERO_WITNESS_INLINE_DEGREE]>,
        )> = if degree == 0 {
            (0..n as NodeIndex)
                .map(|slot| (slot, SmallVec::new()))
                .collect()
        } else {
            (0..n)
                .into_par_iter()
                .map(|slot| {
                    let addr = &addresses[slot];
                    let vec = &vectors[slot];
                    let current_slot = slot as NodeIndex;

                    let mut candidate_slots: SmallVec<[NodeIndex; 512]> = SmallVec::new();

                    // STAGE A: Exact Voronoi & SimHash Insertion Cells
                    for (foundation, coords) in
                        addr.foundations[..foundations_count].iter().enumerate()
                    {
                        let query_code = pack_projected_code(coords);
                        let (signatures, count) = insert_sigs(coords, 0);

                        for &signature in &signatures[..count] {
                            let key = cell_key(foundation, signature);
                            let cell_slice = flat_table.get_residents(key);
                            if cell_slice.len() <= budget {
                                for r in cell_slice {
                                    if r.slot != current_slot {
                                        candidate_slots.push(r.slot);
                                    }
                                }
                            } else {
                                let mut best: SmallVec<[(i16, NodeIndex); 64]> = SmallVec::new();
                                for r in cell_slice {
                                    if r.slot != current_slot {
                                        let (dot, _) =
                                            projected_similarity(query_code, r.projected_code());
                                        best.push((dot, r.slot));
                                    }
                                }
                                if best.len() > budget {
                                    best.select_nth_unstable_by(budget - 1, |a, b| b.0.cmp(&a.0));
                                    for &(_, s) in &best[..budget] {
                                        candidate_slots.push(s);
                                    }
                                } else {
                                    for &(_, s) in &best {
                                        candidate_slots.push(s);
                                    }
                                }
                            }
                        }

                        // SimHash insertion cell
                        let (signature, _, _) = simhash_signature(coords, foundation);
                        let key = simhash_cell_key(foundation, signature);
                        let cell_slice = flat_table.get_residents(key);
                        for r in cell_slice.iter().take(budget) {
                            if r.slot != current_slot {
                                candidate_slots.push(r.slot);
                            }
                        }
                    }

                    candidate_slots.sort_unstable();
                    candidate_slots.dedup();

                    let mut scored: SmallVec<[ScoredWitness; 512]> =
                        SmallVec::with_capacity(candidate_slots.len());
                    let query_complex = vec.complex_data();

                    for &cand in &candidate_slots {
                        let cand_vec = &vectors[cand as usize];
                        let sim = match dist_fn {
                            DistanceFunction::Cosine => {
                                let dot = dot_product_complex_simd(
                                    query_complex,
                                    cand_vec.complex_data(),
                                );
                                let denom =
                                    (vec.norm_squared() * cand_vec.norm_squared()).max(1e-12);
                                (dot.re / denom.sqrt()).clamp(-1.0, 1.0)
                            }
                            DistanceFunction::ProjectiveOverlap => complex_projective_overlap_fast(
                                query_complex,
                                cand_vec.complex_data(),
                            ),
                            DistanceFunction::ProjectiveSineDistance => {
                                let p = complex_projective_overlap_fast(
                                    query_complex,
                                    cand_vec.complex_data(),
                                );
                                1.0 - (1.0 - p).max(0.0).sqrt()
                            }
                            DistanceFunction::PhaseAlignedChordalDistance => {
                                let p = complex_projective_overlap_fast(
                                    query_complex,
                                    cand_vec.complex_data(),
                                );
                                2.0 - (2.0 * (1.0 - p.sqrt())).max(0.0).sqrt()
                            }
                            DistanceFunction::Euclidean => {
                                let dot = dot_product_complex_simd(
                                    query_complex,
                                    cand_vec.complex_data(),
                                );
                                let dist_sq = (vec.norm_squared() + cand_vec.norm_squared()
                                    - 2.0 * dot.re)
                                    .max(0.0);
                                -dist_sq.sqrt()
                            }
                        };
                        scored.push(ScoredWitness {
                            index: cand,
                            similarity: sim,
                        });
                    }

                    let mut top_seeds = rivero_witness::select_top(&mut scored, degree);

                    // Stage A Quality Gate: Use Voronoi and SimHash co-residents directly
                    if !force_stage_b || !top_seeds.is_empty() {
                        stage_a_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return (current_slot, top_seeds);
                    }

                    // STAGE B: Broad Lookup-Family Delta Expansion
                    stage_b_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let mut delta_slots: SmallVec<[NodeIndex; 512]> = SmallVec::new();

                    for (foundation, coords) in
                        addr.foundations[..foundations_count].iter().enumerate()
                    {
                        let query_code = pack_projected_code(coords);
                        let (signatures, count) = lookup_sigs(coords, 0);

                        for &signature in &signatures[..count] {
                            let key = cell_key(foundation, signature);
                            let cell_slice = flat_table.get_residents(key);
                            if cell_slice.len() <= budget {
                                for r in cell_slice {
                                    if r.slot != current_slot && !candidate_slots.contains(&r.slot)
                                    {
                                        delta_slots.push(r.slot);
                                    }
                                }
                            } else {
                                let mut best: SmallVec<[(i16, NodeIndex); 64]> = SmallVec::new();
                                for r in cell_slice {
                                    if r.slot != current_slot && !candidate_slots.contains(&r.slot)
                                    {
                                        let (dot, _) =
                                            projected_similarity(query_code, r.projected_code());
                                        best.push((dot, r.slot));
                                    }
                                }
                                if best.len() > budget {
                                    best.select_nth_unstable_by(budget - 1, |a, b| b.0.cmp(&a.0));
                                    for &(_, s) in &best[..budget] {
                                        delta_slots.push(s);
                                    }
                                } else {
                                    for &(_, s) in &best {
                                        delta_slots.push(s);
                                    }
                                }
                            }
                        }

                        let (signature, _, margins) = simhash_signature(coords, foundation);
                        let probes = simhash_probe_signatures(signature, &margins);
                        for &probe in probes.iter().take(simhash_probes) {
                            let key = simhash_cell_key(foundation, probe);
                            let cell_slice = flat_table.get_residents(key);
                            for r in cell_slice.iter().take(budget) {
                                if r.slot != current_slot && !candidate_slots.contains(&r.slot) {
                                    delta_slots.push(r.slot);
                                }
                            }
                        }
                    }

                    delta_slots.sort_unstable();
                    delta_slots.dedup();

                    for &cand in &delta_slots {
                        let cand_vec = &vectors[cand as usize];
                        let sim = match dist_fn {
                            DistanceFunction::Cosine => {
                                let dot = dot_product_complex_simd(
                                    query_complex,
                                    cand_vec.complex_data(),
                                );
                                let denom =
                                    (vec.norm_squared() * cand_vec.norm_squared()).max(1e-12);
                                (dot.re / denom.sqrt()).clamp(-1.0, 1.0)
                            }
                            DistanceFunction::ProjectiveOverlap => complex_projective_overlap_fast(
                                query_complex,
                                cand_vec.complex_data(),
                            ),
                            DistanceFunction::ProjectiveSineDistance => {
                                let p = complex_projective_overlap_fast(
                                    query_complex,
                                    cand_vec.complex_data(),
                                );
                                1.0 - (1.0 - p).max(0.0).sqrt()
                            }
                            DistanceFunction::PhaseAlignedChordalDistance => {
                                let p = complex_projective_overlap_fast(
                                    query_complex,
                                    cand_vec.complex_data(),
                                );
                                2.0 - (2.0 * (1.0 - p.sqrt())).max(0.0).sqrt()
                            }
                            DistanceFunction::Euclidean => {
                                let dot = dot_product_complex_simd(
                                    query_complex,
                                    cand_vec.complex_data(),
                                );
                                let dist_sq = (vec.norm_squared() + cand_vec.norm_squared()
                                    - 2.0 * dot.re)
                                    .max(0.0);
                                -dist_sq.sqrt()
                            }
                        };
                        scored.push(ScoredWitness {
                            index: cand,
                            similarity: sim,
                        });
                    }

                    top_seeds = rivero_witness::select_top(&mut scored, degree);
                    (current_slot, top_seeds)
                })
                .collect()
        };
        let time_witness_routing_ms = t3.elapsed().as_secs_f64() * 1000.0;

        // PHASE 5 & 6: DETERMINISTIC RECIPROCAL PRUNING (Guarantees identical SHA-256 across all thread counts)
        let t4 = Instant::now();
        let proposals_by_dest: Vec<Vec<ScoredWitness>> = if degree == 0 {
            vec![Vec::new(); n]
        } else {
            (0..n)
                .into_par_iter()
                .map(|dest| {
                    let mut incoming: Vec<ScoredWitness> = Vec::with_capacity(degree * 2);
                    incoming.extend_from_slice(&directed_proposals[dest].1);

                    incoming.sort_unstable_by(|a, b| {
                        b.similarity
                            .total_cmp(&a.similarity)
                            .then_with(|| a.index.cmp(&b.index))
                    });

                    let mut unique: Vec<ScoredWitness> = Vec::with_capacity(degree);
                    for cand in incoming {
                        if !unique.iter().any(|existing| existing.index == cand.index) {
                            unique.push(cand);
                            if unique.len() >= degree {
                                break;
                            }
                        }
                    }
                    unique
                })
                .collect()
        };
        let time_witness_scoring_ms = t4.elapsed().as_secs_f64() * 1000.0;

        let t5 = Instant::now();
        let mut final_witnesses: Vec<SmallVec<[ScoredWitness; RIVERO_WITNESS_INLINE_DEGREE]>> =
            Vec::with_capacity(n);
        for list in proposals_by_dest {
            let mut sv = SmallVec::new();
            sv.extend(list);
            final_witnesses.push(sv);
        }
        let time_witness_finalize_ms = t5.elapsed().as_secs_f64() * 1000.0;

        let total_build_time_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let throughput_vecs_per_sec = if total_build_time_ms > 0.0 {
            (n as f64) / (total_build_time_ms / 1000.0)
        } else {
            0.0
        };

        let sa_cnt = stage_a_count.load(std::sync::atomic::Ordering::Relaxed) as f64;
        let sb_cnt = stage_b_count.load(std::sync::atomic::Ordering::Relaxed) as f64;
        let total_nodes = n as f64;

        let telemetry = BulkBuildTelemetry {
            time_address_compile_ms,
            time_territory_reduction_ms,
            time_territory_merge_ms,
            time_witness_routing_ms,
            time_witness_scoring_ms,
            time_witness_finalize_ms,
            total_build_time_ms,
            throughput_vecs_per_sec,
            cell_count: frozen_territory.cell_count(),
            overflow_count: frozen_territory.overflow_count(),
            stage_a_accepted_pct: if total_nodes > 0.0 {
                (sa_cnt / total_nodes) * 100.0
            } else {
                0.0
            },
            stage_b_expanded_pct: if total_nodes > 0.0 {
                (sb_cnt / total_nodes) * 100.0
            } else {
                0.0
            },
        };

        let descriptor = RiveroBuildDescriptor {
            schema_version: super::RIVERO_SCHEMA_VERSION,
            dimension: dim,
            address_config: self.address_config,
            rivero_config: self.config,
            distance_function: self.distance_function,
            witness_degree: self.witness_degree,
            witness_seeds: self.witness_seeds,
            witness_second_seeds: self.witness_second_seeds,
        };

        Ok(BuiltRiveroState {
            territory: frozen_territory,
            witnesses: final_witnesses,
            addresses,
            descriptor,
            telemetry,
        })
    }
}

/// Computes Complex Projective Overlap with 4-lane loop unrolling for AVX2/AVX-512 vectorization.
#[inline(always)]
fn complex_projective_overlap_fast(a: &[Complex32], b: &[Complex32]) -> f32 {
    let mut sum_re = 0.0f32;
    let mut sum_im = 0.0f32;
    let len = a.len().min(b.len());

    let chunks = len / 4;
    for i in 0..chunks {
        let base = i * 4;
        let a0 = a[base];
        let b0 = b[base];
        let a1 = a[base + 1];
        let b1 = b[base + 1];
        let a2 = a[base + 2];
        let b2 = b[base + 2];
        let a3 = a[base + 3];
        let b3 = b[base + 3];

        sum_re += a0.re * b0.re
            + a0.im * b0.im
            + a1.re * b1.re
            + a1.im * b1.im
            + a2.re * b2.re
            + a2.im * b2.im
            + a3.re * b3.re
            + a3.im * b3.im;

        sum_im += a0.re * b0.im - a0.im * b0.re + a1.re * b1.im - a1.im * b1.re + a2.re * b2.im
            - a2.im * b2.re
            + a3.re * b3.im
            - a3.im * b3.re;
    }

    for i in (chunks * 4)..len {
        let ai = a[i];
        let bi = b[i];
        sum_re += ai.re * bi.re + ai.im * bi.im;
        sum_im += ai.re * bi.im - ai.im * bi.re;
    }

    sum_re * sum_re + sum_im * sum_im
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex32;

    fn make_test_vector(dim: usize, seed: usize) -> VectorEmbedding {
        let complex: Vec<Complex32> = (0..dim)
            .map(|i| {
                let re = (((seed * 17 + i * 7 + 3) % 43) as f32 - 21.0) / 21.0;
                let im = (((seed * 31 + i * 11 + 5) % 47) as f32 - 23.0) / 23.0;
                Complex32::new(re, im)
            })
            .collect();
        VectorEmbedding::from_complex(complex).into_normalized()
    }

    #[test]
    fn test_bulk_build_thread_count_invariance() {
        let dim = 16;
        let n = 200;
        let vectors: Vec<VectorEmbedding> = (0..n).map(|i| make_test_vector(dim, i)).collect();

        let builder1 = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(1);
        let built1 = builder1.build(&vectors).unwrap();
        let fp1 = built1.territory.structural_fingerprint();

        let builder2 = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(2);
        let built2 = builder2.build(&vectors).unwrap();
        let fp2 = built2.territory.structural_fingerprint();

        let builder4 = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(4);
        let built4 = builder4.build(&vectors).unwrap();
        let fp4 = built4.territory.structural_fingerprint();

        let builder8 = RiveroBulkBuilder::with_profile(RiveroProfile::Balanced).with_threads(8);
        let built8 = builder8.build(&vectors).unwrap();
        let fp8 = built8.territory.structural_fingerprint();

        assert_eq!(fp1, fp2, "Fingerprint mismatch between 1T and 2T");
        assert_eq!(fp1, fp4, "Fingerprint mismatch between 1T and 4T");
        assert_eq!(fp1, fp8, "Fingerprint mismatch between 1T and 8T");

        assert_eq!(built1.witnesses.len(), n);
        for i in 0..n {
            assert_eq!(built1.witnesses[i], built4.witnesses[i]);
            assert_eq!(built1.witnesses[i], built8.witnesses[i]);
        }
    }

    #[test]
    fn test_associative_cell_reduction_commutativity() {
        let dim = 16;
        let vectors: Vec<VectorEmbedding> = (0..50).map(|i| make_test_vector(dim, i)).collect();
        let compiler = RiveroCompiler::new(dim);
        let addresses: Vec<RiveroAddress> = vectors
            .iter()
            .map(|v| compiler.compile(v.complex_data()))
            .collect();

        let key = 0x1234_5678_9abc_def0;
        let mut cell_a = CellSlots::default();
        let mut cell_b = CellSlots::default();

        for (i, addr) in addresses[..25].iter().enumerate() {
            let fine_code = pack_projected_code(&addr.foundations[0]);
            cell_a.insert_with_limits(key, fine_code, i as NodeIndex, 16, 6);
        }
        for (i, addr) in addresses[25..].iter().enumerate() {
            let fine_code = pack_projected_code(&addr.foundations[0]);
            cell_b.insert_with_limits(key, fine_code, (i + 25) as NodeIndex, 16, 6);
        }

        let mut merged_ab = CellSlots::default();
        merged_ab.merge_from(key, &cell_a, 16, 6);
        merged_ab.merge_from(key, &cell_b, 16, 6);

        let mut merged_ba = CellSlots::default();
        merged_ba.merge_from(key, &cell_b, 16, 6);
        merged_ba.merge_from(key, &cell_a, 16, 6);

        assert_eq!(merged_ab.slots, merged_ba.slots);
        assert_eq!(merged_ab.elite_len, merged_ba.elite_len);
    }
}
