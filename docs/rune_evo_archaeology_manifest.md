# Rune-EVO Source Archaeology Manifest

**Subsystem:** `rune-substrate/crates/rune-evo`  
**License / Attribution:** `MIT OR Apache-2.0`, © 2026 ArcMoon Studios, Author: Lord Xyn ✶  
**Target Subsystem:** `holosphere/src/learning/inference/`, `holosphere/src/learning/collective/`, & `holosphere/src/learning/`

---

## 1. Module Inventory & Classification Table

| Rune-EVO Source Module | Actual Purpose | Public Entry Points | Stateful? | Deterministic? | Numeric Contract | HoloSphere Destination | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `src/analogy.rs` | $SO(8)$ Givens rotor alignment & structural isomorphism search | `AnalogyScanAgent`, `RotorAlignmentResult`, `align_regions` | No (stateless core) | Yes (explicit seed) | `f32` (Givens line search, residual $\in [0, 1]$) | `src/learning/inference/rune_evo/analogy.rs` | **PORT** |
| `src/inference.rs` | Barycentric centroid triangulation & reference friction | `BarycentricResolver`, `DerivedInsight`, `resolve`, `blend_equal` | No | Yes | `f32` (weighted centroid + reference friction) | `src/learning/inference/rune_evo/barycentric.rs` | **PORT math / ADAPT integration** |
| `src/causal.rs` | Directed Clifford wedge $Cl(8)$ bivector encoding & geometric counterfactuals | `causal_bivector`, `bivector_strength`, `build_causal_edge`, `counterfactual_coords` | No | Yes | `f32` (28-dim grade-2 bivector) | `src/learning/inference/rune_evo/causal.rs` | **PORT math / ADAPT semantics** |
| `src/overlay.rs` | Harmonic sub-layer stacks & semantic drift metric | `OverlayStack`, `HarmonicLayer`, `semantic_drift` | No (in HoloSphere) | Yes | `f32` Euclidean distance | `src/learning/inference/rune_evo/evolution.rs` (`EvolutionHistoryView`) | **REPLACE storage, PORT drift / view semantics** |
| `src/manifold.rs` | E8 projection codecs, phase-shifts, and distances | `decode_projection8`, `encode_projection8`, `apply_phase_shift`, `euclidean_distance` | No | Yes | `f32` 8D coordinate vectors | `src/learning/inference/rune_evo/evolution.rs` | **PORT phase shift & E8 snapping math** |
| `src/hypergraph.rs` | Ad-hoc dynamic node/edge store with Supermemory update chains | `DynamicHypergraph`, `upsert_node`, `connect` | Yes | Yes | `u64` IDs / floats | `src/entity/` & `src/relation/` | **REPLACE** |
| `src/evolution.rs` | Phase shift parameters and state transitions | `EvolutionaryState`, `PhaseShift` | No | Yes | `f32` 8D phase shifts | `src/learning/inference/rune_evo/evolution.rs` | **PORT PhaseShift math; REPLACE EvolutionaryState storage with VersionTable + EvolutionProposal** |
| `src/reasoning.rs` | $Cl(24)$ Multivector blade operations & closure reasoning | `compile_closure`, `execute_operator_chain` | No | Yes | `f32` / Sparse $Cl(24)$ geometric product + Top-K energy truncation | `src/learning/inference/rune_evo/reasoning/` | **PORT $Cl(24)$ math / ADAPT grounded closure compilation; DO NOT PORT hot-path Cobb routing, English connectors, or TerritorySurfaceRegistry** |
| `src/traverse.rs` | E8 lattice-steered navigational routing | `E8HypergraphNavigator`, `NavigationPath` | No | Yes | $E_8$ kissing graph | `src/learning/inference/rune_evo/traverse.rs` | **ADAPT** |
| `src/timeline.rs` | Concept trajectory interpolation over timestamps | `SemanticTimeline`, `TemporalSnapshot` | Yes | Yes | `f32` LERP on E8 | Pinned LSN snapshots in `src/entity/read.rs` | **REPLACE** |
| `src/valence.rs` | Geometric curiosity and motivation fields | `ValenceField`, `LiveValenceField` | Yes | Yes | `f32` gradient potential | `src/learning/` | **ADAPT** |
| `src/hive/consensus.rs`| Swarm multi-agent belief arbitration & conflict detection | `compute_consensus`, `ConsensusResult`, `ConflictPair` | No (pure scan) | Yes | `f32` decayed weighted centroid | `src/learning/collective/consensus.rs` | **ADAPT / PORT** |
| `src/hive/decay.rs` | Exponential confidence decay ($c(t) = c_0 e^{-\lambda \Delta t}$) | `DecayScheduler`, `effective_confidence`, `sweep_below` | No | Yes | `f32` exponential decay | `src/learning/collective/belief.rs` | **ADAPT** |
| `src/hive/agent.rs` | Swarm participant identity & metadata | `AgentRegistry`, `AgentMeta`, `AgentHandle` | Yes | Yes | `u64` monotonic IDs | `src/learning/collective/belief.rs` | **ADAPT** |
| `src/hive/provenance.rs`| Per-node author provenance & reinforcement counts | `ProvenanceStore`, `NodeProvenance` | Yes | Yes | `u64` IDs | `src/entity/provenance.rs` | **REPLACE storage, preserve semantics** |
| `src/hive/event.rs` | Event bus notifications for node updates/conflicts | `EventBus`, `MemoryEvent` | Yes | Yes | Channels | HoloSphere consensus & telemetry | **REPLACE transport, preserve semantics** |
| `src/hive/query.rs` | Proximity & author-filtered swarm memory queries | `HiveQueryEngine` | No | Yes | Filters | `src/learning/query.rs` & `src/experience/query.rs` | **ADAPT** |
| `src/hive/sync.rs` | Distributed OpenRaft replication prototype (unfinished beta) | `HiveNode`, `openraft::Raft` | Yes | Non-det stub | Network wire | HoloSphere Raft engine (`src/cluster/`) | **REPLACE** |
| `src/cobb_*` | COBB memory-mapped anchor bridge | `cobb_bridge`, `cobb_mmap`, `cobb_territory` | Yes | Yes | Mmap | HoloSphere Rivero & Entity Storage | **REPLACE** |
| `src/accelerator.rs`| Fixed-layout mailbox SIMD dispatch | `EvoAccelerator`, `InProcessMailboxAccelerator` | No | Yes | AVX2/AVX-512 | `src/entity/exact/` / SIMD kernels | **REPLACE** |
| `src/error.rs` | Rune-EVO error enum | `EvoError` | No | N/A | N/A | `src/learning/inference/contract.rs` | **ADAPT** |

---

## 2. Distributed Consensus (Raft) vs. Swarm Epistemic Consensus

- **HoloSphere Raft (`src/cluster/`):**
  - **Purpose:** Distributed state-machine replication (SMR), log durability, quorum commit, leader election, crash recovery, linearizable read indices, and partition tolerance.
  - **Question answered:** *"Which mutations are durably committed, in what order, and what world-state are replicas allowed to expose?"*
- **Rune-EVO Swarm Consensus (`src/learning/collective/`):**
  - **Purpose:** Multi-agent epistemic/semantic belief arbitration, decayed confidence weighting, and explicit inter-agent conflict preservation.
  - **Question answered:** *"Given several autonomous agents holding proximate but potentially conflicting beliefs, what semantic resolution does their evidence support?"*
- **Composition:**
  - Swarm Consensus produces an epistemically `Provisional` collective hypothesis with explicit `DERIVED_FROM` relation bindings and retained `ConflictPair` edges.
  - HoloSphere Raft durably replicates and commits the resulting mutation, making the multi-agent hypothesis and audit trail consensus-consistent across the cluster.
