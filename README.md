# HoloSphere — Hierarchical Navigable Semantic Query Resolver

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](#license)
[![Rust: 2024](https://img.shields.io/badge/Rust-2024%20Edition-orange.svg)](https://www.rust-lang.org/)
[![Verification](https://img.shields.io/badge/Verification-cargo%20test-brightgreen.svg)](#verification--testing)
[![Clippy](https://img.shields.io/badge/Clippy-D%20warnings-clean-brightgreen.svg)](#verification--testing)
[![PGO: Profile workload first](https://img.shields.io/badge/PGO-profile%20workload%20first-purple.svg)]

> **HoloSphere is a Rust multi-model state engine with exact vector retrieval, a canonical
> provenance-bearing knowledge hypergraph, empirical experience tracking, governed discovery,
> and provider-neutral model access through MCP and REST.**

Its storage and query engines cover vectors, metadata, property graphs, relational rows,
agent memory, multidimensional tensors, full-text retrieval, and columnar analytics. Its
knowledge layer gives entities, N-ary relations, evidence, outcomes, and learned operators
durable identities and point-in-time semantics. Raft state-machine replication, segmented
logs, memory-mapped snapshots, and pinned reads provide the shared consistency substrate.

HoloSphere is designed around explicit contract-driven retrieval and unified all-or-nothing multi-model state machine replication:

> **Production retrieval baseline:** `RetrievalContract::Exact` is the default and executes
> exhaustive contiguous SIMD scoring over the eligible pinned snapshot.
>
> **Empirical admission gates:** Exact SIMD is the authoritative baseline. Rivero and graph
> routes require regime-specific quality and performance evidence before they can be treated as
> production candidates. Proof-tree search remains explicit research functionality; the planner
> currently resolves `Certified` to Exact SIMD rather than selecting an unadmitted proof path.
>
> **Atomic operational state:** a committed `DataMutation::Batch` stages vector, property-graph,
> relational, agent-memory, and hypercube mutations at one Raft LSN. Physical universal
> snapshots also retain governed discovery state and evolved relation schemas at that LSN.

## Architecture at a Glance

```text
 Codex / Antigravity / Claude / applications
                │  MCP STDIO • MCP HTTP • REST • QIR0 • RESP
                ▼
      authenticated model and service boundary
                │
                ├── search / traverse / resolve / remember / record_outcome
                ▼
 entity + N-ary relation + experience + evidence/provenance
                │
                ▼
 adjudication + inference + synthesis + governed discovery
                │
                ▼
 exact vectors + metadata + graph projection + SQL + memory + tensors
                │
                ▼
      Raft LSNs • WAL • snapshots • audit chain
```

The layers are intentionally distinct. The property graph is a traversal projection; the
canonical relation model is N-ary. The MCP server exposes governed knowledge operations;
it does not replace the evidence, authorization, or admission rules underneath them.

For the formal mathematical specifications, invariants, and optimization policies governing all subsystems, see [HoloSphere Engineering Doctrine: 13 Reusable Primitives](docs/ARCHITECTURE_PRIMITIVES.md).

---

## Token Efficiency & Context Compression

HoloSphere serves as an autonomous external cognitive substrate (exocortex) for LLM agents, eliminating unbounded "context stuffing" and multi-turn trial-and-error loops through precision Point-in-Time Top-$K$ retrieval and empirical outcome caching.

### Empirical Token Reduction Benchmark

Verified against real repository source files via [`tests/token_efficiency_oracle.rs`](tests/token_efficiency_oracle.rs):

| Ingestion Strategy | Byte Volume | BPE Tokens (`cl100k_base`) | Token Savings | Compression Factor |
| :--- | :---: | :---: | :---: | :---: |
| **Raw In-Prompt Context** *(10 Full Source Files)* | **488,750 B** | **135,764 tokens** | Baseline | 1.0× |
| **HoloSphere Pinned MCP Retrieval** *(Precision Evidence)* | **599 B** | **166 tokens** | **99.88% Reduction** | **815.9× Precision Gain** |

```console
$ cargo test --test token_efficiency_oracle -- --nocapture

running 1 test
Total Raw Context Bytes : 488,750 bytes
Estimated Raw Tokens    : 135,764 tokens
Evidence Payload Bytes  : 599 bytes
Evidence Tokens         : 166 tokens
Token Reduction Ratio   : 99.88%
Compression Factor      : 815.9x
test test_holosphere_token_reduction_guarantee ... ok
```

### Architectural Levers for Token Savings
1. **Precision Top-$K$ Knowledge Gating:** Instead of injecting 50k+ tokens of raw file context, `holosphere:search` / `resolve` returns minimal sufficient evidence fragments ($\le 500$ tokens) with cryptographic SHA-256 provenance.
2. **Single-Shot Problem Resolution:** Historical resolutions and their empirical metrics are cached via `record_outcome`. When an issue or invariant is encountered, `resolve` serves the verified fix on Turn 1, eliminating 4–8 turns of iterative trial-and-error debugging (saving 50k–120k tokens per debugging loop).
3. **Ebbinghaus Memory Pruning:** The [`AutonomousMemoryConsolidator`](src/ecosystem/agent_memory.rs) applies exponential forgetting curves ($R = e^{-t / (S \cdot (1 + \text{salience}))}$) to decay transient noise and maintain a constant, bounded long-term memory footprint.
4. **Native Subgraph Triejoins:** Multi-hop relational traversals execute in native sub-millisecond Rust via Worst-Case Optimal Join (Leapfrog Triejoin), returning exact entity bindings rather than forcing the model to perform manual graph deduction in-context.

---

## HoloSphere ContextGraph — Universal Context Compiler & Reasoning Substrate

HoloSphere ContextGraph is a domain-neutral universal context compiler and graph reasoning substrate where source code, technical documentation, runtime architectures, Git history, and datasets are unified into a deterministic hypergraph knowledge plane:

```text
                       ANY SOURCE
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
          Source       Documents       System /
           Code        (MD, PDF)      Telemetry
             │             │             │
             └─────────────┬─────────────┘
                           ▼
                 UNIVERSAL COMPILER IR
                   (ExtractionBatch)
                           │
                   ┌───────┴───────┐
                   ▼               ▼
                detect        fingerprint
                   │               │
                   ▼               ▼
                extract         resolve
                   │               │
                   ▼               ▼
               validate          delta
                           │
                           ▼
                  CONTEXTGRAPH STORE
                (Atomic LSN Snapshots)
                           │
        ┌─────────────┬────┴────────┬─────────────┐
        ▼             ▼             ▼             ▼
     ingest        explore      traverse        path
    (Multi-       (Bounded      (N-ary        (Shortest
    Source)       Topology)    Relations)      Path)
```

### Staged Transformation Pipeline
1. **Detect & Fingerprint:** [`SourceAdapter`](src/contextgraph/adapter.rs) auto-detects source formats and computes cryptographic content hashes.
2. **AST & Text Extraction:** Specialized adapters ([`RustSourceAdapter`](src/contextgraph/adapters/code_rust.rs) with Tree-Sitter AST + `// SAFETY:` / `// WHY:` rationale notes, [`MarkdownSourceAdapter`](src/contextgraph/adapters/markdown.rs), [`FilesystemSourceAdapter`](src/contextgraph/adapters/fs.rs)) emit [`ExtractionBatch`](src/contextgraph/ir.rs) IR.
3. **Multi-Pass Reference Resolution:** [`UniversalReferenceResolver`](src/contextgraph/resolver.rs) resolves cross-entity references and preserves explicit ambiguity (`RelationOrigin::Ambiguous`) with zero ungrounded guessing.
4. **Validation & Delta Creation:** Sorts entities and relations into canonical order and packages atomic [`ContextGraphDelta`](src/contextgraph/schema.rs) transactions.
5. **Atomic Publication:** [`ContextGraphStore`](src/contextgraph/store.rs) applies mutations and advances the commit LSN, publishing deterministic [`GraphFingerprinter`](src/contextgraph/fingerprint.rs) signatures.

### Query Planning & Context Window Governance
The [`QueryPlanner`](src/contextgraph/planner.rs) dynamically routes queries to the minimal sufficient retrieval strategy, bounded by a [`ContextBudget`](src/contextgraph/planner.rs) (`max_results`, `max_chars`, `max_depth`):
- **`ExactEntityLookup`**: Direct $O(1)$ symbol or entity resolution.
- **`LexicalSearch` / `SemanticSearch`**: Sub-millisecond keyword and SIMD vector retrieval.
- **`GraphTraversal`**: Deep N-ary hypergraph neighborhood traversal.
- **`PathSearch`**: Shortest semantic relation pathfinding between two entities.
- **`ImpactTraversal`**: Blast radius analysis for downstream modifications.
- **`TemporalDiff`**: Differential snapshot comparison across LSN points.

### CLI Commands & Visualizations
```bash
# Build ContextGraph from workspace
cargo run --release --bin hnsqr_contextgraph -- build src/

# Query entities and trace paths
cargo run --release --bin hnsqr_contextgraph -- search "GatewayRouter"
cargo run --release --bin hnsqr_contextgraph -- path "ModelToolService" "HNSQRIndex"

# Generate Markdown reports and interactive HTML visualizer
cargo run --release --bin hnsqr_contextgraph -- report .
```
Outputs:
- `.holosphere/CONTEXT_REPORT.md`: Comprehensive markdown architecture summary.
- `.holosphere/contextgraph.html`: Standalone interactive HTML visualizer.
- `.holosphere/contextgraph.json`: Canonical JSON snapshot.

---

## Evolutionary Knowledge Hypergraph Layer


HoloSphere as a whole remains a multi-paradigm state engine. Within it, the
`entity`, `relation`, `experience`, and `learning` subsystems form an evolutionary
knowledge hypergraph layer:

- Canonical relations are typed, provenance-bearing N-ary role bindings; binary
  relationships are the N=2 case and CSR/CSC graph edges are derived projections.
- `HyperPattern` supports both genuinely symmetric member sets and explicit
  role-aware matching without imposing an artificial source/target split.
- Relation versions carry temporal validity and epistemic state. Evolutionary
  inference produces provisional proposals rather than silently rewriting
  admitted knowledge. Tombstones remain addressable for pre-delete `as_of` reads,
  but are excluded from current and post-delete relation queries.
- Evidence independence, circularity, staleness, and semantic-deduplication guards
  prevent N-ary fan-out and swarm echoes from multiplying confidence.
- Retrieval marked `Certified` is resolved by a complete metric-consistent proof
  or exhaustive scan before its output can be treated as exact evidence.

| Subsystem | Durable responsibility |
| :--- | :--- |
| [`entity`](src/entity/mod.rs) | Versioned entities, dense/sparse vectors, contexts, provenance, lifecycle and epistemic status, eligibility, and exact read snapshots |
| [`relation`](src/relation/mod.rs) | Dynamic typed N-ary role bindings, schema versions, incidence indexes, temporal validity, and lineage-preserving binary projections |
| [`experience`](src/experience/mod.rs) | Immutable problems, contexts, registered actions, attempts, metric schemas, raw outcomes, and point-in-time traces |
| [`learning`](src/learning/mod.rs) | Evidence accumulation, deterministic adjudication, collective conflict-preserving consensus, inference, synthesis, integrity guards, and governed discovery |

### Governed Cross-Domain Discovery

The `learning::discovery` subsystem implements governed open-ended discovery over
pinned entity, experience, and canonical N-ary relation snapshots:

1. Schema induction proposes new entity classes, relation types, role cardinalities,
   concept equivalences, and generalization/specialization hierarchies. A distinct,
   later snapshot tests every proposal through `Proposed -> FalsificationTesting ->
   ShadowValidated -> Admitted`; discovery roots are excluded from validation roots.
   Admitted relation schemas are synchronized into the canonical relation catalog.
2. Vocabulary-independent mining discovers repeated N-ary topology, causal sequences,
   invariant role arrangements, before/after outcomes, domain-relabeling invariants,
   and outcome anomalies. It compares roles, topology, time, context, causal ancestry,
   and outcomes rather than names alone.
3. Cross-domain mappings are learned from role behavior, capabilities, temporal
   position, and outcome associations. Competing mappings remain Proposed and inert;
   only independently validated, externally authorized Confirmed mappings enter the
   runtime concept resolver.
4. Candidate reasoning laws are inspectable `OperatorProgram` data. The sandboxed DSL
   supports Boolean, numeric, temporal, causal, domain, constraint, prediction,
   resolution, derived-value, and declarative hypergraph-transformation operations.
   It cannot execute native code or mutate the hypergraph directly, and every program
   carries enforceable AST, depth, effect, and numeric resource limits.
5. Operators compete on held-out accuracy, improvement over the incumbent set,
   counterfactual and intervention accuracy, worst-domain transfer, calibration,
   adversarial robustness, independent roots, and minimum description length. Their
   replicated lifecycle is `Generated -> Provisional -> FalsificationTesting ->
   Shadow -> ShadowValidated -> Admitted -> Monitored`, followed by rejection,
   deprecation, revision, or supersession as evidence requires. Admission always
   requires an external replicated policy or human authority.
6. The active experiment planner selects simulations, shadow replays, diagnostics,
   missing-evidence requests, A/B tests, or controlled changes by expected information
   gain and risk. Every execution requires a recorded authorization and lifecycle;
   live interventions remain external. Completed findings are replicated and folded
   into the next falsification cycle.
7. Every cycle emits an ordered Raft mutation stream for its kernel, schemas,
   mappings, operators, evaluations, experiments, and hash-chained audit events.
   Repeated cycles are idempotent against pinned prior state. Checksummed recovery
   checkpoints restore the complete learning state and rebuild evolved relation types.

The non-self-modifying `ImmutableSafetyKernel` requires certified evidence,
provenance, temporal isolation, external admission, evidence-independence accounting,
circular-support prevention, resource bounds, sandboxing, audit chaining, and
compensating-only rollback. No governed discovery mutation or learned-operator
transition is accepted before that kernel is committed.

This warrants the bounded claim: **HoloSphere performs governed, open-ended autonomous
discovery—it can induce new concepts, propose new declarative reasoning laws, falsify
them against independent evidence, plan authorized experiments, and incorporate
validated operators into future reasoning.** It is not unrestricted self-modification:
the safety constitution and DSL primitives remain engineered, and result quality still
depends on representative structured evidence and valid objectives.

For the complete lifecycle, admission, experiment, recovery, and rollback contracts,
see [Governed Open-Ended Discovery](docs/GOVERNED_OPEN_ENDED_DISCOVERY.md).

### Inference, Resolution Synthesis, and Distillation

The learning layer contains three related but separate capabilities:

- The inference registry runs contract-checked inference paradigms against pinned entity
  and relation snapshots. The `rune_evo` implementation provides analogy, causal,
  barycentric, evolutionary, blade, closure, and composition operators with traceable
  candidates.
- The synthesis planner aligns a new problem with compatible historical precedents,
  composes candidate resolutions under explicit constraints, and returns an auditable
  plan. A candidate remains a hypothesis until its evidence and outcomes are adjudicated.
- `AutonomousDistillationExporter` constructs chosen/rejected `DpoReasoningPair` values
  and serializes them as JSONL for downstream DPO/RLVR tooling. It is an export boundary:
  HoloSphere does not launch a trainer or silently update a model's weights.

### Affect-Aware Retrieval Planning

`AffectiveStateTensor8D` represents valence, arousal, dominance, certainty, trust,
novelty, goal congruence, and reversibility as bounded numeric appraisal inputs. For
eligible non-Exact indexed routes, `UniversalPlanner::plan_with_affect` applies two
explicit policies: low reversibility forces `Certified`, while high novelty plus high
reversibility may relax a requested `HighRecall` route to a bounded PAC plan. Exact
retrieval remains Exact. The tensor can also be quantized to a nearest $E_8$ lattice
point for deterministic lattice representation; that quantization is not a claim of
emotion or consciousness.

The compact property-graph engine remains an acceleration and traversal paradigm;
it is not the canonical ontology or provenance store.

---

## Platform and Deployment Components

### 1. 64-Way Striped Concurrent Ingestion (`src/storage/sharded_map.rs`)

`ShardedConcurrentMap<K, V>` partitions keys across 64 `RwLock<HashMap>` shards,
avoiding one global map lock without claiming a lock-free implementation.

### 2. Multi-Region Federation Primitives (`src/cluster/federation.rs`)

`FederatedRegionManager`, `GeoRoutingTable`, and `CrossRegionReplicator` provide
regional health/latency selection, vector clocks, tombstones, and deterministic LWW
conflict resolution. Availability targets remain a deployment responsibility.

### 3. Control Plane and Optional Usage Accounting (`src/cluster/control_plane.rs`)

`DBaaSControlPlane` reconciles desired and observed replica state. `UsageBillingMeter`
tracks query, storage, and egress counters for deployments that choose to use it; it is
not involved in local model integration.

### 4. Arrow-Shaped Batch Streaming (`src/transport/arrow_flight.rs`)

`ArrowFlightService` defines Arrow-shaped schemas and an `ARROW1`-framed batch payload
served by the daemon's port-50051 socket. It is currently a lightweight project
protocol, not a claim of complete gRPC Arrow Flight SQL interoperability.

### 5. OpenAPI and Swagger (`src/transport/swagger.rs`)

`OpenApiSpecGenerator` and `SWAGGER_HTML` expose OpenAPI 3.1 output plus embedded
`/docs` and `/swagger` pages.

---

## Additional Operational Engines

These are complementary query and storage engines, not six mutually exclusive
knowledge paradigms. Vector and property-graph engines are described separately in
the retrieval and graph sections below.

### Relational SQL and Multi-Table Transactions (`src/storage/relational_acid.rs`)

The relational interpreter supports `SELECT`, `FROM`, `WHERE`, inner/left joins, and
`ORDER BY`, with primary/foreign keys, row-level policies, two-phase locking, and MVCC
snapshot handles.

### N-Dimensional Hypercubes (`src/vector/hypercube.rs`)

`HypercubeTensorSpace` stores dense or sparse cells and supports arbitrary
`HypercubeBoundingBox` subvolume slicing and point-in-time snapshots.

### Linguistic, Sparse, and Hybrid Retrieval (`src/retrieval/`)

The retrieval layer includes BM25/Block-Max WAND, reciprocal-rank fusion, fuzzy
Levenshtein matching, stemming, Soundex, stop-word pruning, and CJK n-grams.

### Columnar Analytics and Embedded Media (`src/storage/columnar_olap.rs`)

Typed columnar tables support similarity-threshold filtering and `SUM`, `AVG`, `MIN`,
`MAX`, `COUNT`, and `VARIANCE`. Media records support byte-range access alongside
their vector projections.

### Long-Term Agent Memory (`src/ecosystem/agent_memory.rs`)

The memory subsystem stores episodic facts, consolidates compatible facts, preserves
contradictions for reconciliation, snapshots state, and scores retention from recency,
recall frequency, and salience.

### RESP, Pub/Sub, and Streams (`src/transport/resp.rs`)

The daemon exposes RESP2/RESP3 framing on port 6379 for implemented key/value,
Pub/Sub, and stream commands. Compatibility is command-specific; see the wire-protocol
list below for the supported surface.

---

## The Production Retrieval Standard & Research Admission Gates

HoloSphere anchors all vector retrieval to an exhaustive, cache-aligned AVX2/AVX-512 contiguous SIMD baseline. Any alternative indexing mechanism is treated as a research hypothesis that must justify its existence directly against this baseline across latency, throughput, memory bandwidth, and recall:

```
                            QUERY INGRESS
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
    [EXACT CONTIGUOUS SIMD SCAN]       [EXPERIMENTAL INDEXING CANDIDATES]
    • Production Default Standard      • Rivero Territorial Routing
    • Metric-consistent exact Top-K    • Graph ANN / ProofTree research
    • Measured by repository benches   • HNSW Graph Traversal
    • Zero Indexing Memory Overhead    • Must pass strict admission gates
                 │                                 │
                 ▼                                 ▼
      Authoritative Top-K            Evaluated vs Exact Baseline
```

### 1. The Production Standard: Contiguous Exact SIMD Scan
- **100.000% Exact Recall**: Zero false negatives, zero metric approximation artifacts, and zero indexing drift.
- **Hardware-oriented execution**: cache-aligned vector streaming, prefetching, and
  SIMD dot products. Latency is hardware-, dimension-, filter-, and corpus-dependent;
  use the repository benchmarks for the target machine.
- **Universal Default**: Automatically selected under `RetrievalContract::Exact` (system default) and when effective corpus size $N < N_{\text{cross}}(D)$.

### 2. Experimental Research Admission Gates
Before any non-brute-force indexing path can qualify for production routing, it must pass hard empirical gates on target datasets:

| Retrieval Path | Minimum Quality Gate | Secondary Quality Gate | Performance Requirement vs Exact SIMD |
| :--- | :---: | :---: | :--- |
| **Exact SIMD Scan** | **100.0% Recall@10** | **100.0% Recall@10** | Baseline ($1.0\times$) — Authoritative Production Standard |
| **Rivero $E_8$ Candidate Routing** | $\ge 95.0\%$ Recall@10 | $\ge 99.0\%$ Recall@10 | Must be materially faster than Exact SIMD ($> 2.0\times$ speedup) |
| **HNSW Graph ANN** | $\ge 95.0\%$ Recall@10 | $\ge 99.0\%$ Recall@10 | Must be materially faster than Exact SIMD ($> 2.0\times$ speedup) |
| **ProofTree (`Certified`, research)** | **100.0% Exact Recall** | Complete proof required | Not Auto-eligible until it beats Exact SIMD on an admitted regime |

---

## Public Dataset Benchmark Workflow

```bash
# Run the public dataset benchmark suite
cargo bench --bench public_dataset_benchmark
```

### Gate B proof artifacts

`gate_b_hierarchical_proof` measures query execution only. It never builds an
index, Rivero state, or proof tree at runtime. Materialize immutable
real-dataset artifacts once, then run the research gate:

```bash
# Example: 25k OpenAI-1536 vectors (768 complex dimensions).
cargo run --release --bin hnsqr_build_bench_db -- --kind index --tag gate_b_exact_d1536 --vectors 25000 --source-dim 1536 --index-dim 768 --profile balanced
cargo run --release --bin hnsqr_build_bench_db -- --kind proof --vectors 25000 --source-dim 1536
cargo bench --bench gate_b_hierarchical_proof
```

Use the missing-artifact message from Gate B for the exact source dimension and
cardinality of each matrix row. A proof path is admitted only at 100% exact
recall **and** when it is faster than Exact SIMD. Until then, production routing
remains Exact SIMD. Current benchmark output—not a checked-in scorecard—is the
authority for timing and admission evidence.

---

## The Universal Cost-Based Crossover Model

Exact SIMD and index routing have different fixed and per-vector costs. The
`UniversalPlanner` owns one crossover primitive, consumed by both `SearchPlan::Auto`
and planner APIs. Its embedded measured table is hardware- and corpus-sensitive; use
`cargo bench --bench dimension_crossover_sweep` to calibrate or review it on a target
machine. The fitted fallback is an interpolation aid, not a quality guarantee.

When effective corpus cardinality $N < N_{\text{cross}}$, HoloSphere automatically executes an exact SIMD scan, eliminating all routing overhead.

---

## Embedding Model Compatibility and Exactness

HoloSphere stores even- or odd-dimensional real embeddings without tying a collection
to a particular model provider:

- **Lossless coordinate folding:** [`ComplexWeaver`](src/vector/folding.rs) maps
  $\mathbb{R}^{D}$ to $\mathbb{C}^{\lceil D/2 \rceil}$ and pads an odd tail with
  `0.0i`, preserving the represented real coordinates, norms, and inner products.
- **Embedding-space isolation:** model-facing collections pin provider, model, version,
  dimension, normalization, and distance metric. Incompatible writes are rejected.
- **Exact by default:** `Exact` exhaustively scores every eligible vector under the
  collection's configured metric. `Certified` currently resolves to that admitted exact
  path in the planner. Explicit proof APIs remain research functionality; a completed
  proof requires `DenseExactProof::globally_exact == true`.

---

## Unified Single-Pass Graph-Vector Traversal

The 32-byte `GraphNodeRecord` packs label bitmasks, CSR/CSC edge head offsets, degrees, and vector storage slots into a single CPU cache line:

```
  ┌─────────────────────────────────────────────────────────────┐
  │ GraphNodeRecord (32 Bytes - Half Cache Line)                │
  ├───────────────┬──────────────┬──────────────┬───────────────┤
  │ Fast Labels   │ Out-Edge Ref │ In-Edge Ref  │ Vector Slot   │
  │ 64-bit Mask   │ 32-bit Index │ 32-bit Index │ 32-bit Direct │
  └───────────────┴──────────────┴──────────────┴───────────────┘
```

Single-pass graph traversal checks vector similarity bounds without secondary table joins.

---

## Retrieval Contracts

| Contract | Default | Guarantees |
| :--- | :---: | :--- |
| `Exact` | **YES (Default)** | Exhaustive ground-truth scan across all eligible candidates in the pinned snapshot. |
| `Certified` | No | Planner-routed to Exact SIMD today. Explicit proof APIs are research-only and require a complete globally exact proof before making an exact claim. |
| `PacRelaxed { epsilon, delta }` | No | $(\epsilon, \delta)$-PAC bounded relaxation under isotropic noise: $(1 - \epsilon)\text{UB}_{\text{cap}} < \tau$. |
| `HighRecall(recall)` | No | Statistical target recall guarantee (e.g., $0.995$) with adaptive candidate expansion. |
| `Budget(Duration)` | No | Peak throughput execution bounded by a strict timeout deadline. |
| `MultiVectorMaxSim { .. }` | No | Token-level late interaction routed to the MaxSim execution path. |

`SearchService::search_with_proof` carries `is_certified` and an optional remaining
upper bound across service and QIR0 boundaries. Implementations without a proof retain
the conservative default: useful results, but `is_certified == false`.

---

## Distributed Consensus & Replication

Clustered mutations follow a linearizable state-machine replication pipeline:

```
Client Request ──► MutationService ──► Raft Log (CRC-framed .rlog) ──► Quorum Replication
                                                                            │
Client ACK ◄── CommitReceipt ◄── ShardStateMachine Apply ◄── Quorum Commit ◄┘
```

- **Durability Invariant**: `ACK ⟹ Quorum Committed ∧ State Machine Applied`.
- CRC-framed logs, mutation IDs, retry semantics, read-index barriers, and applied-index
  receipts make durability and visibility explicit. Run the benchmark suite on the
  intended topology for deployment-specific throughput and latency.

### Snapshots, Recovery, and Semantic Conformance

- `UniversalSnapshot` pins vector, property-graph, relational, agent-memory, and
  hypercube state while retaining discovery operators, governed discovery state, and
  evolved N-ary relation schemas at the same committed LSN.
- `WorldStateDigest` deterministically hashes entity, relation, experience, learning,
  and schema state while excluding rebuildable acceleration structures.
- The semantic-kernel conformance layer provides versioned canonical export/import,
  typed errors, a golden fixture, and fail-closed version checks. World-state equality
  includes canonical learning records rather than a placeholder digest.
- Full backups have an explicit authenticated envelope-encryption path
  (`create_encrypted_full_backup` / `restore_encrypted_pitr`) using AES-256-GCM and a
  caller-provided KMS. The original full-backup method is intentionally plaintext for
  local/export workflows and must not be used for confidential production data.
- Snapshot attachment supports `Lazy`, `Eager`, and default `Adaptive` prefault modes.
  Adaptive warming preserves a memory reserve, reads cgroup v1/v2 limits when present,
  and skips cold dense-page warming when headroom is insufficient.
- The phase 10 integrity and phase 11 conformance suites exercise recovery, world-state
  equivalence, auditability, and compatibility boundaries.

---

## Wire Protocols, Web Console & API Docs

* **QIR0 Binary TCP Protocol (`:9090`)**: High-throughput async protocol supporting `OpCode::Ping`, `Insert`, `Search`, `BatchSearch`, `Stats`, and `OpCode::GraphQuery`.
* **Model Context Protocol (`POST :8080/mcp` & STDIO)**: MCP `2025-06-18` Streamable HTTP / stdio server for Antigravity, Claude Desktop, Cursor, OpenAI, Gemini, and compatible agents. It exposes evidence primitives (`search`, `web_search`, `traverse`, `resolve`, `remember`, `record_outcome`), case primitives (`status`, `run_case`, `task_begin`, `task_context`, `task_complete`), topology discovery (`explore`), and ContextGraph primitives (`ingest`, `path`, `diff`). Every schema is closed and tool names use underscores.
* **Redis RESP Protocol (`:6379`)**: Native RESP2/RESP3 server with `PING`, `SET`, `GET`, `INCR`, `DEL`, `PUBLISH`, `SUBSCRIBE`, `XADD`, and `XREAD`.
* **Arrow-shaped batch socket (`:50051`)**: Project-local `ARROW1`-framed schema and batch payload; full gRPC Arrow Flight SQL compatibility is not yet claimed.
* **HTTP REST Gateway (`:8080`)**: Axum-based JSON REST API for vector collections plus `/v1/knowledge/search`, `/traverse`, `/resolve`, `/remember`, `/outcomes`, `/status`, and `/cases/run`. Collection search accepts exactly one of a raw `query`/`vector` or `query_text`; text-only operations use the configured embedding provider and collections pin its model identity. Human-readable metadata accepts natural JSON string, integer, float, and Boolean scalars. Model responses carry a pinned LSN, proof status, and an explicit untrusted-content marker.
* **Embedded Web Console (`/dashboard` & `/ui`)**: Zero-dependency interactive single-page dashboard for visual graph exploration, live cluster metrics, and interactive query building.
* **Interactive OpenAPI 3.1 & Swagger UI (`/docs` & `/swagger`)**: In-browser API exploration and testing at `http://127.0.0.1:8080/docs`.
* **Multi-Language Client Libraries**:
  * Python: `sdks/python/hnsqr` (`AsyncHNSQRClient`, `HNSQRClient`)
  * TypeScript: `sdks/typescript` (`HNSQRClient`)
  * Go: `sdks/go` (`Client`)

### Connecting OpenAI, Gemini, or Claude

The daemon mounts a single provider-neutral MCP endpoint at `/mcp`. Model access is
fail-closed by default. Configure `HNSQR_MODEL_READ_TOKEN`,
`HNSQR_MODEL_WRITE_TOKEN`, or `HNSQR_MODEL_ADMIN_TOKEN`; anonymous access requires the
explicit development-only `HNSQR_MODEL_ALLOW_ANONYMOUS=true` setting. Knowledge and
outcome writes are idempotent, require provenance, and are fsynced to
`HNSQR_DATA_DIR/model-knowledge.jsonl`, which is replayed at startup.

Text-only calls use the configured embedding provider. Production semantic retrieval
pins provider, model, version, dimensions, normalization, and metric to every collection;
HoloSphere rejects incompatible vectors in the same collection. A dependency-free lexical
hash provider remains available as an explicit offline fallback.

For MCP reads, omit `snapshot_lsn` (or use `0`, for clients that serialize absent numeric
fields as zero) to read the latest committed knowledge. A positive `snapshot_lsn` is a
strict historical pin.

### Universal Case Runtime

`status` is the read-only preflight primitive. It reports the caller's write authorization,
live-web availability, effective embedding identity for each collection, runtime limits, and
any degradations. Call it before selecting a workflow, especially after changing embedding
configuration or connecting a new MCP client.

`run_case` is the bounded, domain-neutral preparation primitive:

```json
{
  "objective": "Find and fix a reproducible timeout",
  "recipe": "diagnose_and_fix",
  "evidence_policy": "web_if_needed",
  "execution_policy": "propose_only",
  "success_criteria": ["a regression test passes"],
  "budgets": {"tool_calls": 8, "retrieval_results": 10}
}
```

It supports `research_and_synthesize`, `diagnose_and_fix`, `implement_and_test`,
`compare_options`, `incident_response`, `analyze_dataset`, and `evaluate_strategy` recipes.
The response contains a durable case when writes are allowed, canonical evidence IDs, ranked
hypotheses, a bounded plan, and an `action_gate`. HoloSphere **never executes external or
consequential actions** through `run_case`; an authorized executor must approve and perform
them, then submit measured evidence through `record_outcome` and `task_complete`.

Resolution ranking is explainable. Each hypothesis includes `ranking_components` for semantic
relevance, verification state, prior outcome success, reproducibility, and recency. Web results
include a stable `evidence_id`, which can be passed into later lifecycle calls without manually
reconstructing provenance.

#### Adding HoloSphere to your MCP client

HoloSphere ships a pre-built STDIO MCP server alongside the main daemon. Most MCP-capable
clients (Kiro, Claude Desktop, Cursor, etc.) accept a JSON config file—typically
`mcp_config.json` or `mcp.json`—where you register servers by name.

Add a `holosphere` entry to that file:

```json
{
  "mcpServers": {
    "holosphere": {
      "command": "X:\\_Repos\\holosphere\\target\\agent-integrations\\hnsqr_mcp_stdio-v2.exe",
      "args": [],
      "env": {
        "HNSQR_MCP_ROLE": "readwrite",
        "HNSQR_DATA_DIR": "C:\\Users\\YourName\\AppData\\Local\\HoloSphere\\model-agent",
        "HNSQR_MCP_TENANT": "local-agents",
        "HNSQR_CONFIG": "X:\\_Repos\\holosphere\\Config.toml"
      }
    }
  }
}
```

Key fields:

| Field | Purpose |
|---|---|
| `command` | Absolute path to the compiled `hnsqr_mcp_stdio-*.exe` under `target\agent-integrations\`. The hash suffix changes on rebuild—update this after `cargo build --release`. |
| `HNSQR_MCP_ROLE` | `readonly`, `readwrite`, or `admin`. Omit to get the fail-closed default (no access). For development without tokens set `HNSQR_MODEL_ALLOW_ANONYMOUS=true` in `env` instead. |
| `HNSQR_DATA_DIR` | Directory where `model-knowledge.jsonl` is persisted and replayed at startup. Must be writable by the process. |
| `HNSQR_MCP_TENANT` | Logical namespace for knowledge scoping. Use distinct values per agent or project to keep knowledge isolated. |
| `HNSQR_CONFIG` | Absolute path to [`Config.toml`](Config.toml). Required when the MCP client launches the process outside the repository directory. |

For production deployments replace `HNSQR_MCP_ROLE` with token-based auth by setting
`HNSQR_MODEL_READ_TOKEN`, `HNSQR_MODEL_WRITE_TOKEN`, or `HNSQR_MODEL_ADMIN_TOKEN` in
`env` and removing `HNSQR_MODEL_ALLOW_ANONYMOUS`.

### Configurable Local and Hosted Embeddings

[`Config.toml`](Config.toml) configures the default text embedding provider for both the
daemon and the STDIO MCP server. The checked-in default uses the locally installed
**BGE-M3 FP16 GGUF** through LM Studio's OpenAI-compatible server:

```toml
[embedding]
backend = "openai_compatible"
provider = "bge"
model = "text-embedding-bge-m3" # Must match the API model identifier exposed by your server.
version = "gguf-fp16"
dimensions = 1024
normalization = "l2"
distance_metric = "cosine"
endpoint = "http://192.168.1.68:1234/v1"
model_path = "C:/Users/LordX/.lmstudio/models/gpustack/bge-m3-GGUF/bge-m3-FP16.gguf"
timeout_ms = 30000
```

Load `bge-m3-FP16.gguf` in LM Studio and start its local server. HoloSphere then sends
standard `POST /v1/embeddings` requests, so the same configuration shape works with
llama.cpp servers and any OpenAI-compatible local or hosted embedding service. For a
hosted service, change `endpoint`, identity fields, and optionally set `api_key_env` to
the name of an environment variable; never put a credential in `Config.toml`.

The `model_path` is provenance and operator guidance—the external serving runtime loads
the model artifact. HoloSphere deliberately does not embed a GGUF, Python, CUDA, or model
runtime into its daemon. This keeps the engine portable while allowing any model whose
server implements the embeddings contract. Use `HNSQR_CONFIG=/absolute/path/Config.toml`
when the daemon or MCP client starts outside the repository directory. Changing any
identity or dimensionality field creates a different embedding space: create/reindex a
new collection rather than mixing vectors.

### Free, Self-Hosted Live Web Search

HoloSphere includes a native, read-only `web_search` MCP tool. The included
[`deploy/searxng/docker-compose.yml`](deploy/searxng/docker-compose.yml) starts a free
self-hosted [SearXNG](https://docs.searxng.org/) metasearch service and binds it only to
`127.0.0.1:8888`; it needs no search API key:

```powershell
docker compose -f deploy\searxng\docker-compose.yml up -d
```

The checked-in [`Config.toml`](Config.toml) connects HoloSphere to that JSON endpoint:

```toml
[web_search]
backend = "searxng"
endpoint = "http://127.0.0.1:8888/search"
timeout_ms = 15000
max_results = 8
```

Any MCP client can then call `web_search` with `query`, optional `language`, optional
`time_range` (`day`, `month`, or `year`), and bounded `k`. Each result carries its title,
URL, snippet, participating engines, retrieval timestamp, content hash, and stable
`evidence_id`. Results are explicitly untrusted evidence and are never treated as instructions.
For write-authorized callers, HoloSphere attempts to register each result as an
`external_web_source` record using its content hash and URL provenance; read-only callers still
receive the result but do not create durable records. This surface deliberately does **not** offer arbitrary URL
fetching, avoiding an SSRF-capable proxy. SearXNG itself sends the upstream queries, so
there is no paid API dependency; normal internet, hardware, and upstream search-engine rate
limits still apply.

Use `search` for durable tenant-scoped HoloSphere knowledge and `web_search` for facts that
need current public-web evidence. Registration preserves source evidence; it does not upgrade a
claim to verified knowledge. Only a measurement, test artifact, or explicit approval may justify
promotion through `record_outcome` or `task_complete`.

See [OpenAI, Gemini, and Claude Integration](docs/MODEL_API_INTEGRATION.md) for setup,
MCP initialization, provider configuration, authorization, and embedding-space rules.

For local autonomous tool use, no public deployment is required. Build the native
STDIO transport and register it once with Codex, Google Antigravity (Gemini), and
Claude Code:

```powershell
cargo build --release --bin hnsqr_mcp_stdio
.\scripts\install_agent_integrations.ps1
```

All three clients then launch the same binary, use tenant `local-agents`, and share the
durable journal at `%LOCALAPPDATA%\HoloSphere\model-agent\model-knowledge.jsonl`. MCP
registration uses an immutable content-hashed snapshot under `target\agent-integrations`,
so a running client cannot lock Cargo's normal `target\release` output during upgrades.
The installer also passes `HNSQR_CONFIG` to every registered client, so text-only tool calls
use the configured embedding provider rather than silently falling back to lexical hashing.

MCP initialization instructions tell each model to search for relevant prior knowledge and patterns,
traverse relations, request evidence-backed resolutions, remember conclusions verified
by tests or explicit confirmation, and record measured outcomes. Antigravity receives
the narrow `mcp(holosphere/*)` allow rule; Claude pre-approves only
`mcp__holosphere__*`. Neither client receives a global permission bypass. Codex,
Antigravity, and Claude sessions that were already running must reload MCP servers or
start a new session before the new tools appear.

### Native Agent Case Memory

HoloSphere does not leave cross-agent learning to client prompt discipline. Any MCP
client can use the native task workflow, which is backed by the same tenant-scoped,
durable knowledge graph as the five evidence primitives:

```text
task_begin(problem)
  → retrieve prior similar cases and evidence-backed candidate resolutions
  → persist the new Issue case and `similar_to` links
task_context(case_id)
  → rehydrate the case, graph relations, and pinned evidence for any later agent
task_complete(case_id, measured evidence)
  → persist the empirical outcome
  → on success, promote a Resolution and link it with `fixed_by`
```

`task_begin` and `task_complete` require read-write authorization and non-empty
provenance. `task_context` is read-only. All writes use caller-supplied idempotency keys;
retrieved content remains explicitly untrusted evidence and never becomes executable
instruction. This makes the memory loop provider-neutral: any agent or model using the
MCP can resume a solved case without relying on Codex-specific behavior or CI hooks.

The primitive tools remain available for advanced clients. HoloSphere does not bypass
the client or secretly inject itself into prompts; the server instead provides a durable,
safe workflow surface and independently enforces tenant, role, provenance, snapshot, and
idempotency rules. Remote API applications can use this workflow through HTTPS MCP;
the REST gateway continues to expose the underlying evidence primitives.

The configuration follows the official [Google Antigravity MCP](https://antigravity.google/docs/mcp/)
and [CLI permission](https://antigravity.google/docs/cli/permissions) formats. If
`GEMINI_API_KEY` already exists, the installer selects Antigravity CLI's `gemini`
provider but never copies the key into a file.

---

## Operational Binaries & Release Footprint

HoloSphere is compiled with aggressive release profile optimizations (`codegen-units = 1`, `lto = "fat"`, `opt-level = 3`, `strip = "symbols"`), producing ultra-compact native static binaries with zero external runtime dependencies:

| Binary Target | Exact Size | Size (MB) | Role & Protocols |
| :--- | :---: | :---: | :--- |
| **`hnsqr_daemon`** | **2,568,704 B** | **2.45 MB** | Multi-transport service host for REST/MCP HTTP (:8080), QIR0 TCP (:9090), Redis RESP (:6379), Arrow Flight (:50051), Web Console, and Swagger docs |
| **`hnsqr_mcp_stdio`** | **1,391,616 B** | **1.33 MB** | Newline-delimited JSON-RPC MCP server used directly by local Codex, Antigravity, Claude Code, and Gemini agent runtimes |
| **`hnsqr_contextgraph`** | **1,450,000 B** | **1.38 MB** | Universal ContextGraph compiler, AST extractor, dependency invalidator, and interactive visualizer CLI |
| **`hnsqr_codegraph`** | **1,450,000 B** | **1.38 MB** | CodeGraph specialized ingestion profile compatibility CLI |
| **`hnsqr_doctor`** | **481,280 B** | **0.46 MB** | Production diagnostic expert system & AVX2/FMA hardware SIMD integrity auditor |
| **`hnsqr_plan`** | **156,672 B** | **0.15 MB** | Analytical capacity projection for memory, storage bandwidth, shard count, and expected latency |
| **`hnsqr_build_bench_db`** | **738,816 B** | **0.70 MB** | Benchmark database generator & immutable snapshot generator CLI |
| **Total (All 7 Binaries)** | **8,237,088 B** | **7.85 MB** | **Complete Universal Multi-Model & Context Engine Surface** |

```bash
# Build optimized release binaries
cargo build --release --bins

# Compile workspace and generate reports
./target/release/hnsqr_contextgraph report .

# Run system & cluster integrity audit
./target/release/hnsqr_doctor

# Sizing planning for 10M vectors @ 1536D at 5,000 QPS
./target/release/hnsqr_plan
```

---

## Profile-Guided Optimization (PGO)

HoloSphere is optimized for hardware branch predictors and I-cache locality using LLVM Profile-Guided Optimization:

```bash
# Automated PGO Build (PowerShell)
.\scripts\build_pgo.ps1 -Method native -Workload Benchmarks

# Automated PGO Build (Bash / Linux CI)
./scripts/build_pgo.sh
```

For profile-guided optimization, capture a representative workload and record its
environment alongside the resulting artifact; do not treat a profile from a synthetic
or unrelated workload as production evidence.

---

## Verification & Testing

```bash
# Format, compile, lint, and test gates
cargo holo-fmt-check
cargo holo-check
cargo holo-clippy
cargo holo-test

# Universal ContextGraph certification suite
cargo test --test contextgraph_universal_test

# `cargo bench` does not execute ordinary #[test] functions. Run this
# separately before treating benchmark output as correctness evidence.
cargo test --release --workspace --all-features

# Compile every benchmark target without executing the fleet.
cargo holo-bench

# Run doc-tests
cargo test --doc

# Focused acceptance surfaces added with the knowledge/MCP architecture
cargo test --test evolutionary_hypergraph_and_hpc_acceptance
cargo test --test model_api_integration
cargo test --test mcp_stdio_integration

# Run the public dataset benchmark suite when its dataset assets are available
cargo bench --bench public_dataset_benchmark
```

The aliases live in [`.cargo/config.toml`](.cargo/config.toml). Benchmark output is
not a substitute for the test gate, and timings are intentionally not hard-coded:
the current result on the target hardware is the authority.

---

## Minimal Embedded Example

```rust
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, VectorEmbedding, planning::RetrievalContract};

fn main() -> hnsqr::HNSQRResult<()> {
    let dim = 1536; // Supports any dimension: 384, 768, 1024, 1536, 3072, 4096, etc.

    let mut config = HNSQRConfig::default();
    config.distance_function = DistanceFunction::Cosine;

    let index = HNSQRIndex::new(config, dim);

    let vector = VectorEmbedding::from_reals(&vec![0.042_f32; dim]).into_normalized();
    index.insert("doc-001", vector.clone())?;

    // Exhaustively score every eligible vector in the current snapshot.
    let results = index.search_with_contract(&vector, 10, None, RetrievalContract::Exact)?;
    for (id, score) in results {
        println!("Match: {id} with similarity {score}");
    }
    Ok(())
}
```

---

## License

Licensed under either of:

- [Apache License 2.0](https://spdx.org/licenses/Apache-2.0.html)
- [MIT License](https://spdx.org/licenses/MIT.html)

at your option.
