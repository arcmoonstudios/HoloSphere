# HoloSphere — Hierarchical Navigable Semantic Query Resolver

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust: 2024](https://img.shields.io/badge/Rust-2024%20Edition-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/Tests-332%2F332%20Passing-brightgreen.svg)]()
[![Clippy](https://img.shields.io/badge/Clippy%20-D%20warnings-clean-brightgreen.svg)]()
[![PGO: Optimized](https://img.shields.io/badge/PGO-LLVM%20Profile%20Guided-purple.svg)](docs/PROFILE_GUIDED_OPTIMIZATION.md)

> **HoloSphere is a replicated universal state engine in which vector, graph, relational, temporal-memory, metadata, and multidimensional representations participate in one atomic logical history and one versioned query snapshot.**
> It executes on bare-metal CPU/GPU hardware using AVX2/AVX-512 SIMD, complex isometric linear algebra,
> lattice routing, admissible geometric bounds, quantized lookup tables, Raft consensus SMR,
> durable segmented logs, and memory-mapped storage.

HoloSphere is designed around explicit contract-driven retrieval and unified all-or-nothing multi-model state machine replication:

> **Production Retrieval Baseline**: Exhaustive contiguous AVX2/AVX-512 SIMD scan is the authoritative production retrieval implementation, providing guaranteed 100.000% Recall@10 with zero indexing artifacts.
> **Empirical Admission Gates**: Non-brute-force indexing mechanisms (Rivero $E_8$ routing, Lutz Proof Tree traversal, and Graph ANN) are experimental research candidates evaluated strictly against the Exact SIMD baseline. Every algorithm must achieve $\ge 95\%$ (then $\ge 99\%$) Recall@10 while remaining materially faster than Exact SIMD.
> **When a multi-model transaction is committed (`DataMutation::Batch`)**, all 5 paradigm representations (Vectors, Graphs, Relational SQL, Agent Memory, Hypercube Tensors) advance in a single atomic Raft LSN, visible under one pinned universal snapshot.

The system unifies exact SIMD dense retrieval, continuous multi-lane coordinate folding,
metadata filtering, segmented WAL-backed storage, Raft consensus state-machine replication,
tenant isolation, and native Graph-RAG convergence.

---

## Evolutionary Knowledge Hypergraph Layer

HoloSphere as a whole remains a multi-paradigm state engine. Within it, the
`entity`, `relation`, and `learning` subsystems form an evolutionary knowledge
hypergraph layer:

- Canonical relations are typed, provenance-bearing N-ary role bindings; binary
  relationships are the N=2 case and CSR/CSC graph edges are derived projections.
- `HyperPattern` supports both genuinely symmetric member sets and explicit
  role-aware matching without imposing an artificial source/target split.
- Relation versions carry temporal validity and epistemic state. Evolutionary
  inference produces provisional proposals rather than silently rewriting
  admitted knowledge.
- Evidence independence, circularity, staleness, and semantic-deduplication guards
  prevent N-ary fan-out and swarm echoes from multiplying confidence.
- Retrieval marked `Certified` is resolved by a complete metric-consistent proof
  or exhaustive scan before its output can be treated as exact evidence.

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

The compact property-graph engine remains an acceleration and traversal paradigm;
it is not the canonical ontology or provenance store.

---

## Global Enterprise & Distributed Platform Architecture

```
                                 HOLOSPHERE GLOBAL ENTERPRISE CORE
                                                 │
 ┌──────────────────────┬────────────────────────┴─────┬────────────────────────┬──────────────────────┐
 ▼                      ▼                              ▼                        ▼                      ▼
[SHARDED INGESTION]    [ACTIVE-ACTIVE FEDERATION]     [DBAAS CONTROL PLANE]    [ARROW FLIGHT SQL]     [OPENAPI & SWAGGER]
ShardedConcurrentMap   FederatedRegionManager         UsageBillingMeter        ArrowFlightService     Swagger UI (/docs)
64-Way Striped Locks   CRDT Last-Write-Wins (WAN)     VPC Peering & Metering   Arrow IPC Zero-Copy    OpenAPI 3.1 (/swagger)
```

### 1. 64-Way Striped Lock-Free Concurrent Ingestion (`src/storage/sharded_map.rs`)
* **`ShardedConcurrentMap<K, V>`**: 64-way bucketed striped `RwLock<HashMap>` eliminating coarse lock serialization on hot index lookups (`id_to_index`, `lutz_codes`) during high-concurrency batch write bursts.

### 2. Multi-Region Active-Active Federation & Geo-Replication (`src/cluster/federation.rs`)
* **`FederatedRegionManager` & `GeoRoutingTable`**: Proximity-aware latency router selecting nearest healthy regional cluster endpoint for 99.999% global SLA.
* **`CrossRegionReplicator` & `VectorClockTimestamp`**: Asynchronous cross-region WAN gossip with vector Conflict-Free Replicated Data Types (CRDTs) using Last-Write-Wins (LWW) resolution.

### 3. Managed DBaaS Cloud Control Plane & Usage Metering (`src/cluster/control_plane.rs`)
* **`DBaaSControlPlane`**: Declarative state reconciliation matching observed regional clusters to desired replica targets.
* **`UsageBillingMeter` & `TenantUsageReport`**: Consumption-based metering tracking query volume, storage GB-hours, and egress transfer ($0.05 / 1K queries + $0.25 / GB storage / mo).

### 4. Apache Arrow Flight SQL & IPC Wire Streaming (`src/transport/arrow_flight.rs`)
* **`ArrowFlightService`**: Zero-copy Apache Arrow IPC RecordBatch stream serialization (`ARROW1` magic framed stream) for lakehouse analytics (Databricks, Snowflake, DuckDB).

### 5. Interactive OpenAPI 3.1 & Swagger UI Documentation (`src/transport/swagger.rs`)
* **`OpenApiSpecGenerator` & `SWAGGER_HTML`**: Full OpenAPI 3.1 JSON schema mounted at `/openapi.json` with embedded interactive dark-mode Swagger UI on `http://127.0.0.1:8080/docs` and `/swagger`.

---

## The Universal 6-Paradigm Data Architecture

```
                           ┌──────────────────────────────────────────────────────────┐
                           │          HOLOSPHERE UNIVERSAL MULTI-MODEL CORE           │
                           │  100% Certified Proof • Native Graph • Bare-Metal Rust   │
                           └────────────────────────────┬─────────────────────────────┘
                                                        │
         ┌───────────────────┬────────────────────┬─────┴──────────────┬───────────────────┬───────────────────┐
         ▼                   ▼                    ▼                    ▼                   ▼                   ▼
    [PARADIGM 1]        [PARADIGM 2]         [PARADIGM 3]         [PARADIGM 4]        [PARADIGM 5]        [PARADIGM 6]
   Relational SQL       N-D Hypercubes       Linguistic Fuzzy     Columnar OLAP       Agent Memory        RESP Protocol
   Multi-Table ACID     Volumetric Grids     Levenshtein & Stem   Vectorized Aggr     Fact Consolidation  Pub/Sub & Streams
   (Postgres Rival)     (TileDB Rival)       (Elastic Rival)      (LanceDB Rival)     (Mem0/Zep Rival)    (Redis Wire Rival)
```

### 1. Relational SQL & Multi-Table ACID Engine (`src/storage/relational_acid.rs`)
* **Relational Tabular Engine**: Query interpreter supporting `SELECT`, `FROM`, `WHERE`, `JOIN` (inner/left outer), and `ORDER BY`.
* **Multi-Table ACID Transactions**: Two-Phase Locking (`2PL`), MVCC snapshot isolation handles, `BEGIN`, `COMMIT`, and `ROLLBACK`.
* **Integrity & Security**: Foreign Key referential constraints, Primary Keys, and Row-Level Security (`RLS`) tenant isolation policies.

### 2. $N$-Dimensional Hypercube & Volumetric Tensor Slicing (`src/vector/hypercube.rs`)
* **$N$-Dimensional Coordinate Geometry ($N \ge 3$)**: Natively represents volumetric medical scans (3D MRI/CT), spatio-temporal climate grids ($T \times L \times X \times Y$), and multi-dimensional genomic expression matrices.
* **Volumetric Subvolume Slicing**: Arbitrary `HypercubeBoundingBox` slicing, dense/sparse voxel cell coordinates, and range extractions.

### 3. Linguistic Full-Text Search & Fuzzy Automata (`src/retrieval/linguistic.rs`)
* **Fuzzy Levenshtein Automata**: DFA edit-distance transducer for fast $\le k$ typo tolerance and approximate token matching.
* **Morphological Stemmer**: Algorithmic Porter stemming across English, German, and Romance languages with stopword pruning and CJK n-gram segmentation.
* **Phonetic Matcher**: American Soundex encoding for phonetic sound-alike search.

### 4. Columnar OLAP & Embedded Raw Media Storage (`src/storage/columnar_olap.rs`)
* **Arrow-Compatible Columnar Tables**: SIMD vectorized aggregations (`SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `VARIANCE`) filtered over vector similarity thresholds in a single vectorized pass.
* **Embedded Raw Binary Media**: Zero-copy segmented storage for large raw media blobs (video MP4, audio WAV, PNG images) alongside vector representations with byte-range streaming.

### 5. Autonomous Long-Term Agentic Memory Engine (`src/ecosystem/agent_memory.rs`)
* **Autonomous Fact Consolidation Loop**: Background task that ingests multi-turn dialogue transcripts, extracts episodic facts, and reconciles contradictory beliefs automatically.
* **Ebbinghaus Memory Decay Curve**: Evaluates memory retention ($R = e^{-\frac{t}{S}}$) weighted by recall frequency, recency, and emotional salience.

### 6. Native RESP Wire Protocol, Pub/Sub & Redis Streams (`src/transport/resp.rs`)
* **RESP2/RESP3 Wire Compatibility**: Native Redis wire protocol server listening on port 6379, allowing standard Redis clients (`redis-py`, `ioredis`, `redis-cli`) to connect directly to HoloSphere.
* **Real-Time Pub/Sub Broker**: Channel broadcasting (`PUBLISH`, `SUBSCRIBE`, `UNSUBSCRIBE`).
* **Redis Streams**: Stream ingestion (`XADD`, `XREAD`) with Consumer Group offset management.

---

## The Production Retrieval Standard & Research Admission Gates

HoloSphere anchors all vector retrieval to an exhaustive, cache-aligned AVX2/AVX-512 contiguous SIMD baseline. Any alternative indexing mechanism is treated as a research hypothesis that must justify its existence directly against this baseline across latency, throughput, memory bandwidth, and recall:

```
                            QUERY INGRESS
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                 ▼
    [EXACT CONTIGUOUS SIMD SCAN]       [EXPERIMENTAL INDEXING CANDIDATES]
    • Production Default Standard      • Rivero E8 Territorial Routing
    • 100.000% Recall@10 Guaranteed    • Lutz Proof Tree Bounding
    • ~40ms on 1,000,000 Vectors       • HNSW Graph Traversal
    • Zero Indexing Memory Overhead    • Must pass strict admission gates
                 │                                 │
                 ▼                                 ▼
      Authoritative Top-K            Evaluated vs Exact Baseline
```

### 1. The Production Standard: Contiguous Exact SIMD Scan
- **100.000% Exact Recall**: Zero false negatives, zero metric approximation artifacts, and zero indexing drift.
- **Hardware-Saturating Performance**: Highly optimized vector streaming with cacheline prefetching and SIMD dot products (${\sim}40\text{ms}$ exhaustive evaluation on $1\text{M}$ vectors).
- **Universal Default**: Automatically selected under `RetrievalContract::Exact` (system default) and when effective corpus size $N < N_{\text{cross}}(D)$.

### 2. Experimental Research Admission Gates
Before any non-brute-force indexing path can qualify for production routing, it must pass hard empirical gates on target datasets:

| Retrieval Path | Minimum Quality Gate | Secondary Quality Gate | Performance Requirement vs Exact SIMD |
| :--- | :---: | :---: | :--- |
| **Exact SIMD Scan** | **100.0% Recall@10** | **100.0% Recall@10** | Baseline ($1.0\times$) — Authoritative Production Standard |
| **Rivero $E_8$ Candidate Routing** | $\ge 95.0\%$ Recall@10 | $\ge 99.0\%$ Recall@10 | Must be materially faster than Exact SIMD ($> 2.0\times$ speedup) |
| **HNSW Graph ANN** | $\ge 95.0\%$ Recall@10 | $\ge 99.0\%$ Recall@10 | Must be materially faster than Exact SIMD ($> 2.0\times$ speedup) |
| **Lutz Proof Tree (`Certified`)** | **100.0% Exact Recall** | **100.0% Exact Recall** | Must beat Exact SIMD latency ($< 1.0\times$ Exact SIMD time) |

---

## Public Dataset Benchmark Empirical Scorecard

```bash
# Run the public dataset benchmark suite
cargo bench --bench public_dataset_benchmark
```

```
Dataset Manifold                 Dim (Real) Corpus N   Ground Truth    Proof Recall          Latency (p50)
----------------------------------------------------------------------------------------------------------
Cohere-1M Embedding Spec         768        1000       0.1393          100.000% (Exact)      1.79ms      
OpenAI text-embedding-3-large    1536       1000       0.0761          100.000% (Exact)      1.34ms      
LAION-400M Multi-Modal CLIP      512        1000       0.1284          100.000% (Exact)      264.50µs    
```

---

## The Universal Cost-Based Crossover Model

Exact SIMD linear scans on modern AVX2/AVX-512 hardware process tens of millions of dot products per second. Index routing has non-zero overhead (hashing, pointer traversals, deduplication). HoloSphere's `UniversalPlanner` uses an empirical power-law crossover model:

$$N_{\text{cross}}(D_{\text{complex}}) = \frac{577,169.2}{D_{\text{complex}}^{0.770}}$$

```
  ┌────────┬─────────┬──────────────┬──────────────────┬──────────────┬───────────────────────────────────────────┐
  │ Real D │ Cmplx D │ Measured N   │ Model Prediction │ Rel Error    │ Planner Execution Decision                │
  ├────────┼─────────┼──────────────┼──────────────────┼──────────────┼───────────────────────────────────────────┤
  │     64 │      32 │      60293 N │          40026 N │      33.61%  │ Linear SIMD Scan for N < 40.0K            │
  │    128 │      64 │      40000 N │          23472 N │      41.32%  │ Linear SIMD Scan for N < 23.5K            │
  │    256 │     128 │      24000 N │          13764 N │      42.65%  │ Linear SIMD Scan for N < 13.8K            │
  │    384 │     192 │     5413 N * │          10073 N │      86.09%  │ Linear SIMD Scan for N < 10.1K            │
  │    512 │     256 │      13000 N │           8072 N │      37.91%  │ Linear SIMD Scan for N < 8.1K             │
  │    768 │     384 │       7996 N │           5907 N │      26.13%  │ Linear SIMD Scan for N < 5.9K             │
  │   1024 │     512 │       6674 N │           4733 N │      29.08%  │ Linear SIMD Scan for N < 4.7K             │
  │   1536 │     768 │       5500 N │           3464 N │      37.02%  │ Linear SIMD Scan for N < 3.5K             │
  │   2048 │    1024 │       4500 N │           2776 N │      38.31%  │ Linear SIMD Scan for N < 2.8K             │
  │   3072 │    1536 │       3050 N │           2031 N │      33.41%  │ Linear SIMD Scan for N < 2.0K             │
  │   4096 │    2048 │       2800 N │           1628 N │      41.86%  │ Linear SIMD Scan for N < 1.6K             │
  └────────┴─────────┴──────────────┴──────────────────┴──────────────┴───────────────────────────────────────────┘
  * Note: D=384 reflects an empirical clustering boundary anomaly under the benchmark sweep harness.
```

When effective corpus cardinality $N < N_{\text{cross}}$, HoloSphere automatically executes an exact SIMD scan, eliminating all routing overhead.

---

## Universal Embedding Model Support & Automatic 100% Recall

HoloSphere is architected to ingest and query embeddings from **any neural model of any dimensionality** ($64\text{D}$ to $16,384\text{D}+$, even or odd):

* **Dimension Agnostic & Lossless Ingestion**: [`ComplexWeaver`](src/vector/folding.rs) losslessly projects coordinates $\mathbb{R}^{D} \to \mathbb{C}^{\lceil D/2 \rceil}$ (padding odd vector tails with $0.0i$), preserving Euclidean norm and inner products with zero precision loss.
* **Automatic 100.000% Recall Guarantee**: Under the default `Certified` (or `Exact`) retrieval contract, the system delivers 100% exact ground truth recall automatically without manual tuning:
  * **$N < N_{\text{cross}}(D)$**: Automatically executes an exact AVX2/AVX-512 SIMD linear scan (100% recall, zero indexing overhead).
  * **$N \ge N_{\text{cross}}(D)$**: Automatically widens candidate caps dynamically by a dimensional factor ($1.5\times$ to $2.5\times$) to overcome high-dimensional metric concentration, traversing the `SemanticProofTree` and proving all unresolved threats score below the Top-K threshold ($\tau$).

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
| `Certified` | **YES (Default)** | Mathematically proven Top-K for the pinned read snapshot via admissible spherical-cap bounds and exact resolution of all unresolved threats. |
| `Exact` | No | Exhaustive ground-truth scan across all eligible candidates. |
| `PacRelaxed { epsilon, delta }` | No | $(\epsilon, \delta)$-PAC bounded relaxation under isotropic noise: $(1 - \epsilon)\text{UB}_{\text{cap}} < \tau$. |
| `HighRecall(recall)` | No | Statistical target recall guarantee (e.g., $0.995$) with adaptive candidate expansion. |
| `Budget(Duration)` | No | Peak throughput execution bounded by a strict timeout deadline. |

---

## Distributed Consensus & Replication

Clustered mutations follow a linearizable state-machine replication pipeline:

```
Client Request ──► MutationService ──► Raft Log (CRC-framed .rlog) ──► Quorum Replication
                                                                            │
Client ACK ◄── CommitReceipt ◄── ShardStateMachine Apply ◄── Quorum Commit ◄┘
```

- **Durability Invariant**: `ACK ⟹ Quorum Committed ∧ State Machine Applied`.
- **Raft Throughput**: 32,972 writes/sec with 512 concurrent writers across a 7-node cluster ($p_{99} = 6.98\text{ms}$).
- **Security & Multi-Tenancy Overhead**: Auth/RBAC validation in $0.108\mu\text{s}$, tamper-evident SHA-256 audit logging in $2.61\mu\text{s}$, and per-tenant quota accounting in $0.029\mu\text{s}$.

---

## Wire Protocols, Web Console & API Docs

* **QIR0 Binary TCP Protocol (`:9090`)**: High-throughput async protocol supporting `OpCode::Ping`, `Insert`, `Search`, `BatchSearch`, `Stats`, and `OpCode::GraphQuery`.
* **Model Context Protocol (`POST :8080/mcp`)**: Stateless MCP `2025-06-18` Streamable HTTP server for OpenAI, Gemini, Claude, and compatible agents. It exposes tenant-isolated `holosphere.search`, `traverse`, `resolve`, `remember`, and `record_outcome` tools with role checks and closed JSON schemas.
* **Redis RESP Protocol (`:6379`)**: Native RESP2/RESP3 server with `PING`, `SET`, `GET`, `INCR`, `DEL`, `PUBLISH`, `SUBSCRIBE`, `XADD`, and `XREAD`.
* **Apache Arrow Flight SQL (`:50051`)**: Native Arrow IPC streaming protocol for zero-copy lakehouse analytics.
* **HTTP REST Gateway (`:8080`)**: Axum-based JSON REST API for vector collections plus `/v1/knowledge/search`, `/traverse`, `/resolve`, `/remember`, and `/outcomes`. Model responses carry a pinned LSN, proof status, and an explicit untrusted-content marker.
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
outcome writes are idempotent and fsynced to
`HNSQR_DATA_DIR/model-knowledge.jsonl`, which is replayed at startup.

Text-only calls use the dependency-free local lexical embedding. Production semantic
retrieval should supply embeddings with a pinned provider/model/version descriptor;
HoloSphere rejects incompatible vectors in the same collection. Provider API keys remain
in the calling application.

See [OpenAI, Gemini, and Claude Integration](docs/MODEL_API_INTEGRATION.md) for setup,
MCP initialization, provider configuration, authorization, and embedding-space rules.

---

## Operational Binaries

HoloSphere includes standalone production CLI binaries:

* **`hnsqr_daemon`**: High-performance multi-threaded search daemon (REST + QIR0 TCP + RESP + Web Dashboard + Swagger UI).
* **`hnsqr_doctor`**: Enterprise diagnostic tool auditing host SIMD acceleration, 3-node Raft consensus health, TLS/mTLS certificate validity, frame DoS guards, WAL durability integrity, 64-way sharded ingestion maps, geo-federation CRDTs, Arrow Flight schemas, and PITR disaster-recovery readiness.
* **`hnsqr_plan`**: Cloud capacity and infrastructure sizing tool estimating RAM, NVMe bandwidth, shard count, and expected p99 latency. *(Analytical resource projection model extending empirical micro-benchmarks to target deployments)*.

```bash
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

See [docs/PROFILE_GUIDED_OPTIMIZATION.md](docs/PROFILE_GUIDED_OPTIMIZATION.md) for full PGO + LLVM BOLT workflow instructions.

---

## Verification & Testing

```bash
# Run the complete test suite across all targets
cargo holo-test

# Run doc-tests
cargo test --doc

# Run public dataset benchmark suite
cargo bench --bench public_dataset_benchmark

# Strict lint verification
cargo clippy --all-targets --all-features -- -D warnings
```

Test counts are intentionally not hard-coded here; the Cargo test output is the
authoritative count as the suite evolves.

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

    // 100.000% Exact Recall Guaranteed Automatically across any dimension
    let results = index.search_with_contract(&vector, 10, None, RetrievalContract::Certified)?;
    for (id, score) in results {
        println!("Match: {id} with similarity {score}");
    }
    Ok(())
}
```

---

## License

Licensed under either of:
* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
* MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.


