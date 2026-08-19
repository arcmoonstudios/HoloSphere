# HoloSphere — Hierarchical Navigable Semantic Query Resolver

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust: 2024](https://img.shields.io/badge/Rust-2024%20Edition-orange.svg)](https://www.rust-lang.org/)
[![Tests](https://img.shields.io/badge/Tests-147%2F147%20Passing-brightgreen.svg)]()
[![Clippy](https://img.shields.io/badge/Clippy%20-D%20warnings-clean-brightgreen.svg)]()

> **HoloSphere is a classical retrieval and storage engine.**
> It runs on conventional CPU/GPU hardware using SIMD, complex-valued linear algebra,
> lattice routing, admissible geometric bounds, quantized lookup tables, Raft consensus,
> durable segmented logs, and memory-mapped storage.
> It does **not** require quantum hardware and makes no claim of quantum computational speedup.

HoloSphere is a proof-carrying multimodal retrieval engine built around one unusually strict contract:

> **When `Certified` retrieval is requested, HoloSphere returns the exact Top-K for the
> pinned corpus snapshot, or returns an explicit failure instead of silently degrading
> correctness.**

The system combines exact dense retrieval, Rivero candidate routing, SemanticProofTree
pruning, LUTz progressive upper-bound filtering, SIMD exact scoring, sparse/hybrid
retrieval, multi-vector late interaction, metadata filtering, segmented WAL-backed storage,
Raft consensus mutation ordering, tenant isolation, and production-oriented operational
tooling.

The mathematical search core is optimized for the common case, but HoloSphere does **not**
claim universal constant-time globally exact search. Certified search is data-dependent
and can approach exhaustive work in adversarial cases.

---

## Why HoloSphere Exists

Most high-throughput vector search systems are approximate: they trade recall for
predictable latency. That trade is often reasonable. It is also unacceptable for
workloads where a missed nearest neighbor is itself a correctness failure — legal
document retrieval, compliance search, precision medicine, financial audit trails.

HoloSphere separates those concerns explicitly:

| Contract | Meaning |
|---|---|
| `Bounded` | Fixed work ceiling. Approximate retrieval is permitted. |
| `HighRecall` | Planner-selected high-recall retrieval without a global exactness proof. |
| `Certified` | Global exact Top-K for the selected read snapshot, established by admissible proof bounds and exact SIMD evaluation of every unresolved threat. |

---

## Architecture

```
        CLIENT
          │
          ▼
┌─────────────────────┐
│  Service Boundary   │
│ Auth / Tenant / SLA │
└──────────┬──────────┘
           │
  ┌────────┴────────────────────┐
  │                             │
  ▼                             ▼
┌─────────────────────┐   ┌─────────────────────┐
│    SearchService    │   │   MutationService   │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
           ▼                         ▼
┌─────────────────────┐   ┌─────────────────────┐
│  Universal Planner  │   │        Raft         │
│  Cost + Proof Model │   │ Persist / Replicate │
└──────────┬──────────┘   └──────────┬──────────┘
           │                         │
  ┌────────┼───────┐                 ▼
  │        │       │       ┌─────────────────────┐
  ▼        ▼       ▼       │  Shard State Machine│
┌──────┐ ┌──────┐ ┌──────┐ └──────────┬──────────┘
│Rivero│ │Proof │ │Sparse│            │
│Route │ │Tree  │ │/Hybr.│            ▼
└──┬───┘ └──┬───┘ └──────┘  ┌─────────────────────┐
   │        │               │ Segmented Storage   │
   └───┬────┘               │ Snapshots / WAL     │
       ▼                    └─────────────────────┘
  ┌──────────┐
  │ LUTz L0  │
  │ / L1 UB  │
  └────┬─────┘
       ▼
  ┌───────────────┐
  │ Exact SIMD    │
  │ unresolved    │
  └──────┬────────┘
         ▼
  ┌───────────────┐
  │ Exact Top-K   │
  │ + Proof State │
  └───────────────┘
```

### Dense Certified Retrieval

The certified dense path is:

```
query
↓
Rivero proposal ordering
↓
exact seed → kth threshold τ
↓
SemanticProofTree max-UB frontier
↓
region pruning
↓
LUTz L0 upper bounds
↓
LUTz L1 upper bounds where useful
↓
exact SIMD for unresolved vectors
↓
terminate when every unresolved upper bound < τ
```

The critical termination invariant is:

```
max { UB(u) | u ∈ U } < τ
```

where `U` is the unresolved corpus frontier and `τ` is the current k-th exact score.
At that point no unresolved vector can enter the Top-K. Strict `< τ` termination
preserves deterministic tie semantics unless a stronger tie certificate is available.

### Spherical-Cap Proof Bounds

For normalized query `q`, normalized territory centroid `c`, angular radius `θ`, and
scalar `s = qᵀc`:

```
         ⎧ 1,                                         s ≥ cos θ
UB_cap = ⎨
         ⎩ s·cos θ + √max(0, 1 − s²)·sin θ,          s < cos θ
```

HoloSphere combines admissible geometric bounds conservatively and escalates to exact SIMD
scoring whenever the proof hierarchy cannot safely eliminate a candidate.
The proof tree covers the eligible corpus. Rivero prioritizes work; it is not trusted
as the sole source of exactness.

### Pairwise Complex Isometric Folding

An even-dimensional real vector can be represented as half as many complex coordinates:

```
Φ(x)ⱼ = x₂ⱼ + i·x₂ⱼ₊₁
```

This preserves the Euclidean geometry relevant to real cosine / dot-product retrieval:

```
‖Φ(x)‖₂ = ‖x‖₂     Re⟨Φ(x), Φ(y)⟩ = xᵀy
```

This is an isometric representation change, **not compression**. Memory reduction comes
from quantized representations such as CPQ/LUTz codes, not from folding two `f32` values
into one `Complex32`.

### Projective Similarity

For applications intentionally requiring global-phase invariance, HoloSphere also supports
normalized complex projective overlap:

```
P(z, w) = |⟨z, w⟩|² / (‖z‖₂² · ‖w‖₂²)
```

This metric is implemented classically. It is mathematically related to projective
geometry in complex Hilbert spaces but does not imply quantum computation.
For conventional LLM embedding retrieval, cosine similarity remains the normal reference
metric.

### LUTz Progressive Filtering

LUTz provides compact candidate-side codes and query-time lookup tables used to derive
progressively tighter upper bounds before touching full vectors:

```
candidate
↓
L0 bound
├─ UB < τ → prune
└─ unresolved
   ↓
   L1 bound
   ├─ UB < τ → prune
   └─ unresolved
      ↓
      exact SIMD
```

LUTz is a proof/filtering layer, not the source of global completeness. Global exactness
comes from the corpus-covering proof frontier plus exact resolution of all remaining
threats.

---

## Distributed Mutation Semantics

Clustered mutations follow a state-machine replication pipeline:

```
client request
↓
MutationService
↓
Raft proposal
↓
durable local log (CRC-framed, append-only segments)
↓
replication to voting quorum
↓
commit
↓
ShardStateMachine apply
↓
CommitReceipt
↓
client ACK
```

The intended invariant is:

```
ACK  ⟹  quorum committed  ∧  state-machine applied
```

Standalone mode uses a local durability path and does not pay distributed consensus
overhead.

---

## Read Consistency

HoloSphere distinguishes read semantics explicitly:

| Mode | Contract |
|---|---|
| `Linearizable` | Read is established against a quorum-confirmed Raft position applied locally before serving. |
| `Committed` | Reads locally applied committed state according to the replica contract. |
| `BoundedStaleness` | Replica may serve only while observed lag stays within the requested bound. |

Certified search additionally pins the storage generations required by the read so
compaction and segment rotation cannot invalidate the proof universe during execution.

---

## Storage

HoloSphere uses segmented storage rather than a monolithic mutable index:

- active mutable segments
- immutable search segments
- background compaction
- memory-mapped immutable data
- CRC32-checksummed snapshots
- WAL / Raft recovery
- tombstone-aware search
- proof-tree and LUTz sidecars
- remote immutable segment caching for cloud-oriented deployments

Snapshot and recovery semantics are designed so immutable search structures can be
attached without rebuilding the entire retrieval index.

The Raft log uses an append-only segmented format (`.rlog` files) with CRC-framed
entries, bounded rotation, surgical suffix truncation, and snapshot-driven prefix
reclamation. Normal append cost is proportional to the new batch size, not to total
historical log size.

---

## Multi-Tenancy and Metadata

HoloSphere includes:

- namespace-qualified IDs
- RBAC-aware request contexts
- per-tenant quotas
- string interning
- metadata cardinality governance
- adaptive posting representations (sorted postings → Roaring bitmaps → dense bitmaps → compact dictionaries)
- tenant-aware resource accounting

The purpose is not merely performance: bounded metadata growth prevents one tenant from
exhausting node memory.

---

## Security and Operations

The repository contains subsystem support for:

- TLS / mTLS configuration
- OIDC / JWKS validation and RBAC
- KMS-style envelope encryption abstractions
- tamper-evident audit logging (hash chain)
- Prometheus / OpenMetrics telemetry
- OpenTelemetry-compatible tracing
- certificate lifecycle management
- Kubernetes operator with PodDisruptionBudget and learner-first rolling upgrades
- backup / point-in-time recovery
- capacity planning
- system diagnostics

Operational binaries:

- `hnsqr_doctor` — surfaces system-health and integrity problems
- `hnsqr_plan` — estimates capacity and deployment requirements from corpus size,
  dimensionality, QPS, write rate, durability, and replication requirements

---

## Repository Layout

`src/lib.rs` is intentionally the **only** Rust source file directly under `src/`.
All implementation modules belong to purpose-oriented subsystem directories.

```
src/
├── lib.rs
├── capacity/
├── cluster/
├── consensus/
├── ecosystem/
├── federation/
├── kubernetes/
├── metadata/
├── planning/
├── proof/
├── retrieval/
├── rivero/
├── security/
├── service/
├── storage/
├── telemetry/
├── transport/
└── vector/
```

This is a repository invariant, not a style preference. New modules must be placed under
the subsystem that owns their behavior. New `src/foo.rs` files at the root level are not
accepted.

---

## Operating Modes

### Standalone

Use standalone mode when you need:

- maximum local throughput
- embedded or edge deployment
- offline / air-gapped operation
- single-node certified retrieval without consensus overhead

### Clustered

Use clustered mode when you need:

- replicated writes with quorum durability guarantees
- shard routing and scatter-gather search
- automatic failover and leader election
- read replicas / non-voting learners
- online topology migration
- stronger availability guarantees

Cluster mode intentionally sacrifices some raw write throughput in exchange for
consensus semantics.

---

## Current Verification Status

At the current development checkpoint:

```
Unit tests:          58 passing
Doc-tests:            7 passing
Integration tests:   82 passing
────────────────────────────────
Total:              147 passing
Failures:             0
```

Quality gates:

```bash
cargo fmt --all -- --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-targets
```

Gate B dense-search oracle suites require exact Top-K equality against exhaustive SIMD
scoring for Certified retrieval. Passing unit tests alone is not treated as proof of
distributed correctness; the deterministic consensus suite and the process-level chaos
harness serve different purposes.

---

## Current Engineering Focus

Phase 5.2 distributed runtime closure is complete:

```
Segmented Raft log                              PASS
Per-request ReadIndex (context-bound)           PASS
Async mutation runtime (zero busy-spin)         PASS
Multi-process chaos (in-process, real storage)  PASS
Reference architecture gate                     PASS
```

Current hardening work: **Certified Deadline Semantics**

```
CERTIFIED DEADLINE CONTRACT
─────────────────────────────────────────────────────────
no deadline configured → exact result always             PASS
deadline expiry → globally_exact = false                 PASS
deadline expiry → deadline_exceeded = true               PASS
deadline expiry → elapsed_us populated                   PASS
deadline expiry → frontier_nodes_remaining populated     PASS
deadline expiry → region_prune_ratio populated           PASS
typed API (certified_search) → DeadlineExceeded variant  PASS
  cannot be confused with Exact at type boundary
legacy API (search_indices_with_proof) → flat tuple,     PASS
  deadline_exceeded field must be inspected manually
query-wide deadline (stages 1–3 + amortised frontier)    PASS
```

Remaining release work: production benchmarks under real network transport
and multi-region deployment validation.

---

## Build

Requirements:

- stable Rust toolchain supporting the 2024 edition
- a supported CPU target
- optional AVX2/FMA or AArch64 NEON acceleration

```bash
git clone <repository-url>
cd hnsqr
cargo build --release
cargo test --all-targets
```

Strict linting:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

---

## Benchmarks

HoloSphere benchmarks should always document:

- corpus size N
- real / complex dimensionality
- Top-K
- metric and retrieval contract (Bounded / HighRecall / Certified)
- warm vs. cold state
- hardware (CPU, memory, NVMe model)
- exact-evaluation fraction and proof/LUTz pruning ratio
- bytes touched per query
- p50 / p95 / p99 latency

Do not compare a Certified exact configuration to a competitor's approximate
configuration without labeling the semantic difference.

```bash
cargo bench --bench phase4_cloud_scale_benchmark -- --nocapture
cargo bench --bench universal_scorecard_benchmark -- --nocapture
```

Benchmark results are empirical measurements, not universal complexity guarantees.

---

## Minimal Embedded Example

```rust
use hnsqr::{DistanceFunction, HNSQRConfig, HNSQRIndex, VectorEmbedding};

fn main() -> hnsqr::HNSQRResult<()> {
    let mut config = HNSQRConfig::default();
    config.distance_function = DistanceFunction::Cosine;

    // 768 complex dimensions = 1536 real dimensions
    let index = HNSQRIndex::new(config, 768);

    let vector = VectorEmbedding::from_real(&vec![0.042_f32; 1536])?.into_normalized();
    index.insert("doc-001", vector.clone())?;

    let results = index.search(&vector, 10)?;
    for result in &results {
        println!("{result:?}");
    }
    Ok(())
}
```

For production clustered applications, prefer the service interfaces rather than calling
low-level mutable index primitives directly.

---

## Design Principles

- Certified means exact or explicit failure.
- Routing heuristics may prioritize work but never define proof completeness.
- A mutation acknowledgement must correspond to its documented durability and consistency
  level.
- Recovery must never promote uncommitted state into committed application state.
- No public service may bypass the authoritative mutation or read-consistency path.
- No swallowed durability or state-machine errors.
- No fake SDK responses or test-only production claims.
- No root-level Rust modules except `src/lib.rs`.
- No quantum-computing claims.
- No universal O(1) claim for globally Certified exact search.
- Pairwise complex folding is an isometry, not compression.
- Benchmarks must describe the semantics they measure.

---

## Non-Goals

HoloSphere is not trying to be:

- a quantum computer
- an eventually-consistent AP database
- an HNSW clone
- a relational database replacement
- a system that hides approximate results behind an "exact" label

The project prioritizes explicit retrieval contracts, measurable performance,
deterministic correctness, and failure transparency.

---

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
