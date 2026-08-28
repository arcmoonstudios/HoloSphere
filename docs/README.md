# HoloSphere documentation

This directory contains documentation that describes the current implementation. The
repository [README](../README.md) is the product overview and primary API/operations
guide; public Rust interfaces are documented in the source and can be rendered with
`cargo doc --no-deps`.

## Subsystem guides

- [Architecture primitives](ARCHITECTURE_PRIMITIVES.md) — reusable design and
  verification invariants used across planning, retrieval, persistence, and ContextGraph.
- [Governed open-ended discovery](GOVERNED_OPEN_ENDED_DISCOVERY.md) — the constrained
  learning/discovery lifecycle, safety boundary, replication, and recovery contract.
- [Model API integration](MODEL_API_INTEGRATION.md) — local and remote MCP integration,
  authorization, persistence, and embedding-identity requirements.

## Current retrieval contract

`Exact` is the production retrieval authority. A request marked `Certified` is currently
routed to Exact SIMD unless a proof path has passed its explicit admission gate. Rivero,
ProofTree, and related execution plans remain independently observable capabilities;
they are not a blanket claim of constant-time or universally exact retrieval. See the
[root README](../README.md#retrieval-contracts) and
[`src/planning/planner.rs`](../src/planning/planner.rs) for the executable contract.

## Verification

Run the gates appropriate to a change; benchmarks are measurements, not correctness
proofs:

```powershell
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
```

Benchmark targets are declared in [`Cargo.toml`](../Cargo.toml). Record the command,
hardware, corpus/workload identity, and raw result when making a performance claim.

## Documentation policy

Documentation must describe existing, verifiable behavior. Historical delivery reports,
unimplemented roadmaps, superseded benchmark scorecards, and agent prompts are removed
rather than retained as competing sources of truth. Use Git history when the rationale
or chronology of a deleted document is needed.
