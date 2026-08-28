# Changelog

All notable changes to HoloSphere are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **HoloSphere ContextGraph Subsystem (`src/contextgraph/`)**:
  - Domain-neutral `Entity` and `Relation` hypergraph model supporting N-ary participant roles and extensible namespaces (`code:*`, `document:*`, `system:*`, `git:*`).
  - Staged universal transformation pipeline: $\text{detect} \to \text{fingerprint} \to \text{extract} \to \text{resolve} \to \text{validate} \to \text{delta} \to \text{commit}$.
  - Tree-sitter Rust AST adapter (`RustSourceAdapter`) extracting functions, structs, traits, enums, impls, tests, and architectural rationale comments (`// SAFETY:`, `// WHY:`, `// NOTE:`).
  - Markdown document adapter (`MarkdownSourceAdapter`) extracting document sections, claims, notes, and citations.
  - Filesystem crawler (`FilesystemSourceAdapter`) for directory ingestion.
  - Multi-pass reference resolution (`UniversalReferenceResolver`) with explicit ambiguity preservation (`RelationOrigin::Ambiguous`).
  - Monotonically advancing LSN snapshots and atomic delta publication in `ContextGraphStore`.
  - Deterministic canonical graph fingerprint engine (`GraphFingerprinter`) guaranteeing bit-exact signatures ($1\text{ thread} \equiv N\text{ threads}$; $\text{Full} \equiv \text{Incremental}$).
  - Fine-grained dependency invalidation graph (`InvalidationGraph`) for incremental re-compilation.
  - Dynamic query planner (`QueryPlanner`) and context budget governor (`ContextBudget`) preventing context bloat.
  - First-class snapshot diffing engine (`ContextGraphDiff`) and retrieval methods (`search`, `explore`, `traverse`, `path`, `impact`, `diff`).
  - Scope clustering (`ScopeClustering`) and architectural analytics (`ContextAnalytics` for hubs, cycles, and orphans).
  - Multi-view visualizer exports: `MarkdownReportView` (`CONTEXT_REPORT.md`), interactive canvas `HtmlVisualizerView` (`contextgraph.html`), and `JsonExportView` (`contextgraph.json`).
  - Live workspace watcher (`ContextGraphWatcher`) for background re-compilation.
- **Operational CLI Tools (`src/bin/`)**:
  - `hnsqr_contextgraph`: Universal ContextGraph CLI supporting `build`, `search`, `explore`, `path`, and `report`.
  - `hnsqr_codegraph`: Specialized codebase ingestion profile compatibility wrapper.
- **Expanded Model Gateway & MCP Surface (`src/transport/`)**:
  - Added `ingest`, `explore`, `path`, and `diff` tool endpoints in `ModelToolService` and `mcp.rs`.
  - Added `status` preflight for authorization, live-web availability, collection embedding identities, limits, and degradations.
  - Added bounded `run_case` preparation with domain-neutral recipes, evidence policies, explicit action gates, and no external-action execution.
  - Added canonical web `evidence_id` values, `max_results` compatibility for `web_search`, and automatic source registration for write-authorized callers.
  - Added explainable outcome-aware resolution ranking using semantic relevance, verification, measured reproducibility, prior success, and recency.
  - Added TypeScript and Python SDK support for `status` and `run_case`; orchestration no longer labels callback-only outcomes as empirical verification.
  - Generalized MCP integration tests to assert required capabilities rather than a fixed tool count.
- **Universal Test & Certification Suite (`tests/contextgraph_universal_test.rs`)**:
  - 8 rigorous integration tests verifying all 10 architectural gates, determinism, AST rationale extraction, atomic LSN snapshots, query planning, differential analysis, scalability hardening, and workspace self-compilation.
- **Engineering Doctrine & Reusable Primitives (`docs/ARCHITECTURE_PRIMITIVES.md`)**:
  - Formalized the 13 domain-neutral engineering primitives (P1–P13) across planning, multi-objective admission, funnel tracing, counterexample corpus, baseline regression gates, contract equivalence, and incremental state parity.
  - Documented core state and persistence invariants ($\text{FULL} \equiv \text{INCREMENTAL}$ and $\text{ATTACH} \ne \text{EAGER TOUCH}$) and the governing process rule of evidence-ordered optimization.
- Repository governance, support, security, contribution, and GitHub automation artifacts.

### Changed

- Documented Exact SIMD as the current production retrieval authority and clarified that
  `Certified` is planner-routed to Exact SIMD until a proof path passes its admission gate.
- Replaced stale fixed benchmark scorecards and crossover tables with reproducible benchmark
  commands and explicit hardware/corpus calibration guidance.
- Documented the mandatory separation between the correctness test gate and benchmark runs.
- Completely removed all legacy experimental quantization paths from the primary index and segmented write/compaction paths in favor of Exact SIMD and ProofTree hierarchies.
- Optimized the Exact cosine/Euclidean hot path by hoisting the metric lookup and introducing a
  real-component-only SIMD inner-product primitive.

## [0.1.0] - 2026-08-26

### Added

- Initial public HoloSphere release.
