# HNSQR

HNSQR is a Rust similarity engine for complex-valued embeddings. It combines native
quantum-fidelity metrics with Rivero's Resolve, an optional HNSW fallback graph,
metadata masks, quantized mmap storage, and TCP/HTTP serving.

The central performance contract is deliberately narrow and testable:

> Strict Rivero search performs fixed work with respect to corpus size `N`.

The current selective profile compiles 24 independent E8 foundations. At serving time,
each foundation probes 84 E8 territories plus 32 margin-ordered SimHash buckets:
`24 × (84 + 32) = 2,784` fixed cell probes. A populated cell contains at most 64 compact
eight-byte resident records. The query scans those records, admits at most the best 16,
collision-votes the admitted slots down to at most 2,048 route candidates, then performs
two bounded scored witness hops. Strict mode never falls back to HNSW or a flat scan.

With the production witness profile, one strict query therefore has these hard
corpus-independent ceilings:

| Work stage | Fixed serving ceiling |
| --- | ---: |
| Territory cells probed | 2,784 |
| Compact Q3 resident records inspected | 178,176 |
| Residents admitted before deduplication | 44,544 |
| Route candidates exposed to exact scoring | 2,048 |
| Scored witness edges inspected, two hops | 4,096 |
| Full-dimensional exact score evaluations | 6,144 |

The construction profile is deliberately broader but still fixed in `N`: 9,192 cell
probes, 588,288 compact-resident inspections, 147,072 admissions, and a 1,024-candidate
exact-scoring cap before bounded witness construction.

This is not a claim of universal exact search or constant time in every input:

- Address compilation is `Θ(D)` in complex embedding dimension.
- Exact serving reranking is `Θ(C × D)`, where `C ≤ 6,144` under the production route and
  witness caps.
- Result ranking is bounded by that same constant candidate set.
- Construction and storage remain linear in the number and dimension of vectors.
- The striped Rust `HashMap` directory is expected/amortized `O(1)`, not an
  adversarial worst-case constant-time hash table.
- Rivero is approximate. Recall depends on whether the bounded E8/SimHash routes and
  witness repair admit the true neighbors.

## Search modes

For a published release:

```toml
[dependencies]
hnsqr = "0.1"
```

### Strict Rivero

Use this for corpus-size-independent serving. It skips HNSW construction and forbids
fallback:

```rust
use hnsqr::{HNSQRConfig, HNSQRIndex, VectorEmbedding};

let config = HNSQRConfig::strict_rivero_for_dim(3);
let index = HNSQRIndex::new(config, 3);

index.insert("document-a", VectorEmbedding::new(vec![1.0, 0.0, 0.0]))?;
index.insert("document-b", VectorEmbedding::new(vec![0.9, 0.1, 0.0]))?;

let query = VectorEmbedding::new(vec![1.0, 0.0, 0.0]);
let (results, proof) = index.search_indices_o1_with_diagnostics(&query, 2, None)?;

assert_eq!(proof.cells_probed, 2_784);
assert!(proof.resident_reads <= proof.candidate_read_bound);
assert!(proof.resident_scans <= proof.resident_scan_bound);
assert!(proof.route_candidates_selected <= proof.selected_candidate_bound);
assert!(proof.witness_edges_scanned <= proof.witness_edge_scan_bound);
assert!(!proof.fallback_used);
# Ok::<(), hnsqr::HNSQRError>(())
```

For repeated queries, compile the fixed-size address once and call
`search_indices_with_rivero_address_and_diagnostics`. This removes address compilation
from the route, though full-dimensional exact scoring of the bounded candidates remains.

### Hybrid Rivero + HNSW

`HNSQRConfig::default()` preserves the graph as a recall fallback. Rivero is attempted
first; if it cannot fill the requested `k`, HNSW completes the request. This mode is
usually more forgiving, but it does not have a worst-case `O(1)`-in-`N` guarantee because
fallback graph traversal is allowed. `IndexStats::rivero_fallbacks` makes that visible.

Rivero can also be disabled with `rivero_enabled = false` for graph-only behavior.

## How the selective address works

An arbitrary-dimensional complex vector is compiled into 24 normalized E8 foundations.
Each foundation uses its own deterministic smooth phase reference, so multiplication by
a global complex phase leaves the address unchanged without relying on an unstable
largest-component anchor. Signed CountSketch projections allow any source dimension.

The dependency-free `rune-rivero-core` crate owns canonical E8 root generation and the
`C(7,3)` insert / `C(9,3)` lookup primitives. Equal root scores use a deterministic
score-descending, root-ID-ascending order. Each vector enters 35 E8 cells plus one exact
12-bit SimHash bucket per foundation.

Each cell stores at most 64 eight-byte records: an arena slot plus eight signed Q3
projected coordinates and an affinity byte. Retention is intentionally mixed: the 24
strongest cell-affinity residents remain as elites, while 40 cell-keyed deterministic
minhash residents preserve distributional diversity. This prevents a full cell from
becoming only an insertion-affinity leaderboard.

Lookup is query-adaptive inside each cell. It scans at most 64 compact Q3 records and
admits the best 16 by projected dot product, L1 distance, and stable slot tie-break.
Separately, 12-bit SimHash generates all Hamming-radius-zero-through-four flip masks,
orders them by the query's hyperplane margins, and uses the 32 least-cost probes for
serving (299 for bounded witness construction). Slots encountered in multiple cells are
then ranked by collision votes, accumulated projected dot score, accumulated L1
distance, and slot ID before the 2,048 serving cap is applied.

Each strict node also retains at most 64 deterministic, exact-similarity-scored
reciprocal witnesses. The production profile expands the best 48 route candidates,
exactly scores their first-hop witnesses, and expands the best 16 newly found candidates
for one second hop. Requested seed counts are mechanically clamped to 64. The production
repair remains bounded at `(48 + 16) × 64 = 4,096` witness edges, so the 2,048 direct
route cap plus witness repair yields at most 6,144 full-dimensional exact evaluations.

Publication uses per-slot `WRITING → LIVE → DELETED` atomic state. A node enters Rivero
territories only after its vector, ID, metadata, and node record are initialized; search
checks `LIVE` before reading vector memory. Removal tombstones the slot before evicting
its routes and metadata postings. `clear`, removal, and graph maintenance are excluded
against serving by a lifecycle gate.

## Features

- Complex `VectorEmbedding` values with quantum fidelity, trace distance, Bures,
  complex cosine, and Euclidean scoring.
- Strict fixed-budget Rivero serving plus optional HNSW fallback.
- Global-phase-invariant, dimension-dynamic Rivero address compilation.
- Per-query proof diagnostics for probes, compact scans, admissions, raw/vote-selected
  candidates, witness seeds/edges/additions, exact scores, rejections, results, and
  fallback. Aggregate statistics track routed searches/fallbacks, peak exact candidates,
  cells, scans, admissions, witness work, exact scores, empty routes, and cell overflow.
- Structured metadata filters compiled into `RoaringBitmap` masks.
- AVX2/FMA complex dot-product kernels where supported.
- 8-bit amplitude/phase quantization and asymmetric scoring.
- Optional mmap-backed quantized vector storage.
- Pairwise real-to-complex LLM embedding gateway.
- Async binary TCP protocol and Axum HTTP endpoints.

## Deterministic Rivero quality audit

Run:

```powershell
cargo bench --bench rivero_scaling
```

The hardened release-mode suite completed successfully on 2026-08-14 with deterministic
seed `0x52495645524f2026`, 64 queries, `k = 10`, 24 foundations, and a per-cell admission
budget of 16. Every row made exactly 2,784 probes and passed the fixed compact-scan,
admission, vote-selection, witness-edge, exact-score, and no-fallback assertions. Memory
is the approximate process working-set delta after construction, not an
allocator-isolated heap measurement. Timings are deterministic local-host observations;
the fixed counters, not latency, establish the corpus-size-independent work ceiling.

### Corpus scaling, clustered 64-dimensional complex embeddings

| Vectors | Build | WS delta | Compile p50 | Route mean ± SD | Route p50 / p95 / p99 | Top-1 / R@10 / contain / self | Exact scores avg / max | Exact corpus fraction avg / max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1.577 s | 46.1 MiB | 35.0 µs | 1,170.652 ± 131.647 µs | 1,140.4 / 1,439.8 / 1,731.6 µs | 100% / 100% / 100% / 100% | 859.8 / 953 | 83.96% / 93.07% |
| 4,096 | 10.828 s | 81.6 MiB | 35.7 µs | 2,383.232 ± 159.082 µs | 2,361.6 / 2,643.3 / 2,838.9 µs | 100% / 100% / 100% / 100% | 1,959.0 / 2,147 | 47.83% / 52.42% |
| 16,384 | 73.681 s | 220.2 MiB | 35.5 µs | 3,841.835 ± 239.472 µs | 3,807.0 / 4,163.4 / 4,610.4 µs | 100% / 100% / 100% / 100% | 2,048.0 / 2,048 | 12.50% / 12.50% |
| 65,536 | 450.863 s | 444.2 MiB | 35.9 µs | 4,921.293 ± 347.437 µs | 4,850.9 / 5,567.2 / 6,188.0 µs | 100% / 100% / 100% / 100% | 2,048.6 / 2,053 | 3.13% / 3.13% |

The strict route retained 100% top-1, Recall@10, exact top-10 containment, and
self-recall through 65,536 clustered vectors while the average exactly scored corpus
fraction fell to 3.13%. The p50 log-log slope was 0.3478 and the p50 max/min ratio was
4.2537. Latency is not claimed to be flat: cell occupancy, witness usefulness, and cache
working set grow before the fixed ceilings dominate.

### Workload stress at `N = 16,384`, `D = 64`

| Workload | Build | WS delta | Route p50 / p99 | Top-1 | Recall@10 | Exact containment | Self | Exact scores avg / max | Exact fraction avg / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Perturbed-anchor isotropic | 79.207 s | 348.3 MiB | 3,836.6 / 4,417.9 µs | 100% | 91.88% | 40.62% | 100% | 5,063.1 / 5,169 | 30.90% / 31.55% |
| Independent isotropic | 77.524 s | 349.2 MiB | 3,791.6 / 4,538.1 µs | 84.38% | 78.28% | 4.69% | 100% | 5,107.9 / 5,178 | 31.18% / 31.60% |
| Blended cluster-boundary | 72.191 s | 153.4 MiB | 2,135.2 / 3,729.3 µs | 100% | 99.688% | 96.875% | 100% | 2,308.1 / 2,475 | 14.09% / 15.11% |

The final perturbed-anchor result improves the earlier pre-selective 66.56% Recall@10
to 91.88% at the same corpus size, while preserving 100% top-1 and self-recall. That is
a material repair, not a universal solution. Independent isotropic queries remain the
hardest case because exact random ranks have little reusable structure for a bounded
router.

### Dimension audit, clustered `N = 4,096`

| Complex D | Build | WS delta | Compile p50 | Route p50 / p99 | Top-1 / R@10 / contain / self | Exact scores avg / max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 10.527 s | 55.6 MiB | 4.2 µs | 2,270.8 / 2,916.6 µs | 100% / 100% / 100% / 100% | 1,647.0 / 1,920 |
| 256 | 10.698 s | 114.5 MiB | 140.1 µs | 2,272.2 / 3,576.3 µs | 100% / 100% / 100% / 100% | 2,168.2 / 2,394 |
| 768 | 14.238 s | 190.5 MiB | 426.4 µs | 2,524.3 / 3,046.1 µs | 100% / 100% / 100% / 100% | 2,321.2 / 2,489 |

The near-linear address-compile growth with dimension is the explicit `O(D)` boundary;
the route probe count remains 2,784.

Universal arbitrary-isotropic Recall@10 ≥ 99% remains incompatible with the fixed-work
contract without additional corpus structure or growing replication. As `N` grows while
probes, retained residents, candidate caps, and witness degree remain fixed, the
inspectable fraction shrinks. Use hybrid fallback, a larger deliberately fixed profile,
or an exact/graph index when unstructured top-k recall dominates the work ceiling.

## Current end-to-end benchmark

The full suite completed successfully against the final code:

```powershell
cargo bench --bench benchmark_suite
```

Configuration: 5,000 clustered vectors, 64 complex dimensions, 200 queries, `k = 10`,
hybrid Rivero + HNSW. These are deterministic local-host current-state observations,
not baseline deltas, multi-host variance, or strict-only Rivero measurements.

| Workload | Current result |
| --- | ---: |
| Single-threaded build | 3.90 s; 1,283 vectors/s |
| 8-thread build | 690.04 ms; 7,246 vectors/s; 5.65× suite scaling |
| Search (`ef_search = 64`) | 2,796.2 µs average; 2,775.6 µs p50; 3,411.9 µs p99; 358 QPS; 100% Recall@10 |
| Parallel batch search | 54.82 ms; 3,649 QPS |
| AVX2/FMA complex dot product, D=64 | 127.19 million/s; 65.12 GFLOPS |
| Quantized fidelity error | 0.0001 mean absolute; 0.0012 maximum |
| Mmap quantized ingest + flush | 3.68 s; 1,357 vectors/s |
| Mmap file attach | 12.74 ms |
| Filtered index search | 3,154.1 µs/query; 317 QPS |
| TCP health round trip | 46.4 µs; 21,534 QPS |
| TCP search round trip | 1,487.6 µs; 672 QPS |
| LLM folding, 1,536 real dimensions | 2.53 µs/vector; 395,169 vectors/s |
| Gateway sequential ingest | 1,051.6 µs/vector; 951 vectors/s |
| Gateway batch ingest | 248.9 µs/vector; 4,018 vectors/s |
| Gateway filtered search | 2,916.8 µs/query; 343 QPS |

## Metadata and services

For metadata filtering, insert with `insert_with_metadata` and use a `SearchIntent`
containing a `FilterExpr`, or compile a mask with `evaluate_filter` and pass it to the
index-level filtered search APIs.

Run the TCP and HTTP services:

```powershell
cargo run --release --bin hnsqr_daemon
```

Defaults are TCP `127.0.0.1:9090` and HTTP `127.0.0.1:8080`. Override them with
`HNSQR_TCP_ADDR`, `HNSQR_HTTP_ADDR`, `HNSQR_DATA_DIR`, and `HNSQR_DIM`.

HTTP routes include collection insert, search, batch search, stats, and health checks.
The LLM gateway folds consecutive real values into complex components. This is a
mechanical storage transform, not evidence that phase semantics improve ordinary LLM
embeddings.

## Current limitations

- Strict Rivero is approximate and cannot guarantee universal nearest-neighbor recall.
- The verified `N = 16,384` isotropic-anchor stress reached 91.88% Recall@10, not the
  universal 99% target; preserving that target as `N` grows requires a different
  quality/work contract or assumptions about corpus structure.
- Hybrid fallback invalidates a strict worst-case `O(1)`-in-`N` claim.
- The hash directory provides expected/amortized rather than adversarial worst-case
  constant access.
- Arbitrary metadata-expression compilation is outside the fixed routing budget.
- Mmap currently stores quantized vectors, but `open_mmap` does not reconstruct external
  IDs, metadata, graph state, slot liveness, or Rivero territories. The measured attach
  timing is file mapping/attachment, not a complete persistent-index recovery
  measurement.
- Working-set figures are process-level observations; allocator tracing and TSan/loom
  remain separate verification tasks.

## Development gates

```powershell
cargo fmt --check
cargo check --all-targets
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo bench --bench rivero_scaling
cargo bench --bench benchmark_suite
```

Final status: format, all-target check, strict Clippy, 29 unit tests, 6 doctests,
`rivero_scaling`, and `benchmark_suite` all pass.

See [ELEVATION_REPORT.md](ELEVATION_REPORT.md) for the implementation and verification
record.

## License

Licensed under MIT.
