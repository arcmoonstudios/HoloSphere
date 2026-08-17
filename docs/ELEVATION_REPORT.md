# HNSQR Elevation Report — Zero-Copy / HPC Review

Date: 2026-08-14

## Scope and composition

- `src/lib.rs` — IV → III → II → I → RVG
- `src/gateway.rs` — IV → III → II → I → RVG
- `src/metadata_index.rs` — IV → III → II → I → RVG
- `src/server.rs` — III → II → I → RVG (inspected; no edit)
- `src/quantization.rs` — II → I → RVG (inspected; no edit)
- `src/mmap_arena.rs` — III → II → I → RVG (documentation corrected)
- `src/rivero.rs` — new fixed-budget routing layer → RVG
- `benches/rivero_scaling.rs` — new deterministic complexity/quality audit → RVG
- `../rune-substrate/crates/rune-rivero-core` — extracted canonical E8 core → RVG

## Verification status

The complete Rust physical gate was run against the final selective profile:

- `cargo fmt --check`: PASS
- `cargo check --all-targets`: PASS
- `cargo clippy --all-targets --all-features -- -D warnings`: PASS
- `cargo test --all-features`: PASS — 29 unit tests and 6 doctests
- `cargo bench --bench rivero_scaling`: PASS — hardened fixed-work matrix
- `cargo bench --bench benchmark_suite`: PASS — current hybrid end-to-end suite

Benchmark timings are deterministic local-host current-state observations. They do not
establish a before/after speedup or multi-host variance; fixed diagnostic assertions,
not latency, establish the strict route's corpus-size-independent work ceiling.

Concurrency tooling (TSan/loom) was not run. No race-freedom claim is made here.

## Applied elevation

### Rivero fixed-budget architecture

HNSQR now has a strict graphless serving mode constructed with
`HNSQRConfig::strict_rivero_for_dim`. Arbitrary-dimensional complex embeddings compile
into 24 deterministic, global-phase-invariant E8 foundations. Each insert registers 35
E8 territories plus one exact 12-bit SimHash bucket per foundation, for 864 fixed route
registrations. Each non-empty serving query probes 84 E8 territories plus 32
query-margin-ordered SimHash buckets per foundation, exactly 2,784 cells.

Every cell retains at most 64 compact eight-byte records. The low 24 code bits contain
eight signed Q3 projected coordinates; the high byte retains insertion affinity. Cell
retention is deterministic and mixed: 24 affinity elites plus 40 cell-keyed minhash
residents. This prevents dense territories from retaining only vectors nearest the cell
center while preserving the strongest local representatives.

Admission is query-adaptive. A dense cell scans all of its at-most-64 compact records,
ranks them by Q3 projected dot product, L1 distance, and slot ID, and admits at most 16.
Across one serving query this gives fixed ceilings of 178,176 compact-record scans and
44,544 admissions. These are inexpensive route records; full-dimensional vectors are
not loaded at this stage.

The supplementary 12-bit SimHash lane generates the fixed pool of all flip masks
through Hamming radius four and orders them by the sum of query hyperplane margins.
Serving uses the 32 lowest-cost signatures; bounded witness construction uses 299.
Repeatedly admitted slots are collision-voted by hit count, accumulated Q3 dot score,
accumulated L1 distance, and stable slot ID. At most 2,048 vote-ranked serving candidates
or 1,024 construction candidates cross into exact scoring.

Strict nodes retain at most 64 deterministic reciprocal witnesses ranked by exact
similarity. The production profile exactly ranks up to 48 route seeds, scans their
first-hop witnesses, then uses up to 16 newly scored first-hop candidates for one second
hop. Requested seed counts are mechanically clamped to 64. The production graphless
repair is bounded at `(48 + 16) × 64 = 4,096` witness-edge inspections. Consequently,
full-dimensional serving scores are capped at 2,048 route candidates plus at most 4,096
newly admitted witnesses: 6,144 evaluations. Strict mode has no HNSW or flat fallback.

The higher fixed construction profile probes 9,192 cells, scans at most 588,288 compact
records, admits at most 147,072 residents, and vote-selects at most 1,024 route
candidates before its independently bounded scored-witness construction. These larger
build constants do not change the lower serving proof.

The initial largest-component phase anchor was rejected by the benchmark because
nearby vectors could switch anchors and compile to unrelated routes. It was replaced
with independent smooth deterministic phase references per foundation. The corrected
compiler is invariant under global complex phase; the verified perturbed-anchor quick
audit produced 100% top-1 and self-recall.

`RiveroSearchDiagnostics` now distinguishes compact resident scans from admitted
resident reads, raw unique route slots from vote-selected route candidates, and route
candidates from witness additions. It also exposes probe/admission/scan/selection
bounds, liveness and filter rejections, first- and second-hop seeds, witness edges and
their bound, exact score evaluations, returned results, and fallback use. Aggregate
`IndexStats` track routed searches/fallbacks, peak exact candidates, cells, compact
scans, admissions, witness edges/additions, exact evaluations, empty routes, populated
cells, and overflows. The deterministic benchmark asserts the fixed-work ceilings for
every query rather than inferring complexity from latency.

Canonical E8 primitives were extracted from `rune-hydron` into the zero-dependency
`rune-rivero-core` crate. Hydron re-exports the same API, preserving `rune-evo`
compatibility. Equal-score root ranking now has a stable root-ID tie-break. Core tests,
strict Clippy, rustdoc, Hydron Rivero tests, and a `rune-evo` library check passed.

The precise complexity claim is expected/amortized `O(1)` in corpus size for strict
resolution. Address compilation is `Θ(D)`; bounded exact reranking is `Θ(CD)`, with
`C ≤ 6,144` under the production serving caps; and the striped Rust `HashMap` directory
is not an adversarial worst-case constant-time table. Hybrid fallback mode is not
worst-case `O(1)` in corpus size.

### Concurrent lifecycle correction

Arena slots now publish through atomic `EMPTY / WRITING / LIVE / DELETED` state.
Search verifies `LIVE` before accessing a routed vector. Live cardinality is tracked
separately from the high-water allocation cursor, so removal updates `size()` and empty
state correctly. Duplicate external IDs are reserved under one map transaction.

Removal tombstones before route eviction, removes metadata postings without retaining a
second metadata copy, and cannot return a deleted slot. `clear`, removal, and optimize
take an exclusive lifecycle gate while serving and insertion take the shared side.
Structured and JSON metadata are indexed inside the insertion publication boundary;
batch metadata no longer has a visible route-before-metadata window.

### `src/lib.rs`

Structural reconstruction: the gateway creates an owned folded embedding, so the
normalization stage can retain and mutate that allocation instead of creating a second
same-sized vector. `VectorEmbedding::into_normalized` now performs that in-place
normalization. `normalize(&self)` retains its cloning API semantics.

Structural reconstruction: batch gateway metadata was cloned solely to cross an
insertion boundary. `insert_with_metadata_ref` retains the vector by value but borrows
metadata until the metadata index has materialized its durable index keys and bitmaps.
No metadata reference escapes the call.

### `src/gateway.rs`

Structural reconstruction: batch LLM ingestion previously constructed temporary tuples
by cloning every ID and, for metadata batches, each metadata map. The revised Rayon
paths fold each borrowed input and immediately insert it, retaining only the one
required folded `VectorEmbedding` allocation.

Correctness fortress: `fold_llm_embedding` no longer reinterprets an arbitrary `f32`
slice as `Complex32` through a raw pointer. Safe pairwise construction preserves the
same real/imaginary mapping and removes the alignment/layout safety assumption.

The folding transform necessarily materializes complex output. It is minimal-copy, not
literal zero-copy; there is one output allocation and no intermediate vector.

### `src/metadata_index.rs`

Structural reconstruction: equality and set-membership filtering only require a
temporary lookup key. String metadata now borrows its stored string directly; numeric
and boolean metadata render into a 384-byte stack buffer through the new
`MetadataValue::write_key_to` API. This preserves the pre-existing canonical key
format, including six decimal places for floats, without allocating a temporary
`String` on the filter path.

The 384-byte capacity covers a signed fixed-six-decimal representation of finite
`f64` values. The formatter returns an error rather than overflowing if a future
representation violates that bound.

### `src/server.rs`

Inspected, no edit. The TCP receive buffer is byte-addressed and does not guarantee
`Complex32` alignment or a Rust-compatible typed layout. Materializing aligned complex
vectors before graph operations is therefore structurally required by the current wire
protocol. The existing direct `BytesMut` response receive avoids an additional
response-buffer copy.

### `src/quantization.rs`

Inspected, no edit. Quantization borrows the input and makes one required owned byte
output. The two-pass range scan does not allocate intermediate amplitude or phase
arrays. SIMD eligibility remains a physical hypothesis: the numeric slice is contiguous
and homogeneous, but no measured A/B result justifies a new intrinsic path.

### `src/mmap_arena.rs`

Inspected, no edit. Mmap reads operate directly against mapped pages. Quantized writes
are a required representation transform into the mapped storage, not a redundant
intermediate copy. Unsafe regions remain bounded by mapping validity and computed
offset invariants; concurrency verification remains open pending TSan/loom or an
equivalent stress harness.

The mmap benchmark wording was corrected. `open_mmap` currently attaches the quantized
file but does not restore external IDs, metadata, liveness, graph state, or Rivero
territories. Its timing is a mapping-attach measurement, not complete persistent-index
recovery.

### `benches/benchmark_suite.rs`

Tool-grounded correction: a benchmark run failed on Windows with OS error 1224 because
the benchmark reopened a file while its initial mapped index was still live, and a
prior interrupted run could leave the fixed temporary filename unavailable. The
benchmark now uses a process-unique temporary filename, drops each mapped index before
the next file operation, and removes the file afterward.

Correctness fortress: the benchmark previously claimed that an allocation-free filtered
bit-test had been verified, although its timing harness cannot establish allocation
behavior. The message now reports only what the harness proves.

## Physical observations

The hardened deterministic audit used seed `0x52495645524f2026`, 64 queries, `k = 10`,
24 foundations, and resident budget 16. Every workload made exactly 2,784 probes and
passed its compact-scan, admission, vote-selection, witness-edge, exact-score, and
no-fallback assertions.

### Clustered corpus scaling, `D = 64`

| N | Build | WS delta | Compile p50 | Route mean ± SD | p50 / p95 / p99 | Top-1 / R@10 / contain / self | Exact avg / max | Exact fraction avg / max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1.577 s | 46.1 MiB | 35.0 µs | 1,170.652 ± 131.647 µs | 1,140.4 / 1,439.8 / 1,731.6 µs | 100% / 100% / 100% / 100% | 859.8 / 953 | 83.96% / 93.07% |
| 4,096 | 10.828 s | 81.6 MiB | 35.7 µs | 2,383.232 ± 159.082 µs | 2,361.6 / 2,643.3 / 2,838.9 µs | 100% / 100% / 100% / 100% | 1,959.0 / 2,147 | 47.83% / 52.42% |
| 16,384 | 73.681 s | 220.2 MiB | 35.5 µs | 3,841.835 ± 239.472 µs | 3,807.0 / 4,163.4 / 4,610.4 µs | 100% / 100% / 100% / 100% | 2,048.0 / 2,048 | 12.50% / 12.50% |
| 65,536 | 450.863 s | 444.2 MiB | 35.9 µs | 4,921.293 ± 347.437 µs | 4,850.9 / 5,567.2 / 6,188.0 µs | 100% / 100% / 100% / 100% | 2,048.6 / 2,053 | 3.13% / 3.13% |

The p50 log-log slope was 0.3478 and the p50 max/min ratio was 4.2537. Fixed counters,
not the rising latency, prove the complexity boundary. The clustered suite retained
100% top-1, Recall@10, exact containment, and self-recall through 65,536 vectors while
the exactly scored fraction fell to 3.13%.

### Workload stress, `N = 16,384`, `D = 64`

| Workload | Build | WS delta | Route p50 / p99 | Top-1 | R@10 | Contain | Self | Exact avg / max | Exact fraction avg / max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Perturbed-anchor isotropic | 79.207 s | 348.3 MiB | 3,836.6 / 4,417.9 µs | 100% | 91.88% | 40.62% | 100% | 5,063.1 / 5,169 | 30.90% / 31.55% |
| Independent isotropic | 77.524 s | 349.2 MiB | 3,791.6 / 4,538.1 µs | 84.38% | 78.28% | 4.69% | 100% | 5,107.9 / 5,178 | 31.18% / 31.60% |
| Cluster boundary | 72.191 s | 153.4 MiB | 2,135.2 / 3,729.3 µs | 100% | 99.688% | 96.875% | 100% | 2,308.1 / 2,475 | 14.09% / 15.11% |

The final perturbed-anchor result improves the old profile's 66.56% Recall@10 to
91.88% while retaining 100% top-1 and self-recall. The universal isotropic problem is
not described as solved: exact random lower ranks offer little stable structure, and a
fixed candidate ceiling inspects a shrinking corpus fraction as `N` grows. Universal
Recall@10 ≥ 99% requires additional data assumptions, growing replication, or
corpus-growing query work; the latter conflicts with the strict fixed-`N` contract.

### Dimension audit, clustered `N = 4,096`

| D | Build | WS delta | Compile p50 | Route p50 / p99 | Quality | Exact avg / max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 10.527 s | 55.6 MiB | 4.2 µs | 2,270.8 / 2,916.6 µs | all four metrics 100% | 1,647.0 / 1,920 |
| 256 | 10.698 s | 114.5 MiB | 140.1 µs | 2,272.2 / 3,576.3 µs | all four metrics 100% | 2,168.2 / 2,394 |
| 768 | 14.238 s | 190.5 MiB | 426.4 µs | 2,524.3 / 3,046.1 µs | all four metrics 100% | 2,321.2 / 2,489 |

Address compilation scales with dimension as stated by the `O(D)` boundary while route
probes remain fixed at 2,784.

### Current hybrid end-to-end suite

The 5,000-vector, `D = 64`, 200-query suite completed with 1,283 vectors/s
single-threaded build, 7,246 vectors/s eight-thread build (5.65×), 2,796.2 µs average
search latency, 358 QPS, and 100% Recall@10. Parallel batch search reached 3,649 QPS.
The AVX2/FMA kernel measured 127.19 million dot products/s and 65.12 GFLOPS;
quantization error was 0.0001 mean absolute and 0.0012 maximum.

Mmap ingest measured 1,357 vectors/s and attach 12.74 ms. Filtered index search measured
3,154.1 µs/query (317 QPS). TCP health measured 46.4 µs (21,534 QPS) and TCP search
1,487.6 µs (672 QPS). LLM folding measured 2.53 µs/vector (395,169 vectors/s); gateway
sequential ingest 1,051.6 µs/vector (951/s), batch ingest 248.9 µs/vector (4,018/s), and
filtered search 2,916.8 µs/query (343 QPS).

These are deterministic current-state local-host observations, not before/after deltas
or multi-host variance. The end-to-end search suite uses hybrid Rivero + HNSW and is not
a strict-only Rivero measurement.

## Open items

- `[QUALITY/CONTRACT DECISION]` Strict fixed-work Rivero cannot promise universal
  arbitrary-isotropic exact Recall@10 ≥ 99% as `N` grows. Use hybrid fallback, publish a
  larger fixed profile with its explicit ceilings, or relax the corpus-independent work
  contract for workloads requiring that guarantee.
- `[BENCH REQUIRED]` Compare safe pairwise folding against the former raw-copy folding
  implementation on representative CPU targets before claiming a throughput change.
- `[BENCH REQUIRED]` Measure numeric `Eq` and `In` metadata filters with allocation
  tracing or an A/B benchmark to quantify the stack-key change.
- `[BENCH REQUIRED]` Benchmark the `split_fold_llm_embedding` transcendental path at
  dimensions above 512 before considering a phase lookup table.
- `[TSAN/LOOM REQUIRED]` Exercise shared arena, mmap, and TCP server concurrency
  surfaces. This review does not prove race-freedom.
- `[PERSISTENCE V2 REQUIRED]` Persist or rebuild Rivero addresses, territories, external
  IDs, metadata, liveness, and a committed high-water record before advertising mmap
  reopen as a recovered strict index.
- `[BOUNDED HASH REQUIRED]` Replace the striped standard `HashMap` with a fixed-probe
  directory before claiming adversarial worst-case rather than expected/amortized
  `O(1)` directory access.
- The current TCP protocol cannot be proven end-to-end zero-copy without a specified
  aligned, layout-stable wire representation. The present aligned decode is retained
  for correctness.

## Physical Gate Checklist

### Final selective profile

- [x] `cargo fmt --check` — PASS
- [x] `cargo check --all-targets` — PASS
- [x] `cargo clippy --all-targets --all-features -- -D warnings` — PASS
- [x] `cargo test --all-features` — PASS, 29 unit tests + 6 doctests
- [x] `cargo bench --bench rivero_scaling` — PASS, complete hardened matrix and every
  fixed-work assertion
- [x] `cargo bench --bench benchmark_suite` — PASS, current hybrid end-to-end suite
- [ ] ThreadSanitizer / loom — NOT RUN
- [ ] `cargo udeps` / semgrep / rust-analyzer diagnostics — NOT RUN
