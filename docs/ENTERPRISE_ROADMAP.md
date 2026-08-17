# HNSQR Enterprise Competitive Roadmap

**Target:** Beat Milvus and Pinecone on single-node performance, then scale horizontally

**Date:** 2026-08-15

**Status:** Post-Rivero hardening; moving from verified methodology to enterprise viability

---

## Executive Summary

HNSQR has successfully proven **bounded-latency-at-scale** (O(1) in corpus size) up to 65K vectors with a real methodology. This addresses the "toy-scale" objection and moves the needle from "asserted, unverified" to "asserted, verified within tested bounds."

**The core differentiation claim—quantum fidelity beats cosine on real retrieval—remains unproven.** This is the gap that decides whether HNSQR is a novel retrieval system or a fast implementation of an unproven metric.

**Three critical missing pieces prevent enterprise competition:**

1. **Metric validation** — Does fidelity actually beat cosine on real embeddings?
2. **Enterprise features** — Filtered search at scale, persistence, verified concurrency
3. **Distributed architecture** — Multi-node sharding, replication, and coordination

This roadmap prioritizes proving the core value proposition first, then building enterprise features, then scaling horizontally.

---

## Phase 0: The Falsifiable Question (HIGHEST PRIORITY)

**Goal:** Determine if quantum fidelity provides retrieval value over cosine similarity

**Why this matters:** Everything downstream depends on this answer. If fidelity doesn't beat cosine on real retrieval tasks, the entire differentiation story collapses to "fast implementation of an unproven metric."

### Critical Analysis: What Fidelity Actually Computes

Given the fold operation `z_i = x_{2i} + i·x_{2i+1}`, the quantum fidelity reduces to:

```
F(z,w) = cos²(x,y) + Cross²(x,y) / (||x||²·||y||²)
```

where `Cross(x,y) = Σᵢ[x_{2i}·y_{2i+1} − x_{2i+1}·y_{2i}]` is a sum of 2D cross products.

**The decisive question:** Does `Cross(x,y)` carry relevance signal beyond what `cos(x,y)` already captures, or is it noise from arbitrary adjacent-index pairing?

Adjacent-index pairing has **no principled basis**:
- Embedding dimensions from a linear projection head carry no canonical "this dim rotates with the next dim" structure
- Any orthogonal rotation of the embedding space is cosine-equivalent but changes Cross completely
- If the pairing is arbitrary, Cross is structured noise, and fidelity will match or underperform cosine

### P0.1: Cross-term Correlation Diagnostic (1-2 days)

**This is the cheapest, most decisive experiment. Do this first.**

**Method:**
1. Take a real embedding model (sentence-transformer or actual OpenAI/Cohere API)
2. Use a labeled relevance set (BEIR subset like SciFact or NFCorpus — small enough for brute force, has real judgments)
3. For all query/candidate pairs:
   - Compute `cos(x,y)` (standard dot product normalized)
   - Compute `Cross(x,y)` from adjacent-pair 2D cross products
4. Regress relevance labels against `Cross²` **controlling for** `cos²`
5. Measure:
   - Correlation coefficient of Cross with relevance
   - Regression coefficient (partial effect after controlling for cosine)
   - Statistical significance (p-value)

**Exit criteria:**
- **If Cross is uncorrelated with relevance once cos is accounted for:** Fidelity-over-cosine is dead on arrival. Stop. Pivot to:
  - Option A: Try learned pairing (train a model to find optimal dimension pairings)
  - Option B: Drop the differentiation claim, sell on latency contract alone
  - Option C: Investigate different quantum-inspired metrics with principled basis
- **If Cross shows significant signal:** Proceed to P0.2

**Deliverable:** `notebooks/cross_term_analysis.ipynb` with statistical results and visualizations

### P0.2: Real-Embeddings IR Benchmark — Exact Scoring Only (3-5 days)

**No Rivero. No approximation. Pure metric comparison.**

**Method:**
1. Same real embedding model and corpus from P0.1
2. Compute rankings two ways:
   - Baseline: Standard cosine similarity (industry standard)
   - Experimental: Quantum fidelity on folded pairs
3. Score both against real relevance judgments:
   - NDCG@10 (normalized discounted cumulative gain)
   - Recall@10
   - MRR@10 (mean reciprocal rank)
4. Run **paired significance test** across queries (not just single aggregate)
5. Test on multiple BEIR datasets:
   - SciFact (scientific claims)
   - NFCorpus (medical)
   - FiQA (financial QA)
   - At least one high-dimensional production embedding model (768-4096D)

**Exit criteria:**
- **If fidelity underperforms or matches cosine:** Same pivot decision as P0.1
- **If fidelity consistently outperforms with statistical significance:** This is the proof needed. Proceed to Phase 1.

**Deliverable:** 
- `benchmarks/metric_comparison.py` 
- Technical report: `FIDELITY_VALIDATION.md` with:
  - Statistical results across datasets
  - Per-query performance distributions
  - Analysis of when fidelity helps vs hurts

### P0.3: Fix the Synthetic Benchmark Lie (1 day)

**Problem:** `benchmark_suite.rs` claims to use "OpenAI 1536-dimensional real float embeddings" but generates `((i + j) as f32).sin()`. This is a smooth deterministic sinusoid with no relationship to actual embedding statistics.

**Action:**
1. Relabel honestly: "synthetic sinusoidal 1536-dim float vectors"
2. OR replace with actual model output from a real embedding model
3. Update all documentation referencing this benchmark

**This must be fixed before any external presentation.**

---

## Phase 1: Enterprise Feature Parity (AFTER P0 validation)

**Assumption:** Phase 0 proved fidelity provides real retrieval value. Now build enterprise features.

### P1.1: Filtered Search at Scale (2-3 weeks)

**Problem:** Current `resolve_rivero_candidates` applies filters as post-hoc rejection within the fixed 2,784-cell / 2,048-candidate budget. At high selectivity on large corpus, this causes silent recall collapse.

**Example failure mode:**
- 65K corpus: Fixed route exposes 3.13% (2,048 vectors)
- Filter selectivity: 1% of corpus matches
- Expected surviving candidates: 2,048 × 0.01 = ~20 vectors
- As N grows, this approaches zero — exactly backwards from O(1) promise

**Enterprise competitors (Pinecone, Milvus, Qdrant) all do filter-aware retrieval:**
- Pre-filtering: Only insert/index vectors matching common filters
- Adaptive expansion: Dynamically increase search budget until k filtered results found
- Hybrid: Combine both strategies

**Implementation path:**

#### P1.1.1: Filter-Aware Route Expansion
- Add `filter_aware_search` mode that tracks filtered candidate count
- If filtered candidates < k, expand search:
  - Increase SimHash Hamming radius probes (32 → 64 → 128)
  - Add secondary witness hops (16 → 32 → 64)
  - Track "expansion rounds" in diagnostics
- Hard ceiling: Max 3× base budget before fallback
- Preserve strict mode for unfiltered queries

#### P1.1.2: Selective Route Compilation
- For high-selectivity filters (< 5% corpus), compile Rivero address only for matching subset
- Maintain separate routing structures per "filter partition"
- Trade: More storage for better filtered recall
- Measure: Storage overhead vs recall improvement

#### P1.1.3: Benchmark Matrix
- Test filtered search with varying:
  - Corpus size: 4K → 16K → 65K → 256K
  - Filter selectivity: 50% → 10% → 1% → 0.1%
  - Dimensionality: 64 → 256 → 768
- Assert: Recall@10 ≥ 95% across all combinations
- Measure: Latency P50/P95/P99, memory overhead

**Deliverable:**
- `src/filtered_rivero.rs` — Filter-aware routing
- `benches/filtered_scaling.rs` — Comprehensive filtered search audit
- README section documenting filter performance guarantees

### P1.2: Production-Grade Persistence (3-4 weeks)

**Problem:** Current mmap only stores quantized vectors. `open_mmap` attaches the file but doesn't restore external IDs, metadata, liveness, graph state, or Rivero territories. The 12.74ms attach timing is misleading — it's not a complete persistent-index recovery.

**Enterprise requirement:** Crash recovery, hot backup, version migration

**Implementation path:**

#### P1.2.1: Serialization Format Design
- Design versioned binary format:
  ```
  Header: [magic: u32, version: u16, flags: u16, checksum: u64]
  Metadata block: [external_ids, metadata_index, roaring_bitmaps]
  Arena block: [slot_states, high_water, size, deleted_count]
  Vectors block: [quantized_vectors] (already exists)
  Rivero block: [addresses, territories, cell_occupancy]
  HNSW block: [graph_edges, entry_point] (if enabled)
  Footer: [block_offsets, total_size, checksum]
  ```
- Use explicit versioning for backward compatibility
- Add integrity checks (checksums per block + global)

#### P1.2.2: Atomic Write Protocol
- Write to temp file: `{name}.hnsqr.tmp`
- Flush and fsync all blocks
- Atomic rename: `{name}.hnsqr.tmp` → `{name}.hnsqr`
- On crash: Detect incomplete writes, fallback to last good snapshot

#### P1.2.3: Incremental Snapshots
- Support "checkpoint + WAL" pattern:
  - Base snapshot: Full index state
  - Write-ahead log: Incremental inserts/deletes since last snapshot
  - Recovery: Load base + replay WAL
- Configurable snapshot interval (time-based or mutation-count-based)

#### P1.2.4: Hot Backup
- Read-only snapshot while serving continues
- Copy-on-write for modified pages
- Stream to backup location without blocking queries

**Deliverable:**
- `src/persistence.rs` — Serialization and recovery
- `src/snapshot.rs` — Enhanced with full state (currently stub)
- Integration tests for crash recovery scenarios
- Documentation: Recovery time objectives (RTO) and point objectives (RPO)

### P1.3: Verified Concurrency (2-3 weeks)

**Problem:** The concurrent arena has disciplined design (SLOT_EMPTY → WRITING → LIVE/DELETED with Acquire/Release ordering), but no verification. ThreadSanitizer and Loom haven't run.

**Enterprise requirement:** Proven race-freedom under adversarial interleaving

**Implementation path:**

#### P1.3.1: Loom Model Checking
- Add `loom` dev-dependency
- Write focused concurrency tests:
  - Concurrent insert + search
  - Concurrent insert + delete
  - Concurrent search + optimize
  - Concurrent batch operations
- Run loom exhaustive exploration on small scenarios
- Target: Proof of correctness for core state machine

#### P1.3.2: ThreadSanitizer Stress Testing
- Add CI job with `-Zsanitizer=thread`
- Run extended stress tests:
  - 16-thread insert + search hammering
  - Random insert/delete/search mix
  - Metadata updates during search
- Run for extended duration (hours, not seconds)

#### P1.3.3: Formal Invariants Documentation
- Document all concurrency invariants in code:
  - Memory ordering requirements
  - Lock ordering (if any)
  - State machine transitions
- Add `debug_assert!` checks for invariants in hot paths (already started)
- Consider formal verification tool (TLA+ or Ivy) for core protocols

**Deliverable:**
- `tests/concurrency_loom.rs` — Loom verification suite
- CI: ThreadSanitizer job (must pass on every PR)
- `CONCURRENCY.md` — Formal invariants and verification results

### P1.4: Joint N×D Scaling Validation (1-2 weeks)

**Problem:** Benchmarks sweep N at fixed D=64, or D at fixed N=4,096. Production embeddings sit at D=768-4096 with N in millions. The untested joint corner is where "corpus-size-independent" is stressed hardest.

**Implementation path:**

#### P1.4.1: Extended Scaling Benchmark
- Add `benches/joint_scaling.rs`:
  ```
  D=256:  N=16K, 65K, 256K, 1M
  D=768:  N=16K, 65K, 256K, 1M
  D=1536: N=16K, 65K, 256K, 1M
  D=3072: N=16K, 65K (if memory permits)
  ```
- Measure:
  - Address compilation time (expect Θ(D))
  - Route latency P50/P95/P99
  - Memory working set
  - Recall@10 (assert ≥ 95% for clustered)

#### P1.4.2: Real Embedding Geometry
- Test on actual production embeddings:
  - OpenAI ada-002 (1536D)
  - Cohere embed-v3 (1024D)
  - E5-large (1024D)
  - Multimodal CLIP (768D)
- Validate recall on real anisotropic structure ("cone effect")
- Current benchmarks only use synthetic clustered/isotropic phase data

**Deliverable:**
- `benches/joint_scaling.rs` — Full N×D matrix
- `REAL_EMBEDDING_VALIDATION.md` — Results on production models

---

## Phase 2: Single-Node Performance Domination

**Goal:** Beat Pinecone and Milvus on single-node raw ANN quality and throughput

### P2.1: Quantitative Competitive Benchmark (2 weeks)

**Set up apples-to-apples comparison:**

#### P2.1.1: Unified Benchmark Suite
- Common datasets:
  - SIFT1M (1M vectors, 128D) — standard ANN benchmark
  - GIST1M (1M vectors, 960D) — high-dimensional
  - Deep1B subset (10M vectors, 96D) — scale test
- Common metrics:
  - Recall@10, Recall@100
  - Queries per second (QPS)
  - Index build time
  - Memory footprint
  - P50/P95/P99 latency

#### P2.1.2: Competitor Baselines
- Deploy locally:
  - Milvus (latest stable)
  - Qdrant (latest stable)
  - Pinecone (via API, for reference)
  - FAISS IVF (Facebook's baseline)
- Use default/recommended configs for each
- Match hardware: Same machine, same resource limits

#### P2.1.3: HNSQR Optimization Pass
- Profile hot paths with perf/flamegraph
- Optimize critical sections:
  - SIMD utilization in scoring
  - Cache efficiency in cell probes
  - Lock contention in concurrent ops
- Set competitive ef_search / nprobes parameters

**Target:** 
- **2-5× faster P50 latency** at same recall level
- **3-10× higher QPS** for batch queries
- **2× lower memory** per vector

**Deliverable:**
- `benchmarks/competitive_suite/` — Unified harness for all systems
- `COMPETITIVE_RESULTS.md` — Head-to-head comparison tables

### P2.2: Advanced Query Features (2-3 weeks)

**Feature parity with enterprise competitors:**

#### P2.2.1: Hybrid Search
- Combine vector similarity + keyword search
- BM25 or similar for text fields
- Fused ranking (weighted combination)
- Used by: Pinecone Hybrid, Weaviate, Qdrant

#### P2.2.2: Multi-Vector Search
- Multiple query vectors in single request
- Use cases: Multi-modal (image + text), ensembles
- Aggregation strategies: max, avg, weighted
- Used by: Weaviate, Milvus

#### P2.2.3: Range Search
- Find all vectors within distance threshold
- Different from k-NN (unknown result count)
- Critical for clustering, deduplication
- Used by: Milvus, Qdrant

#### P2.2.4: Batch Operations
- Bulk insert with guaranteed ordering
- Bulk delete
- Batch upsert (update or insert)
- Transaction-like semantics

**Deliverable:**
- API extensions in `src/lib.rs`
- HTTP endpoints in `src/server.rs`
- Integration tests and benchmarks

### P2.3: Production Observability (1-2 weeks)

**Enterprise requirement:** Deep introspection and diagnostics

#### P2.3.1: Metrics Export
- Prometheus endpoint (standard for monitoring)
- Key metrics:
  - Query latency histograms (P50/P90/P95/P99)
  - Insert/delete throughput
  - Index size and memory usage
  - Cache hit rates
  - Rivero vs HNSW fallback rates
  - Error rates by type

#### P2.3.2: Structured Logging
- Use tracing spans for request tracing
- Correlation IDs across distributed calls
- Configurable log levels (per module)
- JSON output for log aggregation (ELK, Loki)

#### P2.3.3: Health and Readiness
- Kubernetes-style health checks:
  - `/healthz` — basic liveness
  - `/readyz` — ready to serve traffic
  - `/metrics` — Prometheus scrape endpoint
- Graceful shutdown with connection draining

**Deliverable:**
- `src/observability.rs` — Metrics and tracing
- `examples/prometheus_grafana/` — Sample dashboards
- Documentation: Monitoring guide

---

## Phase 3: Horizontal Scale — The Distributed Gap

**This is the largest lift.** Milvus and Pinecone are sharded, replicated systems for 100M+ vectors. Nothing in current HNSQR has a multi-node story.

### P3.1: Architecture Design (3-4 weeks research)

**Key decisions:**

#### P3.1.1: Sharding Strategy
- Hash-based sharding (consistent hashing)
- Range-based sharding (partition by ID range)
- Learned sharding (cluster-aware partitioning)
- Trade-offs:
  - Hash: Simple, balanced, but queries hit all shards
  - Range: Locality-aware, but rebalancing hard
  - Learned: Best for clustered data, complex

#### P3.1.2: Query Routing
- Scatter-gather (query all shards, merge results)
- Cluster routing (learn cluster centers, route to relevant shards)
- Hybrid (route to subset, expand if needed)

#### P3.1.3: Replication and Consistency
- Replication factor (typically 3)
- Consensus protocol (Raft, Paxos, or Viewstamped Replication)
- Read consistency models:
  - Strong (linearizable)
  - Eventual (higher throughput)
  - Configurable (per-query choice)

#### P3.1.4: Failure Handling
- Shard failure detection (heartbeats, gossip)
- Automatic failover (promote replica to primary)
- Rebalancing (add/remove nodes)
- Split/merge shards (dynamic partitioning)

**Study existing architectures:**
- Milvus: Segment-based, separate query/index nodes
- Qdrant: Collections + shards, gRPC coordination
- Vespa: Content clusters + container clusters

**Deliverable:**
- `DISTRIBUTED_ARCHITECTURE.md` — Design document with trade-offs
- Consensus on approach (present options, decide collaboratively)

### P3.2: Coordination Layer (4-6 weeks)

**Implementation of distributed primitives:**

#### P3.2.1: Cluster Membership
- Use etcd or Consul for service discovery
- Node registration and health monitoring
- Shard assignment tracking
- Configuration propagation

#### P3.2.2: Metadata Store
- Store collection schemas, shard mappings, replica sets
- Strongly consistent (use etcd or similar)
- Cached locally with invalidation

#### P3.2.3: Shard Router
- Accept queries, determine target shards
- Parallel RPC to shards, merge results
- Load balancing across replicas
- Circuit breaker for failing shards

**Deliverable:**
- `hnsqr-coordinator` binary — Coordination service
- `src/distributed/` — Sharding, routing, membership
- Integration tests (multi-process, simulated failures)

### P3.3: Data Plane (6-8 weeks)

**Distributed data operations:**

#### P3.3.1: Shard Server
- Extend `hnsqr_daemon` to be shard-aware
- Replication protocol (Raft or similar)
- Leader election per shard
- Follower read support (optional)

#### P3.3.2: Distributed Insert
- Route to correct shard by hash/range
- Replicate to R nodes (quorum write)
- Acknowledge after W confirmations
- Handle partial failures (retry, compensate)

#### P3.3.3: Distributed Search
- Query routing (scatter-gather or selective)
- Parallel search across shards
- Global top-k merge (heap-based)
- Timeout and partial result handling

#### P3.3.4: Rebalancing
- Detect imbalanced shards (size, load)
- Trigger shard split or migration
- Stream data to new shard
- Atomic cutover (minimize downtime)

**Deliverable:**
- `hnsqr-shard-server` binary — Distributed shard node
- End-to-end distributed tests (Docker Compose cluster)
- Chaos testing (kill nodes, network partitions)

### P3.4: Distributed Benchmarks (2-3 weeks)

**Prove horizontal scalability:**

#### P3.4.1: Scale-Out Performance
- Deploy 1, 3, 5, 10 shard clusters
- Load 100M vectors (10M per shard for 10-shard)
- Measure:
  - QPS as cluster size increases
  - Latency impact (network overhead)
  - Rebalancing time
  - Failure recovery time

#### P3.4.2: Consistency Validation
- Concurrent insert + search stress test
- Verify all replicas converge
- Test shard failover (kill primary, promote replica)
- Validate no data loss

**Target:**
- **Linear scale-out** for QPS (5× shards = 5× QPS)
- **Sub-linear latency growth** (network adds <20% overhead)
- **Sub-second failover** (RPO < 1s, RTO < 5s)

**Deliverable:**
- `benchmarks/distributed_scaling.rs` — Multi-node benchmark suite
- `DISTRIBUTED_RESULTS.md` — Scale-out performance data

---

## Phase 4: The 340× Pinecone Claim Reconciliation

**Current standing claim:** "340× Pinecone at 100K vectors with exact results"

**Audit findings:**
- Current hardened audit: 65,536 vectors (not 100K)
- Exact-scored-fraction: 3.13% at 65K (approximate by construction, not exact)
- These describe different measurement regimes

**Required actions:**

### P4.1: Extend to 100K+ (1 week)
- Run `rivero_scaling.rs` at:
  - 100K vectors (claimed threshold)
  - 256K vectors (stress test)
  - 1M vectors (enterprise threshold)
- Document:
  - Actual recall at these scales
  - Exact-scored fraction (will shrink further)
  - Latency P50/P95/P99

### P4.2: Pinecone Baseline (1 week)
- Deploy Pinecone via API (or local if available)
- Same corpus, same queries, same k
- Measure:
  - Latency (include network for API)
  - QPS
  - Cost (for API usage)
- **Document methodology transparently**

### P4.3: Apples-to-Apples Comparison
- Compare like-to-like:
  - If HNSQR is approximate (3.13% exact), Pinecone comparison must use similar recall@k
  - If claiming "exact results," must achieve Recall@10 = 100%
- State clearly:
  - What "340×" refers to (latency? QPS? cost?)
  - At what scale (100K confirmed)
  - At what recall level (90%? 95%? 100%?)

### P4.4: Update Marketing Claims
- Replace "340× Pinecone" with measured results:
  - "X× faster P50 latency at 95% Recall@10, 100K vectors"
  - "Y QPS vs Z QPS on [specific hardware]"
- Add caveats:
  - Single-node vs distributed comparison
  - Local vs API network overhead
  - Cost comparison if relevant

**Deliverable:**
- `PINECONE_COMPARISON.md` — Transparent methodology and results
- Updated README with accurate performance claims

---

## Implementation Timeline

### Sprint 0: Validation (2-3 weeks) — **DO THIS FIRST**
- [ ] P0.1: Cross-term correlation diagnostic (2 days)
- [ ] P0.2: Real-embeddings IR benchmark (3-5 days)
- [ ] P0.3: Fix synthetic benchmark lie (1 day)
- **GATE:** Does fidelity beat cosine? If no, pivot. If yes, continue.

### Sprint 1-4: Enterprise Features (8-12 weeks)
- [ ] P1.1: Filtered search at scale (2-3 weeks)
- [ ] P1.2: Production-grade persistence (3-4 weeks)
- [ ] P1.3: Verified concurrency (2-3 weeks)
- [ ] P1.4: Joint N×D scaling validation (1-2 weeks)

### Sprint 5-8: Single-Node Domination (6-8 weeks)
- [ ] P2.1: Quantitative competitive benchmark (2 weeks)
- [ ] P2.2: Advanced query features (2-3 weeks)
- [ ] P2.3: Production observability (1-2 weeks)
- [ ] P4.1-P4.4: 340× Pinecone claim reconciliation (2-3 weeks)

### Sprint 9-20: Distributed Scale (12-18 weeks)
- [ ] P3.1: Architecture design (3-4 weeks)
- [ ] P3.2: Coordination layer (4-6 weeks)
- [ ] P3.3: Data plane (6-8 weeks)
- [ ] P3.4: Distributed benchmarks (2-3 weeks)

**Total estimated timeline: 6-9 months to enterprise-ready distributed system**

---

## Success Criteria

### Minimum Viable Enterprise Product (after Phase 1+2):
- ✅ Metric validation proves fidelity advantage (P0)
- ✅ Filtered search maintains 95%+ recall at 1% selectivity, 1M vectors (P1.1)
- ✅ Persistence with crash recovery and hot backup (P1.2)
- ✅ Zero race conditions under ThreadSanitizer (P1.3)
- ✅ Validated at D=1536, N=1M (P1.4)
- ✅ 2-5× faster than Milvus/Qdrant at same recall (P2.1)
- ✅ Feature parity: hybrid search, multi-vector, range, batch ops (P2.2)
- ✅ Production observability with Prometheus (P2.3)

### Distributed Enterprise Product (after Phase 3):
- ✅ 10-node cluster handling 100M+ vectors
- ✅ Linear QPS scale-out (10× nodes = 10× QPS)
- ✅ Sub-second failover with zero data loss
- ✅ Automatic rebalancing
- ✅ Chaos testing passes (network partitions, node failures)

### Market Positioning:
- ✅ Transparent, reproducible benchmarks against Pinecone/Milvus
- ✅ Technical reports and blog posts documenting advantages
- ✅ Reference architectures for common use cases
- ✅ Community adoption (GitHub stars, production deployments)

---

## Risk Mitigation

### Risk 1: Fidelity doesn't beat cosine (P0)
**Probability:** Medium (unproven claim)
**Impact:** High (invalidates core differentiation)
**Mitigation:**
- Front-load validation (Sprint 0)
- Prepare pivot options:
  - Learned pairing (train on relevance data)
  - Alternative quantum metrics (principled basis)
  - Pure latency positioning (drop metric claim)

### Risk 2: Filtered search recall collapse (P1.1)
**Probability:** Medium (known issue)
**Impact:** High (enterprise dealbreaker)
**Mitigation:**
- Adaptive expansion strategy
- Selective route compilation
- Comprehensive benchmark matrix
- Document limitations transparently

### Risk 3: Distributed complexity underestimated (P3)
**Probability:** Medium-High (large project)
**Impact:** Medium (delays market entry)
**Mitigation:**
- Sell single-node first (Phase 1+2)
- Partner with cloud providers for managed offering
- Open-source single-node, commercialize distributed tier

### Risk 4: Performance claims don't hold at scale (P4)
**Probability:** Low-Medium (methodology is sound)
**Impact:** Medium (credibility damage)
**Mitigation:**
- Conservative claims (state exact conditions)
- Transparent methodology (reproducible)
- Independent validation (academic benchmarks)

---

## Next Immediate Actions

1. **Start Sprint 0 validation** (this week):
   - Implement cross-term correlation diagnostic
   - Acquire real embedding model and BEIR dataset
   - Run statistical analysis

2. **Set up development infrastructure** (parallel):
   - CI/CD with ThreadSanitizer
   - Benchmark harness for competitor comparison
   - Documentation site structure

3. **Assemble team/resources**:
   - Who owns Phase 0 validation?
   - Who owns distributed architecture design?
   - Budget for cloud infrastructure (benchmark clusters)

4. **Stakeholder alignment**:
   - Present this roadmap
   - Get buy-in on priorities
   - Decide on pivot criteria if P0 fails

---

## Open Questions for Decision

1. **Metric validation:** If fidelity doesn't beat cosine, which pivot option?
2. **Distribution strategy:** Etcd/Consul vs custom consensus? Raft vs Paxos?
3. **Go-to-market:** Open-source everything, or dual-license (OSS single-node, commercial distributed)?
4. **Target customers:** RAG/LLM applications, recommendation systems, or general vector search?
5. **Cloud strategy:** Self-hosted only, or managed service (AWS/GCP/Azure)?

---

**This roadmap is a living document. Update after each sprint based on learnings.**
