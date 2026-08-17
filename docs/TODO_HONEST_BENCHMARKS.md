# TODO: Honest Benchmark Suite

## Immediate Fixes Applied

- ✅ Fixed ComplexWeaver "50% compression" → "lossless coordinate transformation"
- ✅ Renamed "Metric Superiority" → "Metric Comparative Analysis"
- ✅ Added parallel brute-force baseline with comparison
- ✅ Added analysis showing when HNSQR is slower than brute force

## Critical Tasks Remaining

### 1. Find the ANN Crossover Point

**Problem:** At N=5K, HNSQR is 6.79× slower than brute force.

**Task:** Create `benches/crossover_analysis.rs`:

```rust
// Test sizes: 5K, 10K, 25K, 50K, 100K, 250K, 500K, 1M
// For each size, measure:
//   - Sequential brute force (baseline)
//   - Parallel brute force (Rayon)
//   - HNSQR ef=10, 32, 64
//   - Record crossover N where HNSQR becomes competitive
```

Expected result: **"HNSQR becomes competitive at N > ~X"**

### 2. Fix Build Scaling

**Problem:** 65K build takes 483 seconds (136 vec/sec). Empirically T(N) ∝ N^1.38

**Tasks:**
- Profile the 65K build to find hotspots
- Implement separate bulk construction algorithm
- Target: Bring closer to O(N log N)

### 3. Test Real Embeddings

**Problem:** Current tests use synthetic "OpenAI-shaped" vectors

**Task:** Benchmark with actual production embeddings:
- OpenAI text-embedding-ada-002 (1536-dim)
- Cohere embed-english-v3.0 (1024-dim)  
- BERT base (768-dim)
- Sentence-Transformers

**Test:** Does real data have the cluster structure Rivero exploits?

### 4. Compare vs HNSW Head-to-Head

**Problem:** No direct comparison with conventional HNSW

**Task:** Add hnswlib or faiss as comparison:

```rust
// Same dataset, queries, k
// Measure:
//   - Build time
//   - Query latency
//   - Recall@k
//   - Memory usage
```

### 5. Instrument ef_search Behavior

**Problem:** ef=10 and ef=256 behave suspiciously similarly

**Task:** Add diagnostics to each query:
- `visited_nodes`
- `distance_evaluations`
- `candidate_pushes / pops`
- `graph_edges_traversed`
- `rerank_count`
- Timing breakdown: `routing_ns`, `distance_ns`, `postprocess_ns`

### 6. Characterize Isotropic vs Clustered Performance

**Problem:** Rivero drops from 100% → 78% recall on isotropic data

**Task:** Create structured test suite:

```rust
// Dataset variants:
1. High clustering (current 50-cluster synthetic)
2. Medium clustering (5-cluster synthetic)
3. Low clustering (isotropic with slight structure)
4. Pure isotropic (completely independent)

// For each:
- Build time
- Query latency
- Recall@k
- Plot: cluster_score vs recall
```

**Goal:** Define Rivero's operating regime precisely

### 7. Investigate Global-Phase Invariance

**Critical Research Question:**

In HNSQR's quantum fidelity metric:
```
F(x, -x) = 1.0  (maximally similar)
```

Is this correct for semantic embeddings?

**Test:**
1. Take real embeddings for: "happy", "sad"
2. Negate "sad" embedding: -sad
3. Check if Rivero considers "happy" and "-sad" equivalent
4. Determine if this is semantic collapse or brilliant invariance

### 8. Full Quantization Validation

**Current metrics:** MAE, MAX fidelity error only

**Add:**
- Recall@1 degradation (full vs quantized)
- Recall@10 degradation
- NDCG@10
- Rank inversions (how often do neighbors swap order?)
- p95/p99 errors
- Worst-query recall

**Critical test:** Cluster boundary queries (where candidates are close)

### 9. Complete Mmap Cold-Start Benchmark

**Current:** "929 µs attach (routing state not restored)"

**Need:** Full cold-start pipeline:

```rust
// Measure each step:
1. Process spawn
2. File open
3. Mmap attach (what's currently measured)
4. Deserialize routing metadata
5. Restore Rivero graph state
6. Page fault warmup (touch all pages)
7. First query
8. 10th query
9. Steady-state (100th+ query)
```

**Target metric:** "Cold process → first successful query"

### 10. Fix TCP Latency Measurement

**Current:** "717 QPS, avg 1394.4 µs RTT"

**Problem:** If pipelined/concurrent, this is throughput not RTT

**Fix:** Measure individual request latencies:
- p50, p90, p95, p99, max RTT
- At defined concurrency (1, 10, 100 concurrent)
- Report as: "717 aggregate QPS at concurrency=N"

### 11. Native Linux Filesystem Benchmark

**Problem:** Currently running from `/mnt/x/` (Windows drive mounted in WSL)

**Task:**
```bash
# Move to native Linux filesystem
mkdir -p ~/bench
cp -r /mnt/x/_Repos/hnsqr ~/bench/
cd ~/bench/hnsqr

# Ensure target/ and temp files are on Linux filesystem
cargo clean
cargo bench --bench benchmark_suite
cargo bench --bench rivero_scaling
```

Compare results to detect Windows↔WSL overhead

### 12. Clarify PQ-C Bit Depth

**Current:** "8-Bit Polar Phase Quantization"

**Reality:** 64 complex dims × ? = 128 bytes = 2 bytes/dim

**Likely:** `(u8 amplitude, u8 phase)` per complex coordinate

**Fix:** Update all docs and benchmarks to:
- "16-bit polar complex representation"
- "Composed of two 8-bit quantizers (amplitude, phase)"

---

## New Benchmark Structure

Create: `benches/honest_comparison.rs`

```rust
fn main() {
    println!("HNSQR Honest Performance Analysis\n");
    
    // 1. WHERE HNSQR LOSES
    println!("═══ Small Corpus (N < crossover) ═══");
    // Show brute force winning
    
    // 2. WHERE HNSQR WINS  
    println!("═══ Large Clustered Corpus ═══");
    // Show sub-linear scaling
    
    // 3. CROSSOVER ANALYSIS
    println!("═══ Finding the Crossover Point ═══");
    // Plot N vs relative performance
    
    // 4. STRUCTURE SENSITIVITY
    println!("═══ Clustered vs Isotropic ═══");
    // Show recall degradation
    
    // 5. BUILD COST
    println!("═══ Construction Overhead ═══");
    // Show current O(N^1.38) problem
    
    // 6. QUANTIZATION TRADEOFFS
    println!("═══ Memory vs Accuracy ═══");
    // Show recall degradation from PQ-C
}
```

---

## Research Questions to Answer

1. **What is Rivero's minimum viable corpus size?**
   - Below this, brute force wins

2. **What cluster structure does Rivero need?**
   - Quantify: intra-cluster distance, inter-cluster separation

3. **Is global-phase invariance helping or hurting?**
   - Test on real embeddings with known semantic relationships

4. **Can we predict when to fall back to HNSW?**
   - Build confidence score based on query diffuseness

5. **What's the maximum practical N given current build cost?**
   - At N=1M, build would take ~140 hours at current rate

---

## Target Positioning Statement

After completing these benchmarks:

> **HNSQR/Rivero is a structure-exploiting semantic routing system optimized for large-scale (N > 25K), clustered embedding manifolds.**
>
> **Key characteristics:**
> - Sub-linear search scaling: O(log^0.36 N) observed
> - Bounded exact evaluations: ~2K regardless of N
> - Perfect recall on clustered data up to N=65K
> - Degrades to ~78% recall on isotropic/diffuse queries
> - Global-phase invariant fidelity metric
> - Competitive above N ≈ 25K (vs brute force)
>
> **Not recommended for:**
> - Small corpora (N < 25K)
> - Uniformly distributed / isotropic data
> - Applications requiring guarantees on diffuse queries
>
> **Best suited for:**
> - Large-scale semantic search (N > 100K)
> - Domain-specific embeddings with natural clustering
> - Workloads where 2K exact evals << N
> - Phase-invariant similarity semantics

---

## Files to Create

1. `benches/crossover_analysis.rs` - Find minimum viable N
2. `benches/honest_comparison.rs` - Comprehensive realistic benchmark
3. `benches/real_embeddings.rs` - Test with actual OpenAI/Cohere data
4. `benches/structure_sensitivity.rs` - Quantify clustering requirements
5. `docs/FAILURE_REGIMES.md` - Document where Rivero fails
6. `docs/OPERATING_ENVELOPE.md` - Define recommended usage
7. `docs/GLOBAL_PHASE_ANALYSIS.md` - Investigate semantic implications

---

## Success Metrics

After completing this work, we should be able to honestly say:

- ✓ We know exactly where HNSQR wins and loses
- ✓ We can predict performance on new datasets
- ✓ We understand the theoretical tradeoffs
- ✓ We have evidence on real production embeddings
- ✓ We can recommend when to use Rivero vs alternatives
- ✓ The benchmark doesn't oversell or hide limitations

That's far more valuable than claiming universal superiority.
