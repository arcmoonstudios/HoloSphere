# HNSQR Benchmark Critique & Action Plan

## Executive Summary

The current benchmark suite **oversells some results and undersells the actually interesting parts**. This document addresses each critique and outlines fixes.

---

## Critical Issues Identified

### 1. **HNSQR is 6.79× SLOWER than brute force at N=5K** ⚠️

**The Data:**
- Brute force (200 queries): 84.26ms → **421.3 µs/query**
- HNSQR ef=10: **2859.7 µs/query**
- Ratio: **6.79× slower**

**Reality Check:**
- Rayon parallel batch: 3046 QPS (328.33 µs/query effective)
- Sequential brute force: ~2374 QPS (421.3 µs/query)
- Parallel ANN only beats sequential exact by **1.28×**

**Conclusion:** At N=5K, graph traversal overhead **dominates**. Parallelize brute force with Rayon and it will absolutely destroy the current index.

**Action Required:**
- Find the actual crossover point where ANN becomes worthwhile
- Test at: 5K, 10K, 25K, 50K, 100K, 250K, 500K, 1M
- Same threads, metric, queries, hardware
- Report honestly: "HNSQR becomes competitive at N > X"

---

### 2. **100% Recall is Suspiciously Easy**

**The Data:**
```
ef=10   → Recall@10: 100%, latency: 2.860ms
ef=256  → Recall@10: 100%, latency: 3.386ms
```

25.6× increase in ef_search → only 18% latency increase

**Likely Causes:**
1. 50-cluster synthetic dataset is **ridiculously easy**
2. ef_search doesn't dominate actual work
3. Both

**Action Required:** Instrument:
- `visited_nodes`
- `distance_evaluations`
- `candidate_pushes/pops`
- `graph_edges_traversed`
- `rerank_count`
- `lock_wait_ns`
- `routing_ns`
- `distance_ns`
- `postprocess_ns`

---

### 3. **Rivero is the REAL Result** ✓

**The Actually Interesting Data:**
```
N      p50 Latency    Corpus Growth    Latency Growth
1K     1.143 ms       1×               1×
4K     2.454 ms       4×               2.15×
16K    4.047 ms       16×              3.54×
65K    5.084 ms       64×              4.45×

Log-log slope: 0.3591 (not 0.5566 as initially computed)
```

**This is genuinely sub-linear scaling.**

But there's a critical detail:

**Routing work is NOT constant:**
```
N       scans/query    exact_evals/query
1K      9,848          860
4K      38,621         1,959
16K     88,784         2,048
65K     118,996        ~2,049
```

**Correct Property:** "Corpus-independent bounded exact distance evaluations" (~2K ceiling)

**Incorrect Property:** "Fixed-work search" (total scans grow with N)

---

### 4. **Isotropic Dataset DESTROYS Rivero**

```
Clustered (16K):
  top1: 100%, recall@10: 100%

Independent Isotropic (16K):
  top1: 84.38%, recall@10: 78.28%, contain@10: 4.69%
```

**Conclusion:** Rivero is NOT a universal nearest-neighbor index.

**Rivero is a structure-exploiting semantic routing index.**

**This is MORE interesting than claiming universality.**

**Action Required:**
- Stop pretending it dominates arbitrary distributions
- Position as: "Exploits semantic clustering in real embedding manifolds"
- Prove real embeddings have this structure
- Consider hybrid: Rivero (high confidence) + HNSW fallback (diffuse queries)

---

### 5. **Build Scaling is Terrible**

```
N       build_time    throughput
1K      1.600s        640 vec/sec
4K      10.844s       378 vec/sec
16K     80.299s       204 vec/sec
65K     482.943s      136 vec/sec

Empirical fit: T(N) ∝ N^1.38
```

Per-vector cost increased **4.72× over 64× corpus growth**.

**Action Required:**
- Separate: `online_incremental_insert()` ≠ `offline_bulk_construction()`
- Profile the 65K build (483s is the ugliest number in the report)
- Bulk construction needs a fundamentally different algorithm

---

### 6. **"Quantum Metric Superiority" is False Advertising** 😂

```
COSINE Separation Margin:          0.9785
QUANTUM FIDELITY Separation Margin: 0.9551
Margin Improvement:                -2.39%
```

**COSINE WON BY 2.39%.**

**Action Required:**
- Rename to: "HNSQR Metric Comparative Analysis"
- Respect for printing the negative result instead of hiding it

---

### 7. **ComplexWeaver Memory Claim is Wrong** (Fixed)

~~"50% footprint reduction"~~

**Reality:**
- 1536 × f32 = 6144 bytes
- 768 × Complex32 (2×f32) = 6144 bytes
- **0% memory reduction**

**Correct Description:** "Lossless coordinate transformation"

**Action:** ✓ Fixed in benchmark output

---

### 8. **PQ-C Naming is Imprecise**

You call it: "8-Bit Polar Phase Quantization"

But: 64 complex dims × ? = 128 bytes → **2 bytes per complex dim**

**Reality:** Likely `(u8 amplitude, u8 phase)` per complex coordinate

**Correct Name:** "16-bit polar complex representation (two 8-bit quantizers)"

**Action Required:** Clarify terminology

---

### 9. **Quantization Validation is Incomplete**

Current metrics:
- MAE: 0.0001
- MAX: 0.0011
- "99.99% fidelity"

**Missing Metrics:**
- Recall@1 full vs quantized
- Recall@10 full vs quantized  
- NDCG@10
- Top-k rank inversions
- p95/p99 fidelity error
- Worst-query recall

**Action Required:** Test quantization on **cluster-boundary** workload where candidates are close together

---

### 10. **Mmap Benchmark is Misleading**

```
Quantized Mmap File Attach: 929.79 µs
(routing state not restored)
```

**Reality:** You mapped vector storage, not an operational index.

**Action Required:** Benchmark full cold-start:
1. File open
2. Mmap attach
3. Routing metadata restore
4. Rivero state restore
5. Page-fault warmup
6. First query
7. 10th query
8. Steady-state query

**Target metric:** "Cold process → first successful search"

---

### 11. **TCP RTT is Mislabeled**

```
Async TCP Network Searches: 100 Queries in 139.44ms (717 QPS, avg 1394.4 µs RTT)
```

If requests are concurrent/pipelined:
- 139.44ms / 100 = 1.394ms is **throughput-equivalent time**, not RTT latency

**Action Required:**
- Change to: "717 aggregate QPS"
- OR measure actual p50/p90/p95/p99/max RTT under defined concurrency

---

### 12. **WSL /mnt/x/ Performance Warning**

Current path: `/mnt/x/_Repos/hnsqr` (Windows mounted drive)

**Issue:** Windows↔WSL filesystem can pollute:
- Disk/mmap/build benchmarks
- Cargo compilation
- Cold-start numbers

**Action Required:** Rerun serious benchmarks from:
```bash
~/bench/hnsqr  # Native Linux filesystem
```

---

## What HNSQR Actually Is

Stripping away marketing:

> **Complex-valued semantic retrieval with global-phase-invariant fidelity, aggressive polar quantization, bounded expensive scoring, deterministic structure-aware routing, witness-based recovery, excellent behavior on clustered manifolds, and noticeably degraded behavior on isotropic manifolds.**

This is **more interesting** than "quantum HNSW but faster."

---

## Priority Action Items

### Immediate (Fix Dishonest Labels)
1. ✅ Fix ComplexWeaver "compression" claim
2. ☐ Rename "Metric Superiority" benchmark
3. ☐ Fix mmap "attach" vs "operational load"
4. ☐ Fix TCP "RTT" vs "aggregate throughput"
5. ☐ Clarify PQ-C bit depth (16-bit polar, two 8-bit components)

### Critical (Find Real Performance Characteristics)
6. ☐ Parallelize brute-force ground truth baseline
7. ☐ Find actual ANN crossover N (test: 5K, 10K, 25K, 50K, 100K, 250K, 500K, 1M)
8. ☐ Run real embeddings (OpenAI ada-002, Cohere, BERT), not synthetic
9. ☐ Compare Rivero vs HNSW head-to-head (same metric, dataset, hardware)
10. ☐ Profile 65K Rivero build (483s is unacceptable)

### Research (Understand What Rivero Is)
11. ☐ Investigate global-phase invariance semantically
    - Does `x` vs `-x` being maximally similar make sense for embeddings?
    - Is this brilliant or catastrophic for semantic retrieval?
12. ☐ Characterize real embedding manifold structure
    - Do production embeddings have exploitable clustering?
    - How often do queries fall into "diffuse/isotropic" regime?
13. ☐ Build hybrid system:
    ```
    Query → confidence_classifier → { Rivero (high conf)
                                     { HNSW (low conf / diffuse)
    ```

### Engineering (Make It Production-Ready)
14. ☐ Implement bulk construction algorithm (separate from incremental insert)
15. ☐ Full quantization validation (recall degradation, rank inversions, boundary cases)
16. ☐ Complete cold-start benchmark (process spawn → first successful query)
17. ☐ Proper concurrency/latency measurement for TCP server

---

## The Two Most Interesting Findings

1. **Rivero's 64× corpus → 4.45× latency growth** with log-log slope ~0.36
2. **Dramatic clustered vs isotropic split** (100% recall → 78% recall)

These look like **the beginning of an actual research result**, not just another HNSW implementation.

---

## Testing Note

The 33 ignored unit tests are fine. `cargo bench` runs benchmark-mode targets only.

Run separately:
```bash
cargo test --release
```

---

## Next Benchmark Document

Create: `HONEST_BENCHMARK_SUITE.md`

Structure:
1. **Where HNSQR Loses** (N < crossover, isotropic data)
2. **Where HNSQR Wins** (N > crossover, clustered manifolds)
3. **Crossover Analysis** (find the exact N)
4. **Build Cost** (current problem + roadmap)
5. **Memory vs Accuracy Tradeoffs** (quantization analysis)
6. **Real-World Performance** (actual embeddings, not synthetic)

---

## Positioning Statement (Draft)

**HNSQR/Rivero is not a universal ANN index.**

It's a **structure-exploiting semantic routing system** that achieves bounded exact scoring and sub-linear search scaling when embedding manifolds have exploitable cluster structure.

For diffuse/isotropic queries or small corpora (N < ~25K), conventional methods may outperform it.

The research question is: **Do real production embedding workloads have the structural characteristics Rivero exploits?**

If yes, this is transformative.
If no, it's an interesting corner case.

Let's find out.
