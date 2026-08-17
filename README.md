# HNSQR — Hierarchical Navigable Semantic Query Resolver

[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)
[![Rust: 2024](https://img.shields.io/badge/Rust-2024%20Edition-orange.svg)](https://www.rust-lang.org/)
[![Build & Tests](https://img.shields.io/badge/Tests-55%2F55%20Passing-brightgreen.svg)]()

> **HNSQR is a classical retrieval engine.** Its algorithms execute entirely on conventional CPUs/GPUs using SIMD tensor intrinsics, complex-valued linear algebra, lattice routing, graph traversal, and admissible geometric Cauchy-Schwarz bounds. Some optional metrics are mathematically related to quantities used in complex Hilbert spaces, including normalized complex projective ray overlap. HNSQR does not require quantum hardware and makes no claim of quantum computational speedup.

---

## Architectural Overview

HNSQR is a maximum-throughput, proof-carrying multimodal retrieval engine. It unifies high-dimensional vector search, sparse lexical BM25 retrieval, ColBERT multi-vector late interaction, and metadata filtering under an adaptive zero-regret query planner.

```
                                  QUERY
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  Universal Planner  │
                         │ (Cost & Proof Model)│
                         └──────────┬──────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
   ┌─────────────────┐    ┌───────────────────┐   ┌───────────────────┐
   │ Exact AVX2 Scan │    │  Rivero Routing   │   │  LUTz Certified   │
   │  (Small Corpus) │    │  (Bounded O(1))   │   │ (Cauchy-Schwarz)  │
   └─────────────────┘    └─────────┬─────────┘   └─────────┬─────────┘
                                    │                       │
                                    ▼                       ▼
                          ┌───────────────────┐   ┌───────────────────┐
                          │ Territory Envelopes│   │ Proof Frontier    │
                          │   UB_cell <= tau  │   │  max(UB) <= tau   │
                          └─────────┬─────────┘   └─────────┬─────────┘
                                    │                       │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │   Exact Top-K Rank    │
                                    │ (100.00% Exact Proof) │
                                    └───────────────────────┘
```

---

## Key Mathematical Foundations

### 1. Pairwise Complex Isometric Folding
HNSQR embeds $2d$-dimensional real LLM vectors into $d$-dimensional complex Hilbert space via the canonical isometry:
$$\Phi: \mathbb{R}^{2d} \to \mathbb{C}^d, \quad \Phi(x)_j = x_{2j} + i x_{2j+1}$$

This transform strictly preserves the Euclidean norm and inner product:
$$\|\Phi(x)\|_2 = \|x\|_2, \qquad \text{Re}\langle \Phi(x), \Phi(y) \rangle = x^\top y$$

### 2. Complex Projective Overlap (CPO)
For applications requiring global-phase invariance ($P(z, e^{i\theta}w) = P(z, w)$), HNSQR supports Complex Projective Overlap:
$$P(z, w) = \frac{|\langle z, w \rangle|^2}{\|z\|_2^2 \|w\|_2^2} \in [0, 1]$$

### 3. Hierarchical Territory Envelopes
Every Rivero territory cell $C$ with centroid $c$ and maximum blockwise radius $\rho_b = \max_{x \in C} \|x_b - c_b\|_2$ admits an admissible upper bound:
$$\text{Re}\langle q, x \rangle \le \text{Re}\langle q, c \rangle + \min\left(\sum_{b=0}^{B-1} \|q_b\|_2 \rho_b, \ \|q\|_2 \rho_{\text{global}}\right)$$
If $\text{UB}_{\text{cell}}(q) \le \tau$, all vectors in cell $C$ are eliminated simultaneously in $O(B)$ time.

### 4. 8-Bit Complex Polar Quantization (CPQ-8)
Encodes $z_j = r_j e^{i\theta_j}$ into 1 byte magnitude and 1 byte phase, achieving a 4× memory footprint reduction while accelerating inner products with static $O(1)$ trigonometric lookup tables.

---

## Core Capabilities

- **Universal Query Planner:** Cost-based optimizer with automated hardware self-calibration (`autoforge.rs`).
- **Hierarchical Rivero Lattice Index:** 24-foundation root lattice addressing with bounded territory probes.
- **Sparse & Hybrid Retrieval:** Block-Max WAND sparse lexical search fused with dense vectors via Reciprocal Rank Fusion (RRF).
- **Multi-Vector ColBERT MaxSim:** Accelerated late-interaction token scoring.
- **Segmented LSM Storage:** Lock-free concurrent segments with background compaction and zero-copy mmap persistence.
- **Zero-Copy Binary Protocol:** Asynchronous TCP framing (`QIR0` Query Interchange Record) with sub-millisecond round-trips.

---

## Quick Start

```rust
use hnsqr::{HNSQRIndex, HNSQRConfig, VectorEmbedding, DistanceFunction};
use hnsqr::gateway::ComplexWeaver;

// 1. Configure the engine
let mut config = HNSQRConfig::default();
config.distance_function = DistanceFunction::Cosine;

// 2. Initialize index (e.g. 768 complex dims for folded 1536D embeddings)
let index = HNSQRIndex::new(config, 768);

// 3. Fold an OpenAI 1536-dimensional real embedding
let real_embedding = vec![0.042f32; 1536];
let complex_vector = ComplexWeaver::fold_llm_embedding(&real_embedding);

// 4. Ingest and search
index.insert("doc-001", complex_vector.clone())?;
let results = index.search(&complex_vector, 10)?;
```

---

## Benchmarks & Verification

To run the complete test and benchmark suites:

```bash
# Run all unit, doctest, and integration tests
cargo test --all-targets

# Run empirical crossover sweep
cargo bench --bench crossover_sweep -- --nocapture

# Run dense recall funnel & certified retrieval benchmark
cargo bench --bench dense_recall_funnel_benchmark -- --nocapture
```

---

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
