# Profile-Guided Optimization (PGO) & LLVM BOLT Guide for HoloSphere (`hnsqr`)

High-performance retrieval and vector database engines like **HoloSphere (`hnsqr`)** are prime candidates for **Profile-Guided Optimization (PGO)**.

Because HoloSphere relies heavily on:
1. **Branch-heavy proof tree traversal** (evaluating admissible bounds $\text{UB}_{\text{cap}} < \tau$ and early pruning),
2. **Progressive LUTz $L_0 / L_1$ filtering** and SIMD dispatch,
3. **Rivero signature hashing and territorial cell routing**,
4. **Async networking and Raft consensus loops**,

PGO gives LLVM real execution data so it can optimize **branch prediction hints, function inlining decisions, register allocation, and code layout (I-cache locality)**. In retrieval workloads, this typically translates to **10% to 25% higher QPS and noticeably tighter p95/p99 tail latencies**.

---

## Why PGO Works Exceptionally Well for HoloSphere

* **Hot/Cold Block Separation:** Code paths for rare corner cases (e.g., proof frontier deadline aborts, OOD graph fallbacks, and slow disk fallbacks) are moved to cold pages, keeping L1 instruction cache packed with hot search loops (`dot_product_complex_simd`, `score_candidate_l0`).
* **Precise Branch Weighting:** LLVM knows which branches of the spherical-cap bounding checks ($s \ge \cos \theta$) and LUTz pruning cascades actually trigger most often.
* **Inline Optimization:** Critical vector folding, address compilation, and bitmask manipulations are aggressively inlined across module boundaries.

---

## Method 1: Using `cargo-pgo` (Fastest / Recommended for Development)

### 1. Install Prerequisites

Ensure you have LLVM tools and `cargo-pgo` installed:

```bash
rustup component add llvm-tools-preview
cargo install cargo-pgo
```

### 2. Step 1: Build the Instrumented Binary

Build an instrumented release build of HoloSphere. This inserts runtime counters that record execution paths:

```bash
cargo pgo build --release
```

*(Optional target CPU tuning: `RUSTFLAGS="-C target-cpu=native" cargo pgo build --release`)*

### 3. Step 2: Run a Representative Workload

To produce an accurate profile, run your typical query and write mix. You can use HoloSphere's built-in benchmarks or run the server with synthetic client queries:

#### Option A: Using HoloSphere's Universal Scorecard & Proof Benchmarks
```bash
# Run representative benchmark suites to generate profile data
cargo pgo test --bench universal_scorecard_benchmark -- --nocapture
cargo pgo test --bench gate_b_hierarchical_proof -- --nocapture
cargo pgo test --bench rivero_search_scaling -- --nocapture
```

#### Option B: Running the TCP Server / REST Gateway under Load
```bash
# Start the instrumented server
cargo pgo run --release --bin hnsqr_daemon -- --config config.toml &
SERVER_PID=$!

# Run your client load generator or SDK benchmark for 2-5 minutes
python benchmarks/load_test.py --qps 5000 --duration 180

# Gracefully stop the server to flush the .profdata profiles to disk
kill -SIGINT $SERVER_PID
```

The profile data will be collected and written to `target/pgo-profiles/`.

### 4. Step 3: Build the Optimized Binary

Compile the final binary using the gathered execution profile:

```bash
cargo pgo optimize --release
```

The final optimized binary will be located in `target/release/`.

---

## Method 2: Native `rustc` & LLVM Workflow (Recommended for Production CI/CD)

If you are building container images or deploying in a CI pipeline without extra Cargo plugins, use the native `rustc` PGO flags directly.

### Step 1: Compile with Profile Generation
```bash
# Set profile output directory and optimization flags
export RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data -Ctarget-cpu=native"

cargo build --release --all-targets
```

### Step 2: Exercise the Workload
```bash
# Run benchmark/service to produce .profraw files
./target/release/benches/universal_scorecard_benchmark
./target/release/benches/gate_b_hierarchical_proof
```

### Step 3: Merge Profile Data
```bash
# Locate llvm-profdata from the Rust toolchain
LLVM_PROFDATA=$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/host: //p')/bin/llvm-profdata

# Merge all generated .profraw files into a single profile
$LLVM_PROFDATA merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/*.profraw
```

### Step 4: Compile with Profile Use & ThinLTO
```bash
# Build the production artifact with profile-guided optimization and ThinLTO
export RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata -Ctarget-cpu=native -Clto=thin"

cargo build --release
```

---

## Best Practices for Profiling HoloSphere

To maximize gains and prevent performance regressions:

1. **Profile Real Dimensions:** Ensure the profile run uses the vector dimensions you actually run in production (e.g., 768 complex / 1536 real for typical LLM embeddings).
2. **Exercise All Active Retrieval Contracts:** Include a mix of:
   * `Certified` exact searches (exercising `SemanticProofTree` and `LutzCertifier`),
   * `HighRecall` / `Bounded` searches (exercising Rivero candidate selection),
   * Filtered searches (exercising Roaring bitmap intersections).
3. **Simulate Concurrent Load:** Run with multiple client threads (`rayon` batch searches / multi-threaded client requests) so LLVM learns lock contention patterns and memory prefetching behaviors.

---

## Advanced Extension: PGO + LLVM BOLT

For latency-critical deployments, you can combine PGO with **LLVM BOLT** (Binary Optimization and Layout Tool) to perform post-link instruction cache reordering:

```bash
# 1. Build with PGO and relocations enabled
RUSTFLAGS="-Cprofile-use=... -Ctarget-cpu=native -Clink-arg=-Wl,-q" cargo build --release

# 2. Instrument binary with BOLT
llvm-bolt ./target/release/hnsqr_daemon -instrument -o ./hnsqr_daemon.bolt.inst

# 3. Run workload on instrumented binary to produce bolt profiles
./hnsqr_daemon.bolt.inst

# 4. Apply BOLT optimizations
llvm-bolt ./target/release/hnsqr_daemon -o ./target/release/hnsqr_daemon.bolt \
    -data=perf.fdata \
    -reorder-blocks=ext-tsp \
    -reorder-functions=cdsort \
    -split-functions \
    -split-all-cold
```
