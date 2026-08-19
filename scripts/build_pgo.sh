#!/usr/bin/env bash
set -euo pipefail

# 🌙 HoloSphere Profile-Guided Optimization (PGO) Builder
# Automates instrumented build, workload execution, profile merging, and optimized binary creation.

PGO_DATA_DIR="${PGO_DATA_DIR:-/tmp/hnsqr-pgo-data}"
TARGET_CPU="${TARGET_CPU:-native}"

echo "🌙 HoloSphere Profile-Guided Optimization (PGO) Builder"
echo "════════════════════════════════════════════════════════════"

rm -rf "$PGO_DATA_DIR"
mkdir -p "$PGO_DATA_DIR"

echo "==> Step 1: Building instrumented binary with profile-generate..."
RUSTFLAGS="-Cprofile-generate=$PGO_DATA_DIR -Ctarget-cpu=$TARGET_CPU" cargo build --release --all-targets

echo "==> Step 2: Running representative benchmarks to generate profiles..."
./target/release/benches/universal_scorecard_benchmark || true
./target/release/benches/gate_b_hierarchical_proof || true
./target/release/benches/rivero_search_scaling || true

echo "==> Step 3: Merging profile data..."
LLVM_PROFDATA=$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/host: //p')/bin/llvm-profdata
if [ ! -f "$LLVM_PROFDATA" ]; then
    echo "llvm-profdata not found, adding component..."
    rustup component add llvm-tools-preview
    LLVM_PROFDATA=$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/host: //p')/bin/llvm-profdata
fi

"$LLVM_PROFDATA" merge -o "$PGO_DATA_DIR/merged.profdata" "$PGO_DATA_DIR"/*.profraw

echo "==> Step 4: Building production artifact with profile-use..."
RUSTFLAGS="-Cprofile-use=$PGO_DATA_DIR/merged.profdata -Ctarget-cpu=$TARGET_CPU" cargo build --release

echo "✅ PGO compilation complete: target/release/hnsqr_daemon"
