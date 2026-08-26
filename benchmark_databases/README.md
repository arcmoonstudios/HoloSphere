# Prebuilt benchmark databases

This directory holds immutable HNSQR vector/graph snapshot databases built from
the local `datasets/` corpus. It is intentionally outside `target/`, so clean
builds never discard the database or cause benchmark processes to rebuild one.

Build an artifact explicitly before a benchmark run:

```powershell
cargo run --release --bin hnsqr_build_bench_db -- --vectors 5000 --queries 64 --source-dim 128 --profile balanced
```

For benchmarks using a named prebuilt index, specify its tag and the index's
complex dimension:

```powershell
cargo run --release --bin hnsqr_build_bench_db -- --kind index --tag crossover_sweep_n5000 --vectors 5000 --source-dim 128 --index-dim 64 --profile balanced
```

Set `HNSQR_BENCH_DATABASE_DIR` to place the immutable artifacts on a shared or
high-capacity volume. The builder refuses overwrites; delete or select a new
directory only when intentionally regenerating an artifact.
