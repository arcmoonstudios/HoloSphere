# Contributing to HoloSphere

Thanks for contributing. Please start with an issue or discussion for changes that affect public APIs, storage formats, query semantics, or benchmark methodology.

## Development setup

Use the stable Rust toolchain and run the repository gates before opening a pull request:

```console
cargo holo-fmt-check
cargo holo-check
cargo holo-clippy
cargo holo-test
```

`cargo holo-bench` compiles benchmark targets without executing them. Benchmark binaries must use checked-in datasets and prebuilt snapshot artifacts; see `benches/common.rs` and `hnsqr_build_bench_db`.

## Pull requests

- Keep each pull request focused and include tests for behavioral changes.
- Preserve compatibility for snapshots, wire protocols, and persisted state, or document a migration.
- Do not commit secrets, generated datasets, benchmark databases, or `target/` output.
- Update `CHANGELOG.md` for user-visible changes.

By contributing, you agree that your contributions are licensed under the repository's Apache-2.0 OR MIT terms.
