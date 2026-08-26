# Release process

1. Ensure `CHANGELOG.md` is complete and version fields in `Cargo.toml` are updated.
2. Run `cargo holo-fmt-check`, `cargo holo-check`, `cargo holo-clippy`, and `cargo holo-test` on a clean checkout.
3. Verify snapshot, WAL, and wire-format compatibility or publish migration guidance.
4. Create an annotated `vX.Y.Z` tag and GitHub release with the matching changelog section.
5. Publish crates and binaries only after the tagged source has passed CI.
