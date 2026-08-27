# Changelog

All notable changes to HoloSphere are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Repository governance, support, security, contribution, and GitHub automation artifacts.

### Changed

- Documented Exact SIMD as the current production retrieval authority and clarified that
  `Certified` is planner-routed to Exact SIMD until a proof path passes its admission gate.
- Replaced stale fixed benchmark scorecards and crossover tables with reproducible benchmark
  commands and explicit hardware/corpus calibration guidance.
- Documented the mandatory separation between the correctness test gate and benchmark runs.
- Renamed internal planner proof-plan variants from LUTz-branded names to proof-tree names and
  removed LUTz encoding/storage from the primary index and segmented write/compaction paths.
- Optimized the Exact cosine/Euclidean hot path by hoisting the metric lookup and introducing a
  real-component-only SIMD inner-product primitive.

## [0.1.0] - 2026-08-26

### Added

- Initial public HoloSphere release.
