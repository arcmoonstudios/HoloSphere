# Bolt's Performance Journal

This journal records critical, architecture-specific performance insights, non-obvious bottlenecks, and lessons learned during optimization runs across HoloSphere.

2026-08-27 - [Batched dependency invalidation]
Learning: Removing several entities from a locator repeatedly scanned every reverse-dependency set, making invalidation scale with removed entities times graph edges.
Action: Collect invalidated IDs first, then prune reverse dependencies in one pass whenever a bulk operation updates the ContextGraph.
