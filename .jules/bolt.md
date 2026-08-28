# Bolt's Performance Journal

This journal records critical, architecture-specific performance insights, non-obvious bottlenecks, and lessons learned during optimization runs across HoloSphere.

2026-08-27 - Monotonic Norm-Squared Amplitude Scanning in Vector Quantization
Learning: Polar quantization does not require computing full Euclidean norms (hardware `sqrtss`) on every dimension to determine minimum and maximum bounding envelopes. Because $f(x) = \sqrt{x}$ is strictly monotonic for $x \ge 0$, tracking $\min(r^2)$ and $\max(r^2)$ via `norm_sqr()` and performing `sqrt()` once at the scalar boundary eliminates $N$ hardware square roots per vector without precision loss.
Action: In high-dimensional quantization loops, find scalar bounds in the squared domain before taking a single scalar root.

2026-08-27 - Per-Node Row Sorting Allocations in Graph CSR/CSC Compaction
Learning: During graph adjacency compaction (CSR/CSC), sorting each node's contiguous outgoing or incoming edge slice by destination and relationship type causes $O(V)$ ephemeral heap vector allocations when using `.collect()`.
Action: Allocate a single reusable `scratch` vector outside the node loop and `.clear()` per node to achieve $O(1)$ allocation overhead across large graph builds.

2026-08-27 - Binary Search Insertion for Invariant-Ordered Degree-Bounded Witness Lists
Learning: Rivero witness lists in `SmallVec` are maintained in exact `witness_order`. Re-sorting the entire array with `sort_unstable_by` on every reciprocal edge insertion adds $O(D \log D)$ comparison and swap overhead over millions of index build calls.
Action: Maintain the sorted invariant with `binary_search_by` and in-place insertion, reducing per-edge reciprocal pruning to $O(D)$.

2026-08-27 - [Batched dependency invalidation]
Learning: Removing several entities from a locator repeatedly scanned every reverse-dependency set, making invalidation scale with removed entities times graph edges.
Action: Collect invalidated IDs first, then prune reverse dependencies in one pass whenever a bulk operation updates the ContextGraph.

2026-08-27 - Double-Buffered Row Memory Reduction in Wagner-Fischer Fuzzy Automata
Learning: Computing Levenshtein edit distance via 2D dynamic programming matrices (`Vec<Vec<usize>>`) causes $O(M)$ heap vector allocations per string comparison. Using a double-buffered 2-row vector swap (`prev` and `curr`) reduces dynamic memory overhead from $M + 1$ heap allocations to 2 row buffers while preserving exact Wagner-Fischer edit distances.
Action: In sequence alignment and string edit distance loops, double-buffer row state using `std::mem::swap` rather than allocating full 2D matrix vectors.
