/* holosphere/src/retrieval/top_k.rs */
//!▫~•◦-------------------------------‣
//! # Bounded Top-K Stream Collector Primitive
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a zero-allocation, bounded min-heap candidate collector for exact
//! and filtered vector retrieval. Eliminates $O(N)$ vector allocation overhead
//! and replaces full corpus sorting with an $O(N \log K)$ streaming bound.
//!
//! ### Mathematical Invariants:
//! 1. **Capacity Ceiling:** `self.len() <= k` at all points in time.
//! 2. **Monotonic Threshold:** The threshold returned by `min_score()` is monotonically
//!    non-decreasing over any stream of pushes.
//! 3. **Exact Equivalence:** `into_sorted_vec()` produces an identical top-$k$ set and
//!    ordering (descending by score, ascending by index tie-break) as exhaustive $O(N \log N)$ sort.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::{NodeIndex, SimilarityScore};

/// An item paired with its score, ordered as a min-heap element.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct MinHeapItem<T, S> {
    pub item: T,
    pub score: S,
}

impl<T: Ord + Copy, S: PartialOrd + Copy> Eq for MinHeapItem<T, S> {}

impl<T: Ord + Copy, S: PartialOrd + Copy> Ord for MinHeapItem<T, S> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse score ordering so BinaryHeap acts as a min-heap (lowest score at top)
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            // On score tie, larger item ID is considered "worse" and placed at top for eviction
            .then_with(|| self.item.cmp(&other.item))
    }
}

impl<T: Ord + Copy, S: PartialOrd + Copy> PartialOrd for MinHeapItem<T, S> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Zero-allocation bounded top-k collector.
#[derive(Clone, Debug)]
pub struct BoundedTopKCollector<T = NodeIndex, S = SimilarityScore> {
    k: usize,
    heap: BinaryHeap<MinHeapItem<T, S>>,
}

impl<T: Ord + Copy, S: PartialOrd + Copy> BoundedTopKCollector<T, S> {
    /// Creates a new collector bounded to at most `k` items.
    #[inline]
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
        }
    }

    /// Returns the capacity limit `k`.
    #[inline(always)]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.k
    }

    /// Returns the current number of collected items.
    #[inline(always)]
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns whether the collector is empty.
    #[inline(always)]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns the current minimum score in the top-k set, if capacity is reached.
    /// Useful for early-exit branch-and-bound pruning in traversal and scan loops.
    #[inline(always)]
    #[must_use]
    pub fn threshold(&self) -> Option<S> {
        if self.heap.len() >= self.k {
            self.heap.peek().map(|entry| entry.score)
        } else {
            None
        }
    }

    /// Inserts an item with its score into the collector.
    ///
    /// If the collector has not yet reached capacity `k`, the item is accepted unconditionally.
    /// Once at capacity, the item is accepted only if its score exceeds the current minimum
    /// (or ties with a better tie-breaker ID).
    #[inline(always)]
    pub fn push(&mut self, item: T, score: S) {
        if self.k == 0 {
            return;
        }

        let entry = MinHeapItem { item, score };
        if self.heap.len() < self.k {
            self.heap.push(entry);
        } else if let Some(min_top) = self.heap.peek() {
            // Check if incoming entry is strictly better than the worst in heap
            let is_better = match score.partial_cmp(&min_top.score) {
                Some(Ordering::Greater) => true,
                Some(Ordering::Equal) => item < min_top.item,
                _ => false,
            };

            if is_better {
                self.heap.pop();
                self.heap.push(entry);
            }
        }
    }

    /// Clears all collected items without deallocating internal buffer.
    #[inline]
    pub fn clear(&mut self) {
        self.heap.clear();
    }

    /// Consumes the collector and returns the final top-k elements sorted in descending
    /// score order (with ascending item ID on tie-break).
    #[must_use]
    pub fn into_sorted_vec(self) -> Vec<(T, S)> {
        let mut items: Vec<(T, S)> = self
            .heap
            .into_iter()
            .map(|entry| (entry.item, entry.score))
            .collect();

        items.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_capacity_invariant() {
        let mut collector = BoundedTopKCollector::new(5);
        for i in 0..100 {
            collector.push(i, (i as f32) * 0.1);
            assert!(collector.len() <= 5, "Capacity invariant violated!");
        }
        assert_eq!(collector.len(), 5);
    }

    #[test]
    fn test_top_k_monotonic_threshold_invariant() {
        let mut collector = BoundedTopKCollector::new(3);
        let mut prev_threshold = f32::MIN;

        let scores = [0.1, 0.5, 0.2, 0.9, 0.8, 0.3, 1.2, 0.4, 1.5, 0.7];
        for (i, &s) in scores.iter().enumerate() {
            collector.push(i as u32, s);
            if let Some(t) = collector.threshold() {
                assert!(
                    t >= prev_threshold,
                    "Monotonic threshold invariant violated: {t} < {prev_threshold}"
                );
                prev_threshold = t;
            }
        }
    }

    #[test]
    fn test_top_k_exact_equivalence_to_full_sort() {
        let mut collector = BoundedTopKCollector::new(4);
        let raw_stream = vec![
            (10u32, 0.45f32),
            (20u32, 0.95f32),
            (30u32, 0.12f32),
            (40u32, 0.88f32),
            (50u32, 0.95f32), // Tied score with 20, 20 should win tie-break
            (60u32, 0.77f32),
            (70u32, 0.99f32),
        ];

        for &(id, score) in &raw_stream {
            collector.push(id, score);
        }

        let collector_results = collector.into_sorted_vec();

        // Exhaustive sorting baseline
        let mut full_sorted = raw_stream;
        full_sorted.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        full_sorted.truncate(4);

        assert_eq!(collector_results, full_sorted);
    }

    #[test]
    fn test_top_k_empty_and_zero_k() {
        let mut collector: BoundedTopKCollector<u32, f32> = BoundedTopKCollector::new(0);
        collector.push(1, 0.9);
        assert_eq!(collector.len(), 0);
        assert!(collector.into_sorted_vec().is_empty());
    }
}
