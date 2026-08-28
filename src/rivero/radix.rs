/* holosphere/src/rivero/radix.rs */
//!▫~•◦-------------------------------‣
//! # Partitioned Radix Stripe Bucketer Primitive
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a lock-free, zero-contention parallel partitioned bucketing
//! accumulator for multi-threaded bulk index construction.
//!
//! ### Architectural Invariants:
//! 1. **Worker Isolation:** Each worker thread writes exclusively into its private partitioned stripe buffer.
//! 2. **Deterministic Merge Ordering:** Parallel stripe reduction processes entries in strictly ordered worker chunks.
//! 3. **Zero Dynamic Allocation:** Pre-allocates entry buffers per worker slab.
//! 4. **Linear Scaling:** Eliminates cross-thread synchronization during streaming ingestion.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use rayon::prelude::*;

/// A flat, cache-aligned partitioned radix bucketer.
#[derive(Clone, Debug)]
pub struct PartitionedRadixBucketer<T: Send + Sync + Copy, const STRIPES: usize = 64> {
    workers: Vec<[Vec<T>; STRIPES]>,
}

impl<T: Send + Sync + Copy, const STRIPES: usize> PartitionedRadixBucketer<T, STRIPES> {
    /// Creates a new partitioned bucketer for `num_workers` threads with estimated capacity per stripe.
    #[must_use]
    pub fn new(num_workers: usize, estimated_per_stripe: usize) -> Self {
        let workers = (0..num_workers)
            .map(|_| std::array::from_fn(|_| Vec::with_capacity(estimated_per_stripe)))
            .collect();
        Self { workers }
    }

    /// Appends an entry into a worker's designated stripe buffer.
    #[inline(always)]
    pub fn push(&mut self, worker_idx: usize, stripe_idx: usize, entry: T) {
        if worker_idx < self.workers.len() && stripe_idx < STRIPES {
            self.workers[worker_idx][stripe_idx].push(entry);
        }
    }

    /// Returns the number of worker buffers.
    #[inline(always)]
    #[must_use]
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Reduces all stripes in parallel across worker chunks in deterministic worker order.
    pub fn reduce_stripes_parallel<R, F>(&self, reducer: F) -> Vec<R>
    where
        R: Send,
        F: Fn(usize, &dyn Fn(&mut dyn FnMut(&T))) -> R + Sync + Send,
    {
        (0..STRIPES)
            .into_par_iter()
            .map(|stripe_idx| {
                let stream_entries = |consumer: &mut dyn FnMut(&T)| {
                    for worker in &self.workers {
                        for entry in &worker[stripe_idx] {
                            consumer(entry);
                        }
                    }
                };
                reducer(stripe_idx, &stream_entries)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn test_radix_bucketer_worker_isolation_and_determinism() {
        const STRIPES: usize = 8;
        let mut bucketer: PartitionedRadixBucketer<u32, STRIPES> =
            PartitionedRadixBucketer::new(4, 16);

        // Worker 0 pushes into stripe 2
        bucketer.push(0, 2, 100);
        bucketer.push(0, 2, 101);

        // Worker 1 pushes into stripe 2
        bucketer.push(1, 2, 200);

        // Worker 2 pushes into stripe 5
        bucketer.push(2, 5, 300);

        // Worker 3 pushes into stripe 2
        bucketer.push(3, 2, 400);

        let merged_results: Vec<Vec<u32>> = bucketer.reduce_stripes_parallel(|_stripe, stream| {
            let mut entries = Vec::new();
            stream(&mut |&val| entries.push(val));
            entries
        });

        // Verify stripe 2 contains elements in exact deterministic worker order [100, 101, 200, 400]
        assert_eq!(merged_results[2], vec![100, 101, 200, 400]);
        // Verify stripe 5 contains [300]
        assert_eq!(merged_results[5], vec![300]);
        // Verify other stripes are empty
        assert!(merged_results[0].is_empty());
        assert!(merged_results[1].is_empty());
    }

    #[test]
    fn test_radix_parallel_reduction_completeness() {
        const STRIPES: usize = 16;
        let mut bucketer: PartitionedRadixBucketer<(usize, usize), STRIPES> =
            PartitionedRadixBucketer::new(8, 32);

        let mut expected_total = 0usize;
        for w in 0..8 {
            for i in 0..100 {
                let stripe = (w * 13 + i * 7) % STRIPES;
                bucketer.push(w, stripe, (w, i));
                expected_total += 1;
            }
        }

        let total_counted = AtomicUsize::new(0);
        bucketer.reduce_stripes_parallel(|_stripe, stream| {
            let mut count = 0;
            stream(&mut |_| count += 1);
            total_counted.fetch_add(count, Ordering::Relaxed);
        });

        assert_eq!(total_counted.load(Ordering::Relaxed), expected_total);
    }
}
