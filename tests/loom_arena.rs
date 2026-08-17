/* hnsqr/tests/loom_arena.rs */
//!▫~•◦-------------------------------‣
//! # Loom Adversarial Concurrency Validation
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Exposes `ConcurrentArena` to `loom`'s exhaustive thread scheduling permutations
//! to guarantee true race-freedom in the memory-mapping and atomic bump allocation paths.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#[cfg(feature = "loom")]
use loom::sync::Arc;
#[cfg(not(feature = "loom"))]
use std::sync::Arc;

#[cfg(feature = "loom")]
use loom::thread;
#[cfg(not(feature = "loom"))]
use std::thread;

use hnsqr::{HNSQRConfig, HNSQRIndex, VectorEmbedding};

#[test]
fn test_concurrent_arena_race_freedom() {
    #[cfg(feature = "loom")]
    loom::model(|| {
        run_concurrent_insertion_test();
    });

    #[cfg(not(feature = "loom"))]
    run_concurrent_insertion_test();
}

fn run_concurrent_insertion_test() {
    let mut config = HNSQRConfig::strict_rivero_for_dim(2);
    config.max_elements = 10;
    let index = Arc::new(HNSQRIndex::new(config, 2));

    let index_clone1 = Arc::clone(&index);
    let t1 = thread::spawn(move || {
        let v = VectorEmbedding::new(vec![1.0, 0.0]);
        let _ = index_clone1.insert("doc_1", v);
    });

    let index_clone2 = Arc::clone(&index);
    let t2 = thread::spawn(move || {
        let v = VectorEmbedding::new(vec![0.0, 1.0]);
        let _ = index_clone2.insert("doc_2", v);
    });

    t1.join().unwrap();
    t2.join().unwrap();

    assert_eq!(index.size(), 2);
}
