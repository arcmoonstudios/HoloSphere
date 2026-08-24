/* holosphere/src/entity/vector.rs */
//!▫~•◦-------------------------------‣
//! # Contiguous Vector Arena Storage
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides dense, contiguous columnar vector storage indexed by `vector_row`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::sync::atomic::{AtomicU32, Ordering};

/// Dense, contiguous vector arena supporting variable layouts and SIMD alignment.
pub struct VectorArena {
    dimension: usize,
    data: RwLock<Vec<f32>>,
    next_row: AtomicU32,
}

impl VectorArena {
    /// Creates a new vector arena with a fixed component dimension.
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            data: RwLock::new(Vec::new()),
            next_row: AtomicU32::new(0),
        }
    }

    /// Returns vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Appends a vector and returns its row index.
    pub fn append(&self, vector: &[f32]) -> Option<u32> {
        if vector.len() != self.dimension {
            return None;
        }

        let row = self.next_row.fetch_add(1, Ordering::Relaxed);
        let mut data = self.data.write();
        data.extend_from_slice(vector);
        Some(row)
    }

    /// Retrieves a contiguous slice of floats for `row`.
    #[inline]
    pub fn get_row(&self, row: u32) -> Option<Vec<f32>> {
        let start = (row as usize) * self.dimension;
        let end = start + self.dimension;
        let data = self.data.read();
        if end <= data.len() {
            Some(data[start..end].to_vec())
        } else {
            None
        }
    }

    /// Executes a closure with a zero-copy read reference to the vector row.
    #[inline]
    pub fn with_row<F, R>(&self, row: u32, f: F) -> Option<R>
    where
        F: FnOnce(&[f32]) -> R,
    {
        let start = (row as usize) * self.dimension;
        let end = start + self.dimension;
        let data = self.data.read();
        if end <= data.len() {
            Some(f(&data[start..end]))
        } else {
            None
        }
    }

    /// Total rows in arena.
    pub fn row_count(&self) -> usize {
        let data = self.data.read();
        data.len() / self.dimension.max(1)
    }

    /// Direct slice access for dense streaming scans.
    pub fn raw_slice(&self) -> Vec<f32> {
        self.data.read().clone()
    }
}
