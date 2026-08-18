/* hnsqr/src/mmap_arena.rs */
//!▫~•◦-------------------------------‣
//! # Zero-Copy Memory-Mapped Quantized Vector Arena (`MmapArena`)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a memory-mapped quantized vector arena using `memmap2`, with direct page
//! attachment and atomic bump allocation in a 64-byte-aligned layout. The current
//! format does not persist HNSQR external IDs, metadata, graph state, or Rivero routes.
//!
//! ## Key Capabilities
//! - **Direct Mapping:** Attaches quantized vector pages without heap deserialization.
//! - **Cache-Aligned Memory Layout:** 64-byte boundary alignment (`#[repr(align(64))]`) for SIMD memory streams.
//! - **Concurrent Lock-Free Growth:** Atomic capacity checks and uncompressed/quantized dual storage modes.
//!
//! ### Architectural Notes
//! `HNSQRIndex::open_mmap` currently attaches this vector store to an empty in-memory
//! routing index. Complete index recovery requires a future format version.
//!
//! #### Example
//! ```rust
//! use hnsqr::storage::mmap_arena::MmapArena;
//! use std::fs;
//!
//! let path = std::env::temp_dir().join(format!("hnsqr-doc-{}.bin", std::process::id()));
//! let arena = MmapArena::create(&path, 10_000, 64).unwrap();
//! drop(arena);
//! fs::remove_file(path).unwrap();
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#![warn(missing_docs)]

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

use memmap2::{MmapMut, MmapOptions};
use num_complex::Complex32;
use crate::vector::quantization::{PolarQuantizedVector, asymmetric_projective_overlap};
use crate::{HNSQRError, HNSQRResult, NodeIndex};

/// Magic header identifier: "HNSQR001" in ASCII.
pub const HNSQR_MAGIC: u64 = 0x484E535152303031;
/// File format version.
pub const HNSQR_VERSION: u32 = 1;
/// Alignment boundary for header and vector regions (64-byte cache line).
pub const HEADER_SIZE: usize = 128;

/// Memory-Mapped Disk Header for the HNSQR Index.
#[repr(C, align(64))]
pub struct MmapHeader {
    /// Magic constant `HNSQR_MAGIC`.
    pub magic: u64,
    /// Format version number.
    pub version: u32,
    /// Complex dimensionality.
    pub dimension: u32,
    /// Maximum allocated node capacity.
    pub max_capacity: u32,
    /// Atomic count of inserted vectors.
    pub current_len: AtomicU32,
    /// Max graph level currently populated.
    pub max_level: AtomicU32,
    /// Bytes occupied per quantized vector payload.
    pub bytes_per_vector: u32,
    /// Reserved space for future expansion.
    pub _reserved: [u8; 84],
}

/// Zero-Copy Memory-Mapped Persistence Arena for billion-scale vector indexes.
pub struct MmapArena {
    mmap: Arc<parking_lot::RwLock<MmapMut>>,
    _file: Option<File>,
    dimension: usize,
    max_capacity: usize,
    bytes_per_vector: usize,
    vectors_offset: usize,
    norms_offset: usize,
    amplitudes_offset: usize,
}

impl MmapArena {
    /// Creates a new persistent disk-backed `MmapArena` at `path`.
    pub fn create<P: AsRef<Path>>(
        path: P,
        max_capacity: usize,
        dimension: usize,
    ) -> HNSQRResult<Self> {
        let bytes_per_vector = dimension * 2; // 8-bit amplitude + 8-bit phase
        let vectors_size = max_capacity * bytes_per_vector;
        let norms_size = max_capacity * std::mem::size_of::<f32>();
        let amplitudes_size = max_capacity * 2 * std::mem::size_of::<f32>();

        let vectors_offset = HEADER_SIZE;
        let norms_offset = vectors_offset + vectors_size;
        let amplitudes_offset = norms_offset + norms_size;
        let total_file_size = amplitudes_offset + amplitudes_size;

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(|e| {
                HNSQRError::SerializationError(format!("Failed to create mmap file: {}", e))
            })?;

        file.seek(SeekFrom::Start((total_file_size - 1) as u64))
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
        file.write_all(&[0])
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
        file.flush()
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;

        let mut mmap = unsafe {
            MmapOptions::new().map_mut(&file).map_err(|e| {
                HNSQRError::SerializationError(format!("Failed to map memory file: {}", e))
            })?
        };

        // Write header
        let header = unsafe { &mut *(mmap.as_mut_ptr() as *mut MmapHeader) };
        header.magic = HNSQR_MAGIC;
        header.version = HNSQR_VERSION;
        header.dimension = dimension as u32;
        header.max_capacity = max_capacity as u32;
        header.current_len.store(0, AtomicOrdering::SeqCst);
        header.max_level.store(0, AtomicOrdering::SeqCst);
        header.bytes_per_vector = bytes_per_vector as u32;

        Ok(Self {
            mmap: Arc::new(parking_lot::RwLock::new(mmap)),
            _file: Some(file),
            dimension,
            max_capacity,
            bytes_per_vector,
            vectors_offset,
            norms_offset,
            amplitudes_offset,
        })
    }

    /// Creates an anonymous (in-memory) memory-mapped arena.
    pub fn create_anonymous(max_capacity: usize, dimension: usize) -> HNSQRResult<Self> {
        let bytes_per_vector = dimension * 2;
        let vectors_size = max_capacity * bytes_per_vector;
        let norms_size = max_capacity * std::mem::size_of::<f32>();
        let amplitudes_size = max_capacity * 2 * std::mem::size_of::<f32>();

        let vectors_offset = HEADER_SIZE;
        let norms_offset = vectors_offset + vectors_size;
        let amplitudes_offset = norms_offset + norms_size;
        let total_size = amplitudes_offset + amplitudes_size;

        let mut mmap = MmapOptions::new().len(total_size).map_anon().map_err(|e| {
            HNSQRError::SerializationError(format!("Failed to allocate anon mmap: {}", e))
        })?;

        let header = unsafe { &mut *(mmap.as_mut_ptr() as *mut MmapHeader) };
        header.magic = HNSQR_MAGIC;
        header.version = HNSQR_VERSION;
        header.dimension = dimension as u32;
        header.max_capacity = max_capacity as u32;
        header.current_len.store(0, AtomicOrdering::SeqCst);
        header.max_level.store(0, AtomicOrdering::SeqCst);
        header.bytes_per_vector = bytes_per_vector as u32;

        Ok(Self {
            mmap: Arc::new(parking_lot::RwLock::new(mmap)),
            _file: None,
            dimension,
            max_capacity,
            bytes_per_vector,
            vectors_offset,
            norms_offset,
            amplitudes_offset,
        })
    }

    /// Opens an existing memory-mapped index file with zero deserialization overhead.
    pub fn open<P: AsRef<Path>>(path: P) -> HNSQRResult<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|e| {
                HNSQRError::SerializationError(format!("Failed to open mmap file: {}", e))
            })?;

        let mmap = unsafe {
            MmapOptions::new().map_mut(&file).map_err(|e| {
                HNSQRError::SerializationError(format!("Failed to map existing file: {}", e))
            })?
        };

        let header = unsafe { &*(mmap.as_ptr() as *const MmapHeader) };
        if header.magic != HNSQR_MAGIC {
            return Err(HNSQRError::SerializationError(
                "Invalid HNSQR magic header".to_string(),
            ));
        }

        let dimension = header.dimension as usize;
        let max_capacity = header.max_capacity as usize;
        let bytes_per_vector = header.bytes_per_vector as usize;

        let vectors_offset = HEADER_SIZE;
        let norms_offset = vectors_offset + (max_capacity * bytes_per_vector);
        let amplitudes_offset = norms_offset + (max_capacity * std::mem::size_of::<f32>());

        Ok(Self {
            mmap: Arc::new(parking_lot::RwLock::new(mmap)),
            _file: Some(file),
            dimension,
            max_capacity,
            bytes_per_vector,
            vectors_offset,
            norms_offset,
            amplitudes_offset,
        })
    }

    /// Atomically claims a slot directly in the file header.
    #[inline(always)]
    pub fn claim_slot(&self) -> HNSQRResult<NodeIndex> {
        let mmap_guard = self.mmap.read();
        let header = unsafe { &*(mmap_guard.as_ptr() as *const MmapHeader) };
        let index = header.current_len.fetch_add(1, AtomicOrdering::Relaxed);
        if index as usize >= self.max_capacity {
            header.current_len.fetch_sub(1, AtomicOrdering::Relaxed);
            return Err(HNSQRError::IndexFull(self.max_capacity));
        }
        Ok(index as NodeIndex)
    }

    /// Writes and quantizes a complex vector slice into the memory-mapped file slot.
    #[inline(always)]
    pub fn write_vector(&self, index: NodeIndex, slice: &[Complex32], norm_sq: f32) {
        let idx = index as usize;
        let mmap_guard = self.mmap.read();

        let vec_ptr = unsafe {
            let base = mmap_guard
                .as_ptr()
                .add(self.vectors_offset + idx * self.bytes_per_vector)
                as *mut u8;
            std::slice::from_raw_parts_mut(base, self.bytes_per_vector)
        };

        let (min_r, max_r) = PolarQuantizedVector::quantize_into_buffer(slice, vec_ptr);

        // Store precomputed norm and amplitudes
        unsafe {
            let norm_ptr = (mmap_guard.as_ptr().add(self.norms_offset) as *mut f32).add(idx);
            *norm_ptr = norm_sq;

            let amp_ptr =
                (mmap_guard.as_ptr().add(self.amplitudes_offset) as *mut (f32, f32)).add(idx);
            *amp_ptr = (min_r, max_r);
        }
    }

    /// Returns the dimensionality configured in this arena.
    #[inline(always)]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the maximum capacity of this arena.
    #[inline(always)]
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Returns the number of elements active in the mmap file.
    #[inline(always)]
    pub fn len(&self) -> usize {
        let mmap_guard = self.mmap.read();
        let header = unsafe { &*(mmap_guard.as_ptr() as *const MmapHeader) };
        header.current_len.load(AtomicOrdering::Acquire) as usize
    }

    /// Returns true if the arena contains no vectors.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evaluates asymmetric quantum fidelity between an uncompressed query and the mmap-backed quantized vector.
    #[inline(always)]
    pub fn compute_fidelity(
        &self,
        query: &[Complex32],
        query_norm_sq: f32,
        index: NodeIndex,
    ) -> f32 {
        let idx = index as usize;
        let mmap_guard = self.mmap.read();

        unsafe {
            let vec_bytes = std::slice::from_raw_parts(
                mmap_guard
                    .as_ptr()
                    .add(self.vectors_offset + idx * self.bytes_per_vector),
                self.bytes_per_vector,
            );
            let norm_sq = *(mmap_guard.as_ptr().add(self.norms_offset) as *const f32).add(idx);
            let (min_r, max_r) =
                *(mmap_guard.as_ptr().add(self.amplitudes_offset) as *const (f32, f32)).add(idx);

            asymmetric_projective_overlap(query, query_norm_sq, vec_bytes, min_r, max_r, norm_sq)
        }
    }

    /// Flushes all memory-mapped dirty pages to disk.
    pub fn flush(&self) -> HNSQRResult<()> {
        let mmap_guard = self.mmap.read();
        mmap_guard.flush().map_err(|e| {
            HNSQRError::SerializationError(format!("Failed to sync mmap pages: {}", e))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmap_arena_roundtrip() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_hnsqr_arena.bin");

        {
            let arena = MmapArena::create(&file_path, 100, 4).unwrap();
            let idx = arena.claim_slot().unwrap();
            let vec = vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 1.0),
                Complex32::new(0.5, 0.5),
                Complex32::new(-1.0, 0.0),
            ];
            arena.write_vector(idx, &vec, 2.5);
            arena.flush().unwrap();
            assert_eq!(arena.len(), 1);
        }

        // Re-open zero-copy from disk
        {
            let arena = MmapArena::open(&file_path).unwrap();
            assert_eq!(arena.len(), 1);
            let query = vec![
                Complex32::new(1.0, 0.0),
                Complex32::new(0.0, 1.0),
                Complex32::new(0.5, 0.5),
                Complex32::new(-1.0, 0.0),
            ];
            let fid = arena.compute_fidelity(&query, 2.5, 0);
            assert!(fid > 0.95);
        }

        let _ = std::fs::remove_file(file_path);
    }
}
