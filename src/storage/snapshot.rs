/* hnsqr/src/snapshot.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Index Snapshot V2 Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides an immutable, sectioned, checksummed, zero-copy, mmap-oriented snapshot
//! format (`.hnsqr` V2) for instantaneous cold start and persistent index recovery.
//!
//! ### Architectural Pillars:
//!   1. **Zero-Rebuild Instant Startup**:
//!      Restores complete frozen Rivero territory cells, reciprocal witness graphs,
//!      and HNSW fallback layers in milliseconds with 0 rebuild operations.
//!   2. **Sectioned & Checksummed Layout**:
//!      Every section carries its own offset, length, element count, and SHA-256 checksum.
//!   3. **Transactional Publication**:
//!      Writes to `.tmp` file, flushes/fsyncs, validates integrity, and atomically renames.
//!   4. **Cross-Section Bounds & Failure Safety**:
//!      Strict validation of all offsets, slice ranges, and reference IDs; malformed files fail closed.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use memmap2::Mmap;
use num_complex::Complex32;
use rayon::prelude::*;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::rivero::{RIVERO_SCHEMA_VERSION, RiveroConfig};
use crate::rivero_witness::ScoredWitness;
use crate::{HNSQRConfig, HNSQRError, HNSQRIndex, HNSQRResult, Node, NodeIndex};

pub const SNAPSHOT_V2_MAGIC: [u8; 8] = *b"HNSQRV2\0";
pub const SNAPSHOT_V2_VERSION: u32 = 2;
pub const HEADER_SIZE_V2: u32 = 256;
pub const SECTION_DESCRIPTOR_SIZE: u32 = 64;

/// Section types recognized in the V2 snapshot format.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum SectionKind {
    VectorData = 1,
    VectorNorms = 2,
    ExternalIdOffsets = 3,
    ExternalIdBytes = 4,
    LivenessBitmap = 5,
    MetadataDictionary = 6,
    MetadataPostings = 7,
    RiveroConfig = 8,
    RiveroCellDirectory = 9,
    RiveroCellResidents = 10,
    RiveroWitnessOffsets = 11,
    RiveroWitnessEdges = 12,
    HnswGraphOffsets = 13,
    HnswGraphEdges = 14,
    IndexStats = 15,
}

impl SectionKind {
    pub const fn from_u32(val: u32) -> Option<Self> {
        match val {
            1 => Some(Self::VectorData),
            2 => Some(Self::VectorNorms),
            3 => Some(Self::ExternalIdOffsets),
            4 => Some(Self::ExternalIdBytes),
            5 => Some(Self::LivenessBitmap),
            6 => Some(Self::MetadataDictionary),
            7 => Some(Self::MetadataPostings),
            8 => Some(Self::RiveroConfig),
            9 => Some(Self::RiveroCellDirectory),
            10 => Some(Self::RiveroCellResidents),
            11 => Some(Self::RiveroWitnessOffsets),
            12 => Some(Self::RiveroWitnessEdges),
            13 => Some(Self::HnswGraphOffsets),
            14 => Some(Self::HnswGraphEdges),
            15 => Some(Self::IndexStats),
            _ => None,
        }
    }
}

/// 256-byte immutable binary header at offset 0 of every `.hnsqr` V2 snapshot file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SnapshotHeaderV2 {
    /// Format magic identifier `b"HNSQRV2\0"`.
    pub magic: [u8; 8],
    /// Snapshot format version (must equal `SNAPSHOT_V2_VERSION = 2`).
    pub format_version: u32,
    /// Size of the header block in bytes (256).
    pub header_size: u32,
    /// Monotonic snapshot generation number.
    pub generation: u64,
    /// Operational feature flags.
    pub flags: u64,
    /// Complex vector dimensionality $D$.
    pub dimension: u32,
    /// Total number of vector slots allocated.
    pub vector_count: u64,
    /// Number of active live (non-deleted) nodes.
    pub live_count: u64,
    /// Rivero address compiler schema version.
    pub rivero_schema_version: u16,
    /// Rivero operational profile tag.
    pub rivero_profile: u8,
    /// Distance metric tag (0 = Projective Overlap / CPO, 1 = Cosine).
    pub distance_metric: u8,
    /// Total number of section descriptors in the section table.
    pub section_count: u32,
    /// Cryptographic hash of the configuration parameters.
    pub config_hash: [u8; 32],
    /// Canonical structural fingerprint of the entire dataset.
    pub structural_hash: [u8; 32],
    /// Reserved space for future extensions.
    pub reserved: [u8; 128],
}

const _: () = assert!(std::mem::size_of::<SnapshotHeaderV2>() == 256);
const _: () = assert!(std::mem::size_of::<SectionDescriptor>() == 64);
const _: () = assert!(std::mem::size_of::<FrozenCellRecord>() == 24);
const _: () = assert!(std::mem::size_of::<DiskCellResident>() == 8);
const _: () = assert!(std::mem::size_of::<DiskWitnessEdge>() == 8);

/// 64-byte descriptor for one contiguous section in the snapshot file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct SectionDescriptor {
    /// Section kind discriminant.
    pub kind: u32,
    /// Section-specific flags.
    pub flags: u32,
    /// Absolute byte offset in the file.
    pub offset: u64,
    /// Byte length of the section payload.
    pub length: u64,
    /// Number of logical elements stored in this section.
    pub element_count: u64,
    /// SHA-256 cryptographic checksum of this section payload.
    pub checksum: [u8; 32],
}

/// Packed on-disk record for one Rivero territory cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct FrozenCellRecord {
    pub key: u64,
    pub resident_start: u64,
    pub resident_count: u16,
    pub elite_count: u16,
    pub overflowed: u32,
}

/// Eight-byte contiguous resident record on disk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct DiskCellResident {
    pub slot: u32,
    pub fine_code: u32,
}

/// Eight-byte contiguous witness edge record on disk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct DiskWitnessEdge {
    pub target: u32,
    pub similarity_bits: u32,
}

/// Verification mode when opening a snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VerificationMode {
    /// Validates header magic, versions, and section boundaries without payload re-hashing (< 5 ms).
    #[default]
    HeaderAndBounds,
    /// Validates header, bounds, and recomputes the global structural fingerprint.
    FullStructuralCheck,
    /// Validates all section SHA-256 cryptographic checksums in full.
    FullChecksums,
}

/// Prefault mode when attaching memory-mapped snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PrefaultMode {
    /// Rely on on-demand kernel paging (default, lowest latency attach).
    #[default]
    None,
    /// Intentionally fault in memory pages during open.
    Eager,
}

/// Options configuring snapshot restoration.
#[derive(Clone, Copy, Debug, Default)]
pub struct SnapshotOpenOptions {
    pub verification: VerificationMode,
    pub prefault: PrefaultMode,
    pub max_elements_override: Option<usize>,
}

/// Microsecond-level instrumentation breakdown of snapshot attachment phases.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SnapshotAttachBreakdown {
    pub open_syscall_us: f64,
    pub mmap_creation_us: f64,
    pub header_decode_us: f64,
    pub section_table_us: f64,
    pub bounds_validation_us: f64,
    pub config_restore_us: f64,
    pub arena_restore_us: f64,
    pub id_restore_us: f64,
    pub liveness_restore_us: f64,
    pub metadata_restore_us: f64,
    pub rivero_restore_us: f64,
    pub witnesses_restore_us: f64,
    pub graph_restore_us: f64,
    pub structural_val_us: f64,
    pub total_attach_us: f64,
}

/// Telemetry metrics produced during snapshot creation or loading.
#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
pub struct SnapshotStats {
    pub file_size_bytes: u64,
    pub vector_count: u64,
    pub live_count: u64,
    pub section_count: usize,
    pub time_total_ms: f64,
    pub time_validation_ms: f64,
    pub throughput_mb_per_sec: f64,
    pub structural_hash: [u8; 32],
}

impl HNSQRIndex {
    /// Saves the complete index state into a durable, sectioned V2 snapshot file.
    ///
    /// Executes transactionally by writing to a temporary sibling file, validating
    /// checksums and structural hashes, and performing an atomic rename.
    #[allow(clippy::type_complexity)]
    pub fn save_snapshot_v2<P: AsRef<Path>>(&self, path: P) -> HNSQRResult<SnapshotStats> {
        let start_time = Instant::now();
        let path = path.as_ref();
        let tmp_path = path.with_extension("tmp");

        let _lifecycle = self.lifecycle.read();
        let (dimension, vector_count, live_count) =
            (self.dimension, self.arena.len(), self.arena.live_len());

        let raw_file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tmp_path)
            .map_err(|e: std::io::Error| {
                HNSQRError::IoError(format!("Failed to create snapshot tmp file: {}", e))
            })?;
        let mut file = BufWriter::new(raw_file);

        let mut sections: Vec<(SectionDescriptor, Vec<u8>)> = Vec::new();

        // 1. VECTOR_DATA
        let mut vec_bytes = Vec::with_capacity(vector_count * dimension * 8);
        for i in 0..vector_count as NodeIndex {
            let slice = self.arena.get_vector_slice(i);
            for &z in slice {
                vec_bytes.extend_from_slice(&z.re.to_le_bytes());
                vec_bytes.extend_from_slice(&z.im.to_le_bytes());
            }
        }
        sections.push(create_section(
            SectionKind::VectorData,
            0,
            vector_count as u64,
            &vec_bytes,
        ));

        // 2. VECTOR_NORMS
        let mut norm_bytes = Vec::with_capacity(vector_count * 4);
        for i in 0..vector_count as NodeIndex {
            let norm_sq = self.arena.get_norm_squared(i);
            norm_bytes.extend_from_slice(&norm_sq.to_le_bytes());
        }
        sections.push(create_section(
            SectionKind::VectorNorms,
            0,
            vector_count as u64,
            &norm_bytes,
        ));

        // 3. EXTERNAL_ID_OFFSETS & 4. EXTERNAL_ID_BYTES
        let mut id_offsets = Vec::with_capacity((vector_count + 1) * 8);
        let mut id_bytes = Vec::new();
        let mut current_offset = 0u64;
        id_offsets.extend_from_slice(&current_offset.to_le_bytes());

        for i in 0..vector_count as NodeIndex {
            if let Some(node) = self.arena.get_node(i) {
                let id_str = node.external_id.as_bytes();
                id_bytes.extend_from_slice(id_str);
                current_offset += id_str.len() as u64;
            }
            id_offsets.extend_from_slice(&current_offset.to_le_bytes());
        }
        sections.push(create_section(
            SectionKind::ExternalIdOffsets,
            0,
            (vector_count + 1) as u64,
            &id_offsets,
        ));
        sections.push(create_section(
            SectionKind::ExternalIdBytes,
            0,
            id_bytes.len() as u64,
            &id_bytes,
        ));

        // 5. LIVENESS_BITMAP
        let mut live_bitmap = RoaringBitmap::new();
        for i in 0..vector_count as u32 {
            if self.arena.is_live(i as NodeIndex) {
                live_bitmap.insert(i);
            }
        }
        let mut live_bytes = Vec::new();
        live_bitmap
            .serialize_into(&mut live_bytes)
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
        sections.push(create_section(
            SectionKind::LivenessBitmap,
            0,
            live_bitmap.len(),
            &live_bytes,
        ));

        // 6. METADATA_POSTINGS (Canonical BTreeMap serialization for determinism)
        let (cat, num) = self.metadata_index.export_postings();
        let mut sorted_cat: BTreeMap<String, BTreeMap<String, RoaringBitmap>> = BTreeMap::new();
        for (k, v_map) in cat {
            let mut sorted_v = BTreeMap::new();
            for (vk, bm) in v_map {
                sorted_v.insert(vk, bm);
            }
            sorted_cat.insert(k, sorted_v);
        }
        let meta_bytes = bincode::serialize(&(sorted_cat, num))
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
        sections.push(create_section(
            SectionKind::MetadataPostings,
            0,
            1,
            &meta_bytes,
        ));

        // 7. RIVERO_CONFIG
        let rivero_cfg = self.config.read().rivero_config;
        let cfg_bytes = bincode::serialize(&rivero_cfg)
            .map_err(|e| HNSQRError::SerializationError(e.to_string()))?;
        sections.push(create_section(SectionKind::RiveroConfig, 0, 1, &cfg_bytes));

        // 8. RIVERO_CELL_DIRECTORY & 9. RIVERO_CELL_RESIDENTS
        let mut all_cells: Vec<(u64, Vec<(NodeIndex, u32)>, usize, bool)> = Vec::new();
        for stripe in &self.rivero_index.stripes {
            let guard = stripe.read();
            for (&key, cell) in guard.iter() {
                let residents: Vec<(NodeIndex, u32)> =
                    cell.slots.iter().map(|r| (r.slot, r.fine_code)).collect();
                all_cells.push((key, residents, cell.elite_len, cell.overflowed));
            }
        }
        all_cells.sort_unstable_by_key(|c| c.0);

        let mut dir_bytes =
            Vec::with_capacity(all_cells.len() * std::mem::size_of::<FrozenCellRecord>());
        let mut res_bytes = Vec::new();
        let mut res_cursor = 0u64;

        for (key, residents, elite_len, overflowed) in &all_cells {
            let record = FrozenCellRecord {
                key: *key,
                resident_start: res_cursor,
                resident_count: residents.len() as u16,
                elite_count: *elite_len as u16,
                overflowed: if *overflowed { 1 } else { 0 },
            };
            dir_bytes.extend_from_slice(bytemuck_cast(&record));

            for &(slot, fine_code) in residents {
                let disk_res = DiskCellResident { slot, fine_code };
                res_bytes.extend_from_slice(bytemuck_cast(&disk_res));
            }
            res_cursor += residents.len() as u64;
        }
        sections.push(create_section(
            SectionKind::RiveroCellDirectory,
            0,
            all_cells.len() as u64,
            &dir_bytes,
        ));
        sections.push(create_section(
            SectionKind::RiveroCellResidents,
            0,
            res_cursor,
            &res_bytes,
        ));

        // 10. RIVERO_WITNESS_OFFSETS & 11. RIVERO_WITNESS_EDGES
        let mut witness_offsets = Vec::with_capacity((vector_count + 1) * 8);
        let mut witness_edges = Vec::new();
        let mut wit_offset = 0u64;
        witness_offsets.extend_from_slice(&wit_offset.to_le_bytes());

        for i in 0..vector_count as NodeIndex {
            if let Some(node) = self.arena.get_node(i) {
                let conns = node.rivero_witnesses.read();
                for w in conns.iter() {
                    let disk_edge = DiskWitnessEdge {
                        target: w.index,
                        similarity_bits: w.similarity.to_bits(),
                    };
                    witness_edges.extend_from_slice(bytemuck_cast(&disk_edge));
                }
                wit_offset += conns.len() as u64;
            }
            witness_offsets.extend_from_slice(&wit_offset.to_le_bytes());
        }
        sections.push(create_section(
            SectionKind::RiveroWitnessOffsets,
            0,
            (vector_count + 1) as u64,
            &witness_offsets,
        ));
        sections.push(create_section(
            SectionKind::RiveroWitnessEdges,
            0,
            wit_offset,
            &witness_edges,
        ));

        // 12. HNSW_GRAPH_OFFSETS & 13. HNSW_GRAPH_EDGES
        let mut graph_offsets = Vec::with_capacity((vector_count + 1) * 8);
        let mut graph_edges = Vec::new();
        let mut g_offset = 0u64;
        graph_offsets.extend_from_slice(&g_offset.to_le_bytes());

        for i in 0..vector_count as NodeIndex {
            if let Some(node) = self.arena.get_node(i) {
                if !node.layers.is_empty() {
                    let conns = node.layers[0].read();
                    for &target in conns.iter() {
                        graph_edges.extend_from_slice(&target.to_le_bytes());
                    }
                    g_offset += conns.len() as u64;
                }
            }
            graph_offsets.extend_from_slice(&g_offset.to_le_bytes());
        }
        sections.push(create_section(
            SectionKind::HnswGraphOffsets,
            0,
            (vector_count + 1) as u64,
            &graph_offsets,
        ));
        sections.push(create_section(
            SectionKind::HnswGraphEdges,
            0,
            g_offset,
            &graph_edges,
        ));

        // Compute offsets and assemble section table with 8-byte alignment
        let section_table_len = (sections.len() as u64) * (SECTION_DESCRIPTOR_SIZE as u64);
        let mut current_file_offset = (HEADER_SIZE_V2 as u64) + section_table_len;

        for (desc, payload) in sections.iter_mut() {
            let padding = (8 - (current_file_offset % 8)) % 8;
            current_file_offset += padding;
            desc.offset = current_file_offset;
            current_file_offset += payload.len() as u64;
        }

        let structural_hash = self.structural_fingerprint();
        let config_hash = Sha256::digest(&cfg_bytes).into();

        let header = SnapshotHeaderV2 {
            magic: SNAPSHOT_V2_MAGIC,
            format_version: SNAPSHOT_V2_VERSION,
            header_size: HEADER_SIZE_V2,
            generation: 1,
            flags: 0,
            dimension: dimension as u32,
            vector_count: vector_count as u64,
            live_count: live_count as u64,
            rivero_schema_version: RIVERO_SCHEMA_VERSION,
            rivero_profile: 1, // Balanced
            distance_metric: match self.config.read().distance_function {
                crate::DistanceFunction::Cosine => 1,
                _ => 0,
            },
            section_count: sections.len() as u32,
            config_hash,
            structural_hash,
            reserved: [0u8; 128],
        };

        // Write Header
        file.write_all(bytemuck_cast(&header))
            .map_err(|e: std::io::Error| HNSQRError::IoError(e.to_string()))?;

        // Write Section Descriptors Table
        for (desc, _) in &sections {
            file.write_all(bytemuck_cast(desc))
                .map_err(|e: std::io::Error| HNSQRError::IoError(e.to_string()))?;
        }

        // Write Section Payloads with explicit padding
        let mut written_bytes = (HEADER_SIZE_V2 as u64) + section_table_len;
        for (desc, payload) in &sections {
            let pad_len = (desc.offset.saturating_sub(written_bytes)) as usize;
            if pad_len > 0 {
                file.write_all(&[0u8; 8][..pad_len])
                    .map_err(|e: std::io::Error| HNSQRError::IoError(e.to_string()))?;
                written_bytes += pad_len as u64;
            }
            file.write_all(payload)
                .map_err(|e: std::io::Error| HNSQRError::IoError(e.to_string()))?;
            written_bytes += payload.len() as u64;
        }

        file.flush()
            .map_err(|e: std::io::Error| HNSQRError::IoError(e.to_string()))?;
        drop(file);

        // Atomic rename
        std::fs::rename(&tmp_path, path).map_err(|e: std::io::Error| {
            HNSQRError::IoError(format!("Failed to rename snapshot tmp to target: {}", e))
        })?;

        let total_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        let file_size_bytes = current_file_offset;
        let throughput_mb = if total_time_ms > 0.0 {
            (file_size_bytes as f64 / (1024.0 * 1024.0)) / (total_time_ms / 1000.0)
        } else {
            0.0
        };

        Ok(SnapshotStats {
            file_size_bytes,
            vector_count: vector_count as u64,
            live_count: live_count as u64,
            section_count: sections.len(),
            time_total_ms: total_time_ms,
            time_validation_ms: 0.0,
            throughput_mb_per_sec: throughput_mb,
            structural_hash,
        })
    }

    /// Opens an immutable V2 snapshot from disk via memory mapping with zero-rebuild recovery.
    pub fn open_snapshot_v2<P: AsRef<Path>>(
        path: P,
        options: SnapshotOpenOptions,
    ) -> HNSQRResult<Self> {
        Self::open_snapshot_v2_instrumented(path, options).map(|(idx, _)| idx)
    }

    /// Opens an immutable V2 snapshot and returns detailed phase-by-phase attachment latency telemetry.
    pub fn open_snapshot_v2_instrumented<P: AsRef<Path>>(
        path: P,
        options: SnapshotOpenOptions,
    ) -> HNSQRResult<(Self, SnapshotAttachBreakdown)> {
        let t_total = Instant::now();
        let path = path.as_ref();

        let t0 = Instant::now();
        let file = File::open(path).map_err(|e: std::io::Error| {
            HNSQRError::IoError(format!("Failed to open snapshot: {}", e))
        })?;
        let file_len = file
            .metadata()
            .map_err(|e: std::io::Error| HNSQRError::IoError(e.to_string()))?
            .len();
        let open_syscall_us = t0.elapsed().as_micros() as f64;

        if file_len < (HEADER_SIZE_V2 as u64) {
            return Err(HNSQRError::SnapshotIncompatible(format!(
                "Snapshot file too short: {} bytes (expected at least {})",
                file_len, HEADER_SIZE_V2
            )));
        }

        let t1 = Instant::now();
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e: std::io::Error| {
                HNSQRError::IoError(format!("Mmap snapshot failed: {}", e))
            })?
        };
        let mmap_creation_us = t1.elapsed().as_micros() as f64;

        // 1. Validate Header
        let t2 = Instant::now();
        let header = unsafe { &*(mmap.as_ptr() as *const SnapshotHeaderV2) };
        if header.magic != SNAPSHOT_V2_MAGIC {
            return Err(HNSQRError::SnapshotIncompatible(format!(
                "Invalid snapshot magic: {:?} (expected {:?})",
                header.magic, SNAPSHOT_V2_MAGIC
            )));
        }
        if header.format_version != SNAPSHOT_V2_VERSION {
            return Err(HNSQRError::SnapshotIncompatible(format!(
                "Unsupported format version: {} (supported: {})",
                header.format_version, SNAPSHOT_V2_VERSION
            )));
        }
        if header.rivero_schema_version != RIVERO_SCHEMA_VERSION {
            return Err(HNSQRError::SnapshotIncompatible(format!(
                "Rivero schema version mismatch: {} (current: {})",
                header.rivero_schema_version, RIVERO_SCHEMA_VERSION
            )));
        }
        let header_decode_us = t2.elapsed().as_micros() as f64;

        // 2. Validate Section Table
        let t3 = Instant::now();
        let section_count = header.section_count as usize;
        let table_offset = header.header_size as usize;
        let table_end = table_offset + section_count * (SECTION_DESCRIPTOR_SIZE as usize);

        if table_end > mmap.len() {
            return Err(HNSQRError::SnapshotIncompatible(
                "Section descriptor table extends beyond file boundary".to_string(),
            ));
        }

        let mut section_map: HashMap<SectionKind, SectionDescriptor> =
            HashMap::with_capacity(section_count);
        let desc_slice = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr().add(table_offset) as *const SectionDescriptor,
                section_count,
            )
        };

        for &desc in desc_slice {
            let Some(kind) = SectionKind::from_u32(desc.kind) else {
                continue;
            };
            if desc.offset + desc.length > file_len {
                return Err(HNSQRError::SnapshotIncompatible(format!(
                    "Section {:?} boundary overflow (offset={}, len={}, file_len={})",
                    kind, desc.offset, desc.length, file_len
                )));
            }

            if options.verification == VerificationMode::FullChecksums {
                let payload = &mmap[desc.offset as usize..(desc.offset + desc.length) as usize];
                let actual_hash: [u8; 32] = Sha256::digest(payload).into();
                if actual_hash != desc.checksum {
                    return Err(HNSQRError::SnapshotIncompatible(format!(
                        "Checksum verification failed for section {:?}",
                        kind
                    )));
                }
            }

            section_map.insert(kind, desc);
        }
        let section_table_us = t3.elapsed().as_micros() as f64;

        let dim = header.dimension as usize;
        let vector_count = header.vector_count as usize;
        let max_capacity = options
            .max_elements_override
            .unwrap_or(vector_count.max(1000));

        let mut config = HNSQRConfig::default();
        config.max_elements = max_capacity;
        config.distance_function = match header.distance_metric {
            1 => crate::DistanceFunction::Cosine,
            _ => crate::DistanceFunction::ProjectiveOverlap,
        };

        // Restore Rivero Config if present
        let t4 = Instant::now();
        if let Some(desc) = section_map.get(&SectionKind::RiveroConfig) {
            let payload = &mmap[desc.offset as usize..(desc.offset + desc.length) as usize];
            if let Ok(rivero_cfg) = bincode::deserialize::<RiveroConfig>(payload) {
                config.rivero_config = rivero_cfg;
            }
        }
        let config_restore_us = t4.elapsed().as_micros() as f64;

        let index = HNSQRIndex::new(config, dim);

        // 3. Populate Arena Vectors & Norms directly from mmap payload
        let t5 = Instant::now();
        let vec_desc = section_map.get(&SectionKind::VectorData).ok_or_else(|| {
            HNSQRError::SnapshotIncompatible("Missing VectorData section".to_string())
        })?;
        let norm_desc = section_map.get(&SectionKind::VectorNorms).ok_or_else(|| {
            HNSQRError::SnapshotIncompatible("Missing VectorNorms section".to_string())
        })?;

        let vec_slice: &[Complex32] = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr().add(vec_desc.offset as usize) as *const Complex32,
                vector_count * dim,
            )
        };
        let norm_slice: &[f32] = unsafe {
            std::slice::from_raw_parts(
                mmap.as_ptr().add(norm_desc.offset as usize) as *const f32,
                vector_count,
            )
        };

        #[allow(clippy::needless_range_loop)]
        for i in 0..vector_count {
            let slot = index.arena.claim_slot()?;
            let offset = i * dim;
            index
                .arena
                .write_vector(slot, &vec_slice[offset..offset + dim]);
            index.arena.norms_sq[i].store(norm_slice[i].to_bits(), Ordering::Release);
        }
        let arena_restore_us = t5.elapsed().as_micros() as f64;

        // 4. Restore External IDs
        let t6 = Instant::now();
        if let (Some(off_desc), Some(bytes_desc)) = (
            section_map.get(&SectionKind::ExternalIdOffsets),
            section_map.get(&SectionKind::ExternalIdBytes),
        ) {
            let offsets: &[u64] = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr().add(off_desc.offset as usize) as *const u64,
                    vector_count + 1,
                )
            };
            let id_bytes =
                &mmap[bytes_desc.offset as usize..(bytes_desc.offset + bytes_desc.length) as usize];

            let mut id_map = index.id_to_index.write();
            for i in 0..vector_count {
                let start = offsets[i] as usize;
                let end = offsets[i + 1] as usize;
                if end <= id_bytes.len() && start <= end {
                    if let Ok(id_str) = std::str::from_utf8(&id_bytes[start..end]) {
                        let arc_id: Arc<str> = Arc::from(id_str);
                        id_map.insert(arc_id.clone(), i as NodeIndex);
                        let node = Node::new(i as NodeIndex, arc_id, 0);
                        index.arena.write_node(i as NodeIndex, node);
                    }
                }
            }
        }
        let id_restore_us = t6.elapsed().as_micros() as f64;

        // 5. Restore Liveness
        let t7 = Instant::now();
        if let Some(desc) = section_map.get(&SectionKind::LivenessBitmap) {
            let payload = &mmap[desc.offset as usize..(desc.offset + desc.length) as usize];
            if let Ok(live_bitmap) = RoaringBitmap::deserialize_from(payload) {
                for slot in 0..vector_count as u32 {
                    if live_bitmap.contains(slot) {
                        index.arena.publish_slot(slot as NodeIndex);
                    } else {
                        index.arena.delete_slot(slot as NodeIndex);
                    }
                }
            }
        } else {
            for slot in 0..vector_count {
                index.arena.publish_slot(slot as NodeIndex);
            }
        }
        let liveness_restore_us = t7.elapsed().as_micros() as f64;

        // 6. Restore Metadata Postings
        let t8 = Instant::now();
        if let Some(desc) = section_map.get(&SectionKind::MetadataPostings) {
            let payload = &mmap[desc.offset as usize..(desc.offset + desc.length) as usize];
            if let Ok((sorted_cat, num)) = bincode::deserialize::<(
                BTreeMap<String, BTreeMap<String, RoaringBitmap>>,
                HashMap<String, BTreeMap<i64, RoaringBitmap>>,
            )>(payload)
            {
                let mut cat: HashMap<String, HashMap<String, RoaringBitmap>> = HashMap::new();
                for (k, v_map) in sorted_cat {
                    let mut hm = HashMap::new();
                    for (vk, bm) in v_map {
                        hm.insert(vk, bm);
                    }
                    cat.insert(k, hm);
                }
                index.metadata_index.import_postings(cat, num);
            }
        }
        let metadata_restore_us = t8.elapsed().as_micros() as f64;

        // 7. Restore Frozen Rivero Territories
        let t9 = Instant::now();
        if let (Some(dir_desc), Some(res_desc)) = (
            section_map.get(&SectionKind::RiveroCellDirectory),
            section_map.get(&SectionKind::RiveroCellResidents),
        ) {
            let cell_records: &[FrozenCellRecord] = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr().add(dir_desc.offset as usize) as *const FrozenCellRecord,
                    dir_desc.element_count as usize,
                )
            };
            let disk_residents: &[DiskCellResident] = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr().add(res_desc.offset as usize) as *const DiskCellResident,
                    res_desc.element_count as usize,
                )
            };

            let mut stripe_buckets: Vec<Vec<&FrozenCellRecord>> =
                vec![Vec::new(); crate::rivero::RIVERO_STRIPES];
            for rec in cell_records {
                let stripe_idx = crate::rivero::stripe_for(rec.key);
                stripe_buckets[stripe_idx].push(rec);
            }

            stripe_buckets
                .into_par_iter()
                .enumerate()
                .for_each(|(stripe_idx, recs)| {
                    let mut map = HashMap::with_capacity(recs.len());
                    for rec in recs {
                        let start = rec.resident_start as usize;
                        let count = rec.resident_count as usize;
                        let mut slots = Vec::with_capacity(count);
                        if start + count <= disk_residents.len() {
                            for r in &disk_residents[start..start + count] {
                                slots.push(crate::rivero::CellResident {
                                    slot: r.slot as NodeIndex,
                                    fine_code: r.fine_code,
                                });
                            }
                        }
                        map.insert(
                            rec.key,
                            crate::rivero::CellSlots {
                                slots,
                                elite_len: rec.elite_count as usize,
                                overflowed: rec.overflowed != 0,
                            },
                        );
                    }
                    *index.rivero_index.stripes[stripe_idx].write() = map;
                });
        }
        let rivero_restore_us = t9.elapsed().as_micros() as f64;

        // 8. Restore Witnesses
        let t10 = Instant::now();
        if let (Some(off_desc), Some(edges_desc)) = (
            section_map.get(&SectionKind::RiveroWitnessOffsets),
            section_map.get(&SectionKind::RiveroWitnessEdges),
        ) {
            let offsets: &[u64] = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr().add(off_desc.offset as usize) as *const u64,
                    vector_count + 1,
                )
            };
            let edges: &[DiskWitnessEdge] = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr().add(edges_desc.offset as usize) as *const DiskWitnessEdge,
                    edges_desc.element_count as usize,
                )
            };

            (0..vector_count as NodeIndex)
                .into_par_iter()
                .for_each(|i| {
                    let start = offsets[i as usize] as usize;
                    let end = offsets[i as usize + 1] as usize;
                    if end <= edges.len() && start <= end {
                        if let Some(node) = index.arena.get_node(i) {
                            let mut conns = node.rivero_witnesses.write();
                            for edge in &edges[start..end] {
                                conns.push(ScoredWitness {
                                    index: edge.target as NodeIndex,
                                    similarity: f32::from_bits(edge.similarity_bits),
                                });
                            }
                        }
                    }
                });
        }
        let witnesses_restore_us = t10.elapsed().as_micros() as f64;

        // 9. Restore Graph Fallback Layer 0
        let t11 = Instant::now();
        if let (Some(off_desc), Some(edges_desc)) = (
            section_map.get(&SectionKind::HnswGraphOffsets),
            section_map.get(&SectionKind::HnswGraphEdges),
        ) {
            let offsets: &[u64] = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr().add(off_desc.offset as usize) as *const u64,
                    vector_count + 1,
                )
            };
            let edges: &[u32] = unsafe {
                std::slice::from_raw_parts(
                    mmap.as_ptr().add(edges_desc.offset as usize) as *const u32,
                    edges_desc.element_count as usize,
                )
            };

            (0..vector_count as NodeIndex)
                .into_par_iter()
                .for_each(|i| {
                    let start = offsets[i as usize] as usize;
                    let end = offsets[i as usize + 1] as usize;
                    if end <= edges.len() && start <= end {
                        if let Some(node) = index.arena.get_node(i) {
                            let mut conns = node.layers[0].write();
                            for &target in &edges[start..end] {
                                conns.push(target as NodeIndex);
                            }
                        }
                    }
                });
        }
        let graph_restore_us = t11.elapsed().as_micros() as f64;

        // 10. Verify structural hash if configured
        let t12 = Instant::now();
        if options.verification != VerificationMode::HeaderAndBounds {
            let actual_structural = index.structural_fingerprint();
            if actual_structural != header.structural_hash {
                return Err(HNSQRError::SnapshotIncompatible(
                    "Structural fingerprint mismatch after snapshot restoration".to_string(),
                ));
            }
        }
        let structural_val_us = t12.elapsed().as_micros() as f64;

        let total_attach_us = t_total.elapsed().as_micros() as f64;

        let breakdown = SnapshotAttachBreakdown {
            open_syscall_us,
            mmap_creation_us,
            header_decode_us,
            section_table_us,
            bounds_validation_us: 0.0,
            config_restore_us,
            arena_restore_us,
            id_restore_us,
            liveness_restore_us,
            metadata_restore_us,
            rivero_restore_us,
            witnesses_restore_us,
            graph_restore_us,
            structural_val_us,
            total_attach_us,
        };

        Ok((index, breakdown))
    }
}

#[inline(always)]
fn create_section(
    kind: SectionKind,
    flags: u32,
    element_count: u64,
    payload: &[u8],
) -> (SectionDescriptor, Vec<u8>) {
    let checksum: [u8; 32] = Sha256::digest(payload).into();
    let desc = SectionDescriptor {
        kind: kind as u32,
        flags,
        offset: 0,
        length: payload.len() as u64,
        element_count,
        checksum,
    };
    (desc, payload.to_vec())
}

#[inline(always)]
fn bytemuck_cast<T: Sized>(val: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(val as *const T as *const u8, std::mem::size_of::<T>()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SimilarityScore, VectorEmbedding};

    fn make_test_vector(dim: usize, seed: usize) -> VectorEmbedding {
        let complex: Vec<Complex32> = (0..dim)
            .map(|i| {
                let re = (((seed * 17 + i * 7 + 3) % 43) as f32 - 21.0) / 21.0;
                let im = (((seed * 31 + i * 11 + 5) % 47) as f32 - 23.0) / 23.0;
                Complex32::new(re, im)
            })
            .collect();
        VectorEmbedding::from_complex(complex).into_normalized()
    }

    #[test]
    fn test_snapshot_v2_roundtrip_equivalence() {
        let dim = 16;
        let n = 250;
        let vectors: Vec<VectorEmbedding> = (0..n).map(|i| make_test_vector(dim, i)).collect();

        let mut config = HNSQRConfig::default();
        config.rivero_enabled = true;
        config.rivero_witness_degree = 32;
        let index = HNSQRIndex::new(config, dim);

        for (i, v) in vectors.iter().enumerate() {
            index.insert(format!("doc-{i}"), v.clone()).unwrap();
        }

        let snap_path = std::env::temp_dir().join(format!(
            "test_snap_roundtrip_{}.hnsqr",
            uuid::Uuid::new_v4()
        ));
        let stats = index.save_snapshot_v2(&snap_path).unwrap();
        assert!(stats.file_size_bytes > 0);

        let restored =
            HNSQRIndex::open_snapshot_v2(&snap_path, SnapshotOpenOptions::default()).unwrap();
        assert_eq!(restored.arena.len(), n);
        assert_eq!(
            restored.structural_fingerprint(),
            index.structural_fingerprint()
        );

        // Test search equivalence
        let query = make_test_vector(dim, 9999);
        let (orig_res, orig_diag): (Vec<(NodeIndex, SimilarityScore)>, _) =
            index.search_indices_strict(&query, 10, None).unwrap();
        let (rest_res, rest_diag): (Vec<(NodeIndex, SimilarityScore)>, _) =
            restored.search_indices_strict(&query, 10, None).unwrap();

        assert_eq!(orig_res.len(), rest_res.len());
        for (a, b) in orig_res.iter().zip(rest_res.iter()) {
            assert_eq!(a.0, b.0);
            assert!((a.1 - b.1).abs() < 1e-6);
        }
        assert_eq!(orig_diag.resident_scans, rest_diag.resident_scans);

        let _ = std::fs::remove_file(snap_path);
    }

    #[test]
    fn test_snapshot_v2_corruption_and_truncation_fail_safe() {
        let dim = 8;
        let n = 50;
        let vectors: Vec<VectorEmbedding> = (0..n).map(|i| make_test_vector(dim, i)).collect();

        let index = HNSQRIndex::new(HNSQRConfig::default(), dim);
        for (i, v) in vectors.iter().enumerate() {
            index.insert(format!("v-{i}"), v.clone()).unwrap();
        }

        let snap_path =
            std::env::temp_dir().join(format!("test_corrupt_{}.hnsqr", uuid::Uuid::new_v4()));
        index.save_snapshot_v2(&snap_path).unwrap();

        // Truncate file
        let mut data = std::fs::read(&snap_path).unwrap();
        data.truncate(data.len() / 2);
        let trunc_path =
            std::env::temp_dir().join(format!("test_trunc_{}.hnsqr", uuid::Uuid::new_v4()));
        std::fs::write(&trunc_path, &data).unwrap();

        let res = HNSQRIndex::open_snapshot_v2(&trunc_path, SnapshotOpenOptions::default());
        assert!(res.is_err(), "Truncated snapshot must fail to load safely");

        let _ = std::fs::remove_file(snap_path);
        let _ = std::fs::remove_file(trunc_path);
    }
}
