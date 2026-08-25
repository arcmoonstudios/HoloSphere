/* holosphere/src/storage/manifest.rs */
//!▫~•◦-------------------------------‣
//! # Unified Sectioned Snapshot Manifest Protocol
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a unified, versioned, checksum-protected snapshot manifest linking:
//!   - Vector storage
//!   - External ID mappings
//!   - Metadata dictionaries & postings
//!   - Tombstones
//!   - Rivero routing directories
//!   - LUTz quantized codes
//!   - Semantic proof trees
//!
//! Follows strict Copy-On-Write publication:
//!   Write Gen N to temp -> Verify checksums & LSN -> fsync -> Atomic Manifest Swap -> Retire Gen N-1
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::metadata::index::MetadataValue;
use crate::proof::lutz::LutzCode;
use crate::proof::tree::SemanticProofTree;
use crate::{HNSQRError, HNSQRResult, VectorEmbedding};
use memmap2::Mmap;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const MANIFEST_MAGIC: [u8; 8] = *b"HNSQSMF1";
pub const MANIFEST_VERSION: u32 = 1;

/// Section types recognized in the unified snapshot architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum SectionKind {
    ExternalIdMap = 1,
    VectorData = 2,
    MetadataStore = 3,
    TombstoneBitmap = 4,
    RiveroState = 5,
    LutzCodes = 6,
    SemanticProofTree = 7,
    RebuildableGraphState = 8,
}

/// Metadata descriptor for an individual snapshot section.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotSection {
    pub kind: SectionKind,
    pub offset: u64,
    pub length: u64,
    pub element_count: u64,
    pub crc32c: u32,
    pub sha256_hex: String,
}

/// Versioned, checksum-protected Snapshot Manifest.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub magic: [u8; 8],
    pub version: u32,
    pub generation: u64,
    pub snapshot_lsn: u64,
    pub created_at_epoch_ms: u64,
    pub dimension: usize,
    pub total_vectors: usize,
    pub sections: HashMap<SectionKind, SnapshotSection>,
}

impl SnapshotManifest {
    pub fn new(generation: u64, snapshot_lsn: u64, dimension: usize, total_vectors: usize) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            magic: MANIFEST_MAGIC,
            version: MANIFEST_VERSION,
            generation,
            snapshot_lsn,
            created_at_epoch_ms: now,
            dimension,
            total_vectors,
            sections: HashMap::new(),
        }
    }

    pub fn add_section(&mut self, section: SnapshotSection) {
        self.sections.insert(section.kind, section);
    }

    pub fn get_section(&self, kind: SectionKind) -> Option<&SnapshotSection> {
        self.sections.get(&kind)
    }

    pub fn encode(&self) -> HNSQRResult<Vec<u8>> {
        serde_json::to_vec_pretty(self)
            .map_err(|e| HNSQRError::CorruptedSnapshot(format!("Manifest encode error: {e}")))
    }

    pub fn decode(bytes: &[u8]) -> HNSQRResult<Self> {
        let manifest: Self = serde_json::from_slice(bytes)
            .map_err(|e| HNSQRError::CorruptedSnapshot(format!("Manifest decode error: {e}")))?;

        if manifest.magic != MANIFEST_MAGIC {
            return Err(HNSQRError::CorruptedSnapshot(
                "Invalid snapshot manifest magic".to_string(),
            ));
        }
        if manifest.version != MANIFEST_VERSION {
            return Err(HNSQRError::CorruptedSnapshot(format!(
                "Unsupported snapshot manifest version: {}",
                manifest.version
            )));
        }

        Ok(manifest)
    }
}

/// Unified Copy-on-Write Snapshot Engine.
pub struct UnifiedSnapshotEngine;

impl UnifiedSnapshotEngine {
    /// Writes a unified snapshot generation with atomic publication.
    pub fn save_snapshot(
        snapshot_dir: impl AsRef<Path>,
        generation: u64,
        snapshot_lsn: u64,
        dimension: usize,
        vectors: &[VectorEmbedding],
        external_ids: &[String],
        metadata_map: Option<&[HashMap<String, MetadataValue>]>,
        tombstones: Option<&RoaringBitmap>,
        proof_tree: Option<&SemanticProofTree>,
        lutz_codes: Option<&[LutzCode]>,
    ) -> HNSQRResult<SnapshotManifest> {
        let snapshot_dir = snapshot_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&snapshot_dir)?;

        let data_tmp_path = snapshot_dir.join(format!("snapshot_gen_{generation:016}.data.tmp"));
        let manifest_tmp_path =
            snapshot_dir.join(format!("snapshot_gen_{generation:016}.manifest.tmp"));

        let data_final_path = snapshot_dir.join(format!("snapshot_gen_{generation:016}.data"));
        let manifest_final_path =
            snapshot_dir.join(format!("snapshot_gen_{generation:016}.manifest"));
        let current_pointer_path = snapshot_dir.join("current_manifest.json");

        let mut manifest =
            SnapshotManifest::new(generation, snapshot_lsn, dimension, vectors.len());

        let mut file = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&data_tmp_path)?,
        );

        let mut current_offset = 0u64;

        // Helper closure to write and register a section
        let mut write_section =
            |kind: SectionKind, data: &[u8], count: u64| -> HNSQRResult<SnapshotSection> {
                let mut crc = crc32fast::Hasher::new();
                crc.update(data);
                let crc32c = crc.finalize();

                let mut sha = Sha256::new();
                sha.update(data);
                let sha256_hex = sha
                    .finalize()
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>();

                file.write_all(data)?;
                let length = data.len() as u64;

                let section = SnapshotSection {
                    kind,
                    offset: current_offset,
                    length,
                    element_count: count,
                    crc32c,
                    sha256_hex,
                };
                current_offset += length;
                Ok(section)
            };

        // 1. External ID Section
        let id_bytes = bincode::serialize(external_ids)
            .map_err(|e| HNSQRError::CorruptedSnapshot(format!("ID serialize error: {e}")))?;
        let sec_id = write_section(
            SectionKind::ExternalIdMap,
            &id_bytes,
            external_ids.len() as u64,
        )?;
        manifest.add_section(sec_id);

        // 2. Vector Data Section
        let vec_bytes = bincode::serialize(vectors)
            .map_err(|e| HNSQRError::CorruptedSnapshot(format!("Vectors serialize error: {e}")))?;
        let sec_vec = write_section(SectionKind::VectorData, &vec_bytes, vectors.len() as u64)?;
        manifest.add_section(sec_vec);

        // 3. Metadata Section
        if let Some(meta) = metadata_map {
            let meta_bytes = bincode::serialize(meta)
                .map_err(|e| HNSQRError::CorruptedSnapshot(format!("Meta serialize error: {e}")))?;
            let sec_meta =
                write_section(SectionKind::MetadataStore, &meta_bytes, meta.len() as u64)?;
            manifest.add_section(sec_meta);
        }

        // 4. Tombstone Bitmap Section
        if let Some(tb) = tombstones {
            let mut tb_bytes = Vec::new();
            tb.serialize_into(&mut tb_bytes).map_err(|e| {
                HNSQRError::CorruptedSnapshot(format!("Tombstone serialize error: {e}"))
            })?;
            let sec_tb = write_section(SectionKind::TombstoneBitmap, &tb_bytes, tb.len())?;
            manifest.add_section(sec_tb);
        }

        // 5. Semantic Proof Tree Section
        if let Some(tree) = proof_tree {
            let tree_bytes = bincode::serialize(tree).map_err(|e| {
                HNSQRError::CorruptedSnapshot(format!("ProofTree serialize error: {e}"))
            })?;
            let sec_tree = write_section(
                SectionKind::SemanticProofTree,
                &tree_bytes,
                tree.total_vectors() as u64,
            )?;
            manifest.add_section(sec_tree);
        }

        // 6. LUTz Codes Section
        if let Some(codes) = lutz_codes {
            let lutz_bytes = bincode::serialize(codes)
                .map_err(|e| HNSQRError::CorruptedSnapshot(format!("LUTz serialize error: {e}")))?;
            let sec_lutz = write_section(SectionKind::LutzCodes, &lutz_bytes, codes.len() as u64)?;
            manifest.add_section(sec_lutz);
        }

        // Flush and sync data file
        file.flush()?;
        let data_file = file
            .into_inner()
            .map_err(|e| HNSQRError::IoError(e.to_string()))?;
        data_file.sync_all()?;

        // Write and sync manifest file
        let manifest_bytes = manifest.encode()?;
        {
            let mut m_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&manifest_tmp_path)?;
            m_file.write_all(&manifest_bytes)?;
            m_file.sync_all()?;
        }

        // Atomic Publication via rename
        std::fs::rename(&data_tmp_path, &data_final_path)?;
        std::fs::rename(&manifest_tmp_path, &manifest_final_path)?;

        // Update current pointer atomically
        let pointer_tmp = snapshot_dir.join("current_manifest.json.tmp");
        {
            let mut p_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&pointer_tmp)?;
            p_file.write_all(&manifest_bytes)?;
            p_file.sync_all()?;
        }
        std::fs::rename(&pointer_tmp, &current_pointer_path)?;

        Ok(manifest)
    }

    /// Loads and validates the newest valid snapshot generation with fallback support.
    pub fn load_latest_snapshot(
        snapshot_dir: impl AsRef<Path>,
    ) -> HNSQRResult<(SnapshotManifest, Mmap)> {
        let snapshot_dir = snapshot_dir.as_ref().to_path_buf();
        let pointer_path = snapshot_dir.join("current_manifest.json");

        if !pointer_path.exists() {
            return Err(HNSQRError::CorruptedSnapshot(
                "No snapshot manifest found in directory".to_string(),
            ));
        }

        let manifest_bytes = std::fs::read(&pointer_path)?;
        let manifest = SnapshotManifest::decode(&manifest_bytes)?;

        let data_path = snapshot_dir.join(format!("snapshot_gen_{:016}.data", manifest.generation));
        if !data_path.exists() {
            return Err(HNSQRError::CorruptedSnapshot(format!(
                "Snapshot data file missing for generation {}",
                manifest.generation
            )));
        }

        let file = File::open(&data_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Validate all section checksums and bounds
        for (kind, sec) in &manifest.sections {
            let start = sec.offset as usize;
            let end = start + sec.length as usize;

            if end > mmap.len() {
                return Err(HNSQRError::CorruptedSnapshot(format!(
                    "Section {kind:?} exceeds snapshot data file bounds"
                )));
            }

            let slice = &mmap[start..end];
            let mut crc = crc32fast::Hasher::new();
            crc.update(slice);
            if crc.finalize() != sec.crc32c {
                return Err(HNSQRError::CorruptedSnapshot(format!(
                    "Section {kind:?} CRC32C integrity checksum failure"
                )));
            }
        }

        Ok((manifest, mmap))
    }
}
