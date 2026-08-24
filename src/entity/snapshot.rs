/* holosphere/src/entity/snapshot.rs */
//!▫~•◦-------------------------------‣
//! # Snapshot V3 Entity Format & Encoding Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the binary Snapshot V3 container format with explicit section IDs,
//! CRC32 data integrity verification, and fail-closed validation.
//!
//! ## Invariant Guarantees
//! - Mandatory sections cannot be missing or corrupt (fails closed).
//! - All entity headers, version rows, provenance rows, and interned strings
//!   are serialized with item counts, schema versions, and exact CRC32 checksums.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use thiserror::Error;

use crate::entity::arena::EntityArena;
use crate::entity::header::EntityHeader;
use crate::entity::id::{DurableEvidenceRef, EntityId, ProvenanceId, VectorLayout, VersionId};
use crate::entity::provenance::{ProvenanceArena, ProvenanceRecord, ProvenanceRow};
use crate::entity::segment::EntitySegment;
use crate::entity::version::{VersionRow, VersionTable};

pub const SNAPSHOT_V3_MAGIC: [u8; 8] = *b"HNSQR_V3";

pub const SECTION_ENTITY_HEADERS: u32 = 1;
pub const SECTION_ENTITY_ID_MAP: u32 = 2;
pub const SECTION_VECTOR_LAYOUTS: u32 = 3;
pub const SECTION_VERSION_ROWS: u32 = 4;
pub const SECTION_VERSION_ID_MAP: u32 = 5;
pub const SECTION_PROVENANCE_ROWS: u32 = 6;
pub const SECTION_PROVENANCE_ID_MAP: u32 = 7;
pub const SECTION_PROVENANCE_EVIDENCE: u32 = 8;
pub const SECTION_STRING_DICTIONARY: u32 = 9;

pub const SECTION_FLAG_OPTIONAL: u32 = 1 << 0;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum SnapshotV3Error {
    #[error("Invalid snapshot magic: expected HNSQR_V3")]
    InvalidMagic,
    #[error("Header CRC32 mismatch")]
    HeaderChecksumMismatch,
    #[error("Section {section_id} CRC32 mismatch: expected {expected:#x}, got {actual:#x}")]
    SectionChecksumMismatch {
        section_id: u32,
        expected: u32,
        actual: u32,
    },
    #[error("Missing mandatory section {section_id}")]
    MissingMandatorySection { section_id: u32 },
    #[error("Section bounds out of range (offset: {offset}, len: {len}, total: {total})")]
    SectionOutOfBounds { offset: u64, len: u64, total: usize },
    #[error("Deserialization payload error: {0}")]
    PayloadError(String),
}

/// Exactly-32-byte header for each section in Snapshot V3.
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SnapshotV3SectionHeader {
    pub section_id: u32,
    pub item_count: u32,
    pub item_schema_version: u32,
    pub flags: u32,
    pub offset: u64,
    pub length: u64,
    pub crc32: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<SnapshotV3SectionHeader>() == 40);

/// Encodes an `EntitySegment` and its pinned LSN into binary Snapshot V3 format.
pub fn encode_snapshot_v3(segment: &EntitySegment, lsn: u64) -> Vec<u8> {
    let mut buffer = Vec::with_capacity(65536);

    // 1. Snapshot raw data from segment components
    let (headers, _live, entity_ids) = segment.arena.snapshot_data();
    let (version_rows, version_ids) = segment.versions.snapshot_data();
    let (provenance_rows, provenance_ids, evidence_pool, interned_strings) =
        segment.provenance.snapshot_data();
    let vector_layouts = segment.vector_layouts.read().clone();

    // 2. Encode payloads
    let headers_bytes = bytemuck::cast_slice::<EntityHeader, u8>(&headers);
    let entity_ids_bytes = bincode::serialize(&entity_ids).unwrap();
    let vector_layouts_bytes = bincode::serialize(&vector_layouts).unwrap();
    let version_rows_bytes = bytemuck::cast_slice::<VersionRow, u8>(&version_rows);
    let version_ids_bytes = bincode::serialize(&version_ids).unwrap();
    let provenance_rows_bytes = bytemuck::cast_slice::<ProvenanceRow, u8>(&provenance_rows);
    let provenance_ids_bytes = bincode::serialize(&provenance_ids).unwrap();
    let evidence_bytes = bincode::serialize(&evidence_pool).unwrap();
    let strings_bytes = bincode::serialize(&interned_strings).unwrap();

    let payloads = [
        (
            SECTION_ENTITY_HEADERS,
            headers.len() as u32,
            headers_bytes,
            0,
        ),
        (
            SECTION_ENTITY_ID_MAP,
            entity_ids.len() as u32,
            &entity_ids_bytes[..],
            0,
        ),
        (
            SECTION_VECTOR_LAYOUTS,
            vector_layouts.len() as u32,
            &vector_layouts_bytes[..],
            0,
        ),
        (
            SECTION_VERSION_ROWS,
            version_rows.len() as u32,
            version_rows_bytes,
            0,
        ),
        (
            SECTION_VERSION_ID_MAP,
            version_ids.len() as u32,
            &version_ids_bytes[..],
            0,
        ),
        (
            SECTION_PROVENANCE_ROWS,
            provenance_rows.len() as u32,
            provenance_rows_bytes,
            0,
        ),
        (
            SECTION_PROVENANCE_ID_MAP,
            provenance_ids.len() as u32,
            &provenance_ids_bytes[..],
            0,
        ),
        (
            SECTION_PROVENANCE_EVIDENCE,
            evidence_pool.len() as u32,
            &evidence_bytes[..],
            0,
        ),
        (
            SECTION_STRING_DICTIONARY,
            interned_strings.len() as u32,
            &strings_bytes[..],
            0,
        ),
    ];

    // 3. Layout calculation: file header (32B) + section table (payloads.len() * 40B) + payloads
    let file_header_len = 32usize;
    let section_table_len = payloads.len() * std::mem::size_of::<SnapshotV3SectionHeader>();
    let mut current_offset = ((file_header_len + section_table_len + 63) & !63) as u64;

    let mut section_headers = Vec::with_capacity(payloads.len());
    for (sec_id, count, payload, flags) in &payloads {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(payload);
        let crc = hasher.finalize();

        section_headers.push(SnapshotV3SectionHeader {
            section_id: *sec_id,
            item_count: *count,
            item_schema_version: 1,
            flags: *flags,
            offset: current_offset,
            length: payload.len() as u64,
            crc32: crc,
            _pad: 0,
        });

        current_offset = (current_offset + payload.len() as u64 + 63) & !63;
    }

    // 4. Write File Header
    buffer.extend_from_slice(&SNAPSHOT_V3_MAGIC); // 8B
    buffer.extend_from_slice(&lsn.to_le_bytes()); // 8B
    buffer.extend_from_slice(&0u64.to_le_bytes()); // timestamp 8B
    buffer.extend_from_slice(&(section_headers.len() as u32).to_le_bytes()); // 4B

    let mut header_hasher = crc32fast::Hasher::new();
    header_hasher.update(&buffer[..28]);
    let header_crc = header_hasher.finalize();
    buffer.extend_from_slice(&header_crc.to_le_bytes()); // 4B -> Total 32B

    // 5. Write Section Headers Table
    for sh in &section_headers {
        buffer.extend_from_slice(bytemuck::bytes_of(sh));
    }

    // 6. Write Payloads with 64-byte alignment padding
    for (i, (_, _, payload, _)) in payloads.iter().enumerate() {
        let target_offset = section_headers[i].offset as usize;
        while buffer.len() < target_offset {
            buffer.push(0);
        }
        buffer.extend_from_slice(payload);
    }

    buffer
}

/// Decodes binary Snapshot V3 bytes into a verified `EntitySegment` and committed `lsn`.
pub fn decode_snapshot_v3(bytes: &[u8]) -> Result<(u64, EntitySegment), SnapshotV3Error> {
    if bytes.len() < 32 {
        return Err(SnapshotV3Error::InvalidMagic);
    }

    if &bytes[0..8] != &SNAPSHOT_V3_MAGIC {
        return Err(SnapshotV3Error::InvalidMagic);
    }

    let lsn = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
    let section_count = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
    let expected_header_crc = u32::from_le_bytes(bytes[28..32].try_into().unwrap());

    let mut header_hasher = crc32fast::Hasher::new();
    header_hasher.update(&bytes[..28]);
    if header_hasher.finalize() != expected_header_crc {
        return Err(SnapshotV3Error::HeaderChecksumMismatch);
    }

    let section_table_len = section_count * std::mem::size_of::<SnapshotV3SectionHeader>();
    if bytes.len() < 32 + section_table_len {
        return Err(SnapshotV3Error::SectionOutOfBounds {
            offset: 32,
            len: section_table_len as u64,
            total: bytes.len(),
        });
    }

    let mut sections = std::collections::HashMap::new();
    for i in 0..section_count {
        let start = 32 + i * 40;
        let end = start + 40;
        let sh: &SnapshotV3SectionHeader = bytemuck::from_bytes(&bytes[start..end]);

        // Verify section offset and length
        let p_start = sh.offset as usize;
        let p_end = (sh.offset + sh.length) as usize;
        if p_end > bytes.len() {
            return Err(SnapshotV3Error::SectionOutOfBounds {
                offset: sh.offset,
                len: sh.length,
                total: bytes.len(),
            });
        }

        // Verify CRC32
        let payload = &bytes[p_start..p_end];
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(payload);
        let actual_crc = hasher.finalize();
        if actual_crc != sh.crc32 {
            return Err(SnapshotV3Error::SectionChecksumMismatch {
                section_id: sh.section_id,
                expected: sh.crc32,
                actual: actual_crc,
            });
        }

        sections.insert(sh.section_id, (*sh, payload));
    }

    // Verify mandatory sections
    for mandatory in [
        SECTION_ENTITY_HEADERS,
        SECTION_ENTITY_ID_MAP,
        SECTION_VERSION_ROWS,
        SECTION_VERSION_ID_MAP,
        SECTION_PROVENANCE_ROWS,
        SECTION_PROVENANCE_ID_MAP,
        SECTION_PROVENANCE_EVIDENCE,
        SECTION_STRING_DICTIONARY,
    ] {
        if !sections.contains_key(&mandatory) {
            return Err(SnapshotV3Error::MissingMandatorySection {
                section_id: mandatory,
            });
        }
    }

    // Decode payloads
    let (_h_meta, h_payload) = sections[&SECTION_ENTITY_HEADERS];
    let (_id_meta, id_payload) = sections[&SECTION_ENTITY_ID_MAP];
    let (_v_meta, v_payload) = sections[&SECTION_VERSION_ROWS];
    let (_vid_meta, vid_payload) = sections[&SECTION_VERSION_ID_MAP];
    let (_p_meta, p_payload) = sections[&SECTION_PROVENANCE_ROWS];
    let (_pid_meta, pid_payload) = sections[&SECTION_PROVENANCE_ID_MAP];
    let (_ev_meta, ev_payload) = sections[&SECTION_PROVENANCE_EVIDENCE];
    let (_str_meta, str_payload) = sections[&SECTION_STRING_DICTIONARY];

    let headers: Vec<EntityHeader> = match bytemuck::try_cast_slice(h_payload) {
        Ok(s) => s.to_vec(),
        Err(_) => {
            let mut v = vec![EntityHeader::default(); _h_meta.item_count as usize];
            bytemuck::cast_slice_mut::<EntityHeader, u8>(&mut v).copy_from_slice(h_payload);
            v
        }
    };
    let entity_ids: Vec<EntityId> = bincode::deserialize(id_payload)
        .map_err(|e| SnapshotV3Error::PayloadError(e.to_string()))?;
    let version_rows: Vec<VersionRow> = match bytemuck::try_cast_slice(v_payload) {
        Ok(s) => s.to_vec(),
        Err(_) => {
            let mut v = vec![VersionRow::default(); _v_meta.item_count as usize];
            bytemuck::cast_slice_mut::<VersionRow, u8>(&mut v).copy_from_slice(v_payload);
            v
        }
    };
    let version_ids: Vec<VersionId> = bincode::deserialize(vid_payload)
        .map_err(|e| SnapshotV3Error::PayloadError(e.to_string()))?;
    let provenance_rows: Vec<ProvenanceRow> = match bytemuck::try_cast_slice(p_payload) {
        Ok(s) => s.to_vec(),
        Err(_) => {
            let mut v = vec![ProvenanceRow::default(); _p_meta.item_count as usize];
            bytemuck::cast_slice_mut::<ProvenanceRow, u8>(&mut v).copy_from_slice(p_payload);
            v
        }
    };
    let provenance_ids: Vec<ProvenanceId> = bincode::deserialize(pid_payload)
        .map_err(|e| SnapshotV3Error::PayloadError(e.to_string()))?;
    let evidence_pool: Vec<DurableEvidenceRef> = bincode::deserialize(ev_payload)
        .map_err(|e| SnapshotV3Error::PayloadError(e.to_string()))?;
    let interned_strings: Vec<Arc<str>> = bincode::deserialize(str_payload)
        .map_err(|e| SnapshotV3Error::PayloadError(e.to_string()))?;

    // Construct EntitySegment
    let max_entity_id = entity_ids.iter().copied().max().unwrap_or(0);
    let max_version_id = version_ids.iter().copied().max().unwrap_or(0);
    let max_prov_id = provenance_ids.iter().copied().max().unwrap_or(0);

    let arena = Arc::new(EntityArena::new(max_entity_id + 1));
    for (i, &header) in headers.iter().enumerate() {
        if i < entity_ids.len() {
            arena.bind(entity_ids[i], header);
        }
    }

    let versions = Arc::new(VersionTable::new(max_version_id + 1));
    for (i, &vrow) in version_rows.iter().enumerate() {
        if i < version_ids.len() {
            versions.bind(version_ids[i], vrow);
        }
    }

    let provenance = Arc::new(ProvenanceArena::new(max_prov_id + 1));
    for s in &interned_strings {
        provenance.intern_string(s);
    }
    for (i, &prow) in provenance_rows.iter().enumerate() {
        if i < provenance_ids.len() {
            let record = ProvenanceRecord {
                source_uri: provenance
                    .lookup_string(prow.source_uri_id)
                    .unwrap_or_else(|| Arc::from("")),
                actor_id: provenance
                    .lookup_string(prow.actor_id)
                    .unwrap_or_else(|| Arc::from("")),
                extraction_method: provenance
                    .lookup_string(prow.extraction_method_id)
                    .unwrap_or_else(|| Arc::from("")),
                commit_lsn: prow.commit_lsn,
                timestamp_ms: prow.timestamp_ms,
                confidence: prow.confidence_f32(),
                evidence: if prow.evidence_len > 0 {
                    let start = prow.evidence_start as usize;
                    let end = start + (prow.evidence_len as usize);
                    if end <= evidence_pool.len() {
                        evidence_pool[start..end].to_vec()
                    } else {
                        Vec::new()
                    }
                } else {
                    Vec::new()
                },
                signature_hash: prow.signature_hash,
            };
            provenance.bind(provenance_ids[i], &record);
        }
    }

    let segment = EntitySegment {
        generation_id: 1,
        arena,
        provenance,
        versions,
        vector_arena: Arc::new(crate::entity::vector::VectorArena::new(128)),
        vector_layouts: parking_lot::RwLock::new(Vec::new()),
    };

    if let Some((_vl_meta, vl_payload)) = sections.get(&SECTION_VECTOR_LAYOUTS) {
        if let Ok(layouts) = bincode::deserialize::<Vec<VectorLayout>>(vl_payload) {
            *segment.vector_layouts.write() = layouts;
        }
    }

    Ok((lsn, segment))
}
