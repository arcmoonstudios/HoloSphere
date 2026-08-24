/* holosphere/src/entity/provenance.rs */
//!▫~•◦-------------------------------‣
//! # Provenance Arena & Immutable Storage
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the immutable provenance storage kernel. Enables HoloSphere to
//! deterministically trace "Why do we believe this?" by traversing structural
//! evidence chains without reverse-engineering explanations from unstructured text.
//!
//! ## Invariant Guarantees
//! - Provenance rows are immutable once appended: row #N never mutates in-place.
//! - Evidence references use stable durable IDs (`DurableEvidenceRef`), surviving compactions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use bytemuck::{Pod, Zeroable};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entity::id::{DurableEvidenceRef, NULL_ROW_REF, ProvenanceId, ProvenanceIndex};

/// Exactly-80-byte, deterministic, padding-free, mmap-safe provenance row.
///
/// Layout (80 bytes, 8-byte aligned):
/// ```text
/// offset 0  — commit_lsn           : u64      (8 bytes)
/// offset 8  — timestamp_ms         : u64      (8 bytes)
/// offset 16 — signature_hash       : [u8; 32] (32 bytes)
/// offset 48 — source_uri_id        : u32      (4 bytes)
/// offset 52 — actor_id             : u32      (4 bytes)
/// offset 56 — extraction_method_id : u32      (4 bytes)
/// offset 60 — evidence_start       : u32      (4 bytes)
/// offset 64 — evidence_len         : u32      (4 bytes)
/// offset 68 — confidence_q16       : u32      (4 bytes) ← fixed-point Q16 (65536 = 1.0)
/// offset 72 — flags                : u32      (4 bytes)
/// offset 76 — reserved             : u32      (4 bytes)
/// total      80 bytes, no padding
/// ```
#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ProvenanceRow {
    pub commit_lsn: u64,
    pub timestamp_ms: u64,
    pub signature_hash: [u8; 32],

    pub source_uri_id: u32,
    pub actor_id: u32,
    pub extraction_method_id: u32,
    pub evidence_start: u32,
    pub evidence_len: u32,

    pub confidence_q16: u32,
    pub flags: u32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<ProvenanceRow>() == 80);
const _: () = assert!(std::mem::align_of::<ProvenanceRow>() == 8);

impl Default for ProvenanceRow {
    fn default() -> Self {
        Self {
            commit_lsn: 0,
            timestamp_ms: 0,
            signature_hash: [0u8; 32],
            source_uri_id: NULL_ROW_REF,
            actor_id: NULL_ROW_REF,
            extraction_method_id: NULL_ROW_REF,
            evidence_start: 0,
            evidence_len: 0,
            confidence_q16: 65536, // 1.0 in Q16
            flags: 0,
            reserved: 0,
        }
    }
}

impl ProvenanceRow {
    #[inline(always)]
    pub fn confidence_f32(&self) -> f32 {
        (self.confidence_q16 as f32) / 65536.0
    }

    #[inline(always)]
    pub fn set_confidence_f32(&mut self, conf: f32) {
        let clamped = conf.clamp(0.0, 1.0);
        self.confidence_q16 = (clamped * 65536.0).round() as u32;
    }
}

/// High-level ergonomic representation of an immutable provenance record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    pub source_uri: Arc<str>,
    pub actor_id: Arc<str>,
    pub extraction_method: Arc<str>,
    pub commit_lsn: u64,
    pub timestamp_ms: u64,
    pub confidence: f32,
    pub evidence: Vec<DurableEvidenceRef>,
    pub signature_hash: [u8; 32],
}

/// Append-only, thread-safe memory arena for storing immutable provenance records and evidence chains.
pub struct ProvenanceArena {
    rows: RwLock<Vec<ProvenanceRow>>,
    id_to_index: RwLock<HashMap<ProvenanceId, ProvenanceIndex>>,
    index_to_id: RwLock<Vec<ProvenanceId>>,
    evidence_pool: RwLock<Vec<DurableEvidenceRef>>,
    interned_strings: RwLock<Vec<Arc<str>>>,
    string_to_id: RwLock<HashMap<Arc<str>, u32>>,
    next_id: AtomicU64,
}

impl Default for ProvenanceArena {
    fn default() -> Self {
        Self::new(1)
    }
}

impl ProvenanceArena {
    pub fn new(start_id: u64) -> Self {
        Self {
            rows: RwLock::new(Vec::new()),
            id_to_index: RwLock::new(HashMap::new()),
            index_to_id: RwLock::new(Vec::new()),
            evidence_pool: RwLock::new(Vec::new()),
            interned_strings: RwLock::new(Vec::new()),
            string_to_id: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(start_id),
        }
    }

    pub fn intern_string(&self, s: &Arc<str>) -> u32 {
        let r = self.string_to_id.read();
        if let Some(&id) = r.get(s) {
            return id;
        }
        drop(r);

        let mut w_map = self.string_to_id.write();
        if let Some(&id) = w_map.get(s) {
            return id;
        }

        let mut w_vec = self.interned_strings.write();
        let id = w_vec.len() as u32;
        w_vec.push(Arc::clone(s));
        w_map.insert(Arc::clone(s), id);
        id
    }

    pub fn lookup_string(&self, id: u32) -> Option<Arc<str>> {
        let r = self.interned_strings.read();
        r.get(id as usize).cloned()
    }

    /// Appends a new immutable provenance record, generating a new `ProvenanceId`.
    pub fn append(&self, record: &ProvenanceRecord) -> (ProvenanceId, ProvenanceIndex) {
        let prov_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let prov_index = self.bind(prov_id, record);
        (prov_id, prov_index)
    }

    /// Binds an existing durable `ProvenanceId` to a new generation row.
    pub fn bind(&self, prov_id: ProvenanceId, record: &ProvenanceRecord) -> ProvenanceIndex {
        let source_uri_id = self.intern_string(&record.source_uri);
        let actor_id = self.intern_string(&record.actor_id);
        let extraction_method_id = self.intern_string(&record.extraction_method);

        let (evidence_start, evidence_len) = if !record.evidence.is_empty() {
            let mut pool = self.evidence_pool.write();
            let start = pool.len() as u32;
            pool.extend_from_slice(&record.evidence);
            (start, record.evidence.len() as u32)
        } else {
            (0, 0)
        };

        let mut row = ProvenanceRow {
            commit_lsn: record.commit_lsn,
            timestamp_ms: record.timestamp_ms,
            signature_hash: record.signature_hash,
            source_uri_id,
            actor_id,
            extraction_method_id,
            evidence_start,
            evidence_len,
            confidence_q16: 0,
            flags: 0,
            reserved: 0,
        };
        row.set_confidence_f32(record.confidence);

        let mut rows = self.rows.write();
        let mut id_map = self.id_to_index.write();
        let mut idx_vec = self.index_to_id.write();

        let index = rows.len() as ProvenanceIndex;
        rows.push(row);
        idx_vec.push(prov_id);
        id_map.insert(prov_id, index);

        index
    }

    /// Resolves `ProvenanceId` to generation-local `ProvenanceIndex`.
    #[inline]
    pub fn id_to_index(&self, id: ProvenanceId) -> Option<ProvenanceIndex> {
        self.id_to_index.read().get(&id).copied()
    }

    /// Resolves generation-local `ProvenanceIndex` to durable `ProvenanceId`.
    #[inline]
    pub fn index_to_id(&self, index: ProvenanceIndex) -> Option<ProvenanceId> {
        self.index_to_id.read().get(index as usize).copied()
    }

    /// Retrieves a raw `ProvenanceRow` by row index.
    pub fn get_row(&self, row_index: ProvenanceIndex) -> Option<ProvenanceRow> {
        let rows = self.rows.read();
        rows.get(row_index as usize).copied()
    }

    /// Resolves a full `ProvenanceRecord` by row index.
    pub fn resolve_record(&self, row_index: ProvenanceIndex) -> Option<ProvenanceRecord> {
        let row = self.get_row(row_index)?;
        let source_uri = self.lookup_string(row.source_uri_id)?;
        let actor_id = self.lookup_string(row.actor_id)?;
        let extraction_method = self.lookup_string(row.extraction_method_id)?;

        let evidence = if row.evidence_len > 0 {
            let pool = self.evidence_pool.read();
            let start = row.evidence_start as usize;
            let end = start + (row.evidence_len as usize);
            if end <= pool.len() {
                pool[start..end].to_vec()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        Some(ProvenanceRecord {
            source_uri,
            actor_id,
            extraction_method,
            commit_lsn: row.commit_lsn,
            timestamp_ms: row.timestamp_ms,
            confidence: row.confidence_f32(),
            evidence,
            signature_hash: row.signature_hash,
        })
    }

    /// Resolves a full `ProvenanceRecord` by durable `ProvenanceId`.
    pub fn resolve_record_by_id(&self, id: ProvenanceId) -> Option<ProvenanceRecord> {
        let row_idx = self.id_to_index(id)?;
        self.resolve_record(row_idx)
    }

    /// Resolves a full `ProvenanceRecord` by durable `ProvenanceId`.
    pub fn resolve_by_id(&self, id: ProvenanceId) -> Option<ProvenanceRecord> {
        let index = self.id_to_index(id)?;
        self.resolve_record(index)
    }

    /// Total count of provenance rows in arena.
    pub fn len(&self) -> usize {
        self.rows.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Internal snapshot of rows, id mappings, strings, and evidence for serialization / compaction.
    pub fn snapshot_data(
        &self,
    ) -> (
        Vec<ProvenanceRow>,
        Vec<ProvenanceId>,
        Vec<DurableEvidenceRef>,
        Vec<Arc<str>>,
    ) {
        (
            self.rows.read().clone(),
            self.index_to_id.read().clone(),
            self.evidence_pool.read().clone(),
            self.interned_strings.read().clone(),
        )
    }
}
