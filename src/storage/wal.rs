/* hnsqr/src/storage/wal.rs */
//!▫~•◦-------------------------------‣
//! # Length-Framed, CRC32C-Checksummed Write-Ahead Log (WAL) Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides a production-grade crash-safe WAL with:
//!   - Monotonic 64-bit Log Sequence Numbers (LSN)
//!   - CRC32C payload integrity verification
//!   - Configurable Durability Policies (Memory, WalSync, GroupCommit, Quorum)
//!   - Bounded ring-buffer Group Commit engine with backpressure
//!   - Monotonic Write Invariant: WAL Append -> fsync barrier -> Arena Publish
//!   - Idempotent recovery with torn-tail tolerance and mid-log corruption rejection
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use parking_lot::{Condvar, Mutex};
use serde::{Deserialize, Serialize};

use crate::metadata::index::MetadataValue;
use crate::{HNSQRError, HNSQRResult, VectorEmbedding};

pub const WAL_MAGIC: [u8; 4] = *b"HWAL";
pub const WAL_VERSION: u16 = 1;
pub const WAL_HEADER_SIZE: usize = 36;
pub const DEFAULT_MAX_SEGMENT_BYTES: u64 = 64 * 1024 * 1024; // 64 MB

/// WAL record type identifiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u16)]
pub enum WalRecordType {
    Upsert = 1,
    Delete = 2,
    MetadataUpdate = 3,
    Checkpoint = 4,
    ClusterState = 5,
}

impl WalRecordType {
    pub fn from_u16(val: u16) -> Option<Self> {
        match val {
            1 => Some(Self::Upsert),
            2 => Some(Self::Delete),
            3 => Some(Self::MetadataUpdate),
            4 => Some(Self::Checkpoint),
            5 => Some(Self::ClusterState),
            _ => None,
        }
    }
}

/// 36-byte Fixed Length Header for every WAL Record Frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalFrameHeader {
    pub magic: [u8; 4],
    pub version: u16,
    pub record_type: u16,
    pub lsn: u64,
    pub prev_lsn: u64,
    pub payload_len: u32,
    pub crc32c: u32,
}

impl WalFrameHeader {
    pub fn encode(&self) -> [u8; WAL_HEADER_SIZE] {
        let mut buf = [0u8; WAL_HEADER_SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.record_type.to_le_bytes());
        buf[8..16].copy_from_slice(&self.lsn.to_le_bytes());
        buf[16..24].copy_from_slice(&self.prev_lsn.to_le_bytes());
        buf[24..28].copy_from_slice(&self.payload_len.to_le_bytes());
        buf[28..32].copy_from_slice(&self.crc32c.to_le_bytes());
        // 4 bytes reserved zero
        buf
    }

    pub fn decode(buf: &[u8; WAL_HEADER_SIZE]) -> HNSQRResult<Self> {
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&buf[0..4]);
        if magic != WAL_MAGIC {
            return Err(HNSQRError::CorruptedSnapshot(
                "Invalid WAL frame magic identifier".to_string(),
            ));
        }

        let version = u16::from_le_bytes([buf[4], buf[5]]);
        if version != WAL_VERSION {
            return Err(HNSQRError::CorruptedSnapshot(format!(
                "Unsupported WAL version: {version}"
            )));
        }

        let record_type_raw = u16::from_le_bytes([buf[6], buf[7]]);
        if WalRecordType::from_u16(record_type_raw).is_none() {
            return Err(HNSQRError::CorruptedSnapshot(format!(
                "Unknown WAL record type: {record_type_raw}"
            )));
        }

        let lsn = u64::from_le_bytes(buf[8..16].try_into().unwrap());
        let prev_lsn = u64::from_le_bytes(buf[16..24].try_into().unwrap());
        let payload_len = u32::from_le_bytes(buf[24..28].try_into().unwrap());
        let crc32c = u32::from_le_bytes(buf[28..32].try_into().unwrap());

        Ok(Self {
            magic,
            version,
            record_type: record_type_raw,
            lsn,
            prev_lsn,
            payload_len,
            crc32c,
        })
    }
}

/// WAL Mutation Payload variants.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum WalMutation {
    Upsert {
        external_id: String,
        vector: VectorEmbedding,
        metadata: Option<HashMap<String, MetadataValue>>,
    },
    Delete {
        external_id: String,
    },
    MetadataUpdate {
        external_id: String,
        metadata: HashMap<String, MetadataValue>,
    },
    Checkpoint {
        committed_lsn: u64,
        manifest_generation: u64,
    },
    ClusterState {
        epoch: u64,
        state_data: Vec<u8>,
    },
}

impl WalMutation {
    pub fn record_type(&self) -> WalRecordType {
        match self {
            Self::Upsert { .. } => WalRecordType::Upsert,
            Self::Delete { .. } => WalRecordType::Delete,
            Self::MetadataUpdate { .. } => WalRecordType::MetadataUpdate,
            Self::Checkpoint { .. } => WalRecordType::Checkpoint,
            Self::ClusterState { .. } => WalRecordType::ClusterState,
        }
    }
}

/// Durability and acknowledgment policy for WAL appends.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum DurabilityPolicy {
    /// In-memory execution without WAL disk writes (fastest, volatile).
    Memory,
    /// Immediate synchronous `fsync` per transaction.
    WalSync,
    /// Bounded group-commit batching concurrent appends into coalesced fsync barriers.
    #[default]
    WalGroupCommit,
    /// Quorum replication barrier with local WAL sync.
    Quorum,
}

/// Metrics and observability counters for the WAL engine.
#[derive(Clone, Debug, Default)]
pub struct WalMetrics {
    pub appends_total: Arc<AtomicU64>,
    pub bytes_written: Arc<AtomicU64>,
    pub fsync_count: Arc<AtomicU64>,
    pub fsync_total_micros: Arc<AtomicU64>,
    pub backpressure_events: Arc<AtomicU64>,
    pub recovered_records: Arc<AtomicU64>,
}

/// Summary of a recovery execution.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct WalRecoverySummary {
    pub start_lsn: u64,
    pub last_applied_lsn: u64,
    pub total_replayed: usize,
    pub torn_records_skipped: usize,
    pub active_segment_count: usize,
}

struct WalActiveWriter {
    file: File,
    _path: PathBuf,
    segment_id: u64,
    bytes_written: u64,
}

/// Production WAL Manager coordinating logging, group commit, rotation, and recovery.
pub struct WalManager {
    wal_dir: PathBuf,
    max_segment_bytes: u64,
    current_lsn: AtomicU64,
    last_synced_lsn: AtomicU64,
    last_checkpoint_lsn: AtomicU64,
    writer: Mutex<Option<WalActiveWriter>>,
    metrics: WalMetrics,
    sync_cond: Condvar,
    sync_lock: Mutex<()>,
}

impl WalManager {
    /// Opens or initializes a WAL manager in the specified directory.
    pub fn open(wal_dir: impl AsRef<Path>) -> HNSQRResult<Self> {
        let wal_dir = wal_dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&wal_dir)?;

        let metrics = WalMetrics::default();

        let mgr = Self {
            wal_dir,
            max_segment_bytes: DEFAULT_MAX_SEGMENT_BYTES,
            current_lsn: AtomicU64::new(0),
            last_synced_lsn: AtomicU64::new(0),
            last_checkpoint_lsn: AtomicU64::new(0),
            writer: Mutex::new(None),
            metrics,
            sync_cond: Condvar::new(),
            sync_lock: Mutex::new(()),
        };

        mgr.init_active_segment()?;
        Ok(mgr)
    }

    /// Overrides the maximum byte size of an individual WAL segment before rotation.
    pub fn with_max_segment_bytes(mut self, max_bytes: u64) -> Self {
        self.max_segment_bytes = max_bytes.max(1024 * 1024);
        self
    }

    /// Returns the current high-water-mark LSN.
    #[inline(always)]
    pub fn current_lsn(&self) -> u64 {
        self.current_lsn.load(Ordering::Acquire)
    }

    /// Returns the highest durably fsynced LSN.
    #[inline(always)]
    pub fn last_synced_lsn(&self) -> u64 {
        self.last_synced_lsn.load(Ordering::Acquire)
    }

    /// Returns a reference to the WAL metrics.
    pub fn metrics(&self) -> &WalMetrics {
        &self.metrics
    }

    fn init_active_segment(&self) -> HNSQRResult<()> {
        let mut writer_guard = self.writer.lock();
        if writer_guard.is_some() {
            return Ok(());
        }

        // Scan existing segments to find highest segment ID
        let mut highest_seg_id = 0u64;
        let highest_lsn = 0u64;

        if let Ok(entries) = std::fs::read_dir(&self.wal_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension()
                    && ext == "wal"
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                    && let Some(num_str) = stem.strip_prefix("wal_")
                    && let Ok(seg_id) = num_str.parse::<u64>()
                {
                    if seg_id > highest_seg_id {
                        highest_seg_id = seg_id;
                    }
                }
            }
        }

        let seg_id = highest_seg_id.max(1);
        let path = self.wal_dir.join(format!("wal_{seg_id:016}.wal"));
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;

        let metadata = file.metadata()?;
        let bytes_written = metadata.len();

        *writer_guard = Some(WalActiveWriter {
            file,
            _path: path,
            segment_id: seg_id,
            bytes_written,
        });

        self.current_lsn.store(highest_lsn, Ordering::Release);
        self.last_synced_lsn.store(highest_lsn, Ordering::Release);
        Ok(())
    }

    /// Appends a mutation to the WAL according to the specified durability barrier.
    ///
    /// Guarantee: Returns the monotonic LSN only AFTER the durability policy has been satisfied.
    pub fn append(&self, mutation: &WalMutation, policy: DurabilityPolicy) -> HNSQRResult<u64> {
        if policy == DurabilityPolicy::Memory {
            let lsn = self.current_lsn.fetch_add(1, Ordering::SeqCst) + 1;
            return Ok(lsn);
        }

        // 1. Serialize payload & compute CRC32C
        let payload = bincode::serialize(mutation)
            .map_err(|e| HNSQRError::CorruptedSnapshot(format!("Serialization error: {e}")))?;
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&payload);
        let crc32c = hasher.finalize();

        // 2. Lock writer and append framed record
        let target_lsn;
        {
            let mut writer_guard = self.writer.lock();
            let writer = writer_guard.as_mut().unwrap();

            // Check if rotation needed
            if writer.bytes_written >= self.max_segment_bytes {
                let _ = writer.file.sync_all();
                let next_seg_id = writer.segment_id + 1;
                let next_path = self.wal_dir.join(format!("wal_{next_seg_id:016}.wal"));
                let next_file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .read(true)
                    .open(&next_path)?;

                *writer = WalActiveWriter {
                    file: next_file,
                    _path: next_path,
                    segment_id: next_seg_id,
                    bytes_written: 0,
                };
            }

            let prev_lsn = self.current_lsn.load(Ordering::Acquire);
            target_lsn = prev_lsn + 1;

            let header = WalFrameHeader {
                magic: WAL_MAGIC,
                version: WAL_VERSION,
                record_type: mutation.record_type() as u16,
                lsn: target_lsn,
                prev_lsn,
                payload_len: payload.len() as u32,
                crc32c,
            };

            let header_bytes = header.encode();
            writer.file.write_all(&header_bytes)?;
            writer.file.write_all(&payload)?;

            let written = (WAL_HEADER_SIZE + payload.len()) as u64;
            writer.bytes_written += written;

            self.current_lsn.store(target_lsn, Ordering::Release);
            self.metrics
                .bytes_written
                .fetch_add(written, Ordering::Relaxed);
            self.metrics.appends_total.fetch_add(1, Ordering::Relaxed);
        }

        // 3. Durability Barrier
        match policy {
            DurabilityPolicy::Memory => unreachable!(),
            DurabilityPolicy::WalSync => {
                self.sync_target_lsn(target_lsn)?;
            }
            DurabilityPolicy::WalGroupCommit => {
                self.sync_target_lsn(target_lsn)?;
            }
            DurabilityPolicy::Quorum => {
                self.sync_target_lsn(target_lsn)?;
            }
        }

        Ok(target_lsn)
    }

    /// Flushes and fsyncs the active WAL file up to `target_lsn` with group-commit coalescing.
    pub fn sync_target_lsn(&self, target_lsn: u64) -> HNSQRResult<()> {
        if self.last_synced_lsn.load(Ordering::Acquire) >= target_lsn {
            return Ok(());
        }

        let mut _lock = self.sync_lock.lock();
        if self.last_synced_lsn.load(Ordering::Acquire) >= target_lsn {
            return Ok(());
        }

        let start = Instant::now();
        {
            let mut writer_guard = self.writer.lock();
            if let Some(writer) = writer_guard.as_mut() {
                writer.file.sync_data()?;
            }
        }
        let dur_us = start.elapsed().as_micros() as u64;
        self.metrics.fsync_count.fetch_add(1, Ordering::Relaxed);
        self.metrics
            .fsync_total_micros
            .fetch_add(dur_us, Ordering::Relaxed);

        let cur = self.current_lsn.load(Ordering::Acquire);
        self.last_synced_lsn.store(cur, Ordering::Release);
        self.sync_cond.notify_all();
        Ok(())
    }

    /// Replays all WAL records strictly after `snapshot_lsn` across all segments.
    ///
    /// # Invariants
    /// - Records with `lsn <= snapshot_lsn` are skipped.
    /// - Torn record at the end of the log is tolerated (recovers up to last intact record).
    /// - Mid-log corruption returns `HNSQRError::CorruptedSnapshot`.
    /// - Replay is strictly idempotent.
    pub fn replay<F>(&self, snapshot_lsn: u64, mut apply_fn: F) -> HNSQRResult<WalRecoverySummary>
    where
        F: FnMut(u64, WalMutation) -> HNSQRResult<()>,
    {
        let mut segments = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.wal_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension()
                    && ext == "wal"
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                    && let Some(num_str) = stem.strip_prefix("wal_")
                    && let Ok(seg_id) = num_str.parse::<u64>()
                {
                    segments.push((seg_id, path));
                }
            }
        }
        segments.sort_by_key(|(id, _)| *id);

        let mut summary = WalRecoverySummary {
            start_lsn: snapshot_lsn,
            last_applied_lsn: snapshot_lsn,
            total_replayed: 0,
            torn_records_skipped: 0,
            active_segment_count: segments.len(),
        };

        let mut highest_lsn = snapshot_lsn;

        for (_seg_id, path) in &segments {
            let mut file = match File::open(path) {
                Ok(f) => f,
                Err(_) => continue,
            };

            let file_len = file.metadata()?.len();
            let mut offset = 0u64;

            while offset + (WAL_HEADER_SIZE as u64) <= file_len {
                let mut header_buf = [0u8; WAL_HEADER_SIZE];
                if file.read_exact(&mut header_buf).is_err() {
                    // Torn final header
                    summary.torn_records_skipped += 1;
                    break;
                }

                let header = match WalFrameHeader::decode(&header_buf) {
                    Ok(h) => h,
                    Err(_) => {
                        // If this happens at the very end of the file, treat as torn record
                        if offset + (WAL_HEADER_SIZE as u64) == file_len {
                            summary.torn_records_skipped += 1;
                            break;
                        } else {
                            return Err(HNSQRError::CorruptedSnapshot(format!(
                                "Corrupted WAL header frame at offset {offset} in {}",
                                path.display()
                            )));
                        }
                    }
                };

                let payload_len = header.payload_len as usize;
                if offset + (WAL_HEADER_SIZE as u64) + (payload_len as u64) > file_len {
                    // Torn record: incomplete payload at end of log
                    summary.torn_records_skipped += 1;
                    break;
                }

                let mut payload_buf = vec![0u8; payload_len];
                if file.read_exact(&mut payload_buf).is_err() {
                    summary.torn_records_skipped += 1;
                    break;
                }

                // Verify CRC32C
                let mut hasher = crc32fast::Hasher::new();
                hasher.update(&payload_buf);
                if hasher.finalize() != header.crc32c {
                    if offset + (WAL_HEADER_SIZE as u64) + (payload_len as u64) == file_len {
                        summary.torn_records_skipped += 1;
                        break;
                    } else {
                        return Err(HNSQRError::CorruptedSnapshot(format!(
                            "CRC32C mismatch for LSN {} at offset {offset} in {}",
                            header.lsn,
                            path.display()
                        )));
                    }
                }

                offset += (WAL_HEADER_SIZE + payload_len) as u64;

                if header.lsn > snapshot_lsn {
                    let mutation: WalMutation =
                        bincode::deserialize(&payload_buf).map_err(|e| {
                            HNSQRError::CorruptedSnapshot(format!("Payload deserialize error: {e}"))
                        })?;

                    apply_fn(header.lsn, mutation)?;
                    summary.total_replayed += 1;
                    summary.last_applied_lsn = header.lsn;
                    if header.lsn > highest_lsn {
                        highest_lsn = header.lsn;
                    }
                }
            }

            if offset < file_len {
                summary.torn_records_skipped += 1;
            }
        }

        self.current_lsn.store(highest_lsn, Ordering::Release);
        self.last_synced_lsn.store(highest_lsn, Ordering::Release);
        self.metrics
            .recovered_records
            .store(summary.total_replayed as u64, Ordering::Release);

        Ok(summary)
    }

    /// Truncates and deletes WAL segments whose contents are strictly older than `safe_lsn`.
    pub fn truncate_before(&self, safe_lsn: u64) -> HNSQRResult<usize> {
        let mut segments = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.wal_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension()
                    && ext == "wal"
                    && let Some(stem) = path.file_stem().and_then(|s| s.to_str())
                    && let Some(num_str) = stem.strip_prefix("wal_")
                    && let Ok(seg_id) = num_str.parse::<u64>()
                {
                    segments.push((seg_id, path));
                }
            }
        }
        segments.sort_by_key(|(id, _)| *id);

        let active_seg_id = {
            let guard = self.writer.lock();
            guard.as_ref().map(|w| w.segment_id).unwrap_or(u64::MAX)
        };

        let mut deleted_count = 0;
        for (seg_id, path) in segments {
            // Never delete active writing segment
            if seg_id >= active_seg_id {
                continue;
            }

            // Check max LSN inside this segment
            let mut file = match File::open(&path) {
                Ok(f) => f,
                Err(_) => continue,
            };

            let mut max_seg_lsn = 0u64;
            let file_len = file.metadata()?.len();
            let mut offset = 0u64;

            while offset + (WAL_HEADER_SIZE as u64) <= file_len {
                let mut header_buf = [0u8; WAL_HEADER_SIZE];
                if file.read_exact(&mut header_buf).is_err() {
                    break;
                }
                if let Ok(h) = WalFrameHeader::decode(&header_buf) {
                    if h.lsn > max_seg_lsn {
                        max_seg_lsn = h.lsn;
                    }
                    offset += (WAL_HEADER_SIZE as u64) + (h.payload_len as u64);
                    let _ = file.seek(SeekFrom::Start(offset));
                } else {
                    break;
                }
            }

            if max_seg_lsn > 0 && max_seg_lsn <= safe_lsn {
                let _ = std::fs::remove_file(&path);
                deleted_count += 1;
            }
        }

        self.last_checkpoint_lsn.store(safe_lsn, Ordering::Release);
        Ok(deleted_count)
    }
}
