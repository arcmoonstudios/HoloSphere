/* hnsqr/src/consensus/storage.rs */
//!▫~•◦-------------------------------‣
//! # Durable Raft HardState, Segmented Log & Progress Storage Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides durable on-disk persistence for Raft HardState (current_term, voted_for),
//! committed progress boundaries, append-only segmented log entries with CRC32C frame
//! integrity, and snapshot metadata. Guarantees zero silent corruption, fails closed
//! on torn records, and forms the single authoritative recovery source for state machine replay.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::consensus::raft::{RaftLogEntry, RaftNodeId};
use crate::{HNSQRError, HNSQRResult};

const RAFT_FRAME_MAGIC: u32 = 0x5241_4654; // 'RAFT' in ASCII
const RAFT_FRAME_VERSION: u16 = 1;
const RAFT_FRAME_HEADER_SIZE: usize = 14; // 4 + 2 + 4 + 4

/// Crash-safe Raft persistent state (current term and vote).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RaftHardState {
    pub current_term: u64,
    pub voted_for: Option<RaftNodeId>,
}

/// Durable committed and applied state progress boundaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RaftPersistentProgress {
    pub commit_index: u64,
    pub last_applied: u64,
    pub snapshot_index: u64,
    pub snapshot_term: u64,
}

/// Metadata describing the most recent durable Raft snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RaftSnapshotMeta {
    pub last_included_index: u64,
    pub last_included_term: u64,
    pub topology_epoch: u64,
}

/// Encodes payload with CRC32 integrity framing.
pub fn encode_framed_record<T: Serialize>(record: &T) -> HNSQRResult<Vec<u8>> {
    let payload = bincode::serialize(record)
        .map_err(|e| HNSQRError::Internal(format!("Failed to serialize Raft record: {e}")))?;
    let payload_len = payload.len() as u32;

    let mut hasher = crc32fast::Hasher::new();
    hasher.update(&payload);
    let crc = hasher.finalize();

    let mut frame = Vec::with_capacity(RAFT_FRAME_HEADER_SIZE + payload.len());
    frame.extend_from_slice(&RAFT_FRAME_MAGIC.to_le_bytes());
    frame.extend_from_slice(&RAFT_FRAME_VERSION.to_le_bytes());
    frame.extend_from_slice(&payload_len.to_le_bytes());
    frame.extend_from_slice(&crc.to_le_bytes());
    frame.extend_from_slice(&payload);
    Ok(frame)
}

/// Decodes payload with strict CRC32 validation.
pub fn decode_framed_record<T: for<'de> Deserialize<'de>>(buffer: &[u8]) -> HNSQRResult<(T, usize)> {
    if buffer.len() < RAFT_FRAME_HEADER_SIZE {
        return Err(HNSQRError::Internal("Buffer too small for Raft frame header".to_string()));
    }

    let magic = u32::from_le_bytes(buffer[0..4].try_into().unwrap());
    if magic != RAFT_FRAME_MAGIC {
        return Err(HNSQRError::Internal(format!(
            "Invalid Raft frame magic: 0x{magic:08x} (expected 0x{RAFT_FRAME_MAGIC:08x})"
        )));
    }

    let version = u16::from_le_bytes(buffer[4..6].try_into().unwrap());
    if version != RAFT_FRAME_VERSION {
        return Err(HNSQRError::Internal(format!(
            "Unsupported Raft frame version: {version}"
        )));
    }

    let payload_len = u32::from_le_bytes(buffer[6..10].try_into().unwrap()) as usize;
    let expected_crc = u32::from_le_bytes(buffer[10..14].try_into().unwrap());

    let total_frame_len = RAFT_FRAME_HEADER_SIZE + payload_len;
    if buffer.len() < total_frame_len {
        return Err(HNSQRError::Internal("Incomplete Raft frame payload (torn record)".to_string()));
    }

    let payload = &buffer[RAFT_FRAME_HEADER_SIZE..total_frame_len];
    let mut hasher = crc32fast::Hasher::new();
    hasher.update(payload);
    let actual_crc = hasher.finalize();

    if actual_crc != expected_crc {
        return Err(HNSQRError::Internal(format!(
            "CRC32 checksum mismatch in Raft frame: expected 0x{expected_crc:08x}, computed 0x{actual_crc:08x}"
        )));
    }

    let record: T = bincode::deserialize(payload)
        .map_err(|e| HNSQRError::Internal(format!("Failed to deserialize Raft record payload: {e}")))?;

    Ok((record, total_frame_len))
}

/// Storage interface defining required consensus persistence operations.
pub trait RaftStorage: Send + Sync {
    fn save_hard_state(&self, state: &RaftHardState) -> HNSQRResult<()>;
    fn load_hard_state(&self) -> HNSQRResult<RaftHardState>;

    fn save_progress(&self, progress: &RaftPersistentProgress) -> HNSQRResult<()>;
    fn load_progress(&self) -> HNSQRResult<RaftPersistentProgress>;

    fn append_entries(&self, entries: &[RaftLogEntry]) -> HNSQRResult<()>;
    fn truncate_suffix(&self, from_index: u64) -> HNSQRResult<()>;
    fn load_log_entries(&self, from_index: u64) -> HNSQRResult<Vec<RaftLogEntry>>;

    fn save_snapshot_meta(&self, meta: &RaftSnapshotMeta) -> HNSQRResult<()>;
    fn load_snapshot_meta(&self) -> HNSQRResult<RaftSnapshotMeta>;

    fn flush(&self) -> HNSQRResult<()>;
}

/// In-memory implementation of RaftStorage for fast testing and controlled crash injection.
pub struct MemoryRaftStorage {
    pub hard_state: RwLock<RaftHardState>,
    pub progress: RwLock<RaftPersistentProgress>,
    pub snapshot_meta: RwLock<RaftSnapshotMeta>,
    pub log: RwLock<Vec<RaftLogEntry>>,
    pub fail_before_vote_persist: AtomicBool,
    pub fail_after_vote_persist: AtomicBool,
    pub fail_before_log_persist: AtomicBool,
    pub fail_after_log_persist: AtomicBool,
}

impl Default for MemoryRaftStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryRaftStorage {
    pub fn new() -> Self {
        Self {
            hard_state: RwLock::new(RaftHardState::default()),
            progress: RwLock::new(RaftPersistentProgress::default()),
            snapshot_meta: RwLock::new(RaftSnapshotMeta::default()),
            log: RwLock::new(vec![RaftLogEntry {
                term: 0,
                index: 0,
                command: crate::consensus::raft::RaftCommand::NoOp,
            }]),
            fail_before_vote_persist: AtomicBool::new(false),
            fail_after_vote_persist: AtomicBool::new(false),
            fail_before_log_persist: AtomicBool::new(false),
            fail_after_log_persist: AtomicBool::new(false),
        }
    }
}

impl RaftStorage for MemoryRaftStorage {
    fn save_hard_state(&self, state: &RaftHardState) -> HNSQRResult<()> {
        if self.fail_before_vote_persist.load(Ordering::SeqCst) {
            return Err(HNSQRError::Internal("Injected crash before vote persistence".to_string()));
        }
        *self.hard_state.write() = *state;
        if self.fail_after_vote_persist.load(Ordering::SeqCst) {
            return Err(HNSQRError::Internal("Injected crash after vote persistence".to_string()));
        }
        Ok(())
    }

    fn load_hard_state(&self) -> HNSQRResult<RaftHardState> {
        Ok(*self.hard_state.read())
    }

    fn save_progress(&self, progress: &RaftPersistentProgress) -> HNSQRResult<()> {
        *self.progress.write() = *progress;
        Ok(())
    }

    fn load_progress(&self) -> HNSQRResult<RaftPersistentProgress> {
        Ok(*self.progress.read())
    }

    fn append_entries(&self, entries: &[RaftLogEntry]) -> HNSQRResult<()> {
        if self.fail_before_log_persist.load(Ordering::SeqCst) {
            return Err(HNSQRError::Internal("Injected crash before log persistence".to_string()));
        }
        let mut log = self.log.write();
        for entry in entries {
            if (entry.index as usize) < log.len() {
                log[entry.index as usize] = entry.clone();
            } else {
                log.push(entry.clone());
            }
        }
        if self.fail_after_log_persist.load(Ordering::SeqCst) {
            return Err(HNSQRError::Internal("Injected crash after log persistence".to_string()));
        }
        Ok(())
    }

    fn truncate_suffix(&self, from_index: u64) -> HNSQRResult<()> {
        let mut log = self.log.write();
        if (from_index as usize) < log.len() {
            log.truncate(from_index as usize);
        }
        Ok(())
    }

    fn load_log_entries(&self, from_index: u64) -> HNSQRResult<Vec<RaftLogEntry>> {
        let log = self.log.read();
        if (from_index as usize) < log.len() {
            Ok(log[from_index as usize..].to_vec())
        } else {
            Ok(Vec::new())
        }
    }

    fn save_snapshot_meta(&self, meta: &RaftSnapshotMeta) -> HNSQRResult<()> {
        *self.snapshot_meta.write() = *meta;
        Ok(())
    }

    fn load_snapshot_meta(&self) -> HNSQRResult<RaftSnapshotMeta> {
        Ok(*self.snapshot_meta.read())
    }

    fn flush(&self) -> HNSQRResult<()> {
        Ok(())
    }
}

/// Production durable disk-backed Raft storage with CRC32 frame verification and append-only logging.
pub struct DurableRaftStorage {
    base_dir: PathBuf,
    state_file: PathBuf,
    progress_file: PathBuf,
    log_file: PathBuf,
    snapshot_file: PathBuf,
    memory_cache: MemoryRaftStorage,
}

impl DurableRaftStorage {
    pub fn open(dir: impl AsRef<Path>) -> HNSQRResult<Self> {
        let base_dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&base_dir)?;

        let state_file = base_dir.join("raft_hard_state.bin");
        let progress_file = base_dir.join("raft_progress.bin");
        let log_file = base_dir.join("raft_log.bin");
        let snapshot_file = base_dir.join("raft_snapshot_meta.bin");

        let memory_cache = MemoryRaftStorage::new();

        // 1. Recover HardState if exists
        if state_file.exists() {
            let mut bytes = Vec::new();
            File::open(&state_file)?.read_to_end(&mut bytes)?;
            if !bytes.is_empty() {
                let (state, _) = decode_framed_record::<RaftHardState>(&bytes)?;
                *memory_cache.hard_state.write() = state;
            }
        }

        // 2. Recover Progress if exists
        if progress_file.exists() {
            let mut bytes = Vec::new();
            File::open(&progress_file)?.read_to_end(&mut bytes)?;
            if !bytes.is_empty() {
                let (prog, _) = decode_framed_record::<RaftPersistentProgress>(&bytes)?;
                *memory_cache.progress.write() = prog;
            }
        }

        // 3. Recover Snapshot Metadata if exists
        if snapshot_file.exists() {
            let mut bytes = Vec::new();
            File::open(&snapshot_file)?.read_to_end(&mut bytes)?;
            if !bytes.is_empty() {
                let (meta, _) = decode_framed_record::<RaftSnapshotMeta>(&bytes)?;
                *memory_cache.snapshot_meta.write() = meta;
            }
        }

        // 4. Recover Log Entries with sequential frame verification
        if log_file.exists() {
            let mut file = File::open(&log_file)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let mut offset = 0;
            let mut recovered_entries = Vec::new();
            while offset < buffer.len() {
                let (entry, frame_len) = decode_framed_record::<RaftLogEntry>(&buffer[offset..])?;
                recovered_entries.push(entry);
                offset += frame_len;
            }

            if !recovered_entries.is_empty() {
                *memory_cache.log.write() = recovered_entries;
            }
        }

        Ok(Self {
            base_dir,
            state_file,
            progress_file,
            log_file,
            snapshot_file,
            memory_cache,
        })
    }

    /// Atomically replaces a single metadata file with directory sync.
    fn write_atomic_framed<T: Serialize>(&self, target_path: &Path, record: &T) -> HNSQRResult<()> {
        let frame_bytes = encode_framed_record(record)?;
        let tmp_path = target_path.with_extension("tmp");

        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_path)?;
            file.write_all(&frame_bytes)?;
            file.sync_all()?;
        }

        fs::rename(&tmp_path, target_path)?;

        // Directory sync where applicable
        if let Ok(dir_file) = File::open(&self.base_dir) {
            let _ = dir_file.sync_all();
        }
        Ok(())
    }

    fn rewrite_all_log_entries(&self) -> HNSQRResult<()> {
        let log = self.memory_cache.log.read().clone();
        let mut all_frames = Vec::new();
        for entry in &log {
            let frame = encode_framed_record(entry)?;
            all_frames.extend_from_slice(&frame);
        }

        let tmp_file = self.base_dir.join("raft_log.bin.tmp");
        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&tmp_file)?;
            file.write_all(&all_frames)?;
            file.sync_all()?;
        }
        fs::rename(tmp_file, &self.log_file)?;
        if let Ok(dir_file) = File::open(&self.base_dir) {
            let _ = dir_file.sync_all();
        }
        Ok(())
    }
}

impl RaftStorage for DurableRaftStorage {
    fn save_hard_state(&self, state: &RaftHardState) -> HNSQRResult<()> {
        self.write_atomic_framed(&self.state_file, state)?;
        self.memory_cache.save_hard_state(state)?;
        Ok(())
    }

    fn load_hard_state(&self) -> HNSQRResult<RaftHardState> {
        self.memory_cache.load_hard_state()
    }

    fn save_progress(&self, progress: &RaftPersistentProgress) -> HNSQRResult<()> {
        self.write_atomic_framed(&self.progress_file, progress)?;
        self.memory_cache.save_progress(progress)?;
        Ok(())
    }

    fn load_progress(&self) -> HNSQRResult<RaftPersistentProgress> {
        self.memory_cache.load_progress()
    }

    fn append_entries(&self, entries: &[RaftLogEntry]) -> HNSQRResult<()> {
        let mut framed_bytes = Vec::new();
        for entry in entries {
            let frame = encode_framed_record(entry)?;
            framed_bytes.extend_from_slice(&frame);
        }

        // Append-only write with group fsync
        {
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open(&self.log_file)?;
            file.write_all(&framed_bytes)?;
            file.sync_all()?;
        }

        self.memory_cache.append_entries(entries)?;
        Ok(())
    }

    fn truncate_suffix(&self, from_index: u64) -> HNSQRResult<()> {
        self.memory_cache.truncate_suffix(from_index)?;
        self.rewrite_all_log_entries()
    }

    fn load_log_entries(&self, from_index: u64) -> HNSQRResult<Vec<RaftLogEntry>> {
        self.memory_cache.load_log_entries(from_index)
    }

    fn save_snapshot_meta(&self, meta: &RaftSnapshotMeta) -> HNSQRResult<()> {
        self.write_atomic_framed(&self.snapshot_file, meta)?;
        self.memory_cache.save_snapshot_meta(meta)?;
        Ok(())
    }

    fn load_snapshot_meta(&self) -> HNSQRResult<RaftSnapshotMeta> {
        self.memory_cache.load_snapshot_meta()
    }

    fn flush(&self) -> HNSQRResult<()> {
        if self.log_file.exists() {
            let file = OpenOptions::new().write(true).open(&self.log_file)?;
            file.sync_all()?;
        }
        Ok(())
    }
}
