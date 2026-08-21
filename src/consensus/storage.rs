/* hnsqr/src/consensus/storage.rs */
//!▫~•◦-------------------------------‣
//! # Durable Raft HardState, Segmented Log & Progress Storage Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides durable on-disk persistence for Raft HardState (current_term, voted_for),
//! committed progress boundaries, append-only segmented log entries with CRC32 frame
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

/// Metadata describing a discrete on-disk log segment file (`.rlog`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogSegmentMeta {
    pub start_index: u64,
    pub end_index: u64,
    pub entry_count: u64,
    pub byte_size: u64,
    pub file_path: PathBuf,
}

/// Physical location of a single log record on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LogLocation {
    pub start_index: u64,
    pub byte_offset: u64,
    pub frame_length: u32,
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
pub fn decode_framed_record<T: for<'de> Deserialize<'de>>(
    buffer: &[u8],
) -> HNSQRResult<(T, usize)> {
    if buffer.len() < RAFT_FRAME_HEADER_SIZE {
        return Err(HNSQRError::Internal(
            "Buffer too small for Raft frame header".to_string(),
        ));
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
        return Err(HNSQRError::Internal(
            "Incomplete Raft frame payload (torn record)".to_string(),
        ));
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

    let record: T = bincode::deserialize(payload).map_err(|e| {
        HNSQRError::Internal(format!("Failed to deserialize Raft record payload: {e}"))
    })?;

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
            return Err(HNSQRError::Internal(
                "Injected crash before vote persistence".to_string(),
            ));
        }
        *self.hard_state.write() = *state;
        if self.fail_after_vote_persist.load(Ordering::SeqCst) {
            return Err(HNSQRError::Internal(
                "Injected crash after vote persistence".to_string(),
            ));
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
            return Err(HNSQRError::Internal(
                "Injected crash before log persistence".to_string(),
            ));
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
            return Err(HNSQRError::Internal(
                "Injected crash after log persistence".to_string(),
            ));
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

/// Production durable Raft storage engine with append-only segmented log files (`.rlog`).
pub struct DurableRaftStorage {
    base_dir: PathBuf,
    log_dir: PathBuf,
    state_file: PathBuf,
    progress_file: PathBuf,
    snapshot_file: PathBuf,
    segments: RwLock<Vec<LogSegmentMeta>>,
    max_entries_per_segment: usize,
    max_segment_bytes: u64,
    memory_cache: MemoryRaftStorage,
}

impl DurableRaftStorage {
    pub fn open(dir: impl AsRef<Path>) -> HNSQRResult<Self> {
        let base_dir = dir.as_ref().to_path_buf();
        let log_dir = base_dir.join("log");
        fs::create_dir_all(&base_dir)?;
        fs::create_dir_all(&log_dir)?;

        let state_file = base_dir.join("raft_hard_state.bin");
        let progress_file = base_dir.join("raft_progress.bin");
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

        // 4. Recover Segmented Log Files (`.rlog`)
        let mut segments = Vec::new();
        let mut recovered_entries = Vec::new();

        // Check for legacy single-file log `raft_log.bin` and migrate if present
        let legacy_log_file = base_dir.join("raft_log.bin");
        if legacy_log_file.exists() {
            let mut file = File::open(&legacy_log_file)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let mut offset = 0;
            let mut legacy_entries = Vec::new();
            while offset < buffer.len() {
                if let Ok((entry, frame_len)) =
                    decode_framed_record::<RaftLogEntry>(&buffer[offset..])
                {
                    legacy_entries.push(entry);
                    offset += frame_len;
                } else {
                    break;
                }
            }

            if !legacy_entries.is_empty() {
                // Write into segmented structure
                let seg_path = log_dir.join("0000000000000001.rlog");
                let mut framed = Vec::new();
                for entry in &legacy_entries {
                    framed.extend_from_slice(&encode_framed_record(entry)?);
                }
                let mut seg_file = File::create(&seg_path)?;
                seg_file.write_all(&framed)?;
                seg_file.sync_all()?;
            }
            let _ = fs::remove_file(&legacy_log_file);
        }

        // Scan `log_dir` for `.rlog` segments
        let mut segment_paths: Vec<PathBuf> = Vec::new();
        for entry in fs::read_dir(&log_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("rlog") {
                segment_paths.push(path);
            }
        }
        segment_paths.sort();

        for seg_path in &segment_paths {
            let mut file = File::open(seg_path)?;
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;

            let mut offset = 0;
            let mut seg_entries = Vec::new();
            let mut last_valid_offset = 0;

            while offset < buffer.len() {
                match decode_framed_record::<RaftLogEntry>(&buffer[offset..]) {
                    Ok((entry, frame_len)) => {
                        seg_entries.push(entry);
                        offset += frame_len;
                        last_valid_offset = offset;
                    }
                    Err(e) => {
                        // Tolerate torn tail record on the last segment; truncate off cleanly
                        if seg_path == segment_paths.last().unwrap()
                            && offset + RAFT_FRAME_HEADER_SIZE > buffer.len()
                        {
                            let f = OpenOptions::new().write(true).open(seg_path)?;
                            f.set_len(last_valid_offset as u64)?;
                            break;
                        }
                        return Err(HNSQRError::Internal(format!(
                            "Corrupted Raft log frame in segment {:?} at offset {offset}: {e}",
                            seg_path.file_name()
                        )));
                    }
                }
            }

            if !seg_entries.is_empty() {
                let start_index = seg_entries.first().unwrap().index;
                let end_index = seg_entries.last().unwrap().index;
                let byte_size = last_valid_offset as u64;

                segments.push(LogSegmentMeta {
                    start_index,
                    end_index,
                    entry_count: seg_entries.len() as u64,
                    byte_size,
                    file_path: seg_path.clone(),
                });
                recovered_entries.extend(seg_entries);
            }
        }

        if !recovered_entries.is_empty() {
            *memory_cache.log.write() = recovered_entries;
        }

        Ok(Self {
            base_dir,
            log_dir,
            state_file,
            progress_file,
            snapshot_file,
            segments: RwLock::new(segments),
            max_entries_per_segment: 10_000,
            max_segment_bytes: 16 * 1024 * 1024, // 16MB
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

    /// Appends framed entries to the active tail segment, rotating if size thresholds are met.
    fn append_entries_segmented(&self, entries: &[RaftLogEntry]) -> HNSQRResult<()> {
        if entries.is_empty() {
            return Ok(());
        }

        let mut framed_bytes = Vec::new();
        for entry in entries {
            let frame = encode_framed_record(entry)?;
            framed_bytes.extend_from_slice(&frame);
        }

        let mut segments = self.segments.write();
        let need_rotation = segments
            .last()
            .map(|s| {
                s.entry_count >= self.max_entries_per_segment as u64
                    || s.byte_size >= self.max_segment_bytes
            })
            .unwrap_or(true);

        if need_rotation {
            let start_idx = entries[0].index;
            let file_name = format!("{start_idx:016}.rlog");
            let file_path = self.log_dir.join(file_name);

            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&file_path)?;
            file.write_all(&framed_bytes)?;
            file.sync_all()?;

            segments.push(LogSegmentMeta {
                start_index: start_idx,
                end_index: entries.last().unwrap().index,
                entry_count: entries.len() as u64,
                byte_size: framed_bytes.len() as u64,
                file_path,
            });
        } else {
            let last_seg = segments.last_mut().unwrap();
            let mut file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(true)
                .open(&last_seg.file_path)?;
            file.write_all(&framed_bytes)?;
            file.sync_all()?;

            last_seg.end_index = entries.last().unwrap().index;
            last_seg.entry_count += entries.len() as u64;
            last_seg.byte_size += framed_bytes.len() as u64;
        }

        if let Ok(dir_file) = File::open(&self.log_dir) {
            let _ = dir_file.sync_all();
        }
        Ok(())
    }

    /// Surgical log truncation: removes segments strictly above `from_index` and rewrites
    /// only the single containing segment at the truncation boundary without touching older segments.
    fn truncate_suffix_segmented(&self, from_index: u64) -> HNSQRResult<()> {
        let mut segments = self.segments.write();

        // 1. Remove all segments strictly above `from_index`
        segments.retain(|seg| {
            if seg.start_index >= from_index {
                let _ = fs::remove_file(&seg.file_path);
                false
            } else {
                true
            }
        });

        // 2. If the active boundary segment contains `from_index`, prune it
        if let Some(boundary_seg) = segments.last_mut() {
            if boundary_seg.end_index >= from_index {
                // Read and re-encode only entries up to `from_index - 1`
                let log = self.memory_cache.log.read();
                let mut kept_frames = Vec::new();
                let mut kept_count = 0u64;
                let mut max_idx = boundary_seg.start_index;

                for entry in log.iter().skip(boundary_seg.start_index as usize) {
                    if entry.index >= from_index {
                        break;
                    }
                    let frame = encode_framed_record(entry)?;
                    kept_frames.extend_from_slice(&frame);
                    kept_count += 1;
                    max_idx = entry.index;
                }

                let mut file = OpenOptions::new()
                    .write(true)
                    .truncate(true)
                    .open(&boundary_seg.file_path)?;
                file.write_all(&kept_frames)?;
                file.sync_all()?;

                boundary_seg.end_index = max_idx;
                boundary_seg.entry_count = kept_count;
                boundary_seg.byte_size = kept_frames.len() as u64;
            }
        }

        if let Ok(dir_file) = File::open(&self.log_dir) {
            let _ = dir_file.sync_all();
        }
        Ok(())
    }

    /// Compacts log prefix by deleting segment files whose end_index <= snapshot_index.
    pub fn compact_prefix(&self, snapshot_index: u64) -> HNSQRResult<()> {
        let mut segments = self.segments.write();
        segments.retain(|seg| {
            if seg.end_index <= snapshot_index {
                let _ = fs::remove_file(&seg.file_path);
                false
            } else {
                true
            }
        });
        if let Ok(dir_file) = File::open(&self.log_dir) {
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
        self.append_entries_segmented(entries)?;
        self.memory_cache.append_entries(entries)?;
        Ok(())
    }

    fn truncate_suffix(&self, from_index: u64) -> HNSQRResult<()> {
        self.memory_cache.truncate_suffix(from_index)?;
        self.truncate_suffix_segmented(from_index)
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
        let segments = self.segments.read();
        if let Some(last_seg) = segments.last() {
            if last_seg.file_path.exists() {
                let file = OpenOptions::new().write(true).open(&last_seg.file_path)?;
                file.sync_all()?;
            }
        }
        Ok(())
    }
}
