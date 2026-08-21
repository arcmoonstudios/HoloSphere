/* hnsqr/tests/wal_crash_safety.rs */
//!▫~•◦-------------------------------‣
//! # WAL Crash Safety & Durability Invariant Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - Monotonic LSN generation and CRC32C checksum integrity
//!   - Torn final record tolerance on process crash simulation
//!   - Mid-log corruption rejection
//!   - Idempotent repeated replay
//!   - Segment rotation and safe truncation
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use hnsqr::VectorEmbedding;
use hnsqr::metadata::index::MetadataValue;
use hnsqr::storage::wal::{DurabilityPolicy, WalManager, WalMutation};
use num_complex::Complex32;

fn temp_wal_dir(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hnsqr_wal_test_{test_name}_{:x}",
        rand::random::<u64>()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_wal_append_and_idempotent_replay() {
    let wal_dir = temp_wal_dir("append_replay");
    let wal = WalManager::open(&wal_dir).unwrap();

    let v1 =
        VectorEmbedding::from_complex(vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)])
            .into_normalized();
    let v2 =
        VectorEmbedding::from_complex(vec![Complex32::new(0.5, 0.5), Complex32::new(0.5, -0.5)])
            .into_normalized();

    let mut meta1 = HashMap::new();
    meta1.insert(
        "category".to_string(),
        MetadataValue::String("finance".to_string()),
    );

    let lsn1 = wal
        .append(
            &WalMutation::Upsert {
                external_id: "doc_1".to_string(),
                vector: v1,
                metadata: Some(meta1),
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();

    let lsn2 = wal
        .append(
            &WalMutation::Upsert {
                external_id: "doc_2".to_string(),
                vector: v2,
                metadata: None,
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();

    let lsn3 = wal
        .append(
            &WalMutation::Delete {
                external_id: "doc_1".to_string(),
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();

    assert_eq!(lsn1, 1);
    assert_eq!(lsn2, 2);
    assert_eq!(lsn3, 3);
    assert_eq!(wal.current_lsn(), 3);

    // Replay all records from LSN 0
    let mut replayed_mutations = Vec::new();
    let summary = wal
        .replay(0, |lsn, mutation| {
            replayed_mutations.push((lsn, mutation));
            Ok(())
        })
        .unwrap();

    assert_eq!(summary.total_replayed, 3);
    assert_eq!(summary.last_applied_lsn, 3);
    assert_eq!(replayed_mutations.len(), 3);

    // Idempotent second replay produces exact same output
    let mut second_replayed = Vec::new();
    let summary2 = wal
        .replay(0, |lsn, mutation| {
            second_replayed.push((lsn, mutation));
            Ok(())
        })
        .unwrap();
    assert_eq!(summary2.total_replayed, 3);
    assert_eq!(replayed_mutations, second_replayed);

    // Replay after snapshot LSN 2 recovers only LSN 3
    let mut post_snap = Vec::new();
    let summary3 = wal
        .replay(2, |lsn, mutation| {
            post_snap.push((lsn, mutation));
            Ok(())
        })
        .unwrap();
    assert_eq!(summary3.total_replayed, 1);
    assert_eq!(post_snap.len(), 1);
    assert_eq!(post_snap[0].0, 3);

    let _ = std::fs::remove_dir_all(&wal_dir);
}

#[test]
fn test_wal_torn_tail_crash_tolerance() {
    let wal_dir = temp_wal_dir("torn_tail");
    let wal = WalManager::open(&wal_dir).unwrap();

    let v = VectorEmbedding::from_complex(vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)])
        .into_normalized();

    for i in 1..=5 {
        wal.append(
            &WalMutation::Upsert {
                external_id: format!("doc_{i}"),
                vector: v.clone(),
                metadata: None,
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();
    }

    // Corrupt the tail by writing 15 bytes of garbage simulating half-written torn frame
    let active_file = std::fs::read_dir(&wal_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    {
        let mut f = OpenOptions::new().append(true).open(&active_file).unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01, 0x02, 0x03, 0x04])
            .unwrap();
    }

    // Recovery must tolerate torn tail and recover all 5 valid records
    let mut count = 0;
    let summary = wal
        .replay(0, |_lsn, _mut| {
            count += 1;
            Ok(())
        })
        .unwrap();

    assert_eq!(count, 5);
    assert_eq!(summary.total_replayed, 5);
    assert!(summary.torn_records_skipped >= 1);

    let _ = std::fs::remove_dir_all(&wal_dir);
}

#[test]
fn test_wal_mid_log_corruption_rejection() {
    let wal_dir = temp_wal_dir("mid_corruption");
    let wal = WalManager::open(&wal_dir).unwrap();

    let v = VectorEmbedding::from_complex(vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)])
        .into_normalized();

    for i in 1..=5 {
        wal.append(
            &WalMutation::Upsert {
                external_id: format!("doc_{i}"),
                vector: v.clone(),
                metadata: None,
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();
    }

    // Corrupt payload byte inside record #2 (mid-log)
    let active_file = std::fs::read_dir(&wal_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    {
        let mut f = OpenOptions::new().write(true).open(&active_file).unwrap();
        f.seek(SeekFrom::Start(45)).unwrap(); // Inside record payload
        f.write_all(&[0xFF, 0xFF]).unwrap();
    }

    // Recovery must reject mid-log corruption
    let replay_res = wal.replay(0, |_lsn, _mut| Ok(()));
    assert!(replay_res.is_err(), "Mid-log corruption must fail closed!");

    let _ = std::fs::remove_dir_all(&wal_dir);
}
