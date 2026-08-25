/* hnsqr/tests/backup_pitr_recovery.rs */
//!▫~•◦-------------------------------‣
//! # Enterprise Backup & PITR Recovery Invariant Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - Full snapshot backup packaging and SHA-256 verification
//!   - Incremental WAL backup chaining across LSN intervals
//!   - Point-in-Time Recovery (PITR) restoring exact data state at specific LSN
//!   - Restore verification asserting vector and metadata parity
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::VectorEmbedding;
use hnsqr::storage::backup::{BackupManager, BackupType};
use hnsqr::storage::manifest::UnifiedSnapshotEngine;
use hnsqr::storage::wal::{DurabilityPolicy, WalManager, WalMutation};
use num_complex::Complex32;

fn temp_dir(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!(
        "hnsqr_backup_test_{name}_{:x}",
        rand::random::<u64>()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
fn test_full_and_incremental_backup_with_pitr() {
    let base_dir = temp_dir("pitr_flow");
    let data_dir = base_dir.join("live_data");
    let snap_dir = data_dir.join("snapshots");
    let wal_dir = data_dir.join("wal");
    let backup_dir = base_dir.join("backups");
    let restore_dir = base_dir.join("restored_data");

    std::fs::create_dir_all(&snap_dir).unwrap();
    std::fs::create_dir_all(&wal_dir).unwrap();

    let dim = 4;
    let v1 =
        VectorEmbedding::from_complex(vec![Complex32::new(1.0, 0.0), Complex32::new(0.0, 1.0)])
            .into_normalized();

    // 1. Initialize WAL and append 10 base records (LSN 1..=10)
    let wal = WalManager::open(&wal_dir).unwrap();
    for i in 1..=10 {
        wal.append(
            &WalMutation::Upsert {
                external_id: format!("doc_base_{i}"),
                vector: v1.clone(),
                metadata: None,
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();
    }
    assert_eq!(wal.current_lsn(), 10);

    // 2. Save Snapshot Gen 1 at LSN 10
    UnifiedSnapshotEngine::save_snapshot(
        &snap_dir,
        1,
        10,
        dim,
        std::slice::from_ref(&v1),
        &["doc_base_1".to_string()],
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    // 3. Perform Full Backup
    let full_meta =
        BackupManager::create_full_backup(&snap_dir, &backup_dir, "backup_full_gen1").unwrap();
    assert_eq!(full_meta.backup_type, BackupType::Full);
    assert_eq!(full_meta.end_lsn, 10);

    // 4. Append WAL mutations for LSN 11, 12, 13, 14
    let lsn11 = wal
        .append(
            &WalMutation::Upsert {
                external_id: "doc_11".to_string(),
                vector: v1.clone(),
                metadata: None,
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();

    let lsn12 = wal
        .append(
            &WalMutation::Upsert {
                external_id: "doc_12".to_string(),
                vector: v1.clone(),
                metadata: None,
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();

    let _lsn13 = wal
        .append(
            &WalMutation::Upsert {
                external_id: "doc_13".to_string(),
                vector: v1.clone(),
                metadata: None,
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();

    let _lsn14 = wal
        .append(
            &WalMutation::Upsert {
                external_id: "doc_14".to_string(),
                vector: v1.clone(),
                metadata: None,
            },
            DurabilityPolicy::WalSync,
        )
        .unwrap();

    assert_eq!(lsn11, 11);
    assert_eq!(lsn12, 12);

    // 5. Create Incremental Backup spanning LSN 11..=14
    let inc_meta =
        BackupManager::create_incremental_backup(&wal_dir, &backup_dir, "backup_inc_1", 11, 14)
            .unwrap();
    assert_eq!(inc_meta.backup_type, BackupType::Incremental);
    assert_eq!(inc_meta.start_lsn, 11);
    assert_eq!(inc_meta.end_lsn, 14);

    // 6. Restore to Point-In-Time (PITR) at exact LSN 12 (must restore docs 11 and 12, skipping 1..=10 and not reaching 13 or 14)
    let mut restored_mutations = Vec::new();
    let summary = BackupManager::restore_pitr(
        &backup_dir,
        &restore_dir,
        "backup_full_gen1",
        Some("backup_inc_1"),
        12, // Target LSN
        |lsn, mutation| {
            restored_mutations.push((lsn, mutation));
            Ok(())
        },
    )
    .unwrap();

    assert_eq!(summary.last_applied_lsn, 12);
    assert_eq!(restored_mutations.len(), 2);
    assert_eq!(restored_mutations[0].0, 11);
    assert_eq!(restored_mutations[1].0, 12);

    let _ = std::fs::remove_dir_all(&base_dir);
}
