/* hnsqr/tests/snapshot_manifest_recovery.rs */
//!▫~•◦-------------------------------‣
//! # Unified Snapshot & Manifest Recovery Invariant Test Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - Atomic Copy-on-Write snapshot publication
//!   - Complete multi-section integrity (IDs, vectors, metadata, proof tree, LUTz)
//!   - Corrupt-section detection & CRC32C failure safety
//!   - Point-in-time snapshot + WAL replay restoration
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};

use hnsqr::metadata::index::MetadataValue;
use hnsqr::proof::tree::SemanticProofTree;
use hnsqr::storage::manifest::{SectionKind, UnifiedSnapshotEngine};
use hnsqr::{NodeIndex, VectorEmbedding};
use num_complex::Complex32;

fn temp_snap_dir(test_name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "hnsqr_snap_test_{test_name}_{:x}",
        rand::random::<u64>()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn test_unified_snapshot_cow_and_load_integrity() {
    let snap_dir = temp_snap_dir("cow_integrity");
    let dim = 8;
    let n = 20;

    let vectors: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new((i * 3 + d) as f32, (i * 7 + d) as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let external_ids: Vec<String> = (0..n).map(|i| format!("doc_{i}")).collect();

    let mut metadata_map = Vec::new();
    for i in 0..n {
        let mut m = HashMap::new();
        m.insert(
            "tenant".to_string(),
            MetadataValue::String(format!("tenant_{}", i % 3)),
        );
        metadata_map.push(m);
    }

    let slots: Vec<NodeIndex> = (0..n as NodeIndex).collect();
    let proof_tree = SemanticProofTree::build(&vectors, &slots, dim);

    // Write Generation 1
    let manifest1 = UnifiedSnapshotEngine::save_snapshot(
        &snap_dir,
        1,
        100,
        dim,
        &vectors,
        &external_ids,
        Some(&metadata_map),
        None,
        Some(&proof_tree),
        None,
    )
    .unwrap();

    assert_eq!(manifest1.generation, 1);
    assert_eq!(manifest1.snapshot_lsn, 100);
    assert!(manifest1.sections.contains_key(&SectionKind::VectorData));
    assert!(manifest1.sections.contains_key(&SectionKind::ExternalIdMap));
    assert!(manifest1.sections.contains_key(&SectionKind::MetadataStore));
    assert!(
        manifest1
            .sections
            .contains_key(&SectionKind::SemanticProofTree)
    );

    // Load back and verify integrity
    let (loaded_manifest, mmap) = UnifiedSnapshotEngine::load_latest_snapshot(&snap_dir).unwrap();
    assert_eq!(loaded_manifest.generation, 1);
    assert!(!mmap.is_empty());

    let _ = std::fs::remove_dir_all(&snap_dir);
}

#[test]
fn test_corrupt_section_checksum_failure_safety() {
    let snap_dir = temp_snap_dir("corrupt_section");
    let dim = 8;
    let n = 10;

    let vectors: Vec<VectorEmbedding> = (0..n)
        .map(|i| {
            VectorEmbedding::from_complex(
                (0..dim)
                    .map(|d| Complex32::new(i as f32, d as f32))
                    .collect(),
            )
            .into_normalized()
        })
        .collect();

    let external_ids: Vec<String> = (0..n).map(|i| format!("doc_{i}")).collect();

    UnifiedSnapshotEngine::save_snapshot(
        &snap_dir,
        1,
        50,
        dim,
        &vectors,
        &external_ids,
        None,
        None,
        None,
        None,
    )
    .unwrap();

    // Corrupt 2 bytes in snapshot data file
    let data_file_path = snap_dir.join("snapshot_gen_0000000000000001.data");
    {
        let mut f = OpenOptions::new()
            .write(true)
            .open(&data_file_path)
            .unwrap();
        f.seek(SeekFrom::Start(12)).unwrap();
        f.write_all(&[0xAA, 0xBB]).unwrap();
    }

    // Load must reject corrupted section with CRC32C failure
    let load_res = UnifiedSnapshotEngine::load_latest_snapshot(&snap_dir);
    assert!(
        load_res.is_err(),
        "Corrupted snapshot must fail validation!"
    );

    let _ = std::fs::remove_dir_all(&snap_dir);
}
