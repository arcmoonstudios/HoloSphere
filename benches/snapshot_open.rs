mod common;

use std::fs;

use common::{BenchScale, DEFAULT_BENCH_SEED, get_or_build_snapshot_v2};
use hnsqr::HNSQRIndex;
use hnsqr::rivero::RiveroProfile;
use hnsqr::storage::snapshot::{SnapshotOpenOptions, VerificationMode};

fn main() {
    let scale = BenchScale::from_env();
    let n = scale.corpus_size();

    println!(
        "╔══════════════════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "║ HNSQR SNAPSHOT V2 MMAP ATTACH & PERSISTENCE BENCHMARK (Scale: {:?}, N = {})            ║",
        scale, n
    );
    println!(
        "╚══════════════════════════════════════════════════════════════════════════════════════╝\n"
    );

    let (snap_path, _) =
        get_or_build_snapshot_v2(scale, RiveroProfile::Balanced, DEFAULT_BENCH_SEED);
    let meta = fs::metadata(&snap_path).unwrap();
    let file_size_mb = meta.len() as f64 / (1024.0 * 1024.0);
    let bytes_per_vec = meta.len() as f64 / n as f64;

    println!(
        "  * Snapshot File: {:.2} MB ({} bytes)",
        file_size_mb,
        meta.len()
    );
    println!(
        "  * Asymptotic Storage: {:.2} bytes / vector\n",
        bytes_per_vec
    );

    let (_, breakdown_hb) = HNSQRIndex::open_snapshot_v2_instrumented(
        &snap_path,
        SnapshotOpenOptions {
            verification: VerificationMode::HeaderAndBounds,
            ..Default::default()
        },
    )
    .unwrap();

    let (_, breakdown_full) = HNSQRIndex::open_snapshot_v2_instrumented(
        &snap_path,
        SnapshotOpenOptions {
            verification: VerificationMode::FullChecksums,
            ..Default::default()
        },
    )
    .unwrap();

    println!("  Microsecond Attach Breakdown Comparison:");
    println!("  ┌─────────────────────────────────────┬──────────────────┬──────────────────┐");
    println!("  │ Phase                               │ HeaderAndBounds  │ FullChecksums    │");
    println!("  ├─────────────────────────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │ open() Syscall                      │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.open_syscall_us, breakdown_full.open_syscall_us
    );
    println!(
        "  │ mmap Creation                       │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.mmap_creation_us, breakdown_full.mmap_creation_us
    );
    println!(
        "  │ Header Decode & Validation          │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.header_decode_us, breakdown_full.header_decode_us
    );
    println!(
        "  │ Section Table Validation            │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.section_table_us, breakdown_full.section_table_us
    );
    println!(
        "  │ Config & Arena Mapping              │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.arena_restore_us, breakdown_full.arena_restore_us
    );
    println!(
        "  │ External IDs & Metadata Map         │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.id_restore_us + breakdown_hb.metadata_restore_us,
        breakdown_full.id_restore_us + breakdown_full.metadata_restore_us
    );
    println!(
        "  │ Frozen Rivero Territory Slices      │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.rivero_restore_us, breakdown_full.rivero_restore_us
    );
    println!(
        "  │ CSR Witness Graph Slices            │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.witnesses_restore_us, breakdown_full.witnesses_restore_us
    );
    println!(
        "  │ Graph Fallback Layer Slices         │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.graph_restore_us, breakdown_full.graph_restore_us
    );
    println!(
        "  │ Full Checksums / Structural Hashing │ {:>14.1} µs │ {:>14.1} µs │",
        breakdown_hb.structural_val_us, breakdown_full.structural_val_us
    );
    println!("  ├─────────────────────────────────────┼──────────────────┼──────────────────┤");
    println!(
        "  │ TOTAL ATTACH TIME                   │ {:>12.2} ms │ {:>12.2} ms │",
        breakdown_hb.total_attach_us / 1000.0,
        breakdown_full.total_attach_us / 1000.0
    );
    println!("  └─────────────────────────────────────┴──────────────────┴──────────────────┘\n");
}
