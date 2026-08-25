/* holosphere/src/storage/mod.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Storage Engine & Durability Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides enterprise-grade crash-safe write-ahead logging (WAL), group commit,
//! unified sectioned snapshot manifests, and quota-bounded metadata persistence.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod adaptive_prefault;
pub mod backpressure;
pub mod backup;
pub mod columnar_olap;
pub mod io_budget;
pub mod manifest;
pub mod mmap_arena;
pub mod predictive_warming;
pub mod relational_acid;
pub mod remote_cache;
pub mod remote_layout;
pub mod segment;
pub mod segment_store;
pub mod sharded_map;
pub mod snapshot;
pub mod two_tier_cache;
pub mod wal;

pub use adaptive_prefault::{AdaptivePrefaultEngine, PrefaultMode};
pub use backpressure::{BackpressureConfig, BackpressureController, MutationPermit};
pub use backup::{BackupManager, BackupMetadata, BackupType};
pub use columnar_olap::{
    ColumnarFloatArray, ColumnarOlapEngine, EmbeddedMediaBlob, OlapAggregationOp,
};
pub use io_budget::{IoBudgetManager, IoMaintenanceClass};
pub use mmap_arena::{MmapArena, MmapHeader};
pub use predictive_warming::{PredictiveWarmer, ProofHeatMap};
pub use relational_acid::{
    AcidTransaction, ColumnDefinition, RelationalRow, RelationalSqlEngine, RowLevelSecurityPolicy,
    SqlType, SqlValue, TableSchema, TransactionState,
};
pub use remote_cache::{CachedChunk, ChunkId, RemoteRangeCache};
pub use remote_layout::{
    ProofAwareLayoutBuilder, ProofLeafBlockMapping, RemoteAmplificationMetrics, RemoteChunkSize,
};
pub use segment::{
    ImmutableSegment, MutableSegment, SegmentId, SegmentState, SegmentStats, SegmentedEngine,
};
pub use segment_store::{
    ImmutableSegmentStore, LocalSegmentStore, S3SegmentStore, SegmentObjectId,
    SegmentObjectMetadata,
};
pub use sharded_map::ShardedConcurrentMap;
pub use snapshot::{
    SectionDescriptor, SnapshotAttachBreakdown, SnapshotHeaderV2, SnapshotOpenOptions,
    SnapshotStats, VerificationMode,
};
pub use two_tier_cache::{CacheBlockId, CachedVectorBlock, TwoTierCache};
pub use wal::{
    DurabilityPolicy, WalFrameHeader, WalManager, WalMetrics, WalMutation, WalRecordType,
    WalRecoverySummary,
};
