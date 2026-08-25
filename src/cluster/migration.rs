/* holosphere/src/cluster/migration.rs */
//!▫~•◦-------------------------------‣
//! # 5-Stage Transactional Shard Migration Protocol
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use super::ring::ShardId;

/// 5-Stage Transactional Shard Migration Protocol states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MigrationPhase {
    /// 1. Prepare: Register migration intent in consensus with source and destination.
    Prepare,
    /// 2. SnapshotTransfer: Stream immutable segment chunks with CRC32C.
    SnapshotTransfer,
    /// 3. WalCatchup: Replay mutation stream from snapshot LSN to current.
    WalCatchup,
    /// 4. OwnershipCommit: Commit ownership flip in consensus state machine.
    OwnershipCommit,
    /// 5. Cleanup: Reclaim source storage after safety grace period.
    Cleanup,
}

/// Active migration task descriptor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MigrationTask {
    pub migration_id: u64,
    pub source_shard: ShardId,
    pub dest_shard: ShardId,
    pub phase: MigrationPhase,
    pub snapshot_lsn: u64,
    pub committed_lsn: u64,
    pub bytes_transferred: u64,
}
