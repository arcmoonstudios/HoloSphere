/* holosphere/src/consensus/mod.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Distributed Consensus & Raft State Machine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides production-grade Raft consensus, durable HardState and log persistence,
//! dynamic joint-membership transitions, leader election, and replicated topology control.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod driver;
pub mod durability_controller;
pub mod pending;
pub mod raft;
pub mod read_index;
pub mod state_machine;
pub mod storage;

pub use driver::{ClientProposal, ClusterLiveness, DriverEvent, RaftDriver, RaftDriverError};
pub use durability_controller::{DurabilityBatchPlan, DurabilityController, StorageTelemetry};
pub use pending::{
    ApplyError, CommitReceipt, CommitStatus, DurabilityLevel, MutationId, PendingProposals,
    ProposalId,
};
pub use raft::{
    AdaptiveMicrobatcher, AppendEntriesArgs, AppendEntriesReply, MembershipMutation, RaftCluster,
    RaftCommand, RaftLogEntry, RaftMessage, RaftNode, RaftPipelineTelemetry, RaftRole,
    RequestVoteArgs, RequestVoteReply, StorageHealthMetrics, TopologyMutation,
};
pub use read_index::{
    LinearizableReadMode, ReadConsistency, ReadContextId, ReadIndexConfirmation, ReadIndexEngine,
    ReadIndexRequest, ReadIndexTelemetry,
};
pub use state_machine::{
    ApplyReceipt, ClientIdentity, DataMutation, DeduplicationHorizon, ReplicatedStateMachine,
    RetrySemantics, ShardStateMachine,
};
pub use storage::{
    DurableRaftStorage, LogLocation, LogSegmentMeta, MemoryRaftStorage, RaftHardState,
    RaftPersistentProgress, RaftSnapshotMeta, RaftStorage,
};
