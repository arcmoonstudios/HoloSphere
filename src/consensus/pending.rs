/* holosphere/src/consensus/pending.rs */
//!▫~•◦-------------------------------‣
//! # Bounded Pending Proposal Registry & Commit Receipts
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides deterministic causal identity, explicit durability diagnostics,
//! and asynchronous non-blocking resolution for in-flight Raft proposals.
//! Guarantees zero memory leaks and resolves completions outside critical locks.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::{HNSQRError, HNSQRResult};

/// Globally unique and collision-immune Raft proposal identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ProposalId {
    pub term: u64,
    pub log_index: u64,
}

impl fmt::Display for ProposalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "T{}:I{}", self.term, self.log_index)
    }
}

/// Client-provided or system-generated idempotent mutation identifier.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MutationId(pub String);

impl MutationId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    pub fn generate() -> Self {
        let id = format!(
            "mut_{:016x}_{:08x}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
            rand::random::<u32>()
        );
        Self(id)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for MutationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl<T: Into<String>> From<T> for MutationId {
    fn from(s: T) -> Self {
        Self(s.into())
    }
}

/// Declared consensus durability level achieved by a committed mutation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DurabilityLevel {
    MemoryOnly,
    #[default]
    QuorumReplicated,
    QuorumDurableFsylog,
}

/// Comprehensive commit & state-machine application receipt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitReceipt {
    pub mutation_id: MutationId,
    pub term: u64,
    pub log_index: u64,
    pub quorum_committed: bool,
    pub state_machine_applied: bool,
    pub applied_index: u64,
    pub topology_epoch: u64,
    pub durability: DurabilityLevel,
}

impl CommitReceipt {
    /// Constructs a fully verified commit receipt enforcing valid applied index.
    pub fn new_verified(
        mutation_id: MutationId,
        term: u64,
        log_index: u64,
        applied_index: u64,
        topology_epoch: u64,
        durability: DurabilityLevel,
    ) -> Self {
        assert!(
            applied_index > 0,
            "State machine must apply before generating ACK"
        );
        Self {
            mutation_id,
            term,
            log_index,
            quorum_committed: true,
            state_machine_applied: true,
            applied_index,
            topology_epoch,
            durability,
        }
    }
}

/// Status of mutation commitment when leadership is lost.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CommitStatus {
    Committed,
    Uncommitted,
    Unknown,
}

/// Errors occurring during state machine application or consensus proposal.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplyError {
    LeadershipLost {
        mutation_id: MutationId,
        term: u64,
        log_index: u64,
        commit_status: CommitStatus,
    },
    StateApplyFailed {
        mutation_id: MutationId,
        reason: String,
        log_index: u64,
    },
    StateMachineUnavailable {
        log_index: u64,
    },
    SequenceViolation {
        reason: String,
    },
    QuorumUnavailable {
        term: u64,
    },
    ProposalTimedOut {
        proposal: ProposalId,
    },
    Deduplicated {
        receipt: CommitReceipt,
    },
}

impl fmt::Display for ApplyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LeadershipLost {
                mutation_id,
                term,
                log_index,
                commit_status,
            } => {
                write!(
                    f,
                    "Leadership lost for mutation {mutation_id} at term {term}, index {log_index} (commit status: {commit_status:?})"
                )
            }
            Self::StateApplyFailed {
                mutation_id,
                reason,
                log_index,
            } => {
                write!(
                    f,
                    "State machine apply failed for mutation {mutation_id} at index {log_index}: {reason}"
                )
            }
            Self::StateMachineUnavailable { log_index } => {
                write!(
                    f,
                    "Authoritative state machine unavailable at log index {log_index}"
                )
            }
            Self::SequenceViolation { reason } => {
                write!(f, "Client sequence violation: {reason}")
            }
            Self::QuorumUnavailable { term } => {
                write!(f, "Raft quorum unavailable in term {term}")
            }
            Self::ProposalTimedOut { proposal } => {
                write!(
                    f,
                    "Proposal {proposal} timed out waiting for quorum commit and state application"
                )
            }
            Self::Deduplicated { receipt } => {
                write!(
                    f,
                    "Mutation {} already applied at index {}",
                    receipt.mutation_id, receipt.applied_index
                )
            }
        }
    }
}

impl std::error::Error for ApplyError {}

impl From<ApplyError> for HNSQRError {
    fn from(err: ApplyError) -> Self {
        match err {
            ApplyError::LeadershipLost {
                mutation_id,
                term,
                log_index,
                commit_status,
            } => HNSQRError::Internal(format!(
                "Leadership lost for {mutation_id} at term {term}, index {log_index} ({commit_status:?})"
            )),
            ApplyError::StateApplyFailed {
                mutation_id,
                reason,
                ..
            } => HNSQRError::Internal(format!("State apply failed for {mutation_id}: {reason}")),
            ApplyError::StateMachineUnavailable { log_index } => HNSQRError::Internal(format!(
                "State machine unavailable at log index {log_index}"
            )),
            ApplyError::SequenceViolation { reason } => {
                HNSQRError::Internal(format!("Sequence violation: {reason}"))
            }
            ApplyError::QuorumUnavailable { term } => {
                HNSQRError::Internal(format!("Quorum unavailable in term {term}"))
            }
            ApplyError::ProposalTimedOut { proposal } => {
                HNSQRError::Internal(format!("Proposal {proposal} timed out"))
            }
            ApplyError::Deduplicated { receipt } => HNSQRError::Internal(format!(
                "Idempotent replay for mutation {}",
                receipt.mutation_id
            )),
        }
    }
}

struct PendingEntry {
    sender: oneshot::Sender<Result<CommitReceipt, ApplyError>>,
    mutation_id: MutationId,
    created_at: Instant,
}

/// Thread-safe bounded registry of pending proposals awaiting quorum commit and apply.
pub struct PendingProposals {
    entries: Mutex<HashMap<ProposalId, PendingEntry>>,
    max_capacity: usize,
}

impl Default for PendingProposals {
    fn default() -> Self {
        Self::new(65_536)
    }
}

impl PendingProposals {
    pub fn new(max_capacity: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::with_capacity(1024)),
            max_capacity,
        }
    }

    /// Registers a new in-flight proposal and returns its receiver.
    pub fn register(
        &self,
        proposal: ProposalId,
        mutation_id: MutationId,
    ) -> HNSQRResult<oneshot::Receiver<Result<CommitReceipt, ApplyError>>> {
        let mut guard = self.entries.lock();
        if guard.len() >= self.max_capacity {
            return Err(HNSQRError::Internal(format!(
                "Pending proposals queue saturated ({}/{})",
                guard.len(),
                self.max_capacity
            )));
        }

        let (tx, rx) = oneshot::channel();
        guard.insert(
            proposal,
            PendingEntry {
                sender: tx,
                mutation_id,
                created_at: Instant::now(),
            },
        );
        Ok(rx)
    }

    /// Notifies a committed and applied proposal with its receipt outside the lock.
    pub fn complete_applied(
        &self,
        proposal: ProposalId,
        term: u64,
        applied_index: u64,
        topology_epoch: u64,
        durability: DurabilityLevel,
    ) -> bool {
        let maybe_entry = {
            let mut guard = self.entries.lock();
            guard.remove(&proposal)
        };

        if let Some(entry) = maybe_entry {
            let receipt = CommitReceipt::new_verified(
                entry.mutation_id,
                term,
                proposal.log_index,
                applied_index,
                topology_epoch,
                durability,
            );
            let _ = entry.sender.send(Ok(receipt));
            true
        } else {
            false
        }
    }

    /// Fails a pending proposal with a specific error outside the lock.
    pub fn fail_proposal(&self, proposal: ProposalId, error: ApplyError) -> bool {
        let maybe_entry = {
            let mut guard = self.entries.lock();
            guard.remove(&proposal)
        };

        if let Some(entry) = maybe_entry {
            let _ = entry.sender.send(Err(error));
            true
        } else {
            false
        }
    }

    /// Immediately cancels and fails all pending proposals when leadership is lost, outside the lock.
    pub fn cancel_all_leadership_lost(&self, current_term: u64) -> usize {
        let drained: Vec<(ProposalId, PendingEntry)> = {
            let mut guard = self.entries.lock();
            guard.drain().collect()
        };

        let count = drained.len();
        for (proposal, entry) in drained {
            let _ = entry.sender.send(Err(ApplyError::LeadershipLost {
                mutation_id: entry.mutation_id,
                term: current_term,
                log_index: proposal.log_index,
                commit_status: CommitStatus::Unknown,
            }));
        }
        count
    }

    /// Removes and fails timed out proposals outside the lock.
    pub fn prune_timeouts(&self, timeout: std::time::Duration) -> usize {
        let timed_out_entries: Vec<(ProposalId, PendingEntry)> = {
            let mut guard = self.entries.lock();
            let now = Instant::now();
            let mut keys = Vec::new();
            for (&proposal, entry) in guard.iter() {
                if now.duration_since(entry.created_at) > timeout {
                    keys.push(proposal);
                }
            }
            keys.into_iter()
                .filter_map(|p| guard.remove(&p).map(|e| (p, e)))
                .collect()
        };

        let count = timed_out_entries.len();
        for (proposal, entry) in timed_out_entries {
            let _ = entry
                .sender
                .send(Err(ApplyError::ProposalTimedOut { proposal }));
        }
        count
    }

    pub fn len(&self) -> usize {
        self.entries.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.lock().is_empty()
    }
}
