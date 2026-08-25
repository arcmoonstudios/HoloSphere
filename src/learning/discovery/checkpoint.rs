/* holosphere/src/learning/discovery/checkpoint.rs */
//!▫~•◦-------------------------------‣
//! # Governed Discovery Checkpointing & Crash Recovery
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Implements SHA-256 verified recovery checkpoints for the complete governed
//! autonomous discovery catalog, operators, schemas, and epistemic audit logs.
//!
//! ## Key Capabilities
//! - **Atomic State Checkpoints:** Serializes discovery state snapshots with cryptographic integrity hashes.
//! - **Idempotent Replay:** Reconstructs continuous discovery cycles cleanly across node restarts.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::learning::discovery::{
    DeclarativeOperator, DiscoveryAuditLog, DiscoveryStateSnapshot, ExperimentStatus,
    MappingLifecycle, OperatorLifecycle, SchemaProposalState, materialize_relation_type,
};
use crate::learning::read::LearningSegment;
use crate::relation::RelationSegment;

pub const DISCOVERY_CHECKPOINT_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GovernedDiscoveryCheckpoint {
    pub format_version: u32,
    pub lsn: u64,
    pub operators: Vec<DeclarativeOperator>,
    pub state: DiscoveryStateSnapshot,
    pub payload_digest: [u8; 32],
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryCheckpointError {
    #[error("unsupported governed-discovery checkpoint version {0}")]
    UnsupportedVersion(u32),
    #[error("governed-discovery checkpoint digest mismatch")]
    DigestMismatch,
    #[error("governed-discovery checkpoint safety kernel is missing or invalid")]
    InvalidSafetyKernel,
    #[error("governed-discovery checkpoint audit chain is invalid")]
    InvalidAuditChain,
    #[error("governed-discovery checkpoint contains an invalid operator")]
    InvalidOperator,
    #[error("governed-discovery checkpoint contains invalid governed lifecycle state")]
    InvalidGovernedState,
    #[error("governed-discovery checkpoint relation schema collides with the target catalog")]
    RelationSchemaCollision,
    #[error("governed-discovery checkpoint payload cannot be decoded")]
    DecodeFailure,
}

impl GovernedDiscoveryCheckpoint {
    pub fn capture(segment: &LearningSegment, lsn: u64) -> Result<Self, DiscoveryCheckpointError> {
        let mut checkpoint = Self {
            format_version: DISCOVERY_CHECKPOINT_VERSION,
            lsn,
            operators: segment.discovery.snapshot_at(lsn),
            state: segment.governed_discovery.snapshot_at(lsn),
            payload_digest: [0; 32],
        };
        checkpoint.operators.sort_by_key(|operator| operator.id);
        checkpoint.payload_digest = checkpoint.compute_digest();
        checkpoint.verify()?;
        Ok(checkpoint)
    }

    pub fn encode(&self) -> Result<Vec<u8>, DiscoveryCheckpointError> {
        self.verify()?;
        bincode::serialize(self).map_err(|_| DiscoveryCheckpointError::DecodeFailure)
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DiscoveryCheckpointError> {
        let checkpoint: Self =
            bincode::deserialize(bytes).map_err(|_| DiscoveryCheckpointError::DecodeFailure)?;
        checkpoint.verify()?;
        Ok(checkpoint)
    }

    pub fn verify(&self) -> Result<(), DiscoveryCheckpointError> {
        if self.format_version != DISCOVERY_CHECKPOINT_VERSION {
            return Err(DiscoveryCheckpointError::UnsupportedVersion(
                self.format_version,
            ));
        }
        if self.compute_digest() != self.payload_digest {
            return Err(DiscoveryCheckpointError::DigestMismatch);
        }
        self.state
            .safety_kernel
            .as_ref()
            .ok_or(DiscoveryCheckpointError::InvalidSafetyKernel)?
            .verify()
            .map_err(|_| DiscoveryCheckpointError::InvalidSafetyKernel)?;
        if self.state.lsn != self.lsn
            || !DiscoveryAuditLog::verify_entries(&self.state.audit_entries)
        {
            return Err(DiscoveryCheckpointError::InvalidAuditChain);
        }
        if self.operators.iter().any(|operator| {
            !operator.has_valid_identity()
                || operator.committed_lsn > self.lsn
                || operator.lifecycle == OperatorLifecycle::Generated
                || (operator.lifecycle == OperatorLifecycle::Admitted
                    && operator.admission_authority.is_none())
        }) {
            return Err(DiscoveryCheckpointError::InvalidOperator);
        }
        if self.state.schemas.iter().any(|record| {
            matches!(
                record.proposal.state,
                SchemaProposalState::ShadowValidated | SchemaProposalState::Admitted
            ) && record
                .validation
                .as_ref()
                .is_none_or(|validation| !validation.passed)
                || (record.proposal.state == SchemaProposalState::Admitted
                    && record.authority.is_none())
        }) || self.state.mappings.iter().any(|record| {
            matches!(
                record.hypothesis.lifecycle,
                MappingLifecycle::ShadowValidated | MappingLifecycle::Confirmed
            ) && record
                .validation
                .as_ref()
                .is_none_or(|validation| !validation.passed)
                || (record.hypothesis.lifecycle == MappingLifecycle::Confirmed
                    && record.authority.is_none())
        }) || self.state.experiments.iter().any(|experiment| {
            matches!(
                experiment.status,
                ExperimentStatus::Authorized
                    | ExperimentStatus::Running
                    | ExperimentStatus::Completed
            ) && experiment.authorization.is_none()
                || (experiment.status == ExperimentStatus::Completed && experiment.result.is_none())
        }) {
            return Err(DiscoveryCheckpointError::InvalidGovernedState);
        }
        let operator_ids: std::collections::BTreeSet<_> =
            self.operators.iter().map(|operator| operator.id).collect();
        if self
            .state
            .evaluations
            .keys()
            .any(|operator| !operator_ids.contains(operator))
        {
            return Err(DiscoveryCheckpointError::InvalidGovernedState);
        }
        Ok(())
    }

    /// Restores a verified Raft snapshot into an empty/replaced learning
    /// segment. This is a state-machine recovery operation, not an admission
    /// bypass: every restored record was already committed at or before `lsn`.
    pub fn restore_into(&self, segment: &LearningSegment) -> Result<(), DiscoveryCheckpointError> {
        self.verify()?;
        segment.discovery.replace_from(self.operators.clone());
        segment
            .governed_discovery
            .replace_from_snapshot(self.state.clone());
        Ok(())
    }

    /// Restores learning state and deterministically rebuilds its canonical
    /// evolved N-ary relation-schema projection.
    pub fn restore_into_with_relations(
        &self,
        segment: &LearningSegment,
        relations: &RelationSegment,
    ) -> Result<(), DiscoveryCheckpointError> {
        self.verify()?;
        let relation_types: Vec<_> = self
            .state
            .schemas
            .iter()
            .filter_map(|record| {
                let provenance_id = record
                    .proposal
                    .empirical_roots
                    .iter()
                    .next()
                    .map_or(0, |root| root.0);
                materialize_relation_type(&record.proposal, provenance_id)
            })
            .collect();
        for rtype in &relation_types {
            relations
                .prevalidate_evolved_type(rtype)
                .map_err(|_| DiscoveryCheckpointError::RelationSchemaCollision)?;
        }
        self.restore_into(segment)?;
        for rtype in relation_types {
            relations.synchronize_evolved_type(rtype);
        }
        Ok(())
    }

    fn compute_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_GOVERNED_DISCOVERY_CHECKPOINT_V2");
        hasher.update(self.format_version.to_le_bytes());
        hasher.update(self.lsn.to_le_bytes());
        hasher.update(
            bincode::serialize(&self.operators).expect("checkpoint operators are serializable"),
        );
        hasher.update(bincode::serialize(&self.state).expect("checkpoint state is serializable"));
        hasher.finalize().into()
    }
}
