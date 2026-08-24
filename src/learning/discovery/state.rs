//! Replicated MVCC state for the complete governed discovery lifecycle.

use std::collections::BTreeMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::learning::discovery::active_experiment::{ActiveExperimentProposal, ExperimentStatus};
use crate::learning::discovery::evaluation::CompetitiveOperatorEvaluation;
use crate::learning::discovery::experiment::ExperimentProposalId;
use crate::learning::discovery::lifecycle::{
    DiscoveryAuditAction, DiscoveryAuditEntry, ImmutableSafetyKernel,
};
use crate::learning::discovery::mapping::{
    ConceptMappingHypothesis, MappingHypothesisId, MappingLifecycle, MappingValidation,
};
use crate::learning::discovery::operator::{DiscoveredOperatorId, GovernanceAuthority};
use crate::learning::discovery::schema::{
    EvolvedSchemaId, EvolvedSchemaProposal, SchemaProposalState, SchemaValidation,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GovernedSchemaRecord {
    pub proposal: EvolvedSchemaProposal,
    pub validation: Option<SchemaValidation>,
    pub authority: Option<GovernanceAuthority>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GovernedMappingRecord {
    pub hypothesis: ConceptMappingHypothesis,
    pub validation: Option<MappingValidation>,
    pub authority: Option<GovernanceAuthority>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiscoveryStateMutation {
    InstallSafetyKernel {
        kernel: ImmutableSafetyKernel,
    },
    UpsertSchema {
        record: GovernedSchemaRecord,
        expected_previous: Option<SchemaProposalState>,
    },
    UpsertMapping {
        record: GovernedMappingRecord,
        expected_previous: Option<MappingLifecycle>,
    },
    RecordEvaluation {
        operator: DiscoveredOperatorId,
        evaluation: CompetitiveOperatorEvaluation,
    },
    UpsertExperiment {
        experiment: ActiveExperimentProposal,
        expected_previous: Option<ExperimentStatus>,
    },
    AppendAudit {
        action: DiscoveryAuditAction,
        expected_previous_hash: [u8; 32],
    },
}

impl DiscoveryStateMutation {
    pub fn conflict_key(&self) -> Vec<u8> {
        match self {
            Self::InstallSafetyKernel { .. } => b"kernel".to_vec(),
            Self::UpsertSchema { record, .. } => {
                [b"schema".as_slice(), &record.proposal.id.0].concat()
            }
            Self::UpsertMapping { record, .. } => {
                [b"mapping".as_slice(), &record.hypothesis.id.0].concat()
            }
            Self::RecordEvaluation { operator, .. } => {
                [b"evaluation".as_slice(), &operator.0].concat()
            }
            Self::UpsertExperiment { experiment, .. } => {
                [b"experiment".as_slice(), &experiment.id.0].concat()
            }
            Self::AppendAudit { .. } => b"audit-head".to_vec(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct Versioned<T> {
    committed_lsn: u64,
    value: T,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryStateSnapshot {
    pub lsn: u64,
    pub safety_kernel: Option<ImmutableSafetyKernel>,
    pub schemas: Vec<GovernedSchemaRecord>,
    pub mappings: Vec<GovernedMappingRecord>,
    pub evaluations: BTreeMap<DiscoveredOperatorId, CompetitiveOperatorEvaluation>,
    pub experiments: Vec<ActiveExperimentProposal>,
    pub audit_entries: Vec<DiscoveryAuditEntry>,
}

#[derive(Default)]
pub struct GovernedDiscoveryState {
    safety_kernel: RwLock<Vec<Versioned<ImmutableSafetyKernel>>>,
    schemas: RwLock<BTreeMap<EvolvedSchemaId, Vec<Versioned<GovernedSchemaRecord>>>>,
    mappings: RwLock<BTreeMap<MappingHypothesisId, Vec<Versioned<GovernedMappingRecord>>>>,
    evaluations:
        RwLock<BTreeMap<DiscoveredOperatorId, Vec<Versioned<CompetitiveOperatorEvaluation>>>>,
    experiments: RwLock<BTreeMap<ExperimentProposalId, Vec<Versioned<ActiveExperimentProposal>>>>,
    audit: RwLock<Vec<DiscoveryAuditEntry>>,
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum DiscoveryStateError {
    #[error("immutable safety kernel cannot be replaced")]
    SafetyKernelReplacement,
    #[error("safety kernel is invalid")]
    InvalidSafetyKernel,
    #[error("immutable safety kernel must be committed before governed discovery state")]
    MissingSafetyKernel,
    #[error("governance authority is required for this transition")]
    MissingAuthority,
    #[error("state transition from {from} to {to} is forbidden")]
    ForbiddenTransition { from: String, to: String },
    #[error("expected previous state does not match replicated state")]
    PreviousStateMismatch,
    #[error("content-addressed discovery definition changed during transition")]
    DefinitionChanged,
    #[error("commit LSN {incoming} is not newer than {current}")]
    NonMonotonicCommit { current: u64, incoming: u64 },
    #[error("audit previous hash does not match the replicated audit head")]
    AuditChainMismatch,
}

impl GovernedDiscoveryState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn prevalidate(
        &self,
        mutation: &DiscoveryStateMutation,
        commit_lsn: u64,
    ) -> Result<(), DiscoveryStateError> {
        if !matches!(mutation, DiscoveryStateMutation::InstallSafetyKernel { .. }) {
            self.verify_kernel_installed()?;
        }
        match mutation {
            DiscoveryStateMutation::InstallSafetyKernel { kernel } => {
                kernel
                    .verify()
                    .map_err(|_| DiscoveryStateError::InvalidSafetyKernel)?;
                if let Some(existing) = self.safety_kernel.read().last() {
                    if existing.value.digest() != kernel.digest() {
                        return Err(DiscoveryStateError::SafetyKernelReplacement);
                    }
                    require_newer(existing.committed_lsn, commit_lsn)?;
                }
            }
            DiscoveryStateMutation::UpsertSchema {
                record,
                expected_previous,
            } => {
                let schemas = self.schemas.read();
                let previous = schemas
                    .get(&record.proposal.id)
                    .and_then(|versions| versions.last());
                validate_schema(record, *expected_previous, previous, commit_lsn)?;
            }
            DiscoveryStateMutation::UpsertMapping {
                record,
                expected_previous,
            } => {
                let mappings = self.mappings.read();
                let previous = mappings
                    .get(&record.hypothesis.id)
                    .and_then(|versions| versions.last());
                validate_mapping(record, *expected_previous, previous, commit_lsn)?;
            }
            DiscoveryStateMutation::RecordEvaluation { operator, .. } => {
                if let Some(previous) = self
                    .evaluations
                    .read()
                    .get(operator)
                    .and_then(|versions| versions.last())
                {
                    require_newer(previous.committed_lsn, commit_lsn)?;
                }
            }
            DiscoveryStateMutation::UpsertExperiment {
                experiment,
                expected_previous,
            } => {
                let experiments = self.experiments.read();
                let previous = experiments
                    .get(&experiment.id)
                    .and_then(|versions| versions.last());
                validate_experiment(experiment, *expected_previous, previous, commit_lsn)?;
            }
            DiscoveryStateMutation::AppendAudit {
                expected_previous_hash,
                ..
            } => {
                let actual = self
                    .audit
                    .read()
                    .last()
                    .map_or([0; 32], |entry| entry.entry_hash);
                if actual != *expected_previous_hash {
                    return Err(DiscoveryStateError::AuditChainMismatch);
                }
                if let Some(previous) = self.audit.read().last() {
                    require_newer_or_equal(previous.lsn, commit_lsn)?;
                }
            }
        }
        Ok(())
    }

    pub fn verify_kernel_installed(&self) -> Result<(), DiscoveryStateError> {
        self.safety_kernel
            .read()
            .last()
            .ok_or(DiscoveryStateError::MissingSafetyKernel)?
            .value
            .verify()
            .map_err(|_| DiscoveryStateError::InvalidSafetyKernel)
    }

    pub fn apply(
        &self,
        mutation: DiscoveryStateMutation,
        commit_lsn: u64,
    ) -> Result<(), DiscoveryStateError> {
        self.prevalidate(&mutation, commit_lsn)?;
        match mutation {
            DiscoveryStateMutation::InstallSafetyKernel { kernel } => {
                self.safety_kernel.write().push(Versioned {
                    committed_lsn: commit_lsn,
                    value: kernel,
                });
            }
            DiscoveryStateMutation::UpsertSchema { record, .. } => {
                self.schemas
                    .write()
                    .entry(record.proposal.id)
                    .or_default()
                    .push(Versioned {
                        committed_lsn: commit_lsn,
                        value: record,
                    });
            }
            DiscoveryStateMutation::UpsertMapping { record, .. } => {
                self.mappings
                    .write()
                    .entry(record.hypothesis.id)
                    .or_default()
                    .push(Versioned {
                        committed_lsn: commit_lsn,
                        value: record,
                    });
            }
            DiscoveryStateMutation::RecordEvaluation {
                operator,
                evaluation,
            } => {
                self.evaluations
                    .write()
                    .entry(operator)
                    .or_default()
                    .push(Versioned {
                        committed_lsn: commit_lsn,
                        value: evaluation,
                    });
            }
            DiscoveryStateMutation::UpsertExperiment { experiment, .. } => {
                self.experiments
                    .write()
                    .entry(experiment.id)
                    .or_default()
                    .push(Versioned {
                        committed_lsn: commit_lsn,
                        value: experiment,
                    });
            }
            DiscoveryStateMutation::AppendAudit {
                action,
                expected_previous_hash,
            } => {
                let mut audit = self.audit.write();
                let sequence = audit.len() as u64;
                let entry_hash = audit_hash(sequence, commit_lsn, &action, expected_previous_hash);
                audit.push(DiscoveryAuditEntry {
                    sequence,
                    lsn: commit_lsn,
                    action,
                    previous_hash: expected_previous_hash,
                    entry_hash,
                });
            }
        }
        Ok(())
    }

    pub fn snapshot_at(&self, lsn: u64) -> DiscoveryStateSnapshot {
        DiscoveryStateSnapshot {
            lsn,
            safety_kernel: latest_at(&self.safety_kernel.read(), lsn),
            schemas: latest_map_at(&self.schemas.read(), lsn),
            mappings: latest_map_at(&self.mappings.read(), lsn),
            evaluations: latest_btree_at(&self.evaluations.read(), lsn),
            experiments: latest_map_at(&self.experiments.read(), lsn),
            audit_entries: self
                .audit
                .read()
                .iter()
                .filter(|entry| entry.lsn <= lsn)
                .cloned()
                .collect(),
        }
    }

    pub fn replace_from_snapshot(&self, snapshot: DiscoveryStateSnapshot) {
        self.safety_kernel.write().clear();
        self.schemas.write().clear();
        self.mappings.write().clear();
        self.evaluations.write().clear();
        self.experiments.write().clear();
        self.audit.write().clear();
        if let Some(kernel) = snapshot.safety_kernel {
            self.safety_kernel.write().push(Versioned {
                committed_lsn: snapshot.lsn,
                value: kernel,
            });
        }
        for schema in snapshot.schemas {
            self.schemas
                .write()
                .entry(schema.proposal.id)
                .or_default()
                .push(Versioned {
                    committed_lsn: snapshot.lsn,
                    value: schema,
                });
        }
        for mapping in snapshot.mappings {
            self.mappings
                .write()
                .entry(mapping.hypothesis.id)
                .or_default()
                .push(Versioned {
                    committed_lsn: snapshot.lsn,
                    value: mapping,
                });
        }
        for (operator, evaluation) in snapshot.evaluations {
            self.evaluations
                .write()
                .entry(operator)
                .or_default()
                .push(Versioned {
                    committed_lsn: snapshot.lsn,
                    value: evaluation,
                });
        }
        for experiment in snapshot.experiments {
            self.experiments
                .write()
                .entry(experiment.id)
                .or_default()
                .push(Versioned {
                    committed_lsn: snapshot.lsn,
                    value: experiment,
                });
        }
        *self.audit.write() = snapshot.audit_entries;
    }

    pub(crate) fn copy_all_to(&self, target: &GovernedDiscoveryState) {
        *target.safety_kernel.write() = self.safety_kernel.read().clone();
        *target.schemas.write() = self.schemas.read().clone();
        *target.mappings.write() = self.mappings.read().clone();
        *target.evaluations.write() = self.evaluations.read().clone();
        *target.experiments.write() = self.experiments.read().clone();
        *target.audit.write() = self.audit.read().clone();
    }
}

fn validate_schema(
    record: &GovernedSchemaRecord,
    expected: Option<SchemaProposalState>,
    previous: Option<&Versioned<GovernedSchemaRecord>>,
    commit_lsn: u64,
) -> Result<(), DiscoveryStateError> {
    match previous {
        None => {
            if expected.is_some() || record.proposal.state != SchemaProposalState::Proposed {
                return Err(DiscoveryStateError::PreviousStateMismatch);
            }
        }
        Some(previous) => {
            require_newer(previous.committed_lsn, commit_lsn)?;
            if previous.value.proposal.kind != record.proposal.kind {
                return Err(DiscoveryStateError::DefinitionChanged);
            }
            if expected != Some(previous.value.proposal.state)
                || !schema_transition(previous.value.proposal.state, record.proposal.state)
            {
                return Err(DiscoveryStateError::ForbiddenTransition {
                    from: format!("{:?}", previous.value.proposal.state),
                    to: format!("{:?}", record.proposal.state),
                });
            }
        }
    }
    if matches!(
        record.proposal.state,
        SchemaProposalState::ShadowValidated | SchemaProposalState::Admitted
    ) && record
        .validation
        .as_ref()
        .is_none_or(|validation| !validation.passed)
    {
        return Err(DiscoveryStateError::ForbiddenTransition {
            from: "unvalidated".to_string(),
            to: format!("{:?}", record.proposal.state),
        });
    }
    if record.proposal.state == SchemaProposalState::Admitted && record.authority.is_none() {
        return Err(DiscoveryStateError::MissingAuthority);
    }
    Ok(())
}

fn validate_mapping(
    record: &GovernedMappingRecord,
    expected: Option<MappingLifecycle>,
    previous: Option<&Versioned<GovernedMappingRecord>>,
    commit_lsn: u64,
) -> Result<(), DiscoveryStateError> {
    match previous {
        None => {
            if expected.is_some() || record.hypothesis.lifecycle != MappingLifecycle::Proposed {
                return Err(DiscoveryStateError::PreviousStateMismatch);
            }
        }
        Some(previous) => {
            require_newer(previous.committed_lsn, commit_lsn)?;
            if previous.value.hypothesis.left != record.hypothesis.left
                || previous.value.hypothesis.right != record.hypothesis.right
            {
                return Err(DiscoveryStateError::DefinitionChanged);
            }
            if expected != Some(previous.value.hypothesis.lifecycle)
                || !mapping_transition(
                    previous.value.hypothesis.lifecycle,
                    record.hypothesis.lifecycle,
                )
            {
                return Err(DiscoveryStateError::ForbiddenTransition {
                    from: format!("{:?}", previous.value.hypothesis.lifecycle),
                    to: format!("{:?}", record.hypothesis.lifecycle),
                });
            }
        }
    }
    if matches!(
        record.hypothesis.lifecycle,
        MappingLifecycle::ShadowValidated | MappingLifecycle::Confirmed
    ) && record
        .validation
        .as_ref()
        .is_none_or(|validation| !validation.passed)
    {
        return Err(DiscoveryStateError::ForbiddenTransition {
            from: "unvalidated".to_string(),
            to: format!("{:?}", record.hypothesis.lifecycle),
        });
    }
    if record.hypothesis.lifecycle == MappingLifecycle::Confirmed && record.authority.is_none() {
        return Err(DiscoveryStateError::MissingAuthority);
    }
    Ok(())
}

fn validate_experiment(
    experiment: &ActiveExperimentProposal,
    expected: Option<ExperimentStatus>,
    previous: Option<&Versioned<ActiveExperimentProposal>>,
    commit_lsn: u64,
) -> Result<(), DiscoveryStateError> {
    match previous {
        None => {
            if expected.is_some() || experiment.status != ExperimentStatus::Proposed {
                return Err(DiscoveryStateError::PreviousStateMismatch);
            }
        }
        Some(previous) => {
            require_newer(previous.committed_lsn, commit_lsn)?;
            if expected != Some(previous.value.status)
                || !experiment_transition(previous.value.status, experiment.status)
            {
                return Err(DiscoveryStateError::ForbiddenTransition {
                    from: format!("{:?}", previous.value.status),
                    to: format!("{:?}", experiment.status),
                });
            }
        }
    }
    if experiment.status == ExperimentStatus::Authorized
        && experiment.requires_external_authorization
        && experiment.authorization.is_none()
    {
        return Err(DiscoveryStateError::MissingAuthority);
    }
    if matches!(
        experiment.status,
        ExperimentStatus::Authorized | ExperimentStatus::Running | ExperimentStatus::Completed
    ) && experiment.authorization.is_none()
    {
        return Err(DiscoveryStateError::MissingAuthority);
    }
    if experiment.status == ExperimentStatus::Completed && experiment.result.is_none() {
        return Err(DiscoveryStateError::ForbiddenTransition {
            from: "running-without-result".to_string(),
            to: "Completed".to_string(),
        });
    }
    Ok(())
}

fn schema_transition(from: SchemaProposalState, to: SchemaProposalState) -> bool {
    matches!(
        (from, to),
        (
            SchemaProposalState::Proposed,
            SchemaProposalState::FalsificationTesting
        ) | (SchemaProposalState::Proposed, SchemaProposalState::Rejected)
            | (
                SchemaProposalState::FalsificationTesting,
                SchemaProposalState::ShadowValidated
            )
            | (
                SchemaProposalState::FalsificationTesting,
                SchemaProposalState::Rejected
            )
            | (
                SchemaProposalState::ShadowValidated,
                SchemaProposalState::Admitted
            )
            | (
                SchemaProposalState::Admitted,
                SchemaProposalState::Deprecated
            )
    ) || from == to
}

fn mapping_transition(from: MappingLifecycle, to: MappingLifecycle) -> bool {
    matches!(
        (from, to),
        (
            MappingLifecycle::Proposed,
            MappingLifecycle::FalsificationTesting
        ) | (MappingLifecycle::Proposed, MappingLifecycle::Rejected)
            | (
                MappingLifecycle::FalsificationTesting,
                MappingLifecycle::ShadowValidated
            )
            | (
                MappingLifecycle::FalsificationTesting,
                MappingLifecycle::Rejected
            )
            | (
                MappingLifecycle::ShadowValidated,
                MappingLifecycle::Confirmed
            )
            | (MappingLifecycle::Confirmed, MappingLifecycle::Deprecated)
    ) || from == to
}

fn experiment_transition(from: ExperimentStatus, to: ExperimentStatus) -> bool {
    matches!(
        (from, to),
        (ExperimentStatus::Proposed, ExperimentStatus::Authorized)
            | (ExperimentStatus::Proposed, ExperimentStatus::Rejected)
            | (ExperimentStatus::Proposed, ExperimentStatus::Expired)
            | (ExperimentStatus::Authorized, ExperimentStatus::Running)
            | (ExperimentStatus::Authorized, ExperimentStatus::Rejected)
            | (ExperimentStatus::Running, ExperimentStatus::Completed)
            | (ExperimentStatus::Running, ExperimentStatus::Rejected)
    ) || from == to
}

fn require_newer(current: u64, incoming: u64) -> Result<(), DiscoveryStateError> {
    if incoming <= current {
        return Err(DiscoveryStateError::NonMonotonicCommit { current, incoming });
    }
    Ok(())
}

fn require_newer_or_equal(current: u64, incoming: u64) -> Result<(), DiscoveryStateError> {
    if incoming < current {
        return Err(DiscoveryStateError::NonMonotonicCommit { current, incoming });
    }
    Ok(())
}

fn latest_at<T: Clone>(versions: &[Versioned<T>], lsn: u64) -> Option<T> {
    versions
        .iter()
        .rev()
        .find(|version| version.committed_lsn <= lsn)
        .map(|version| version.value.clone())
}

fn latest_map_at<K: Ord, T: Clone>(values: &BTreeMap<K, Vec<Versioned<T>>>, lsn: u64) -> Vec<T> {
    values
        .values()
        .filter_map(|versions| latest_at(versions, lsn))
        .collect()
}

fn latest_btree_at<K: Ord + Copy, T: Clone>(
    values: &BTreeMap<K, Vec<Versioned<T>>>,
    lsn: u64,
) -> BTreeMap<K, T> {
    values
        .iter()
        .filter_map(|(key, versions)| latest_at(versions, lsn).map(|value| (*key, value)))
        .collect()
}

fn audit_hash(
    sequence: u64,
    lsn: u64,
    action: &DiscoveryAuditAction,
    previous_hash: [u8; 32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_DISCOVERY_AUDIT_CHAIN_V1");
    hasher.update(sequence.to_le_bytes());
    hasher.update(lsn.to_le_bytes());
    hasher.update(previous_hash);
    hasher.update(bincode::serialize(action).expect("audit actions are serializable"));
    hasher.finalize().into()
}
