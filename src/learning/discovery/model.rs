//! Canonical evidence contracts for governed autonomous discovery.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::learning::integrity::EmpiricalRootId;

#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct DiscoveryCaseId(pub u64);

#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct DomainId(pub u64);

#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct FeatureId(pub u64);

#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ResolutionId(pub u64);

#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ConceptId(pub u64);

/// Empirical outcome of applying a resolution in a structured context.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DiscoveryOutcome {
    Successful,
    Failed,
    Unknown,
}

/// Prevents discovery evidence from being reused as admission evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EvidencePartition {
    Discovery,
    Validation,
}

/// One certified problem-solving episode used for discovery and falsification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryCase {
    pub id: DiscoveryCaseId,
    pub domain: DomainId,
    pub snapshot_lsn: u64,
    /// Canonical structural/context features. These are semantic IDs, not free text.
    pub features: BTreeSet<FeatureId>,
    pub observed_resolution: Option<ResolutionId>,
    pub outcome: DiscoveryOutcome,
    /// Discovery cases may induce operators. Only validation cases may admit them.
    pub evidence_partition: EvidencePartition,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
    /// Only certified evidence is eligible to induce or validate reasoning laws.
    pub certified_evidence: bool,
}

/// A role position independent of domain vocabulary and relation type IDs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct StructuralRole {
    pub relation_arity: u16,
    pub role_ordinal: u16,
    pub peer_role_count: u16,
    pub temporal_position: i8,
}

/// Structural evidence about one domain-local concept.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConceptProfile {
    pub domain: DomainId,
    pub concept: ConceptId,
    pub capabilities: BTreeSet<FeatureId>,
    pub roles: BTreeSet<StructuralRole>,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
    pub certified_evidence: bool,
}

/// Input to one deterministic discovery cycle.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveryCorpus {
    pub cases: Vec<DiscoveryCase>,
    pub concept_profiles: Vec<ConceptProfile>,
}

#[inline]
pub(crate) fn ratio_q32(numerator: usize, denominator: usize) -> i64 {
    if denominator == 0 {
        return 0;
    }
    (((numerator as i128) << 32) / denominator as i128) as i64
}
