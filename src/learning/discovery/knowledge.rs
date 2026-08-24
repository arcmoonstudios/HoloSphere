//! Canonical temporal knowledge snapshot consumed by open-ended discovery.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::discovery::{
    ConceptId, ConceptProfile, DiscoveryCase, DiscoveryOutcome, DomainId, FeatureId, ResolutionId,
};
use crate::learning::integrity::EmpiricalRootId;
use crate::relation::{RelationId, RelationTypeId, RoleId};

#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct NumericAttributeId(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TemporalInterval {
    pub valid_from_lsn: u64,
    pub valid_until_lsn: Option<u64>,
}

impl TemporalInterval {
    pub fn contains(self, lsn: u64) -> bool {
        self.valid_from_lsn <= lsn && self.valid_until_lsn.is_none_or(|until| lsn < until)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HyperedgeMember {
    pub concept: ConceptId,
    pub role: RoleId,
}

/// One provenance-bearing N-ary observation. Concept and relation IDs are local
/// vocabulary; canonical structural signatures deliberately exclude them.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalHyperedge {
    pub id: RelationId,
    pub domain: DomainId,
    pub relation_type: RelationTypeId,
    pub members: Vec<HyperedgeMember>,
    pub interval: TemporalInterval,
    pub causal_predecessors: BTreeSet<RelationId>,
    pub context_features: BTreeSet<FeatureId>,
    pub numeric_context_q32: BTreeMap<NumericAttributeId, i64>,
    pub observed_resolution: Option<ResolutionId>,
    pub outcome: DiscoveryOutcome,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
    pub certified_evidence: bool,
}

impl TemporalHyperedge {
    pub fn arity(&self) -> usize {
        self.members.len()
    }

    /// Vocabulary-independent role/cardinality signature. Role IDs are replaced
    /// by deterministic ordinals derived from multiplicity and first occurrence.
    pub fn structural_signature(&self) -> [u8; 32] {
        let mut role_counts = BTreeMap::<RoleId, u16>::new();
        for member in &self.members {
            *role_counts.entry(member.role).or_default() += 1;
        }
        let mut multiplicities: Vec<_> = role_counts.values().copied().collect();
        multiplicities.sort_unstable();
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_RELABEL_INVARIANT_HYPEREDGE_V1");
        hasher.update((self.members.len() as u64).to_le_bytes());
        for count in multiplicities {
            hasher.update(count.to_le_bytes());
        }
        hasher.finalize().into()
    }

    pub fn canonical_member_roles(&self) -> Vec<u16> {
        let mut counts = BTreeMap::<RoleId, u16>::new();
        for member in &self.members {
            *counts.entry(member.role).or_default() += 1;
        }
        let mut ranked: Vec<_> = counts.into_iter().collect();
        ranked.sort_by_key(|(role, count)| (*count, *role));
        let ordinals: BTreeMap<_, _> = ranked
            .into_iter()
            .enumerate()
            .map(|(ordinal, (role, _))| (role, ordinal as u16))
            .collect();
        let mut roles: Vec<_> = self
            .members
            .iter()
            .map(|member| ordinals[&member.role])
            .collect();
        roles.sort_unstable();
        roles
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeSnapshot {
    pub lsn: u64,
    pub cases: Vec<DiscoveryCase>,
    pub concept_profiles: Vec<ConceptProfile>,
    pub hyperedges: Vec<TemporalHyperedge>,
}

impl KnowledgeSnapshot {
    pub fn certified_hyperedges(&self) -> impl Iterator<Item = &TemporalHyperedge> {
        self.hyperedges
            .iter()
            .filter(|edge| edge.certified_evidence && edge.interval.valid_from_lsn <= self.lsn)
    }
}
