/* holosphere/src/learning/discovery/mining.rs */
//!▫~•◦-------------------------------‣
//! # Deterministic Schema & Hypergraph Motif Mining
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Mines frequent subgraphs, temporal patterns, and candidate entity schema
//! hierarchies from empirical relational hyperedges under strict resource bounds.
//!
//! ## Key Capabilities
//! - **Bounded Search Depth:** Prevents combinatorial explosion during sub-isomorphism checks.
//! - **Support & Confidence Pruning:** Filters low-utility patterns before program synthesis.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::discovery::model::{
    ConceptId, ConceptProfile, DiscoveryCase, DiscoveryOutcome, DomainId, EvidencePartition,
    FeatureId, ResolutionId, StructuralRole, ratio_q32,
};
use crate::learning::integrity::EmpiricalRootId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SchemaProposalId(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InducedSchemaProposal {
    pub id: SchemaProposalId,
    pub defining_capabilities: BTreeSet<FeatureId>,
    pub defining_roles: BTreeSet<StructuralRole>,
    pub members: Vec<(DomainId, ConceptId)>,
    pub supporting_domains: BTreeSet<DomainId>,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
}

/// Groups concepts by vocabulary-independent structural signatures. Results are
/// proposals only; naming and ontology admission remain governed operations.
pub fn induce_schemas(
    profiles: &[ConceptProfile],
    min_domains: usize,
    min_members: usize,
) -> Vec<InducedSchemaProposal> {
    type Signature = (BTreeSet<FeatureId>, BTreeSet<StructuralRole>);
    let mut groups: BTreeMap<Signature, Vec<&ConceptProfile>> = BTreeMap::new();
    for profile in profiles.iter().filter(|profile| profile.certified_evidence) {
        groups
            .entry((profile.capabilities.clone(), profile.roles.clone()))
            .or_default()
            .push(profile);
    }

    let mut proposals = Vec::new();
    for ((capabilities, roles), mut members) in groups {
        members.sort_by_key(|member| (member.domain, member.concept));
        let domains: BTreeSet<_> = members.iter().map(|member| member.domain).collect();
        if domains.len() < min_domains || members.len() < min_members {
            continue;
        }
        let roots = members
            .iter()
            .flat_map(|member| member.empirical_roots.iter().copied())
            .collect();
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_INDUCED_SCHEMA_V1");
        for feature in &capabilities {
            hasher.update(feature.0.to_le_bytes());
        }
        for role in &roles {
            hasher.update(role.relation_arity.to_le_bytes());
            hasher.update(role.role_ordinal.to_le_bytes());
            hasher.update(role.peer_role_count.to_le_bytes());
            hasher.update(role.temporal_position.to_le_bytes());
        }
        let id = SchemaProposalId(hasher.finalize().into());
        proposals.push(InducedSchemaProposal {
            id,
            defining_capabilities: capabilities,
            defining_roles: roles,
            members: members
                .into_iter()
                .map(|member| (member.domain, member.concept))
                .collect(),
            supporting_domains: domains,
            empirical_roots: roots,
        });
    }
    proposals.sort_by_key(|proposal| proposal.id);
    proposals
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MotifMinerConfig {
    pub max_features_per_case: usize,
    pub max_condition_terms: usize,
    pub min_successes: usize,
    pub min_domains: usize,
    pub max_motifs: usize,
}

impl Default for MotifMinerConfig {
    fn default() -> Self {
        Self {
            max_features_per_case: 24,
            max_condition_terms: 4,
            min_successes: 3,
            min_domains: 2,
            max_motifs: 256,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MotifId(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveredMotif {
    pub id: MotifId,
    pub conditions: Vec<FeatureId>,
    pub resolution: ResolutionId,
    pub successes: usize,
    pub contradictions: usize,
    pub supporting_domains: BTreeSet<DomainId>,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
    pub precision_q32: i64,
}

#[derive(Default)]
struct MotifAccumulator {
    successes: usize,
    contradictions: usize,
    domains: BTreeSet<DomainId>,
    roots: BTreeSet<EmpiricalRootId>,
}

pub fn mine_motifs(cases: &[DiscoveryCase], config: MotifMinerConfig) -> Vec<DiscoveredMotif> {
    let max_terms = config.max_condition_terms.clamp(1, 8);
    let mut accumulators: BTreeMap<(Vec<FeatureId>, ResolutionId), MotifAccumulator> =
        BTreeMap::new();

    for case in cases.iter().filter(|case| {
        case.certified_evidence && case.evidence_partition == EvidencePartition::Discovery
    }) {
        let Some(resolution) = case.observed_resolution else {
            continue;
        };
        if matches!(case.outcome, DiscoveryOutcome::Unknown) {
            continue;
        }
        let features: Vec<_> = case
            .features
            .iter()
            .copied()
            .take(config.max_features_per_case)
            .collect();
        for size in 1..=max_terms.min(features.len()) {
            let mut selected = Vec::with_capacity(size);
            enumerate_combinations(&features, size, 0, &mut selected, &mut |conditions| {
                let accumulator = accumulators
                    .entry((conditions.to_vec(), resolution))
                    .or_default();
                match case.outcome {
                    DiscoveryOutcome::Successful => accumulator.successes += 1,
                    DiscoveryOutcome::Failed => accumulator.contradictions += 1,
                    DiscoveryOutcome::Unknown => {}
                }
                accumulator.domains.insert(case.domain);
                accumulator
                    .roots
                    .extend(case.empirical_roots.iter().copied());
            });
        }
    }

    let mut motifs = Vec::new();
    for ((conditions, resolution), accumulator) in accumulators {
        if accumulator.successes < config.min_successes
            || accumulator.domains.len() < config.min_domains
        {
            continue;
        }
        let total = accumulator.successes + accumulator.contradictions;
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_DISCOVERED_MOTIF_V1");
        for condition in &conditions {
            hasher.update(condition.0.to_le_bytes());
        }
        hasher.update(resolution.0.to_le_bytes());
        motifs.push(DiscoveredMotif {
            id: MotifId(hasher.finalize().into()),
            conditions,
            resolution,
            successes: accumulator.successes,
            contradictions: accumulator.contradictions,
            supporting_domains: accumulator.domains,
            empirical_roots: accumulator.roots,
            precision_q32: ratio_q32(accumulator.successes, total),
        });
    }
    motifs.sort_by(|left, right| {
        right
            .precision_q32
            .cmp(&left.precision_q32)
            .then_with(|| right.successes.cmp(&left.successes))
            .then_with(|| left.conditions.len().cmp(&right.conditions.len()))
            .then_with(|| left.id.cmp(&right.id))
    });
    motifs.truncate(config.max_motifs);
    motifs
}

fn enumerate_combinations(
    features: &[FeatureId],
    remaining: usize,
    start: usize,
    selected: &mut Vec<FeatureId>,
    visitor: &mut impl FnMut(&[FeatureId]),
) {
    if remaining == 0 {
        visitor(selected);
        return;
    }
    let last_start = features.len().saturating_sub(remaining);
    for index in start..=last_start {
        selected.push(features[index]);
        enumerate_combinations(features, remaining - 1, index + 1, selected, visitor);
        selected.pop();
    }
}
