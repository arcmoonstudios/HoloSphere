//! Behavioral cross-domain concept mapping without vocabulary dependence.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::discovery::knowledge::KnowledgeSnapshot;
use crate::learning::discovery::model::{ConceptId, DiscoveryOutcome, DomainId, FeatureId};
use crate::learning::integrity::EmpiricalRootId;
use crate::relation::RoleId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct MappingHypothesisId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MappingLifecycle {
    Proposed,
    FalsificationTesting,
    ShadowValidated,
    Confirmed,
    Rejected,
    Deprecated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappingValidationPolicy {
    pub min_independent_roots: usize,
    pub min_role_similarity_q32: i64,
    pub min_outcome_similarity_q32: i64,
    pub min_total_score_q32: i64,
}

impl Default for MappingValidationPolicy {
    fn default() -> Self {
        Self {
            min_independent_roots: 2,
            min_role_similarity_q32: q32(1, 2),
            min_outcome_similarity_q32: q32(1, 2),
            min_total_score_q32: q32(2, 3),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappingValidation {
    pub role_similarity_q32: i64,
    pub outcome_similarity_q32: i64,
    pub capability_similarity_q32: i64,
    pub temporal_similarity_q32: i64,
    pub total_score_q32: i64,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
    pub endpoints_observed: bool,
    pub passed: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConceptBehavior {
    pub domain: DomainId,
    pub concept: ConceptId,
    pub capabilities: BTreeSet<FeatureId>,
    pub structural_roles: BTreeSet<[u8; 32]>,
    pub outcome_associations: BTreeSet<(DiscoveryOutcome, Option<u64>)>,
    pub temporal_positions: BTreeSet<i8>,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConceptMappingHypothesis {
    pub id: MappingHypothesisId,
    pub left: (DomainId, ConceptId),
    pub right: (DomainId, ConceptId),
    pub lifecycle: MappingLifecycle,
    pub role_similarity_q32: i64,
    pub outcome_similarity_q32: i64,
    pub capability_similarity_q32: i64,
    pub total_score_q32: i64,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
    /// Other candidates sharing either endpoint remain live until adjudicated.
    pub competing_hypotheses: BTreeSet<MappingHypothesisId>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappingInductionPolicy {
    pub min_role_similarity_q32: i64,
    pub min_outcome_similarity_q32: i64,
    pub min_total_score_q32: i64,
    pub min_independent_roots: usize,
    pub max_hypotheses: usize,
}

impl Default for MappingInductionPolicy {
    fn default() -> Self {
        Self {
            min_role_similarity_q32: q32(1, 2),
            min_outcome_similarity_q32: q32(1, 2),
            min_total_score_q32: q32(2, 3),
            min_independent_roots: 2,
            max_hypotheses: 1_024,
        }
    }
}

pub fn derive_concept_behaviors(snapshot: &KnowledgeSnapshot) -> Vec<ConceptBehavior> {
    derive_concept_behaviors_excluding_roots(snapshot, &BTreeSet::new())
}

pub(crate) fn derive_concept_behaviors_excluding_roots(
    snapshot: &KnowledgeSnapshot,
    excluded_roots: &BTreeSet<EmpiricalRootId>,
) -> Vec<ConceptBehavior> {
    let mut behaviors = BTreeMap::<(DomainId, ConceptId), ConceptBehavior>::new();
    for profile in snapshot.concept_profiles.iter().filter(|profile| {
        profile.certified_evidence
            && profile
                .empirical_roots
                .iter()
                .any(|root| !excluded_roots.contains(root))
    }) {
        let behavior = behaviors
            .entry((profile.domain, profile.concept))
            .or_insert_with(|| ConceptBehavior {
                domain: profile.domain,
                concept: profile.concept,
                ..ConceptBehavior::default()
            });
        behavior
            .capabilities
            .extend(profile.capabilities.iter().copied());
        behavior
            .empirical_roots
            .extend(profile.empirical_roots.iter().copied());
        behavior
            .temporal_positions
            .extend(profile.roles.iter().map(|role| role.temporal_position));
    }

    for edge in snapshot.certified_hyperedges().filter(|edge| {
        edge.empirical_roots
            .iter()
            .any(|root| !excluded_roots.contains(root))
    }) {
        let role_ordinals = canonical_role_ordinals(edge.members.iter().map(|member| member.role));
        for member in &edge.members {
            let behavior = behaviors
                .entry((edge.domain, member.concept))
                .or_insert_with(|| ConceptBehavior {
                    domain: edge.domain,
                    concept: member.concept,
                    ..ConceptBehavior::default()
                });
            let mut hasher = Sha256::new();
            hasher.update(b"HOLOSPHERE_BEHAVIORAL_ROLE_V1");
            hasher.update(edge.structural_signature());
            hasher.update(role_ordinals[&member.role].to_le_bytes());
            behavior.structural_roles.insert(hasher.finalize().into());
            behavior.outcome_associations.insert((
                edge.outcome,
                edge.observed_resolution.map(|resolution| resolution.0),
            ));
            behavior
                .empirical_roots
                .extend(edge.empirical_roots.iter().copied());
        }
    }
    behaviors.into_values().collect()
}

/// Learns correspondence from role behavior, temporal position, capabilities,
/// and outcomes. Names and local relation/type IDs are never compared.
pub fn learn_concept_mappings(
    behaviors: &[ConceptBehavior],
    policy: MappingInductionPolicy,
) -> Vec<ConceptMappingHypothesis> {
    let mut hypotheses = Vec::new();
    for (index, left) in behaviors.iter().enumerate() {
        for right in behaviors.iter().skip(index + 1) {
            if left.domain == right.domain {
                continue;
            }
            let roots: BTreeSet<_> = left
                .empirical_roots
                .union(&right.empirical_roots)
                .copied()
                .collect();
            if roots.len() < policy.min_independent_roots {
                continue;
            }
            let role_similarity = jaccard_q32(&left.structural_roles, &right.structural_roles);
            let outcome_similarity =
                jaccard_q32(&left.outcome_associations, &right.outcome_associations);
            let capability_similarity = jaccard_q32(&left.capabilities, &right.capabilities);
            let temporal_similarity =
                jaccard_q32(&left.temporal_positions, &right.temporal_positions);
            let total = ((role_similarity as i128 * 4
                + outcome_similarity as i128 * 3
                + capability_similarity as i128 * 2
                + temporal_similarity as i128)
                / 10) as i64;
            if role_similarity < policy.min_role_similarity_q32
                || outcome_similarity < policy.min_outcome_similarity_q32
                || total < policy.min_total_score_q32
            {
                continue;
            }
            let (left_key, right_key) =
                ordered_pair((left.domain, left.concept), (right.domain, right.concept));
            let id = mapping_id(left_key, right_key);
            hypotheses.push(ConceptMappingHypothesis {
                id,
                left: left_key,
                right: right_key,
                lifecycle: MappingLifecycle::Proposed,
                role_similarity_q32: role_similarity,
                outcome_similarity_q32: outcome_similarity,
                capability_similarity_q32: capability_similarity,
                total_score_q32: total,
                empirical_roots: roots,
                competing_hypotheses: BTreeSet::new(),
            });
        }
    }
    let endpoint_index: BTreeMap<_, BTreeSet<_>> = hypotheses.iter().fold(
        BTreeMap::<(DomainId, ConceptId), BTreeSet<MappingHypothesisId>>::new(),
        |mut index, hypothesis| {
            index
                .entry(hypothesis.left)
                .or_default()
                .insert(hypothesis.id);
            index
                .entry(hypothesis.right)
                .or_default()
                .insert(hypothesis.id);
            index
        },
    );
    for hypothesis in &mut hypotheses {
        hypothesis.competing_hypotheses = endpoint_index[&hypothesis.left]
            .union(&endpoint_index[&hypothesis.right])
            .copied()
            .filter(|candidate| *candidate != hypothesis.id)
            .collect();
    }
    hypotheses.sort_by(|left, right| {
        right
            .total_score_q32
            .cmp(&left.total_score_q32)
            .then_with(|| left.id.cmp(&right.id))
    });
    hypotheses.truncate(policy.max_hypotheses);
    hypotheses
}

/// Re-scores a proposed mapping using a separate validation snapshot. This is
/// deliberately endpoint- and behavior-based; names never participate.
pub fn validate_concept_mapping(
    hypothesis: &ConceptMappingHypothesis,
    validation_snapshot: &KnowledgeSnapshot,
    policy: MappingValidationPolicy,
) -> MappingValidation {
    let behaviors =
        derive_concept_behaviors_excluding_roots(validation_snapshot, &hypothesis.empirical_roots);
    let by_key: BTreeMap<_, _> = behaviors
        .iter()
        .map(|behavior| ((behavior.domain, behavior.concept), behavior))
        .collect();
    let (Some(left), Some(right)) = (by_key.get(&hypothesis.left), by_key.get(&hypothesis.right))
    else {
        return MappingValidation::default();
    };
    let role_similarity_q32 = jaccard_q32(&left.structural_roles, &right.structural_roles);
    let outcome_similarity_q32 =
        jaccard_q32(&left.outcome_associations, &right.outcome_associations);
    let capability_similarity_q32 = jaccard_q32(&left.capabilities, &right.capabilities);
    let temporal_similarity_q32 = jaccard_q32(&left.temporal_positions, &right.temporal_positions);
    let total_score_q32 = ((role_similarity_q32 as i128 * 4
        + outcome_similarity_q32 as i128 * 3
        + capability_similarity_q32 as i128 * 2
        + temporal_similarity_q32 as i128)
        / 10) as i64;
    let mut empirical_roots = left
        .empirical_roots
        .union(&right.empirical_roots)
        .copied()
        .collect::<BTreeSet<_>>();
    empirical_roots.retain(|root| !hypothesis.empirical_roots.contains(root));
    let passed = empirical_roots.len() >= policy.min_independent_roots
        && role_similarity_q32 >= policy.min_role_similarity_q32
        && outcome_similarity_q32 >= policy.min_outcome_similarity_q32
        && total_score_q32 >= policy.min_total_score_q32;
    MappingValidation {
        role_similarity_q32,
        outcome_similarity_q32,
        capability_similarity_q32,
        temporal_similarity_q32,
        total_score_q32,
        empirical_roots,
        endpoints_observed: true,
        passed,
    }
}

/// Runtime resolver built only from confirmed mappings. Proposed and competing
/// hypotheses remain inspectable but cannot silently affect reasoning.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfirmedConceptMappingIndex {
    canonical: BTreeMap<(DomainId, ConceptId), (DomainId, ConceptId)>,
}

impl ConfirmedConceptMappingIndex {
    pub fn from_confirmed(hypotheses: &[ConceptMappingHypothesis]) -> Self {
        let mut adjacency = BTreeMap::<_, BTreeSet<_>>::new();
        for hypothesis in hypotheses
            .iter()
            .filter(|hypothesis| hypothesis.lifecycle == MappingLifecycle::Confirmed)
        {
            adjacency
                .entry(hypothesis.left)
                .or_default()
                .insert(hypothesis.right);
            adjacency
                .entry(hypothesis.right)
                .or_default()
                .insert(hypothesis.left);
        }
        let mut canonical = BTreeMap::new();
        let mut visited = BTreeSet::new();
        for start in adjacency.keys().copied() {
            if !visited.insert(start) {
                continue;
            }
            let mut stack = vec![start];
            let mut component = BTreeSet::from([start]);
            while let Some(current) = stack.pop() {
                for peer in adjacency.get(&current).into_iter().flatten() {
                    if visited.insert(*peer) {
                        component.insert(*peer);
                        stack.push(*peer);
                    }
                }
            }
            let representative = *component.iter().next().expect("component is non-empty");
            for member in component {
                canonical.insert(member, representative);
            }
        }
        Self { canonical }
    }

    pub fn resolve(&self, concept: (DomainId, ConceptId)) -> (DomainId, ConceptId) {
        self.canonical.get(&concept).copied().unwrap_or(concept)
    }

    pub fn equivalent(&self, left: (DomainId, ConceptId), right: (DomainId, ConceptId)) -> bool {
        self.resolve(left) == self.resolve(right)
    }
}

fn canonical_role_ordinals(roles: impl Iterator<Item = RoleId>) -> BTreeMap<RoleId, u16> {
    let mut counts = BTreeMap::<RoleId, u16>::new();
    for role in roles {
        *counts.entry(role).or_default() += 1;
    }
    let mut ranked: Vec<_> = counts.into_iter().collect();
    ranked.sort_by_key(|(role, count)| (*count, *role));
    ranked
        .into_iter()
        .enumerate()
        .map(|(ordinal, (role, _))| (role, ordinal as u16))
        .collect()
}

fn jaccard_q32<T: Ord>(left: &BTreeSet<T>, right: &BTreeSet<T>) -> i64 {
    let union = left.union(right).count();
    if union == 0 {
        return 1i64 << 32;
    }
    q32(left.intersection(right).count(), union)
}

const fn q32(numerator: usize, denominator: usize) -> i64 {
    ((numerator as i128 * (1i128 << 32)) / denominator as i128) as i64
}

fn ordered_pair(
    left: (DomainId, ConceptId),
    right: (DomainId, ConceptId),
) -> ((DomainId, ConceptId), (DomainId, ConceptId)) {
    if left <= right {
        (left, right)
    } else {
        (right, left)
    }
}

fn mapping_id(left: (DomainId, ConceptId), right: (DomainId, ConceptId)) -> MappingHypothesisId {
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_CONCEPT_MAPPING_V1");
    hasher.update(left.0.0.to_le_bytes());
    hasher.update(left.1.0.to_le_bytes());
    hasher.update(right.0.0.to_le_bytes());
    hasher.update(right.1.0.to_le_bytes());
    MappingHypothesisId(hasher.finalize().into())
}
