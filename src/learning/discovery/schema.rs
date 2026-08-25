//! Governed induction of entity classes, N-ary relation types, roles, and hierarchies.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::discovery::knowledge::KnowledgeSnapshot;
use crate::learning::discovery::model::{ConceptId, DomainId, FeatureId, StructuralRole};
use crate::learning::integrity::EmpiricalRootId;
use crate::relation::{RelationType, RelationTypeState, RoleSchema};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EvolvedSchemaId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaProposalState {
    Proposed,
    FalsificationTesting,
    ShadowValidated,
    Admitted,
    Rejected,
    Deprecated,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaValidationPolicy {
    pub min_observations: usize,
    pub min_domains: usize,
    pub min_independent_roots: usize,
    pub min_structural_accuracy_q32: i64,
}

impl Default for SchemaValidationPolicy {
    fn default() -> Self {
        Self {
            min_observations: 4,
            min_domains: 2,
            min_independent_roots: 2,
            min_structural_accuracy_q32: q32(3, 4),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaValidation {
    pub observations: usize,
    pub structural_matches: usize,
    pub contradictions: usize,
    pub domains: BTreeSet<DomainId>,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
    pub structural_accuracy_q32: i64,
    pub counterexample_roots: BTreeSet<EmpiricalRootId>,
    pub passed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposedRole {
    pub ordinal: u16,
    pub min_count: u16,
    pub max_count: u16,
    pub required: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvolvedSchemaKind {
    EntityClass {
        capabilities: BTreeSet<FeatureId>,
        structural_roles: BTreeSet<StructuralRole>,
        members: BTreeSet<(DomainId, ConceptId)>,
    },
    RelationType {
        structural_signature: [u8; 32],
        arity: u16,
        roles: Vec<ProposedRole>,
    },
    ConceptEquivalence {
        concepts: BTreeSet<(DomainId, ConceptId)>,
    },
    Generalization {
        general: EvolvedSchemaId,
        specializations: BTreeSet<EvolvedSchemaId>,
    },
    Specialization {
        specialized: EvolvedSchemaId,
        general: EvolvedSchemaId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvolvedSchemaProposal {
    pub id: EvolvedSchemaId,
    pub state: SchemaProposalState,
    pub kind: EvolvedSchemaKind,
    pub supporting_domains: BTreeSet<DomainId>,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
    /// Every empirical root visible in the induction snapshot. Validation may
    /// not reuse any of them, even if a root did not support this proposal.
    pub discovery_snapshot_roots: BTreeSet<EmpiricalRootId>,
    pub proposed_at_lsn: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaInductionPolicy {
    pub min_domains: usize,
    pub min_members: usize,
    pub min_independent_roots: usize,
    pub max_proposals: usize,
}

impl Default for SchemaInductionPolicy {
    fn default() -> Self {
        Self {
            min_domains: 2,
            min_members: 2,
            min_independent_roots: 2,
            max_proposals: 512,
        }
    }
}

/// Induces proposals only. Admission is intentionally impossible in this stage.
pub fn induce_evolved_schemas(
    snapshot: &KnowledgeSnapshot,
    policy: SchemaInductionPolicy,
) -> Vec<EvolvedSchemaProposal> {
    let discovery_snapshot_roots: BTreeSet<_> = snapshot
        .concept_profiles
        .iter()
        .flat_map(|profile| profile.empirical_roots.iter().copied())
        .chain(
            snapshot
                .hyperedges
                .iter()
                .flat_map(|edge| edge.empirical_roots.iter().copied()),
        )
        .chain(
            snapshot
                .cases
                .iter()
                .flat_map(|case| case.empirical_roots.iter().copied()),
        )
        .collect();
    type EntitySignature = (BTreeSet<FeatureId>, BTreeSet<StructuralRole>);
    let mut entity_groups = BTreeMap::<EntitySignature, Vec<_>>::new();
    for profile in snapshot
        .concept_profiles
        .iter()
        .filter(|profile| profile.certified_evidence)
    {
        entity_groups
            .entry((profile.capabilities.clone(), profile.roles.clone()))
            .or_default()
            .push(profile);
    }

    let mut proposals = Vec::new();
    let mut entity_classes = Vec::<(EvolvedSchemaId, BTreeSet<FeatureId>)>::new();
    for ((capabilities, roles), members) in entity_groups {
        let domains: BTreeSet<_> = members.iter().map(|member| member.domain).collect();
        let roots: BTreeSet<_> = members
            .iter()
            .flat_map(|member| member.empirical_roots.iter().copied())
            .collect();
        if domains.len() < policy.min_domains
            || members.len() < policy.min_members
            || roots.len() < policy.min_independent_roots
        {
            continue;
        }
        let concepts: BTreeSet<_> = members
            .iter()
            .map(|member| (member.domain, member.concept))
            .collect();
        let id = schema_id(b"ENTITY_CLASS", |hasher| {
            hash_features(hasher, &capabilities);
            hash_roles(hasher, &roles);
        });
        proposals.push(EvolvedSchemaProposal {
            id,
            state: SchemaProposalState::Proposed,
            kind: EvolvedSchemaKind::EntityClass {
                capabilities: capabilities.clone(),
                structural_roles: roles,
                members: concepts.clone(),
            },
            supporting_domains: domains.clone(),
            empirical_roots: roots.clone(),
            discovery_snapshot_roots: discovery_snapshot_roots.clone(),
            proposed_at_lsn: snapshot.lsn,
        });
        if concepts.len() > 1 {
            let equivalence_id = schema_id(b"CONCEPT_EQUIVALENCE", |hasher| {
                for (domain, concept) in &concepts {
                    hasher.update(domain.0.to_le_bytes());
                    hasher.update(concept.0.to_le_bytes());
                }
            });
            proposals.push(EvolvedSchemaProposal {
                id: equivalence_id,
                state: SchemaProposalState::Proposed,
                kind: EvolvedSchemaKind::ConceptEquivalence { concepts },
                supporting_domains: domains,
                empirical_roots: roots,
                discovery_snapshot_roots: discovery_snapshot_roots.clone(),
                proposed_at_lsn: snapshot.lsn,
            });
        }
        entity_classes.push((id, capabilities));
    }

    let mut relation_groups = BTreeMap::<[u8; 32], Vec<_>>::new();
    for edge in snapshot.certified_hyperedges() {
        relation_groups
            .entry(edge.structural_signature())
            .or_default()
            .push(edge);
    }
    for (signature, edges) in relation_groups {
        let domains: BTreeSet<_> = edges.iter().map(|edge| edge.domain).collect();
        let roots: BTreeSet<_> = edges
            .iter()
            .flat_map(|edge| edge.empirical_roots.iter().copied())
            .collect();
        if domains.len() < policy.min_domains
            || edges.len() < policy.min_members
            || roots.len() < policy.min_independent_roots
        {
            continue;
        }
        let first = edges[0];
        let mut counts = BTreeMap::<u16, Vec<u16>>::new();
        for edge in &edges {
            let mut per_role = BTreeMap::<u16, u16>::new();
            for role in edge.canonical_member_roles() {
                *per_role.entry(role).or_default() += 1;
            }
            for (role, count) in per_role {
                counts.entry(role).or_default().push(count);
            }
        }
        let roles = counts
            .into_iter()
            .map(|(ordinal, values)| ProposedRole {
                ordinal,
                min_count: *values.iter().min().unwrap_or(&0),
                max_count: *values.iter().max().unwrap_or(&0),
                required: values.len() == edges.len() && values.iter().all(|count| *count > 0),
            })
            .collect();
        let id = EvolvedSchemaId(signature);
        proposals.push(EvolvedSchemaProposal {
            id,
            state: SchemaProposalState::Proposed,
            kind: EvolvedSchemaKind::RelationType {
                structural_signature: signature,
                arity: first.arity() as u16,
                roles,
            },
            supporting_domains: domains,
            empirical_roots: roots,
            discovery_snapshot_roots: discovery_snapshot_roots.clone(),
            proposed_at_lsn: snapshot.lsn,
        });
    }

    // Subset structure induces explicit generalization/specialization proposals.
    entity_classes.sort_by_key(|(id, _)| *id);
    for (general_id, general_features) in &entity_classes {
        let specializations: BTreeSet<_> = entity_classes
            .iter()
            .filter(|(candidate_id, candidate_features)| {
                candidate_id != general_id
                    && general_features.len() < candidate_features.len()
                    && general_features.is_subset(candidate_features)
            })
            .map(|(candidate_id, _)| *candidate_id)
            .collect();
        if specializations.is_empty() {
            continue;
        }
        let id = schema_id(b"GENERALIZATION", |hasher| {
            hasher.update(general_id.0);
            for specialized in &specializations {
                hasher.update(specialized.0);
            }
        });
        proposals.push(EvolvedSchemaProposal {
            id,
            state: SchemaProposalState::Proposed,
            kind: EvolvedSchemaKind::Generalization {
                general: *general_id,
                specializations: specializations.clone(),
            },
            supporting_domains: BTreeSet::new(),
            empirical_roots: BTreeSet::new(),
            discovery_snapshot_roots: discovery_snapshot_roots.clone(),
            proposed_at_lsn: snapshot.lsn,
        });
        for specialized in specializations {
            let specialization_id = schema_id(b"SPECIALIZATION", |hasher| {
                hasher.update(specialized.0);
                hasher.update(general_id.0);
            });
            proposals.push(EvolvedSchemaProposal {
                id: specialization_id,
                state: SchemaProposalState::Proposed,
                kind: EvolvedSchemaKind::Specialization {
                    specialized,
                    general: *general_id,
                },
                supporting_domains: BTreeSet::new(),
                empirical_roots: BTreeSet::new(),
                discovery_snapshot_roots: discovery_snapshot_roots.clone(),
                proposed_at_lsn: snapshot.lsn,
            });
        }
    }

    proposals.sort_by_key(|proposal| proposal.id);
    proposals.dedup_by_key(|proposal| proposal.id);
    proposals.truncate(policy.max_proposals);
    proposals
}

/// Converts a relation proposal to the canonical relation-schema subsystem while
/// preserving the mandatory Proposed boundary.
pub fn materialize_proposed_relation_type(
    proposal: &EvolvedSchemaProposal,
    provenance_id: u64,
) -> Option<RelationType> {
    let mut relation = materialize_relation_type(proposal, provenance_id)?;
    relation.state = RelationTypeState::Proposed;
    Some(relation)
}

/// Materializes the governed lifecycle into the canonical relation catalog.
/// Rejected schema hypotheses are deliberately not materialized.
pub fn materialize_relation_type(
    proposal: &EvolvedSchemaProposal,
    provenance_id: u64,
) -> Option<RelationType> {
    let EvolvedSchemaKind::RelationType {
        structural_signature,
        roles,
        ..
    } = &proposal.kind
    else {
        return None;
    };
    let type_id = u32::from_le_bytes(structural_signature[..4].try_into().ok()?);
    let state = match proposal.state {
        SchemaProposalState::Proposed
        | SchemaProposalState::FalsificationTesting
        | SchemaProposalState::ShadowValidated => RelationTypeState::Proposed,
        SchemaProposalState::Admitted => RelationTypeState::Admitted,
        SchemaProposalState::Deprecated => RelationTypeState::Deprecated,
        SchemaProposalState::Rejected => return None,
    };
    let role_schemas: Vec<_> = roles
        .iter()
        .map(|role| RoleSchema {
            role_id: role.ordinal,
            name: Arc::from(format!("induced_role_{}", role.ordinal)),
            min_count: role.min_count,
            max_count: role.max_count,
            required: role.required,
        })
        .collect();
    Some(RelationType {
        id: type_id,
        name: Arc::from(format!("induced_relation_{type_id:016x}")),
        schema_version: 1,
        state,
        structural_fingerprint: RelationType::compute_structural_fingerprint(
            type_id,
            1,
            &role_schemas,
        ),
        roles: role_schemas,
        binary_projection: None,
        provenance_id,
    })
}

/// Tests an induced schema against a distinct pinned knowledge snapshot. The
/// proposal's discovery support is never counted as validation evidence.
pub fn validate_evolved_schema(
    proposal: &EvolvedSchemaProposal,
    validation: &KnowledgeSnapshot,
    known_proposals: &[EvolvedSchemaProposal],
    policy: SchemaValidationPolicy,
) -> SchemaValidation {
    let mut result = SchemaValidation::default();
    match &proposal.kind {
        EvolvedSchemaKind::EntityClass {
            capabilities,
            structural_roles,
            ..
        } => {
            for profile in validation
                .concept_profiles
                .iter()
                .filter(|profile| profile.certified_evidence)
            {
                if profile
                    .empirical_roots
                    .iter()
                    .all(|root| proposal.discovery_snapshot_roots.contains(root))
                {
                    continue;
                }
                let applicable = !capabilities.is_disjoint(&profile.capabilities)
                    || !structural_roles.is_disjoint(&profile.roles);
                if !applicable {
                    continue;
                }
                let matches = capabilities.is_subset(&profile.capabilities)
                    && structural_roles.is_subset(&profile.roles);
                record_schema_observation(
                    &mut result,
                    profile.domain,
                    &profile.empirical_roots,
                    matches,
                );
            }
        }
        EvolvedSchemaKind::RelationType {
            structural_signature,
            arity,
            ..
        } => {
            for edge in validation
                .certified_hyperedges()
                .filter(|edge| edge.arity() == *arity as usize)
            {
                if edge
                    .empirical_roots
                    .iter()
                    .all(|root| proposal.discovery_snapshot_roots.contains(root))
                {
                    continue;
                }
                record_schema_observation(
                    &mut result,
                    edge.domain,
                    &edge.empirical_roots,
                    edge.structural_signature() == *structural_signature,
                );
            }
        }
        EvolvedSchemaKind::ConceptEquivalence { concepts } => {
            let behaviors =
                crate::learning::discovery::mapping::derive_concept_behaviors_excluding_roots(
                    validation,
                    &proposal.discovery_snapshot_roots,
                );
            let by_key: BTreeMap<_, _> = behaviors
                .iter()
                .map(|behavior| ((behavior.domain, behavior.concept), behavior))
                .collect();
            for concept in concepts {
                let Some(behavior) = by_key.get(concept) else {
                    continue;
                };
                let peers: Vec<_> = concepts
                    .iter()
                    .filter(|peer| *peer != concept)
                    .filter_map(|peer| by_key.get(peer))
                    .collect();
                let matches = peers.iter().any(|peer| {
                    behavior.structural_roles == peer.structural_roles
                        && behavior.outcome_associations == peer.outcome_associations
                });
                record_schema_observation(
                    &mut result,
                    behavior.domain,
                    &behavior.empirical_roots,
                    matches,
                );
            }
        }
        EvolvedSchemaKind::Generalization {
            general,
            specializations,
        } => {
            let known: BTreeSet<_> = known_proposals
                .iter()
                .map(|candidate| candidate.id)
                .collect();
            let matches = known.contains(general)
                && specializations
                    .iter()
                    .all(|specialized| known.contains(specialized));
            result.observations = 1;
            result.structural_matches = usize::from(matches);
            result.contradictions = usize::from(!matches);
            result.domains = proposal.supporting_domains.clone();
            result.empirical_roots = proposal.empirical_roots.clone();
        }
        EvolvedSchemaKind::Specialization {
            specialized,
            general,
        } => {
            let known: BTreeSet<_> = known_proposals
                .iter()
                .map(|candidate| candidate.id)
                .collect();
            let matches = known.contains(specialized) && known.contains(general);
            result.observations = 1;
            result.structural_matches = usize::from(matches);
            result.contradictions = usize::from(!matches);
            result.domains = proposal.supporting_domains.clone();
            result.empirical_roots = proposal.empirical_roots.clone();
        }
    }
    result
        .empirical_roots
        .retain(|root| !proposal.discovery_snapshot_roots.contains(root));
    result
        .counterexample_roots
        .retain(|root| !proposal.discovery_snapshot_roots.contains(root));
    result.structural_accuracy_q32 = ratio_q32(result.structural_matches, result.observations);
    result.passed = result.observations >= policy.min_observations
        && result.domains.len() >= policy.min_domains
        && result.empirical_roots.len() >= policy.min_independent_roots
        && result.structural_accuracy_q32 >= policy.min_structural_accuracy_q32;
    result
}

fn record_schema_observation(
    result: &mut SchemaValidation,
    domain: DomainId,
    roots: &BTreeSet<EmpiricalRootId>,
    matches: bool,
) {
    result.observations += 1;
    result.domains.insert(domain);
    result.empirical_roots.extend(roots.iter().copied());
    if matches {
        result.structural_matches += 1;
    } else {
        result.contradictions += 1;
        result.counterexample_roots.extend(roots.iter().copied());
    }
}

fn schema_id(namespace: &[u8], feed: impl FnOnce(&mut Sha256)) -> EvolvedSchemaId {
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_EVOLVED_SCHEMA_V1");
    hasher.update(namespace);
    feed(&mut hasher);
    EvolvedSchemaId(hasher.finalize().into())
}

fn hash_features(hasher: &mut Sha256, features: &BTreeSet<FeatureId>) {
    for feature in features {
        hasher.update(feature.0.to_le_bytes());
    }
}

fn hash_roles(hasher: &mut Sha256, roles: &BTreeSet<StructuralRole>) {
    for role in roles {
        hasher.update(role.relation_arity.to_le_bytes());
        hasher.update(role.role_ordinal.to_le_bytes());
        hasher.update(role.peer_role_count.to_le_bytes());
        hasher.update(role.temporal_position.to_le_bytes());
    }
}

const fn q32(numerator: usize, denominator: usize) -> i64 {
    if denominator == 0 {
        0
    } else {
        ((numerator as i128 * (1i128 << 32)) / denominator as i128) as i64
    }
}

const fn ratio_q32(numerator: usize, denominator: usize) -> i64 {
    q32(numerator, denominator)
}
