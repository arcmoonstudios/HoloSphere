/* holosphere/src/learning/discovery/hyper_motif.rs */
//!▫~•◦-------------------------------‣
//! # Temporal Hypergraph Motif Representation
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Models recurring structural, causal, and temporal patterns extracted across
//! N-ary hyperedges, entity roles, timestamps, and task outcome states.
//!
//! ## Key Capabilities
//! - **N-Ary Topology:** Captures multi-entity relational patterns beyond simple pairwise graphs.
//! - **Causal Annotation:** Links motif structures directly to empirical success/failure utility.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::discovery::knowledge::{KnowledgeSnapshot, TemporalHyperedge};
use crate::learning::discovery::model::{DiscoveryOutcome, DomainId, FeatureId, ResolutionId};
use crate::learning::integrity::EmpiricalRootId;
use crate::relation::RelationId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HypergraphMotifId(pub [u8; 32]);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum HypergraphMotifKind {
    RepeatedNaryStructure {
        structural_signature: [u8; 32],
        arity: u16,
        canonical_roles: Vec<u16>,
    },
    CausalSequence {
        predecessor: [u8; 32],
        successor: [u8; 32],
        max_gap_lsn: u64,
        resulting_outcome: DiscoveryOutcome,
    },
    BeforeAfterOutcome {
        before: [u8; 32],
        after: [u8; 32],
        outcome_before: DiscoveryOutcome,
        outcome_after: DiscoveryOutcome,
    },
    DomainInvariantRoleArrangement {
        structural_signature: [u8; 32],
        canonical_roles: Vec<u16>,
    },
    OutcomeAnomaly {
        structural_signature: [u8; 32],
        expected: DiscoveryOutcome,
        unexpected: DiscoveryOutcome,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalHypergraphMotif {
    pub id: HypergraphMotifId,
    pub kind: HypergraphMotifKind,
    pub supporting_relations: BTreeSet<RelationId>,
    pub supporting_domains: BTreeSet<DomainId>,
    pub context_features: BTreeSet<FeatureId>,
    pub associated_resolutions: BTreeSet<ResolutionId>,
    pub empirical_roots: BTreeSet<EmpiricalRootId>,
    pub support: usize,
    pub contradictions: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HypergraphMotifPolicy {
    pub min_support: usize,
    pub min_domains: usize,
    pub min_independent_roots: usize,
    pub anomaly_min_baseline: usize,
    pub max_temporal_gap_lsn: u64,
    pub max_motifs: usize,
}

impl Default for HypergraphMotifPolicy {
    fn default() -> Self {
        Self {
            min_support: 3,
            min_domains: 2,
            min_independent_roots: 3,
            anomaly_min_baseline: 4,
            max_temporal_gap_lsn: 10_000,
            max_motifs: 1_024,
        }
    }
}

pub fn mine_temporal_hypergraph_motifs(
    snapshot: &KnowledgeSnapshot,
    policy: HypergraphMotifPolicy,
) -> Vec<TemporalHypergraphMotif> {
    let edges: Vec<_> = snapshot.certified_hyperedges().collect();
    let by_id: BTreeMap<_, _> = edges.iter().map(|edge| (edge.id, *edge)).collect();
    let mut by_shape = BTreeMap::<[u8; 32], Vec<&TemporalHyperedge>>::new();
    for edge in &edges {
        by_shape
            .entry(edge.structural_signature())
            .or_default()
            .push(edge);
    }

    let mut motifs = Vec::new();
    for (signature, group) in &by_shape {
        if qualifies(group, policy) {
            let canonical_roles = group[0].canonical_member_roles();
            motifs.push(build_motif(
                HypergraphMotifKind::RepeatedNaryStructure {
                    structural_signature: *signature,
                    arity: group[0].arity() as u16,
                    canonical_roles: canonical_roles.clone(),
                },
                group.iter().copied(),
                0,
            ));
            if group
                .iter()
                .map(|edge| edge.domain)
                .collect::<BTreeSet<_>>()
                .len()
                >= policy.min_domains
            {
                motifs.push(build_motif(
                    HypergraphMotifKind::DomainInvariantRoleArrangement {
                        structural_signature: *signature,
                        canonical_roles,
                    },
                    group.iter().copied(),
                    0,
                ));
            }
        }

        let mut outcome_groups = BTreeMap::<DiscoveryOutcome, Vec<&TemporalHyperedge>>::new();
        for edge in group {
            outcome_groups.entry(edge.outcome).or_default().push(edge);
        }
        if let Some((expected, baseline)) = outcome_groups
            .iter()
            .max_by_key(|(outcome, edges)| (edges.len(), *outcome))
        {
            if baseline.len() >= policy.anomaly_min_baseline {
                for (unexpected, anomalies) in &outcome_groups {
                    if unexpected == expected
                        || anomalies.is_empty()
                        || anomalies.len() >= baseline.len()
                    {
                        continue;
                    }
                    let mut combined = baseline.clone();
                    combined.extend(anomalies.iter().copied());
                    motifs.push(build_motif(
                        HypergraphMotifKind::OutcomeAnomaly {
                            structural_signature: *signature,
                            expected: *expected,
                            unexpected: *unexpected,
                        },
                        combined.into_iter(),
                        anomalies.len(),
                    ));
                }
            }
        }
    }

    let mut causal_groups = BTreeMap::<([u8; 32], [u8; 32], DiscoveryOutcome), Vec<_>>::new();
    for successor in &edges {
        for predecessor_id in &successor.causal_predecessors {
            let Some(predecessor) = by_id.get(predecessor_id) else {
                continue;
            };
            let gap = successor
                .interval
                .valid_from_lsn
                .saturating_sub(predecessor.interval.valid_from_lsn);
            if gap > policy.max_temporal_gap_lsn {
                continue;
            }
            causal_groups
                .entry((
                    predecessor.structural_signature(),
                    successor.structural_signature(),
                    successor.outcome,
                ))
                .or_default()
                .push((*predecessor, *successor, gap));
        }
    }
    for ((predecessor, successor, outcome), sequences) in causal_groups {
        let successors: Vec<_> = sequences
            .iter()
            .map(|(_, successor, _)| *successor)
            .collect();
        if !qualifies(&successors, policy) {
            continue;
        }
        let max_gap = sequences.iter().map(|(_, _, gap)| *gap).max().unwrap_or(0);
        let all_edges: Vec<_> = sequences
            .iter()
            .flat_map(|(left, right, _)| [*left, *right])
            .collect();
        motifs.push(build_motif(
            HypergraphMotifKind::CausalSequence {
                predecessor,
                successor,
                max_gap_lsn: max_gap,
                resulting_outcome: outcome,
            },
            all_edges.into_iter(),
            0,
        ));
    }

    // Before/after pairs are discovered from repeated member sets, not relation names.
    let mut by_members = BTreeMap::<Vec<_>, Vec<&TemporalHyperedge>>::new();
    for edge in &edges {
        let mut members: Vec<_> = edge.members.iter().map(|member| member.concept).collect();
        members.sort_unstable();
        members.dedup();
        by_members.entry(members).or_default().push(edge);
    }
    let mut transition_groups =
        BTreeMap::<([u8; 32], [u8; 32], DiscoveryOutcome, DiscoveryOutcome), Vec<_>>::new();
    for group in by_members.values_mut() {
        group.sort_by_key(|edge| edge.interval.valid_from_lsn);
        for pair in group.windows(2) {
            transition_groups
                .entry((
                    pair[0].structural_signature(),
                    pair[1].structural_signature(),
                    pair[0].outcome,
                    pair[1].outcome,
                ))
                .or_default()
                .push((pair[0], pair[1]));
        }
    }
    for ((before, after, outcome_before, outcome_after), transitions) in transition_groups {
        let after_edges: Vec<_> = transitions.iter().map(|(_, after)| *after).collect();
        if !qualifies(&after_edges, policy) {
            continue;
        }
        let all_edges: Vec<_> = transitions
            .iter()
            .flat_map(|(left, right)| [*left, *right])
            .collect();
        motifs.push(build_motif(
            HypergraphMotifKind::BeforeAfterOutcome {
                before,
                after,
                outcome_before,
                outcome_after,
            },
            all_edges.into_iter(),
            0,
        ));
    }

    motifs.sort_by(|left, right| {
        right
            .support
            .cmp(&left.support)
            .then_with(|| left.id.cmp(&right.id))
    });
    motifs.dedup_by_key(|motif| motif.id);
    motifs.truncate(policy.max_motifs);
    motifs
}

fn qualifies(edges: &[&TemporalHyperedge], policy: HypergraphMotifPolicy) -> bool {
    edges.len() >= policy.min_support
        && edges
            .iter()
            .map(|edge| edge.domain)
            .collect::<BTreeSet<_>>()
            .len()
            >= policy.min_domains
        && edges
            .iter()
            .flat_map(|edge| edge.empirical_roots.iter())
            .collect::<BTreeSet<_>>()
            .len()
            >= policy.min_independent_roots
}

fn build_motif<'a>(
    kind: HypergraphMotifKind,
    edges: impl Iterator<Item = &'a TemporalHyperedge>,
    contradictions: usize,
) -> TemporalHypergraphMotif {
    let edges: Vec<_> = edges.collect();
    let id = motif_id(&kind);
    TemporalHypergraphMotif {
        id,
        kind,
        supporting_relations: edges.iter().map(|edge| edge.id).collect(),
        supporting_domains: edges.iter().map(|edge| edge.domain).collect(),
        context_features: edges
            .iter()
            .flat_map(|edge| edge.context_features.iter().copied())
            .collect(),
        associated_resolutions: edges
            .iter()
            .filter_map(|edge| edge.observed_resolution)
            .collect(),
        empirical_roots: edges
            .iter()
            .flat_map(|edge| edge.empirical_roots.iter().copied())
            .collect(),
        support: edges.len().saturating_sub(contradictions),
        contradictions,
    }
}

fn motif_id(kind: &HypergraphMotifKind) -> HypergraphMotifId {
    let encoded = bincode::serialize(kind).expect("motif kinds are serializable");
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_TEMPORAL_HYPERGRAPH_MOTIF_V1");
    hasher.update(encoded);
    HypergraphMotifId(hasher.finalize().into())
}
