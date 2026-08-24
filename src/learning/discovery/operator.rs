//! Safe declarative reasoning operators synthesized from discovered motifs.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::learning::discovery::mining::{DiscoveredMotif, MotifId};
use crate::learning::discovery::model::{
    DiscoveryCase, DiscoveryCaseId, DomainId, FeatureId, ResolutionId,
};
use crate::learning::discovery::{
    ConditionExpression, DslEffect, HypergraphMotifId, OperatorProgram, ReasoningContext,
    ResourceCostBounds, execute_program,
};
use crate::learning::integrity::EmpiricalRootId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct DiscoveredOperatorId(pub [u8; 32]);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OperatorLifecycle {
    Generated,
    Provisional,
    FalsificationTesting,
    Shadow,
    ShadowValidated,
    Admitted,
    Monitored,
    Rejected,
    Deprecated,
    Superseded,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum OperatorPredicate {
    HasFeature(FeatureId),
    LacksFeature(FeatureId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorEffect {
    ProposeResolution(ResolutionId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GovernanceAuthority {
    ReplicatedPolicy { policy_id: u64, version: u32 },
    HumanApproval { actor_id: u64 },
}

/// Inspectable and resource-bounded reasoning law. It is data, never executable
/// native code, and can only propose a resolution.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeclarativeOperator {
    pub id: DiscoveredOperatorId,
    pub version: u32,
    pub predicates: Vec<OperatorPredicate>,
    pub effect: OperatorEffect,
    pub lifecycle: OperatorLifecycle,
    pub source_motifs: Vec<MotifId>,
    pub source_hypergraph_motifs: Vec<HypergraphMotifId>,
    pub program: OperatorProgram,
    pub epistemic: OperatorEpistemicRecord,
    pub admission_authority: Option<GovernanceAuthority>,
    pub committed_lsn: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorEpistemicRecord {
    pub training_evidence: BTreeSet<DiscoveryCaseId>,
    pub validation_evidence: BTreeSet<DiscoveryCaseId>,
    pub applicable_domains: BTreeSet<DomainId>,
    pub applicable_contexts: BTreeSet<FeatureId>,
    pub counterexamples: BTreeSet<DiscoveryCaseId>,
    pub provenance_roots: BTreeSet<EmpiricalRootId>,
    pub ancestry: BTreeSet<DiscoveredOperatorId>,
    pub previous_version: Option<DiscoveredOperatorId>,
    pub predictive_accuracy_q32: i64,
    pub calibration_error_q32: i64,
    pub uncertainty_q32: i64,
    pub monitoring_observations: u64,
    pub monitoring_failures: u64,
}

impl DeclarativeOperator {
    pub fn from_motif(motif: &DiscoveredMotif) -> Self {
        let predicates: Vec<_> = motif
            .conditions
            .iter()
            .copied()
            .map(OperatorPredicate::HasFeature)
            .collect();
        let effect = OperatorEffect::ProposeResolution(motif.resolution);
        let program = OperatorProgram {
            condition: ConditionExpression::All(
                motif
                    .conditions
                    .iter()
                    .copied()
                    .map(ConditionExpression::FeaturePresent)
                    .collect(),
            ),
            effects: vec![DslEffect::ProposeResolution(motif.resolution)],
            bounds: ResourceCostBounds::default(),
        };
        Self {
            id: canonical_operator_id(&predicates, effect, &program, 1, None),
            version: 1,
            predicates,
            effect,
            lifecycle: OperatorLifecycle::Generated,
            source_motifs: vec![motif.id],
            source_hypergraph_motifs: Vec::new(),
            program,
            epistemic: OperatorEpistemicRecord {
                applicable_domains: motif.supporting_domains.clone(),
                applicable_contexts: motif.conditions.iter().copied().collect(),
                provenance_roots: motif.empirical_roots.clone(),
                predictive_accuracy_q32: motif.precision_q32,
                uncertainty_q32: (1i64 << 32).saturating_sub(motif.precision_q32),
                ..OperatorEpistemicRecord::default()
            },
            admission_authority: None,
            committed_lsn: 0,
        }
    }

    pub fn matches(&self, case: &DiscoveryCase) -> bool {
        execute_program(
            &self.program,
            &ReasoningContext {
                case: Some(case.clone()),
                ..ReasoningContext::default()
            },
        )
        .is_ok_and(|result| result.matched && result.unsatisfied_constraints.is_empty())
    }

    pub fn proposed_resolution(&self) -> ResolutionId {
        match self.effect {
            OperatorEffect::ProposeResolution(resolution) => resolution,
        }
    }

    /// Recomputes the content-derived identity of the executable definition.
    pub fn canonical_id(&self) -> DiscoveredOperatorId {
        canonical_operator_id(
            &self.predicates,
            self.effect,
            &self.program,
            self.version,
            self.epistemic.previous_version,
        )
    }

    pub fn has_valid_identity(&self) -> bool {
        self.id == self.canonical_id()
    }

    pub fn from_program(
        program: OperatorProgram,
        source_motif: HypergraphMotifId,
        provenance_roots: BTreeSet<EmpiricalRootId>,
        applicable_domains: BTreeSet<DomainId>,
    ) -> Option<Self> {
        let resolution = program.effects.iter().find_map(|effect| match effect {
            DslEffect::ProposeResolution(resolution) => Some(*resolution),
            _ => None,
        })?;
        let predicates = flatten_legacy_predicates(&program.condition);
        let effect = OperatorEffect::ProposeResolution(resolution);
        let version = 1;
        let previous_version = None;
        Some(Self {
            id: canonical_operator_id(&predicates, effect, &program, version, previous_version),
            version,
            predicates,
            effect,
            lifecycle: OperatorLifecycle::Generated,
            source_motifs: Vec::new(),
            source_hypergraph_motifs: vec![source_motif],
            program,
            epistemic: OperatorEpistemicRecord {
                provenance_roots,
                applicable_domains,
                previous_version,
                ..OperatorEpistemicRecord::default()
            },
            admission_authority: None,
            committed_lsn: 0,
        })
    }

    pub fn revise_with_program(&self, program: OperatorProgram) -> Option<Self> {
        let resolution = program.effects.iter().find_map(|effect| match effect {
            DslEffect::ProposeResolution(resolution) => Some(*resolution),
            _ => None,
        })?;
        let predicates = flatten_legacy_predicates(&program.condition);
        let effect = OperatorEffect::ProposeResolution(resolution);
        let version = self.version.saturating_add(1);
        let previous_version = Some(self.id);
        let mut ancestry = self.epistemic.ancestry.clone();
        ancestry.insert(self.id);
        Some(Self {
            id: canonical_operator_id(&predicates, effect, &program, version, previous_version),
            version,
            predicates,
            effect,
            lifecycle: OperatorLifecycle::Generated,
            source_motifs: self.source_motifs.clone(),
            source_hypergraph_motifs: self.source_hypergraph_motifs.clone(),
            program,
            epistemic: OperatorEpistemicRecord {
                ancestry,
                previous_version,
                provenance_roots: self.epistemic.provenance_roots.clone(),
                ..OperatorEpistemicRecord::default()
            },
            admission_authority: None,
            committed_lsn: 0,
        })
    }
}

fn canonical_operator_id(
    predicates: &[OperatorPredicate],
    effect: OperatorEffect,
    program: &OperatorProgram,
    version: u32,
    previous_version: Option<DiscoveredOperatorId>,
) -> DiscoveredOperatorId {
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_DECLARATIVE_OPERATOR_V1");
    for predicate in predicates {
        match predicate {
            OperatorPredicate::HasFeature(feature) => {
                hasher.update([1]);
                hasher.update(feature.0.to_le_bytes());
            }
            OperatorPredicate::LacksFeature(feature) => {
                hasher.update([0]);
                hasher.update(feature.0.to_le_bytes());
            }
        }
    }
    match effect {
        OperatorEffect::ProposeResolution(resolution) => {
            hasher.update([1]);
            hasher.update(resolution.0.to_le_bytes());
        }
    }
    hasher.update(bincode::serialize(program).expect("operator programs are serializable"));
    hasher.update(version.to_le_bytes());
    if let Some(previous) = previous_version {
        hasher.update(previous.0);
    }
    DiscoveredOperatorId(hasher.finalize().into())
}

fn flatten_legacy_predicates(condition: &ConditionExpression) -> Vec<OperatorPredicate> {
    match condition {
        ConditionExpression::FeaturePresent(feature) => {
            vec![OperatorPredicate::HasFeature(*feature)]
        }
        ConditionExpression::FeatureAbsent(feature) => {
            vec![OperatorPredicate::LacksFeature(*feature)]
        }
        ConditionExpression::All(children) => children
            .iter()
            .flat_map(flatten_legacy_predicates)
            .collect(),
        _ => Vec::new(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NovelResolution {
    pub operator_id: DiscoveredOperatorId,
    pub resolution: ResolutionId,
    pub matched_predicates: Vec<OperatorPredicate>,
    pub source_motifs: Vec<MotifId>,
    pub source_hypergraph_motifs: Vec<HypergraphMotifId>,
}
