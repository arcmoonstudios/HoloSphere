/* holosphere/src/learning/inference/candidate.rs */
//!▫~•◦-------------------------------‣
//! # Epistemically Constrained Inference Proposals & Candidate Bundles
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the universal output shape emitted by all hypothesis generators.
//! Every generated entity, relation, or bundle strictly begins in the `Provisional`
//! epistemic state.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::id::{EntityId, VersionId};
use crate::entity::status::EpistemicStatus;
use crate::learning::inference::contract::InferenceMethodId;
use crate::learning::inference::trace::InferenceTrace;
use crate::relation::id::{RelationTypeId, RoleId};

/// Unique identifier for an inference candidate.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct InferenceCandidateId(pub u64);

/// Scoped identifier for an entity proposed within an inference bundle.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct CandidateEntityId(pub u64);

/// Reference to an entity in a candidate role binding, pointing either to an existing
/// durable entity or to a candidate entity proposed in the same bundle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CandidateEntityRef {
    Existing(EntityId),
    Proposed(CandidateEntityId),
}

/// Geometric artifact associated with a derived candidate entity or hypothesis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InferenceGeometryArtifact {
    E8Coordinates([f32; 8]),
    BivectorGrade2([f32; 28]),
    Matrix8x8(Vec<f32>),
    None,
}

/// Score associated with an inference candidate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct InferenceScore {
    /// Fixed-point confidence in Q32.32 format.
    pub confidence_q32: i64,
    /// Raw floating-point score (e.g. geometric residual, friction, or similarity).
    pub raw_floating: f32,
}

impl Eq for InferenceScore {}

/// Role binding within a candidate relation proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CandidateRoleBinding {
    pub entity: CandidateEntityRef,
    pub role_id: RoleId,
}

/// Proposed derived entity candidate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DerivedEntityProposal {
    pub local_id: CandidateEntityId,
    pub epistemic_status: EpistemicStatus,
    pub geometry: InferenceGeometryArtifact,
    pub provenance: InferenceTrace,
}

impl DerivedEntityProposal {
    pub fn new_provisional(
        local_id: CandidateEntityId,
        geometry: InferenceGeometryArtifact,
        provenance: InferenceTrace,
    ) -> Self {
        Self {
            local_id,
            epistemic_status: EpistemicStatus::Provisional,
            geometry,
            provenance,
        }
    }
}

/// Proposed relation candidate.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelationProposal {
    pub candidate_id: InferenceCandidateId,
    pub proposed_relation_type: RelationTypeId,
    pub bindings: Vec<CandidateRoleBinding>,
    pub score: InferenceScore,
    pub epistemic_status: EpistemicStatus,
    pub trace: InferenceTrace,
}

impl RelationProposal {
    pub fn new_provisional(
        candidate_id: InferenceCandidateId,
        proposed_relation_type: RelationTypeId,
        bindings: Vec<CandidateRoleBinding>,
        score: InferenceScore,
        trace: InferenceTrace,
    ) -> Self {
        Self {
            candidate_id,
            proposed_relation_type,
            bindings,
            score,
            epistemic_status: EpistemicStatus::Provisional,
            trace,
        }
    }
}

/// Parameters for a phase shift transformation on the E8 manifold.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PhaseShiftArtifact {
    pub axis: [f32; 8],
    pub angle: f32,
    pub gain: f32,
}

/// Immutable geometric artifact recording an evolutionary phase transition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvolutionArtifact {
    pub method: InferenceMethodId,
    pub method_version: u32,
    pub source_entity: EntityId,
    pub source_version: VersionId,
    pub source_coords: [f32; 8],
    pub resulting_coords: [f32; 8],
    pub phase_shift: PhaseShiftArtifact,
    pub state_fingerprint: [u8; 32],
    pub trace: InferenceTrace,
}

/// Proposed evolutionary transition for an existing entity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvolutionProposal {
    pub entity_id: EntityId,
    pub source_version: VersionId,
    pub epistemic_status: EpistemicStatus,
    pub artifact: EvolutionArtifact,
}

impl EvolutionProposal {
    pub fn new_provisional(
        entity_id: EntityId,
        source_version: VersionId,
        artifact: EvolutionArtifact,
    ) -> Self {
        Self {
            entity_id,
            source_version,
            epistemic_status: EpistemicStatus::Provisional,
            artifact,
        }
    }
}

/// Atomic bundle of proposed entities, relations, and evolutionary transitions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InferenceProposalBundle {
    pub entities: Vec<DerivedEntityProposal>,
    pub relations: Vec<RelationProposal>,
    pub evolutions: Vec<EvolutionProposal>,
}

/// Universal candidate proposal emitted by an inference generator.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum InferenceProposal {
    Relation(RelationProposal),
    DerivedEntity(DerivedEntityProposal),
    Evolution(EvolutionProposal),
    Bundle(InferenceProposalBundle),
}

/// Legacy alias for relational inference candidates.
pub type InferenceCandidate = RelationProposal;

impl InferenceProposal {
    pub fn as_relation(&self) -> Option<&RelationProposal> {
        match self {
            Self::Relation(r) => Some(r),
            _ => None,
        }
    }

    pub fn as_bundle(&self) -> Option<&InferenceProposalBundle> {
        match self {
            Self::Bundle(b) => Some(b),
            _ => None,
        }
    }

    pub fn as_evolution(&self) -> Option<&EvolutionProposal> {
        match self {
            Self::Evolution(e) => Some(e),
            _ => None,
        }
    }
}
