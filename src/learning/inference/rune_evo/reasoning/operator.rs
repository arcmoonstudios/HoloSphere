/* holosphere/src/learning/inference/rune_evo/reasoning/operator.rs */
//!▫~•◦-------------------------------‣
//! # Grounded Reasoning Operators & Transform Witnesses
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Models compiled reasoning transforms grounded directly in HoloSphere
//! durable EntityIds, RelationTypeIds, evidence references, and ProvenanceIds.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

pub use crate::entity::id::DurableEvidenceRef;
use crate::entity::id::{EntityId, ProvenanceId};
use crate::learning::inference::rune_evo::reasoning::blade::Cl24Blade;
use crate::relation::id::RelationTypeId;

/// Unique identifier for a compiled reasoning operator.
#[repr(transparent)]
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize,
)]
pub struct ReasoningOperatorId(pub u64);

/// Reference metadata for Rune-EVO operator classification.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuneOperatorClass {
    Identify,
    Cause,
    Constrain,
    Optimize,
    Analogize,
    Contrast,
    Generalize,
    Specialize,
    Justify,
    Synthesize,
}

/// A compiled semantic transform grounded in HoloSphere durable identity and E8 geometry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReasoningOperator {
    pub operator_id: ReasoningOperatorId,
    pub from_entity: EntityId,
    pub to_entity: EntityId,
    pub relation_type: RelationTypeId,
    pub from_coords: [f32; 8],
    pub to_coords: [f32; 8],
    pub transform: Vec<Cl24Blade>,
    pub evidence: Vec<DurableEvidenceRef>,
    pub provenance_id: ProvenanceId,
    pub reference_confidence: f32,
}

impl ReasoningOperator {
    pub fn is_executable(&self) -> bool {
        self.from_coords.iter().all(|c| c.is_finite())
            && self.to_coords.iter().all(|c| c.is_finite())
            && !self.transform.is_empty()
            && self.reference_confidence > 0.0
    }
}
