/* holosphere/src/learning/inference/rune_evo/reasoning/closure.rs */
//!▫~•◦-------------------------------‣
//! # Evidence-Bound Closure Synthesis & Transitive Rule Compilation
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Compiles verified multi-hop relation paths into provisional closure hypotheses
//! with Cl(24) algebraic transform sidecars, identity-enforced link continuity,
//! and explicit evidence bounds.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::entity::id::EntityId;
use crate::entity::status::EpistemicStatus;
use crate::learning::inference::contract::{InferenceError, InferenceMethodId, InferenceSeed};
use crate::learning::inference::rune_evo::analogy::euclidean_dist_8;
use crate::learning::inference::rune_evo::reasoning::composition::{
    Cl24CompositionArtifact, RuneCl24CompositionConfig, execute_operator_chain,
};
use crate::learning::inference::rune_evo::reasoning::operator::ReasoningOperator;
use crate::learning::inference::trace::InferenceTrace;
use crate::relation::id::RelationTypeId;

pub const RUNE_CLOSURE_METHOD_ID: InferenceMethodId = InferenceMethodId(105);
pub const RUNE_CLOSURE_METHOD_VERSION: u32 = 1;

/// Classification of a closure execution by hop count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClosureKind {
    /// Depth = 1: Single-hop evidence recovery.
    EvidenceRecovery,
    /// Depth >= 2: Multi-hop composed reasoning.
    ComposedReasoning,
}

/// Declared semantic mode for composing two relations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompositionSemantics {
    /// Deterministic transitive or deductive schema rule (e.g. PART_OF ∘ PART_OF → PART_OF).
    DeclaredExact,
    /// Learned inductive relational pattern.
    LearnedHypothesis,
    /// Unconstrained geometric exploratory composition without an admitted target schema.
    GeometricExploratory,
}

/// Explicit schema rule governing the composition of two relation types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompositionRule {
    pub lhs: RelationTypeId,
    pub rhs: RelationTypeId,
    pub result: RelationTypeId,
    pub semantics: CompositionSemantics,
}

/// Catalog of admitted relation composition rules.
#[derive(Clone, Debug, Default)]
pub struct CompositionRuleRegistry {
    rules: HashMap<(RelationTypeId, RelationTypeId), CompositionRule>,
}

impl CompositionRuleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, rule: CompositionRule) {
        self.rules.insert((rule.lhs, rule.rhs), rule);
    }

    pub fn find_rule(&self, lhs: RelationTypeId, rhs: RelationTypeId) -> Option<CompositionRule> {
        self.rules.get(&(lhs, rhs)).copied()
    }
}

/// Bounded evidence and novelty metric attached to a compiled closure candidate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RuneClosureEvidenceV1 {
    pub packet_evidence: f32,
    pub branch_shape: f32,
    pub identity_resonance: f32,
    pub composition_novelty: f32,
    pub truncation_loss_ratio: f32,
}

/// Complete synthesized closure candidate representing a multi-hop inference hypothesis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClosureCandidate {
    pub start_entity: EntityId,
    /// The canonical semantic endpoint MUST be the final operator's to_entity.
    /// (The Cl(24) projected coordinate is a transform witness sidecar, NOT the endpoint).
    pub semantic_endpoint: EntityId,
    pub chain_relations: Vec<RelationTypeId>,
    pub chain_entities: Vec<EntityId>,
    pub result_relation_type: Option<RelationTypeId>,
    pub composition_artifact: Cl24CompositionArtifact,
    pub closure_evidence: RuneClosureEvidenceV1,
    pub closure_kind: ClosureKind,
    pub epistemic_status: EpistemicStatus,
    pub trace: InferenceTrace,
}

/// Compiles an operator chain into a provisional ClosureCandidate.
pub fn compile_closure(
    operators: &[ReasoningOperator],
    rules: &CompositionRuleRegistry,
    config: &RuneCl24CompositionConfig,
    snapshot_lsn: u64,
) -> Result<ClosureCandidate, InferenceError> {
    if operators.is_empty() {
        return Err(InferenceError::InvalidParameters(
            "cannot compile closure from empty operator chain".into(),
        ));
    }
    if operators.len() > config.max_operator_chain {
        return Err(InferenceError::InvalidParameters(format!(
            "operator chain length {} exceeds maximum allowed {}",
            operators.len(),
            config.max_operator_chain
        )));
    }

    // Enforce link continuity by canonical EntityId identity
    for i in 0..operators.len().saturating_sub(1) {
        if operators[i].to_entity != operators[i + 1].from_entity {
            return Err(InferenceError::InvalidParameters(format!(
                "chain identity continuity break at hop {}: operator {} target ({}) != operator {} source ({})",
                i,
                operators[i].operator_id.0,
                operators[i].to_entity,
                operators[i + 1].operator_id.0,
                operators[i + 1].from_entity
            )));
        }
    }

    let start_entity = operators[0].from_entity;
    let semantic_endpoint = operators.last().unwrap().to_entity;
    let chain_relations: Vec<RelationTypeId> =
        operators.iter().map(|op| op.relation_type).collect();
    let mut chain_entities = vec![start_entity];
    for op in operators {
        chain_entities.push(op.to_entity);
    }

    // Resolve target relation type from composition rules
    let result_relation_type = if operators.len() == 1 {
        Some(operators[0].relation_type)
    } else if operators.len() == 2 {
        rules
            .find_rule(operators[0].relation_type, operators[1].relation_type)
            .map(|r| r.result)
    } else {
        None
    };

    let composition_artifact = execute_operator_chain(operators, config);

    let packet_evidence = operators
        .iter()
        .map(|op| op.reference_confidence)
        .sum::<f32>()
        / (operators.len() as f32);

    let branch_shape = (1.0
        - euclidean_dist_8(
            &operators[0].from_coords,
            &operators.last().unwrap().to_coords,
        ))
    .clamp(0.0, 1.0);
    let identity_resonance = 1.0f32; // Exact identity continuity verified above

    let closure_evidence = RuneClosureEvidenceV1 {
        packet_evidence: packet_evidence.clamp(0.0, 1.0),
        branch_shape,
        identity_resonance,
        composition_novelty: composition_artifact.composition_delta,
        truncation_loss_ratio: composition_artifact.max_truncation_loss_ratio,
    };

    let closure_kind = if operators.len() == 1 {
        ClosureKind::EvidenceRecovery
    } else {
        ClosureKind::ComposedReasoning
    };

    let trace = InferenceTrace {
        method: RUNE_CLOSURE_METHOD_ID,
        method_version: RUNE_CLOSURE_METHOD_VERSION,
        source_entities: chain_entities.clone(),
        source_relations: Vec::new(),
        source_attempts: Vec::new(),
        snapshot_lsn,
        seed: InferenceSeed::default(),
        parameter_digest: composition_artifact.semantic_fingerprint,
    };

    Ok(ClosureCandidate {
        start_entity,
        semantic_endpoint,
        chain_relations,
        chain_entities,
        result_relation_type,
        composition_artifact,
        closure_evidence,
        closure_kind,
        epistemic_status: EpistemicStatus::Provisional,
        trace,
    })
}
