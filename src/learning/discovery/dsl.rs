//! Inspectable, versioned, resource-bounded reasoning operator language.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::learning::discovery::hyper_motif::{
    HypergraphMotifId, HypergraphMotifKind, TemporalHypergraphMotif,
};
use crate::learning::discovery::knowledge::NumericAttributeId;
use crate::learning::discovery::model::{
    DiscoveryCase, DiscoveryOutcome, DomainId, FeatureId, ResolutionId,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericExpression {
    Attribute(NumericAttributeId),
    ConstantQ32(i64),
    Add(Box<NumericExpression>, Box<NumericExpression>),
    Subtract(Box<NumericExpression>, Box<NumericExpression>),
    MultiplyQ32(Box<NumericExpression>, Box<NumericExpression>),
    Min(Box<NumericExpression>, Box<NumericExpression>),
    Max(Box<NumericExpression>, Box<NumericExpression>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConditionExpression {
    True,
    FeaturePresent(FeatureId),
    FeatureAbsent(FeatureId),
    NumericCompare {
        left: NumericExpression,
        operator: ComparisonOperator,
        right: NumericExpression,
    },
    FeaturePersists {
        feature: FeatureId,
        minimum_duration_lsn: u64,
    },
    HypergraphMotifPresent(HypergraphMotifId),
    CausalMotifPresent(HypergraphMotifId),
    DomainIs(DomainId),
    All(Vec<ConditionExpression>),
    Any(Vec<ConditionExpression>),
    Not(Box<ConditionExpression>),
}

/// Declarative hypergraph rewrite templates. Execution returns these as
/// proposals; applying one still requires normal schema, provenance, and Raft
/// admission. The DSL never mutates the graph directly.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum HypergraphTransformation {
    InstantiateCanonicalMotif {
        motif: HypergraphMotifId,
    },
    ProjectRole {
        motif: HypergraphMotifId,
        role_ordinal: u16,
    },
    ComposeMotifs {
        left: HypergraphMotifId,
        right: HypergraphMotifId,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DslEffect {
    PredictOutcome(DiscoveryOutcome),
    PredictFeature(FeatureId),
    ProposeResolution(ResolutionId),
    SetDerivedNumeric {
        attribute: NumericAttributeId,
        value: NumericExpression,
    },
    RequireConstraint(FeatureId),
    ProposeHypergraphTransformation(HypergraphTransformation),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceCostBounds {
    pub max_ast_nodes: u32,
    pub max_depth: u16,
    pub max_effects: u16,
    pub max_numeric_abs_q32: i64,
}

impl Default for ResourceCostBounds {
    fn default() -> Self {
        Self {
            max_ast_nodes: 256,
            max_depth: 16,
            max_effects: 32,
            max_numeric_abs_q32: 1i64 << 48,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorProgram {
    pub condition: ConditionExpression,
    pub effects: Vec<DslEffect>,
    pub bounds: ResourceCostBounds,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningContext {
    pub case: Option<DiscoveryCase>,
    pub numeric_values_q32: BTreeMap<NumericAttributeId, i64>,
    pub feature_durations_lsn: BTreeMap<FeatureId, u64>,
    pub present_motifs: BTreeSet<HypergraphMotifId>,
    pub causal_motifs: BTreeSet<HypergraphMotifId>,
    pub satisfied_constraints: BTreeSet<FeatureId>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramCost {
    pub ast_nodes: u32,
    pub max_depth: u16,
    pub effects_evaluated: u16,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramResult {
    pub matched: bool,
    pub predicted_outcomes: BTreeSet<DiscoveryOutcome>,
    pub predicted_features: BTreeSet<FeatureId>,
    pub proposed_resolutions: BTreeSet<ResolutionId>,
    pub derived_numeric_q32: BTreeMap<NumericAttributeId, i64>,
    pub unsatisfied_constraints: BTreeSet<FeatureId>,
    pub proposed_hypergraph_transformations: BTreeSet<HypergraphTransformation>,
    pub cost: ProgramCost,
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum OperatorSandboxError {
    #[error("operator AST contains {actual} nodes, exceeding {limit}")]
    AstNodeLimit { actual: u32, limit: u32 },
    #[error("operator AST depth {actual} exceeds {limit}")]
    DepthLimit { actual: u16, limit: u16 },
    #[error("operator has {actual} effects, exceeding {limit}")]
    EffectLimit { actual: u16, limit: u16 },
    #[error("numeric attribute {0:?} is missing")]
    MissingNumericAttribute(NumericAttributeId),
    #[error("numeric result exceeded the configured absolute Q32 bound")]
    NumericBoundExceeded,
}

pub fn validate_program(program: &OperatorProgram) -> Result<ProgramCost, OperatorSandboxError> {
    let mut cost = ProgramCost::default();
    inspect_condition(&program.condition, 1, &mut cost);
    for effect in &program.effects {
        cost.effects_evaluated = cost.effects_evaluated.saturating_add(1);
        cost.ast_nodes = cost.ast_nodes.saturating_add(1);
        if let DslEffect::SetDerivedNumeric { value, .. } = effect {
            inspect_numeric(value, 2, &mut cost);
        }
    }
    if cost.ast_nodes > program.bounds.max_ast_nodes {
        return Err(OperatorSandboxError::AstNodeLimit {
            actual: cost.ast_nodes,
            limit: program.bounds.max_ast_nodes,
        });
    }
    if cost.max_depth > program.bounds.max_depth {
        return Err(OperatorSandboxError::DepthLimit {
            actual: cost.max_depth,
            limit: program.bounds.max_depth,
        });
    }
    if cost.effects_evaluated > program.bounds.max_effects {
        return Err(OperatorSandboxError::EffectLimit {
            actual: cost.effects_evaluated,
            limit: program.bounds.max_effects,
        });
    }
    Ok(cost)
}

pub fn execute_program(
    program: &OperatorProgram,
    context: &ReasoningContext,
) -> Result<ProgramResult, OperatorSandboxError> {
    let cost = validate_program(program)?;
    let matched = evaluate_condition(&program.condition, context, program.bounds)?;
    let mut result = ProgramResult {
        matched,
        cost,
        ..ProgramResult::default()
    };
    if !matched {
        return Ok(result);
    }
    for effect in &program.effects {
        match effect {
            DslEffect::PredictOutcome(outcome) => {
                result.predicted_outcomes.insert(*outcome);
            }
            DslEffect::PredictFeature(feature) => {
                result.predicted_features.insert(*feature);
            }
            DslEffect::ProposeResolution(resolution) => {
                result.proposed_resolutions.insert(*resolution);
            }
            DslEffect::SetDerivedNumeric { attribute, value } => {
                let value = evaluate_numeric(value, context, program.bounds)?;
                result.derived_numeric_q32.insert(*attribute, value);
            }
            DslEffect::RequireConstraint(constraint) => {
                if !context.satisfied_constraints.contains(constraint) {
                    result.unsatisfied_constraints.insert(*constraint);
                }
            }
            DslEffect::ProposeHypergraphTransformation(transformation) => {
                result
                    .proposed_hypergraph_transformations
                    .insert(transformation.clone());
            }
        }
    }
    if !result.unsatisfied_constraints.is_empty() {
        result.proposed_resolutions.clear();
    }
    Ok(result)
}

/// Synthesizes an inspectable program from a motif. This is program generation,
/// not native-code generation; the immutable interpreter remains engineered.
pub fn synthesize_program_from_motif(motif: &TemporalHypergraphMotif) -> OperatorProgram {
    let mut conditions = vec![ConditionExpression::HypergraphMotifPresent(motif.id)];
    conditions.extend(
        motif
            .context_features
            .iter()
            .copied()
            .map(ConditionExpression::FeaturePresent),
    );
    if matches!(motif.kind, HypergraphMotifKind::CausalSequence { .. }) {
        conditions.push(ConditionExpression::CausalMotifPresent(motif.id));
    }
    let mut effects: Vec<_> = motif
        .associated_resolutions
        .iter()
        .copied()
        .map(DslEffect::ProposeResolution)
        .collect();
    effects.push(DslEffect::ProposeHypergraphTransformation(
        HypergraphTransformation::InstantiateCanonicalMotif { motif: motif.id },
    ));
    match motif.kind {
        HypergraphMotifKind::CausalSequence {
            resulting_outcome, ..
        } => effects.push(DslEffect::PredictOutcome(resulting_outcome)),
        HypergraphMotifKind::BeforeAfterOutcome { outcome_after, .. } => {
            effects.push(DslEffect::PredictOutcome(outcome_after));
        }
        HypergraphMotifKind::OutcomeAnomaly { unexpected, .. } => {
            effects.push(DslEffect::PredictOutcome(unexpected));
        }
        _ => {}
    }
    OperatorProgram {
        condition: ConditionExpression::All(conditions),
        effects,
        bounds: ResourceCostBounds::default(),
    }
}

pub fn compose_programs(
    programs: &[OperatorProgram],
    bounds: ResourceCostBounds,
) -> Result<OperatorProgram, OperatorSandboxError> {
    let program = OperatorProgram {
        condition: ConditionExpression::All(
            programs
                .iter()
                .map(|program| program.condition.clone())
                .collect(),
        ),
        effects: programs
            .iter()
            .flat_map(|program| program.effects.iter().cloned())
            .collect(),
        bounds,
    };
    validate_program(&program)?;
    Ok(program)
}

fn inspect_condition(condition: &ConditionExpression, depth: u16, cost: &mut ProgramCost) {
    cost.ast_nodes = cost.ast_nodes.saturating_add(1);
    cost.max_depth = cost.max_depth.max(depth);
    match condition {
        ConditionExpression::NumericCompare { left, right, .. } => {
            inspect_numeric(left, depth + 1, cost);
            inspect_numeric(right, depth + 1, cost);
        }
        ConditionExpression::All(children) | ConditionExpression::Any(children) => {
            for child in children {
                inspect_condition(child, depth + 1, cost);
            }
        }
        ConditionExpression::Not(child) => inspect_condition(child, depth + 1, cost),
        _ => {}
    }
}

fn inspect_numeric(expression: &NumericExpression, depth: u16, cost: &mut ProgramCost) {
    cost.ast_nodes = cost.ast_nodes.saturating_add(1);
    cost.max_depth = cost.max_depth.max(depth);
    match expression {
        NumericExpression::Add(left, right)
        | NumericExpression::Subtract(left, right)
        | NumericExpression::MultiplyQ32(left, right)
        | NumericExpression::Min(left, right)
        | NumericExpression::Max(left, right) => {
            inspect_numeric(left, depth + 1, cost);
            inspect_numeric(right, depth + 1, cost);
        }
        _ => {}
    }
}

fn evaluate_condition(
    condition: &ConditionExpression,
    context: &ReasoningContext,
    bounds: ResourceCostBounds,
) -> Result<bool, OperatorSandboxError> {
    Ok(match condition {
        ConditionExpression::True => true,
        ConditionExpression::FeaturePresent(feature) => context
            .case
            .as_ref()
            .is_some_and(|case| case.features.contains(feature)),
        ConditionExpression::FeatureAbsent(feature) => context
            .case
            .as_ref()
            .is_some_and(|case| !case.features.contains(feature)),
        ConditionExpression::NumericCompare {
            left,
            operator,
            right,
        } => compare(
            evaluate_numeric(left, context, bounds)?,
            *operator,
            evaluate_numeric(right, context, bounds)?,
        ),
        ConditionExpression::FeaturePersists {
            feature,
            minimum_duration_lsn,
        } => context
            .feature_durations_lsn
            .get(feature)
            .is_some_and(|duration| duration >= minimum_duration_lsn),
        ConditionExpression::HypergraphMotifPresent(motif) => {
            context.present_motifs.contains(motif)
        }
        ConditionExpression::CausalMotifPresent(motif) => context.causal_motifs.contains(motif),
        ConditionExpression::DomainIs(domain) => context
            .case
            .as_ref()
            .is_some_and(|case| case.domain == *domain),
        ConditionExpression::All(children) => {
            for child in children {
                if !evaluate_condition(child, context, bounds)? {
                    return Ok(false);
                }
            }
            true
        }
        ConditionExpression::Any(children) => {
            for child in children {
                if evaluate_condition(child, context, bounds)? {
                    return Ok(true);
                }
            }
            false
        }
        ConditionExpression::Not(child) => !evaluate_condition(child, context, bounds)?,
    })
}

fn evaluate_numeric(
    expression: &NumericExpression,
    context: &ReasoningContext,
    bounds: ResourceCostBounds,
) -> Result<i64, OperatorSandboxError> {
    let value = match expression {
        NumericExpression::Attribute(attribute) => *context
            .numeric_values_q32
            .get(attribute)
            .ok_or(OperatorSandboxError::MissingNumericAttribute(*attribute))?,
        NumericExpression::ConstantQ32(value) => *value,
        NumericExpression::Add(left, right) => evaluate_numeric(left, context, bounds)?
            .saturating_add(evaluate_numeric(right, context, bounds)?),
        NumericExpression::Subtract(left, right) => evaluate_numeric(left, context, bounds)?
            .saturating_sub(evaluate_numeric(right, context, bounds)?),
        NumericExpression::MultiplyQ32(left, right) => {
            let product = evaluate_numeric(left, context, bounds)? as i128
                * evaluate_numeric(right, context, bounds)? as i128;
            (product >> 32).clamp(i64::MIN as i128, i64::MAX as i128) as i64
        }
        NumericExpression::Min(left, right) => {
            evaluate_numeric(left, context, bounds)?.min(evaluate_numeric(right, context, bounds)?)
        }
        NumericExpression::Max(left, right) => {
            evaluate_numeric(left, context, bounds)?.max(evaluate_numeric(right, context, bounds)?)
        }
    };
    if value.unsigned_abs() > bounds.max_numeric_abs_q32.unsigned_abs() {
        return Err(OperatorSandboxError::NumericBoundExceeded);
    }
    Ok(value)
}

fn compare(left: i64, operator: ComparisonOperator, right: i64) -> bool {
    match operator {
        ComparisonOperator::Equal => left == right,
        ComparisonOperator::NotEqual => left != right,
        ComparisonOperator::Less => left < right,
        ComparisonOperator::LessOrEqual => left <= right,
        ComparisonOperator::Greater => left > right,
        ComparisonOperator::GreaterOrEqual => left >= right,
    }
}
