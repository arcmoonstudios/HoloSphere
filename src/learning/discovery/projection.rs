/* holosphere/src/learning/discovery/projection.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Experience Projection Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Projects stored empirical experiences, entity mutations, and episodic outcomes
//! into structured discovery cases suitable for motif mining and inductive learning.
//!
//! ## Key Capabilities
//! - **Deterministic Projection:** Losslessly formats raw experience records into feature vectors.
//! - **Partition Assignment:** Assigns discovery cases to deterministic train/validation sets.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::entity::id::EntityId;
use crate::entity::read::EntityReadSnapshot;
use crate::experience::action::{ActionParameterValue, DurableActionParameter};
use crate::experience::attempt::AttemptState;
use crate::experience::context::{ContextValue, DurableContextDimension};
use crate::experience::id::{AttemptId, MetricId};
use crate::experience::read::ExperienceReadSnapshot;
use crate::learning::discovery::knowledge::{
    HyperedgeMember, KnowledgeSnapshot, NumericAttributeId, TemporalHyperedge, TemporalInterval,
};
use crate::learning::discovery::model::{ConceptId, ConceptProfile, StructuralRole};
use crate::learning::discovery::model::{
    DiscoveryCase, DiscoveryCaseId, DiscoveryCorpus, DiscoveryOutcome, DomainId, EvidencePartition,
    FeatureId, ResolutionId,
};
use crate::learning::evidence::stats::MetricEvaluationRule;
use crate::learning::integrity::EmpiricalRootId;
use crate::relation::{RelationId, RelationReadSnapshot};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperienceProjectionPolicy {
    pub metric_rules: Vec<MetricEvaluationRule>,
    pub minimum_total_utility_q32: i64,
    /// Outcomes at or before this LSN induce motifs; later outcomes are reserved
    /// for validation. The caller must pin a snapshot beyond this boundary.
    pub discovery_through_lsn: u64,
    pub minimum_provenance_confidence_q16: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProjectionSkipReason {
    NotCompleted,
    MissingProblem,
    MissingContext,
    MissingOutcome,
    MissingActionDefinition,
    NoActions,
    NoApplicableMetrics,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionSkip {
    pub attempt_id: AttemptId,
    pub reason: ProjectionSkipReason,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperienceProjectionReport {
    pub corpus: DiscoveryCorpus,
    pub skipped: Vec<ProjectionSkip>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeProjectionPolicy {
    pub experience: ExperienceProjectionPolicy,
    pub default_domain: DomainId,
    pub entity_domains: BTreeMap<EntityId, DomainId>,
    pub entity_capabilities: BTreeMap<EntityId, BTreeSet<FeatureId>>,
    pub causal_predecessors: BTreeMap<RelationId, BTreeSet<RelationId>>,
    pub relation_context_features: BTreeMap<RelationId, BTreeSet<FeatureId>>,
    pub relation_numeric_context_q32: BTreeMap<RelationId, BTreeMap<NumericAttributeId, i64>>,
    pub relation_outcomes: BTreeMap<RelationId, (DiscoveryOutcome, Option<ResolutionId>)>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct KnowledgeProjectionReport {
    pub knowledge: KnowledgeSnapshot,
    pub skipped_relations: BTreeSet<RelationId>,
    pub skipped_experience: Vec<ProjectionSkip>,
}

/// Converts pinned, durable experience records into the exact contracts consumed
/// by the discovery engine. Domain, feature, resolution, and empirical-root IDs
/// are content-derived so compaction and replica layout cannot change them.
pub fn project_experience(
    experience: &ExperienceReadSnapshot,
    entities: &EntityReadSnapshot,
    policy: &ExperienceProjectionPolicy,
) -> ExperienceProjectionReport {
    let rules: BTreeMap<MetricId, &MetricEvaluationRule> = policy
        .metric_rules
        .iter()
        .map(|rule| (rule.metric_id, rule))
        .collect();
    let mut attempts: Vec<_> = experience.segment.attempts.read().keys().copied().collect();
    attempts.sort_unstable();

    let mut report = ExperienceProjectionReport::default();
    for attempt_id in attempts {
        let Some(attempt) = experience.attempt(attempt_id) else {
            continue;
        };
        if attempt.state != AttemptState::Completed {
            skip(&mut report, attempt_id, ProjectionSkipReason::NotCompleted);
            continue;
        }
        let Some(problem) = experience.problem(attempt.problem_id) else {
            skip(
                &mut report,
                attempt_id,
                ProjectionSkipReason::MissingProblem,
            );
            continue;
        };
        let Some(context) = experience.context(attempt.context_id) else {
            skip(
                &mut report,
                attempt_id,
                ProjectionSkipReason::MissingContext,
            );
            continue;
        };
        let Some(outcome) = experience.outcomes_for_attempt(attempt_id) else {
            skip(
                &mut report,
                attempt_id,
                ProjectionSkipReason::MissingOutcome,
            );
            continue;
        };
        if attempt.action_invocations.is_empty() {
            skip(&mut report, attempt_id, ProjectionSkipReason::NoActions);
            continue;
        }

        let actions = experience.segment.actions.read();
        let mut invocations = attempt.action_invocations.clone();
        invocations.sort_unstable_by_key(|invocation| invocation.ordinal);
        let Some(resolution) = resolution_id(&invocations, &actions) else {
            skip(
                &mut report,
                attempt_id,
                ProjectionSkipReason::MissingActionDefinition,
            );
            continue;
        };

        let mut total_utility_q32 = 0i64;
        let mut evaluated_metrics = 0usize;
        for observation in &outcome.observations {
            let Some(rule) = rules.get(&observation.metric_id) else {
                continue;
            };
            let Some(utility) = rule.evaluate(&observation.baseline, &observation.observed) else {
                continue;
            };
            total_utility_q32 = total_utility_q32.saturating_add(utility.raw_q32);
            evaluated_metrics += 1;
        }
        if evaluated_metrics == 0 {
            skip(
                &mut report,
                attempt_id,
                ProjectionSkipReason::NoApplicableMetrics,
            );
            continue;
        }

        let provenance = entities
            .segment
            .provenance
            .resolve_record_by_id(outcome.provenance_id);
        let certified_evidence = provenance.as_ref().is_some_and(|record| {
            let confidence_q16 = (record.confidence.clamp(0.0, 1.0) * 65_536.0).round() as u32;
            confidence_q16 >= policy.minimum_provenance_confidence_q16
        });
        let empirical_roots = provenance
            .as_ref()
            .map(|record| BTreeSet::from([EmpiricalRootId(u64_from_digest(record.signature_hash))]))
            .unwrap_or_default();
        let mut features: BTreeSet<_> = context.dimensions.iter().map(feature_id).collect();
        features.insert(named_feature(
            b"HOLOSPHERE_PROBLEM_SYMPTOM_V1",
            problem.symptom.as_bytes(),
        ));

        report.corpus.cases.push(DiscoveryCase {
            id: DiscoveryCaseId(attempt_id.0),
            domain: DomainId(content_id(
                b"HOLOSPHERE_DOMAIN_V1",
                problem.component.as_bytes(),
            )),
            snapshot_lsn: outcome.commit_lsn,
            features,
            observed_resolution: Some(resolution),
            outcome: if total_utility_q32 >= policy.minimum_total_utility_q32 {
                DiscoveryOutcome::Successful
            } else {
                DiscoveryOutcome::Failed
            },
            evidence_partition: if outcome.commit_lsn <= policy.discovery_through_lsn {
                EvidencePartition::Discovery
            } else {
                EvidencePartition::Validation
            },
            empirical_roots,
            certified_evidence,
        });
    }
    report
}

/// Projects the actual N-ary relation store and empirical experience store into
/// one temporally pinned discovery snapshot. Domain/outcome annotations are
/// explicit policy inputs because the storage kernel intentionally does not infer
/// semantics from relation names.
pub fn project_knowledge(
    experience: &ExperienceReadSnapshot,
    entities: &EntityReadSnapshot,
    relations: &RelationReadSnapshot,
    policy: &KnowledgeProjectionPolicy,
) -> KnowledgeProjectionReport {
    let experience_projection = project_experience(experience, entities, &policy.experience);
    let mut report = KnowledgeProjectionReport {
        knowledge: KnowledgeSnapshot {
            lsn: experience.lsn.min(entities.lsn).min(relations.lsn),
            cases: experience_projection.corpus.cases,
            concept_profiles: experience_projection.corpus.concept_profiles,
            hyperedges: Vec::new(),
        },
        skipped_relations: BTreeSet::new(),
        skipped_experience: experience_projection.skipped,
    };
    let mut profiles = BTreeMap::<(DomainId, ConceptId), ConceptProfile>::new();
    let relation_ids = relations.segment.arena.live_ids();
    for relation_id in relation_ids {
        let Some(resolved) = relations.current(relation_id, entities) else {
            report.skipped_relations.insert(relation_id);
            continue;
        };
        let domains: BTreeSet<_> = resolved
            .bindings
            .iter()
            .filter_map(|binding| policy.entity_domains.get(&binding.entity_id).copied())
            .collect();
        let domain = if domains.len() == 1 {
            *domains.iter().next().expect("one domain")
        } else {
            policy.default_domain
        };
        let provenance_root = resolved
            .provenance
            .as_ref()
            .map(|provenance| EmpiricalRootId(u64_from_digest(provenance.signature_hash)));
        let certified = resolved.epistemic_status.is_verified()
            && resolved.lifecycle_status.is_active()
            && provenance_root.is_some();
        let roots: BTreeSet<_> = provenance_root.into_iter().collect();
        let members: Vec<_> = resolved
            .bindings
            .iter()
            .map(|binding| HyperedgeMember {
                concept: ConceptId(binding.entity_id),
                role: binding.role_id,
            })
            .collect();
        let (outcome, resolution) = policy
            .relation_outcomes
            .get(&relation_id)
            .copied()
            .unwrap_or((DiscoveryOutcome::Unknown, None));
        report.knowledge.hyperedges.push(TemporalHyperedge {
            id: relation_id,
            domain,
            relation_type: resolved.type_id,
            members: members.clone(),
            interval: TemporalInterval {
                valid_from_lsn: resolved.valid_from_lsn,
                valid_until_lsn: resolved.valid_until_lsn,
            },
            causal_predecessors: policy
                .causal_predecessors
                .get(&relation_id)
                .cloned()
                .unwrap_or_default(),
            context_features: policy
                .relation_context_features
                .get(&relation_id)
                .cloned()
                .unwrap_or_default(),
            numeric_context_q32: policy
                .relation_numeric_context_q32
                .get(&relation_id)
                .cloned()
                .unwrap_or_default(),
            observed_resolution: resolution,
            outcome,
            empirical_roots: roots.clone(),
            certified_evidence: certified,
        });
        for member in members {
            let key = (domain, member.concept);
            let profile = profiles.entry(key).or_insert_with(|| ConceptProfile {
                domain,
                concept: member.concept,
                capabilities: policy
                    .entity_capabilities
                    .get(&member.concept.0)
                    .cloned()
                    .unwrap_or_default(),
                roles: BTreeSet::new(),
                empirical_roots: BTreeSet::new(),
                certified_evidence: certified,
            });
            profile.roles.insert(StructuralRole {
                relation_arity: resolved.bindings.len() as u16,
                role_ordinal: member.role,
                peer_role_count: resolved.bindings.len().saturating_sub(1) as u16,
                temporal_position: 0,
            });
            profile.empirical_roots.extend(roots.iter().copied());
            profile.certified_evidence &= certified;
        }
    }
    report
        .knowledge
        .concept_profiles
        .extend(profiles.into_values());
    report.knowledge.hyperedges.sort_by_key(|edge| edge.id);
    report
        .knowledge
        .concept_profiles
        .sort_by_key(|profile| (profile.domain, profile.concept));
    report
}

fn skip(
    report: &mut ExperienceProjectionReport,
    attempt_id: AttemptId,
    reason: ProjectionSkipReason,
) {
    report.skipped.push(ProjectionSkip { attempt_id, reason });
}

fn feature_id(dimension: &DurableContextDimension) -> FeatureId {
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_CONTEXT_FEATURE_V1");
    hasher.update(dimension.key.as_bytes());
    encode_context_value(&mut hasher, &dimension.value);
    FeatureId(u64_from_digest(hasher.finalize().into()))
}

fn named_feature(namespace: &[u8], value: &[u8]) -> FeatureId {
    FeatureId(content_id(namespace, value))
}

fn resolution_id(
    invocations: &[crate::experience::action::ActionInvocation],
    actions: &std::collections::HashMap<
        crate::experience::id::ActionId,
        crate::experience::action::ActionDefinition,
    >,
) -> Option<ResolutionId> {
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_RESOLUTION_PLAN_V1");
    for invocation in invocations {
        let action = actions.get(&invocation.action_id)?;
        hasher.update(action.name.as_bytes());
        let mut parameters: Vec<&DurableActionParameter> = invocation.parameters.iter().collect();
        parameters.sort_by(|left, right| left.key.cmp(&right.key));
        for parameter in parameters {
            hasher.update(parameter.key.as_bytes());
            encode_action_value(&mut hasher, &parameter.value);
        }
    }
    Some(ResolutionId(u64_from_digest(hasher.finalize().into())))
}

fn encode_context_value(hasher: &mut Sha256, value: &ContextValue) {
    match value {
        ContextValue::String(value) => {
            hasher.update([0]);
            hasher.update(value.as_bytes());
        }
        ContextValue::Integer(value) => {
            hasher.update([1]);
            hasher.update(value.to_le_bytes());
        }
        ContextValue::Float(value) => {
            hasher.update([2]);
            hasher.update(value.to_bits().to_le_bytes());
        }
        ContextValue::Boolean(value) => hasher.update([3, u8::from(*value)]),
    }
}

fn encode_action_value(hasher: &mut Sha256, value: &ActionParameterValue) {
    match value {
        ActionParameterValue::String(value) => {
            hasher.update([0]);
            hasher.update(value.as_bytes());
        }
        ActionParameterValue::Integer(value) => {
            hasher.update([1]);
            hasher.update(value.to_le_bytes());
        }
        ActionParameterValue::Float(value) => {
            hasher.update([2]);
            hasher.update(value.to_bits().to_le_bytes());
        }
        ActionParameterValue::Boolean(value) => hasher.update([3, u8::from(*value)]),
    }
}

fn content_id(namespace: &[u8], value: &[u8]) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(namespace);
    hasher.update(value);
    u64_from_digest(hasher.finalize().into())
}

fn u64_from_digest(digest: [u8; 32]) -> u64 {
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 prefix is 8 bytes"))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::entity::provenance::ProvenanceRecord;
    use crate::entity::segment::EntitySegment;
    use crate::experience::action::ActionInvocation;
    use crate::experience::attempt::AttemptRecord;
    use crate::experience::context::ContextRecord;
    use crate::experience::id::{ActionId, ContextId, OutcomeId, ProblemId};
    use crate::experience::metric::MetricValue;
    use crate::experience::outcome::{DurableOutcomeObservation, OutcomeRecord};
    use crate::experience::problem::ProblemOccurrence;
    use crate::experience::read::ExperienceSegment;
    use crate::learning::discovery::{
        DiscoveryGovernance, DiscoveryPolicy, GovernedDiscoveryEngine, MotifMinerConfig,
        OperatorLifecycle, OperatorValidationPolicy,
    };
    use crate::learning::evidence::stats::{MetricDirection, NormalizationRule};

    fn record_attempt(
        experience: &ExperienceSegment,
        entities: &EntitySegment,
        id: u64,
        component: &'static str,
        commit_lsn: u64,
    ) {
        let problem_id = ProblemId(id * 10 + 1);
        let context_id = ContextId(id * 10 + 2);
        let action_id = ActionId(id * 10 + 3);
        let attempt_id = AttemptId(id * 10 + 4);
        let outcome_id = OutcomeId(id * 10 + 5);
        let provenance_id = id * 10 + 6;
        entities.provenance.bind(
            provenance_id,
            &ProvenanceRecord {
                source_uri: Arc::from("sensor://test"),
                actor_id: Arc::from("test"),
                extraction_method: Arc::from("direct"),
                commit_lsn,
                timestamp_ms: commit_lsn,
                confidence: 1.0,
                evidence: Vec::new(),
                signature_hash: [id as u8; 32],
            },
        );
        experience.problems.write().insert(
            problem_id,
            ProblemOccurrence {
                problem_id,
                symptom: Arc::from("queue-growth"),
                component: Arc::from(component),
                first_observed_lsn: 1,
                provenance_id,
                context_id,
            },
        );
        experience.contexts.write().insert(
            context_id,
            ContextRecord::new(
                context_id,
                1,
                vec![DurableContextDimension {
                    key: Arc::from("capacity.bounded"),
                    value: ContextValue::Boolean(true),
                }],
                provenance_id,
            ),
        );
        experience.actions.write().insert(
            action_id,
            crate::experience::action::ActionDefinition {
                action_id,
                name: Arc::from("apply-backpressure"),
                description: Arc::from("bound incoming work"),
                provenance_id,
            },
        );
        experience.attempts.write().insert(
            attempt_id,
            AttemptRecord {
                attempt_id,
                problem_id,
                context_id,
                state: AttemptState::Completed,
                action_invocations: vec![ActionInvocation {
                    invocation_id: id,
                    attempt_id,
                    action_id,
                    ordinal: 0,
                    parameters: Vec::new(),
                    started_lsn: commit_lsn - 1,
                    completed_lsn: commit_lsn,
                    provenance_id,
                }],
                outcome_id: Some(outcome_id),
                started_lsn: commit_lsn - 1,
                completed_lsn: Some(commit_lsn),
                abort_reason: None,
                provenance_id,
            },
        );
        experience.outcomes.write().insert(
            outcome_id,
            OutcomeRecord {
                outcome_id,
                attempt_id,
                observations: vec![DurableOutcomeObservation {
                    metric_id: MetricId(1),
                    baseline: MetricValue::Unsigned(100),
                    observed: MetricValue::Unsigned(50),
                    measurement_start_lsn: commit_lsn - 1,
                    measurement_end_lsn: commit_lsn,
                    provenance_id,
                }],
                commit_lsn,
                provenance_id,
            },
        );
    }

    #[test]
    fn stored_experience_projects_and_discovers_across_domains() {
        let experience = Arc::new(ExperienceSegment::new(1));
        let entities = Arc::new(EntitySegment::new(1, 1));
        record_attempt(&experience, &entities, 1, "network", 100);
        record_attempt(&experience, &entities, 2, "database", 100);
        record_attempt(&experience, &entities, 3, "network", 200);
        record_attempt(&experience, &entities, 4, "database", 200);

        let projection = project_experience(
            &experience.read_snapshot(200),
            &entities.read_snapshot(200),
            &ExperienceProjectionPolicy {
                metric_rules: vec![MetricEvaluationRule {
                    metric_id: MetricId(1),
                    weight_q32: 1i64 << 32,
                    direction: MetricDirection::LowerIsBetter,
                    normalization: NormalizationRule::RelativeDelta,
                }],
                minimum_total_utility_q32: 1,
                discovery_through_lsn: 100,
                minimum_provenance_confidence_q16: 65_536,
            },
        );
        assert!(projection.skipped.is_empty());
        assert_eq!(projection.corpus.cases.len(), 4);
        assert_eq!(
            projection.corpus.cases[0].observed_resolution,
            projection.corpus.cases[1].observed_resolution
        );

        let report = GovernedDiscoveryEngine::new(DiscoveryPolicy {
            mining: MotifMinerConfig {
                min_successes: 2,
                min_domains: 2,
                ..MotifMinerConfig::default()
            },
            validation: OperatorValidationPolicy {
                min_evaluated_cases: 2,
                min_supporting_domains: 2,
                min_independent_roots: 2,
                min_precision_q32: 1i64 << 32,
                min_lift_q32: 0,
                max_contradiction_ratio_q32: 0,
                min_held_out_domain_passes: 2,
            },
            governance: DiscoveryGovernance::PolicyAuthorized {
                policy_id: 9,
                version: 1,
            },
            ..DiscoveryPolicy::default()
        })
        .discover(&projection.corpus);
        assert!(report.operators.iter().any(|assessment| {
            assessment.operator.lifecycle == OperatorLifecycle::Shadow
                && assessment.validation.supporting_domains.len() == 2
        }));
    }
}
