/* holosphere/src/experience/mod.rs */
//!▫~•◦-------------------------------‣
//! # Empirical Experience Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the immutable empirical substrate recording Problems, Contexts,
//! Actions, Attempts, and Raw Outcomes.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod action;
pub mod attempt;
pub mod context;
pub mod id;
pub mod metric;
pub mod mutation;
pub mod outcome;
pub mod problem;
pub mod query;
pub mod read;

// Re-exports
pub use action::{
    ActionDefinition, ActionInvocation, ActionParameterValue, DurableActionParameter,
};
pub use attempt::{AttemptRecord, AttemptState};
pub use context::{
    ContextRecord, ContextValue, DurableContextDimension, compute_context_fingerprint,
};
pub use id::{
    ActionId, AttemptId, ContextId, EvaluationPolicyId, MetricId, OutcomeId, ProblemId, SymbolId,
};
pub use metric::{MetricValue, MetricValueKind, OutcomeMetricSchema};
pub use mutation::{ExperienceMutation, ExperienceMutationError};
pub use outcome::{DurableOutcomeObservation, OutcomeRecord};
pub use problem::ProblemOccurrence;
pub use query::ExperienceQuery;
pub use read::{ExperienceReadSnapshot, ExperienceSegment, ExperienceTrace};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::entity::provenance::ProvenanceRecord;
    use crate::entity::segment::EntitySegment;
    use std::sync::Arc;

    #[test]
    fn test_phase5_context_canonicalization_and_fingerprinting() {
        let dim1 = vec![
            DurableContextDimension {
                key: Arc::from("storage.media"),
                value: ContextValue::String(Arc::from("NVMe")),
            },
            DurableContextDimension {
                key: Arc::from("cluster.nodes"),
                value: ContextValue::Integer(5),
            },
            DurableContextDimension {
                key: Arc::from("raft.durability"),
                value: ContextValue::String(Arc::from("quorum_fsync")),
            },
        ];

        let dim2 = vec![
            DurableContextDimension {
                key: Arc::from("raft.durability"),
                value: ContextValue::String(Arc::from("quorum_fsync")),
            },
            DurableContextDimension {
                key: Arc::from("storage.media"),
                value: ContextValue::String(Arc::from("NVMe")),
            },
            DurableContextDimension {
                key: Arc::from("cluster.nodes"),
                value: ContextValue::Integer(5),
            },
        ];

        let (fp1, hash1) = compute_context_fingerprint(1, &dim1);
        let (fp2, hash2) = compute_context_fingerprint(1, &dim2);

        // Deduplication: different input ordering yields identical SHA-256 fingerprint
        assert_eq!(fp1, fp2);
        assert_eq!(hash1, hash2);

        // Modifying any single dimension changes SHA-256 fingerprint
        let dim_modified = vec![
            DurableContextDimension {
                key: Arc::from("storage.media"),
                value: ContextValue::String(Arc::from("SSD")),
            },
            DurableContextDimension {
                key: Arc::from("cluster.nodes"),
                value: ContextValue::Integer(5),
            },
            DurableContextDimension {
                key: Arc::from("raft.durability"),
                value: ContextValue::String(Arc::from("quorum_fsync")),
            },
        ];
        let (fp_mod, hash_mod) = compute_context_fingerprint(1, &dim_modified);
        assert_ne!(fp1, fp_mod);
        assert_ne!(hash1, hash_mod);
    }

    #[test]
    fn test_phase5_metric_raw_observations_and_derived_deltas() {
        let latency_metric = OutcomeMetricSchema {
            metric_id: MetricId(1),
            name: Arc::from("latency_p50_us"),
            unit: Arc::from("us"),
            value_kind: MetricValueKind::UnsignedInteger,
            schema_version: 1,
        };

        let baseline = MetricValue::Unsigned(8400);
        let observed = MetricValue::Unsigned(3100);

        let delta = latency_metric.compute_delta(&baseline, &observed).unwrap();
        assert_eq!(delta, -5300.0);

        let pct = latency_metric
            .compute_percentage_delta(&baseline, &observed)
            .unwrap();
        // -5300 / 8400 = -0.63095238...
        assert!((pct - (-0.630952)).abs() < 0.001);
    }

    #[test]
    fn test_phase5_attempt_lifecycle_and_atomic_snapshot_isolation() {
        let exp_seg = Arc::new(ExperienceSegment::new(1));

        // 1. Create Problem and Context
        let prob_id = ProblemId(101);
        let ctx_id = ContextId(201);
        let att_id = AttemptId(301);
        let act_id = ActionId(401);
        let out_id = OutcomeId(501);

        ExperienceMutation::CreateProblem {
            problem_id: prob_id,
            symptom: Arc::from("WriteLatencySpike"),
            component: Arc::from("RaftWAL"),
            context_id: ctx_id,
            provenance_id: 1,
        }
        .apply(&exp_seg, 10)
        .unwrap();

        ExperienceMutation::CreateContext {
            context_id: ctx_id,
            schema_version: 1,
            dimensions: vec![DurableContextDimension {
                key: Arc::from("cluster.nodes"),
                value: ContextValue::Integer(5),
            }],
            provenance_id: 1,
        }
        .apply(&exp_seg, 10)
        .unwrap();

        ExperienceMutation::RegisterAction {
            action_id: act_id,
            name: Arc::from("GroupCommit"),
            description: Arc::from("Batch writes within group window"),
            provenance_id: 1,
        }
        .apply(&exp_seg, 10)
        .unwrap();

        // 2. Begin Attempt at LSN 100
        ExperienceMutation::BeginAttempt {
            attempt_id: att_id,
            problem_id: prob_id,
            context_id: ctx_id,
            provenance_id: 1,
        }
        .apply(&exp_seg, 100)
        .unwrap();

        ExperienceMutation::RecordActionInvocation {
            attempt_id: att_id,
            invocation_id: 1,
            action_id: act_id,
            ordinal: 0,
            parameters: vec![DurableActionParameter {
                key: Arc::from("window_ms"),
                value: ActionParameterValue::Integer(3),
            }],
            started_lsn: 100,
            completed_lsn: 110,
            provenance_id: 1,
        }
        .apply(&exp_seg, 110)
        .unwrap();

        // Pinned snapshot at LSN 150 (attempt is currently Running, no outcome visible)
        let snap_running = exp_seg.read_snapshot(150);
        let att_running = snap_running.attempt(att_id).expect("must find attempt");
        assert_eq!(att_running.state, AttemptState::Running);
        assert!(snap_running.outcomes_for_attempt(att_id).is_none());

        // 3. Complete Attempt at LSN 200 with raw observations
        ExperienceMutation::CompleteAttempt {
            attempt_id: att_id,
            expected_state: AttemptState::Running,
            outcome_id: out_id,
            observations: vec![DurableOutcomeObservation {
                metric_id: MetricId(1),
                baseline: MetricValue::Unsigned(8400),
                observed: MetricValue::Unsigned(3100),
                measurement_start_lsn: 100,
                measurement_end_lsn: 200,
                provenance_id: 1,
            }],
            provenance_id: 1,
        }
        .apply(&exp_seg, 200)
        .unwrap();

        // Verify that pinned snapshot at LSN 150 STILL sees Running and no outcomes
        let att_old = snap_running.attempt(att_id).unwrap();
        assert_eq!(att_old.state, AttemptState::Running);
        assert!(snap_running.outcomes_for_attempt(att_id).is_none());

        // Snapshot at LSN 200 sees Completed with complete outcome
        let snap_complete = exp_seg.read_snapshot(200);
        let att_done = snap_complete.attempt(att_id).unwrap();
        assert_eq!(att_done.state, AttemptState::Completed);
        let outcome = snap_complete
            .outcomes_for_attempt(att_id)
            .expect("outcome must be visible");
        assert_eq!(outcome.observations.len(), 1);
        assert_eq!(
            outcome.observations[0].observed,
            MetricValue::Unsigned(3100)
        );
    }

    #[test]
    fn test_phase5_engineering_system_test() {
        let ent_seg = Arc::new(EntitySegment::new(1, 1));
        let exp_seg = Arc::new(ExperienceSegment::new(1));

        let prob_id = ProblemId(901);
        let ctx_id = ContextId(902);
        let act_workers = ActionId(910);
        let act_group = ActionId(920);

        let att_118 = AttemptId(118);
        let att_402 = AttemptId(402);

        let out_118 = OutcomeId(518);
        let out_402 = OutcomeId(502);

        // Provenance record
        let prov = ProvenanceRecord {
            source_uri: Arc::from("file:///telemetry/wal_benchmarks.log"),
            actor_id: Arc::from("bench_agent"),
            extraction_method: Arc::from("automated_bench"),
            commit_lsn: 100,
            timestamp_ms: 1718000000,
            confidence: 1.0,
            evidence: vec![],
            signature_hash: [42u8; 32],
        };
        let (prov_id, _) = ent_seg.provenance.append(&prov);

        // Problem & Context setup
        ExperienceMutation::CreateProblem {
            problem_id: prob_id,
            symptom: Arc::from("HighWriteLatency"),
            component: Arc::from("WAL"),
            context_id: ctx_id,
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 100)
        .unwrap();

        ExperienceMutation::CreateContext {
            context_id: ctx_id,
            schema_version: 1,
            dimensions: vec![
                DurableContextDimension {
                    key: Arc::from("storage.media"),
                    value: ContextValue::String(Arc::from("NVMe")),
                },
                DurableContextDimension {
                    key: Arc::from("cluster.nodes"),
                    value: ContextValue::Integer(5),
                },
                DurableContextDimension {
                    key: Arc::from("raft.durability"),
                    value: ContextValue::String(Arc::from("quorum_fsync")),
                },
                DurableContextDimension {
                    key: Arc::from("batch.mean_size"),
                    value: ContextValue::Integer(3),
                },
            ],
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 100)
        .unwrap();

        ExperienceMutation::RegisterAction {
            action_id: act_workers,
            name: Arc::from("IncreaseWorkerCount"),
            description: Arc::from("Scale worker threadpool"),
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 100)
        .unwrap();

        ExperienceMutation::RegisterAction {
            action_id: act_group,
            name: Arc::from("GroupCommit"),
            description: Arc::from("Group commit batch window"),
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 100)
        .unwrap();

        // ── Attempt 118: IncreaseWorkerCount ──
        ExperienceMutation::BeginAttempt {
            attempt_id: att_118,
            problem_id: prob_id,
            context_id: ctx_id,
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 110)
        .unwrap();

        ExperienceMutation::RecordActionInvocation {
            attempt_id: att_118,
            invocation_id: 1,
            action_id: act_workers,
            ordinal: 0,
            parameters: vec![DurableActionParameter {
                key: Arc::from("workers"),
                value: ActionParameterValue::Integer(16),
            }],
            started_lsn: 110,
            completed_lsn: 120,
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 120)
        .unwrap();

        ExperienceMutation::CompleteAttempt {
            attempt_id: att_118,
            expected_state: AttemptState::Running,
            outcome_id: out_118,
            observations: vec![
                DurableOutcomeObservation {
                    metric_id: MetricId(1), // p50 latency (us)
                    baseline: MetricValue::Unsigned(8400),
                    observed: MetricValue::Unsigned(9700),
                    measurement_start_lsn: 110,
                    measurement_end_lsn: 120,
                    provenance_id: prov_id,
                },
                DurableOutcomeObservation {
                    metric_id: MetricId(2), // throughput (qps)
                    baseline: MetricValue::Unsigned(12200),
                    observed: MetricValue::Unsigned(12400),
                    measurement_start_lsn: 110,
                    measurement_end_lsn: 120,
                    provenance_id: prov_id,
                },
                DurableOutcomeObservation {
                    metric_id: MetricId(3), // cpu %
                    baseline: MetricValue::Unsigned(63),
                    observed: MetricValue::Unsigned(91),
                    measurement_start_lsn: 110,
                    measurement_end_lsn: 120,
                    provenance_id: prov_id,
                },
            ],
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 120)
        .unwrap();

        // ── Attempt 402: GroupCommit(window=3ms) ──
        ExperienceMutation::BeginAttempt {
            attempt_id: att_402,
            problem_id: prob_id,
            context_id: ctx_id,
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 200)
        .unwrap();

        ExperienceMutation::RecordActionInvocation {
            attempt_id: att_402,
            invocation_id: 2,
            action_id: act_group,
            ordinal: 0,
            parameters: vec![DurableActionParameter {
                key: Arc::from("window_ms"),
                value: ActionParameterValue::Integer(3),
            }],
            started_lsn: 200,
            completed_lsn: 210,
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 210)
        .unwrap();

        ExperienceMutation::CompleteAttempt {
            attempt_id: att_402,
            expected_state: AttemptState::Running,
            outcome_id: out_402,
            observations: vec![
                DurableOutcomeObservation {
                    metric_id: MetricId(1), // p50 latency (us)
                    baseline: MetricValue::Unsigned(8400),
                    observed: MetricValue::Unsigned(3100),
                    measurement_start_lsn: 200,
                    measurement_end_lsn: 210,
                    provenance_id: prov_id,
                },
                DurableOutcomeObservation {
                    metric_id: MetricId(2), // throughput (qps)
                    baseline: MetricValue::Unsigned(12200),
                    observed: MetricValue::Unsigned(22400),
                    measurement_start_lsn: 200,
                    measurement_end_lsn: 210,
                    provenance_id: prov_id,
                },
                DurableOutcomeObservation {
                    metric_id: MetricId(4), // error count
                    baseline: MetricValue::Unsigned(0),
                    observed: MetricValue::Unsigned(0),
                    measurement_start_lsn: 200,
                    measurement_end_lsn: 210,
                    provenance_id: prov_id,
                },
            ],
            provenance_id: prov_id,
        }
        .apply(&exp_seg, 210)
        .unwrap();

        // Verification closure
        let verify_all = |e_seg: &Arc<ExperienceSegment>, ent: &Arc<EntitySegment>| {
            let snap = e_seg.read_snapshot(300);
            let ent_snap = ent.read_snapshot(300);

            // 1. Verify Attempt 118 Full Trace
            let trace_118 = snap
                .full_trace(att_118, &ent_snap)
                .expect("trace 118 must exist");
            assert_eq!(trace_118.problem.symptom.as_ref(), "HighWriteLatency");
            assert_eq!(trace_118.attempt.state, AttemptState::Completed);
            assert_eq!(trace_118.ordered_actions[0].action_id, act_workers);
            assert_eq!(trace_118.outcome.as_ref().unwrap().observations.len(), 3);
            assert_eq!(
                trace_118.outcome.as_ref().unwrap().observations[0].observed,
                MetricValue::Unsigned(9700)
            );

            // 2. Verify Attempt 402 Full Trace
            let trace_402 = snap
                .full_trace(att_402, &ent_snap)
                .expect("trace 402 must exist");
            assert_eq!(trace_402.problem.symptom.as_ref(), "HighWriteLatency");
            assert_eq!(trace_402.attempt.state, AttemptState::Completed);
            assert_eq!(trace_402.ordered_actions[0].action_id, act_group);
            assert_eq!(trace_402.outcome.as_ref().unwrap().observations.len(), 3);
            assert_eq!(
                trace_402.outcome.as_ref().unwrap().observations[0].observed,
                MetricValue::Unsigned(3100)
            );

            // 3. Verify pattern queries
            let query = ExperienceQuery::new()
                .with_problem(prob_id)
                .with_action(act_group);
            let results = query.execute(&snap);
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].attempt_id, att_402);
        };

        verify_all(&exp_seg, &ent_seg);

        // Compact experience segment
        let compacted_exp = exp_seg.compact(2);
        verify_all(&compacted_exp, &ent_seg);
    }
}
