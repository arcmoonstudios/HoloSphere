# Governed Open-Ended Discovery

HoloSphere's discovery subsystem learns representations and declarative reasoning
rules while keeping the system's safety constitution immutable. It does not generate
or execute native code.

## End-to-end flow

```text
pinned experience + entity + N-ary relation snapshots
                    |
                    v
 schema induction + behavioral concept mapping
                    |
                    v
 temporal hypergraph motif mining
                    |
                    v
 resource-bounded OperatorProgram synthesis
                    |
                    v
 held-out / counterfactual / intervention / adversarial evaluation
                    |
                    v
 Provisional -> FalsificationTesting -> Shadow -> ShadowValidated
                    |
             external authority
                    |
                    v
             Admitted -> Monitored
                    |
          revise / deprecate / supersede
```

`ContinuousDiscoveryReport::replicated_actions` is the commit boundary. Each action
becomes its own Raft entry through `DataMutation::new_discovery_action`; intermediate
epistemic states therefore cannot be collapsed into a single invisible promotion.

## Acceptance contract

| Requirement | Implementation | Fail-closed property |
|---|---|---|
| Induce entity classes, N-ary types, roles/cardinalities, equivalence, and hierarchies | `schema.rs` | All begin Proposed; validation must use a later snapshot and independent empirical roots |
| Discover unknown motifs | `hyper_motif.rs` | Certified hyperedges only; bounded support/domain/root policy |
| Learn cross-domain mappings | `mapping.rs` | Names are unused; competing mappings persist; only Confirmed mappings resolve concepts |
| Synthesize reasoning laws | `dsl.rs` | Serializable AST only; bounded nodes, depth, effects, and numeric magnitude |
| Falsify candidates competitively | `evaluation.rs` | Admission requires every configured predictive, causal, transfer, calibration, robustness, MDL, and independence gate |
| Plan active evidence acquisition | `active_experiment.rs` | Risk filter plus explicit authorization, start, result, and completion lifecycle; live interventions remain external |
| Admit and reuse operators | `operator.rs`, `engine.rs` | Content-addressed identity, immutable definition during transition, external admission authority |
| Monitor and revise | `lifecycle.rs` | Counterexamples create a new version; earlier records are never rewritten; supersession is explicit |
| Replicate all discovery state | `state.rs`, `cluster/state_machine.rs` | Constitution-first ordering, expected-state compare-and-set, one discovery key per committed LSN |
| Recover after compaction/restart | `checkpoint.rs` | Versioned checksum, safety-kernel verification, operator identity verification, audit-chain verification |
| Keep a non-self-modifying constitution | `ImmutableSafetyKernel` | Digest-verified and replacement-protected; every governed mutation requires it first |

## Schema and mapping lifecycles

Schemas use:

```text
Proposed -> FalsificationTesting -> ShadowValidated -> Admitted -> Deprecated
                                  \-> Rejected
```

Mappings use the equivalent lifecycle ending in `Confirmed`. A mapping proposal has
no effect on runtime concept identity until it is Confirmed. Validation roots that
also supported induction are removed before the evidence thresholds are evaluated.

## Operator language and lifecycle

`OperatorProgram` conditions cover features, numeric comparison and transforms,
temporal persistence, motifs, causal motifs, domains, Boolean composition, and
constraints. Effects cover predicted outcomes/features, resolution proposals,
derived numeric values, and proposed hypergraph transformations. A transformation is
only a proposal; applying it still traverses canonical provenance, schema, epistemic,
and Raft admission paths.

Each `DeclarativeOperator` records training and validation cases, applicable domains
and contexts, counterexamples, empirical provenance roots, ancestry and previous
version, accuracy, calibration, uncertainty, monitoring statistics, and the program's
resource bounds. Only Admitted or Monitored operators are returned by future
reasoning through `DiscoveryCatalog::recommend_in_context`.

## Experiments

The planner preserves competing explanations and asks which bounded observation or
intervention has the highest estimated information gain. Authorization changes a
proposal to Authorized; `start_experiment` changes it to Running. Sandboxed replay and
simulation execute only in Running state. A/B tests and controlled configuration
changes always return `ExternalExecutionRequired`, even after authorization. A
Completed experiment must carry its result and exact evidence observations; a later
continuous cycle incorporates those observations into competitive falsification.

## Recovery and rollback

`GovernedDiscoveryCheckpoint` captures operators, schemas, mappings, evaluations,
experiments, the immutable kernel, and the audit chain at one LSN. Decode and restore
verify its digest, kernel, audit linkage, operator identities, and authority boundary.
The canonical evolved relation-schema projection is deterministically rebuilt during
recovery.

Rollback is compensating-only: `propose_compensating_rollback` creates a new operator
version containing an earlier program. It must pass the complete lifecycle again;
history is never deleted or rewritten.

## Verification

The executable acceptance suite is in
`src/learning/discovery/acceptance.rs`. It covers schema/mapping induction and
promotion, all motif classes, DSL sandboxing and transformations, competitive gates,
authorized experiment execution and replication, continuous admission/monitoring/
revision/supersession, ordered Raft replay, canonical relation synchronization,
checkpoint recovery, audit verification, compensating rollback, and safety-kernel
tamper rejection.

The defensible claim is governed open-ended autonomous discovery, not unrestricted
self-modification. Objectives, observations, DSL primitives, and the immutable safety
kernel remain engineered inputs.
