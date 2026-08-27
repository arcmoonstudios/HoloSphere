# HoloSphere Engineering Doctrine: 13 Reusable Primitives & Architectural Invariants

This document formalizes the 13 domain-neutral engineering primitives, their invariants, and the governing optimization process rule that guide HoloSphere's design, verification, indexing, planning, and context compilation subsystems.

---

## 1. The Core Architecture Loop

```text
                         MEASURED SYSTEM
                              │
             ┌────────────────┼────────────────┐
             │                │                │
        correctness       performance      evidence
             │                │                │
             ▼                ▼                ▼
      Contract Harness   Baseline Gate   Artifact Gate
      Counterexamples    Budget Gate     Verification Matrix
             │                │                │
             └────────────────┼────────────────┘
                              ▼
                     Calibration State
                              │
                              ▼
                  Execution Portfolio Planner
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
       Strategy A          Strategy B           Strategy C
          │                   │                    │
          └───────────────────┼────────────────────┘
                              ▼
                     Funnel / Outcome Trace
                              │
                              ▼
                      New Measurements
```

---

## 2. The 13 Reusable Engineering Primitives

| # | Primitive | Formal Interface | Core Invariants | $\Delta$ Online Cost | Breadth | HoloSphere Mapping |
| :- | :--- | :--- | :--- | :-: | :-: | :--- |
| **P1** | **Execution Portfolio Planner** | `PlanRequest -> PlanDecision` | Only eligible strategies considered; requested quality contract never weakened; fallback defined; decisions observable | 1/5 | 5/5 | [`src/planning/`](../src/planning/mod.rs), [`src/contextgraph/planner.rs`](../src/contextgraph/planner.rs) |
| **P2** | **Objective-Aware Stage Admission Gate** | `(StageProfile, ObjectiveVector) -> Admit/Reject` | No stage survives solely because it exists; evaluate latency, memory, quality, throughput, and energy independently | 0/5 | 5/5 | [`src/learning/`](../src/learning/mod.rs), [`src/planning/`](../src/planning/mod.rs) |
| **P3** | **Multi-Stage Funnel Tracer** | `(Input, PipelineStages, GroundTruth?) -> FunnelTrace` | Every candidate loss is attributable to a stage; counts monotonic unless stage can legitimately recover candidates | 0 prod / 4 diag | 4/5 | [`src/retrieval/`](../src/retrieval/), [`src/contextgraph/ir.rs`](../src/contextgraph/ir.rs) |
| **P4** | **Counterexample Regression Corpus** | `FailureObservation -> FrozenCase; run(Cases, Impl) -> Report` | Once a failure becomes reproducible, future versions cannot silently regress on it | 0/5 | 5/5 | [`tests/`](../tests/), [`src/experience/`](../src/experience/mod.rs) |
| **P5** | **Frozen Baseline Regression Gate** | `(WorkloadIdentity, Measurements, Baseline) -> RegressionDecision` | Baseline immutable except explicit promotion; workload identity must match; regressions never auto-blessed | 0/5 | 5/5 | [`benches/`](../benches/), CI Gates |
| **P6** | **Contract Equivalence Harness** | `(ReferenceImpl, CandidateImpl, ContractCases) -> EquivalenceReport` | Same declared semantics $\implies$ bounded numerical/result difference; persistence cannot alter contract | 0/5 | 5/5 | [`tests/`](../tests/), [`src/planning/`](../src/planning/mod.rs) |
| **P7** | **Cross-Cutting Overhead Budget Gate** | `(SubsystemMetrics, Budgets) -> BudgetReport` | Optimize only exceeded budgets; passing components do not become optimization targets without new evidence | $\approx 0$/5 | 5/5 | [`src/storage/`](../src/storage/), [`src/transport/`](../src/transport/) |
| **P8** | **Cost-Ordered Retrieval Cascade** | `Query + RetrievalStages + StopPolicy -> Results` | Cheap stage executes first only when quality contract permits; short-circuit must be explainable; fusion deterministic | 1/5 | 4/5 | [`src/contextgraph/query.rs`](../src/contextgraph/query.rs), [`src/retrieval/`](../src/retrieval/) |
| **P9** | **Lazy Snapshot Attachment** | `SnapshotDescriptor -> AttachedSnapshot` | Attach cost proportional to metadata, not corpus size; untouched pages remain untouched; first-touch semantics exact | 0–1/5 | 4/5 | [`src/storage/`](../src/storage/), [`src/contextgraph/store.rs`](../src/contextgraph/store.rs) |
| **P10**| **Orthogonal Verification Matrix** | `VerificationProfile -> MatrixReport` | Compile, test, benchmark, lint, and proof statuses are independent; one cannot imply another | 0/5 | 5/5 | [`.cargo/config.toml`](../.cargo/config.toml), CI Gates |
| **P11**| **Required Evidence Artifact Gate** | `(GateSpec, ArtifactManifest) -> GateResult` | Missing required evidence $\ne$ pass; artifact identity, version, and hash validated before evaluation | 0/5 | 4/5 | [`benches/`](../benches/), [`src/transport/`](../src/transport/) |
| **P12**| **Mixed-Workload Concurrency Harness** | `(WorkloadMix, ConcurrencySchedule, SLOs) -> ConcurrencyReport` | Correctness under load matters before throughput; p50 alone is insufficient; replayable deterministic seed | 0 prod / high bench | 5/5 | [`tests/`](../tests/), [`benches/`](../benches/) |
| **P13**| **Incremental Derived-State Parity Harness**| `(BaseState, MutationSeq, FullBuilder, IncrBuilder) -> ParityReport` | $\text{FULL}(\text{final}) \equiv \text{INCREMENTAL}(\text{base} + \text{mutations})$ logically; thread count/order must not change result | 0/5 | 4/5 | [`src/contextgraph/compiler.rs`](../src/contextgraph/compiler.rs), [`tests/`](../tests/) |

---

## 3. Detailed Primitive Specifications

### P1 — Execution Portfolio Planner
```rust
pub struct PlanRequest {
    pub workload: WorkloadDescriptor,
    pub objective: ObjectiveVector,
    pub contract: QualityContract,
    pub candidates: Vec<StrategyCapability>,
    pub telemetry: CalibrationSnapshot,
}

pub struct PlanDecision {
    pub strategy: StrategyId,
    pub predicted_cost: CostEstimate,
    pub predicted_quality: QualityEstimate,
    pub fallback: Option<StrategyId>,
    pub basis: DecisionBasis,
}

pub struct WorkloadDescriptor {
    pub effective_items: u64,
    pub feature_width: Option<u32>,
    pub requested_results: usize,
    pub selectivity: Option<f64>,
    pub workload_class: WorkloadClass,
    pub hardware_class: HardwareClass,
}
```
* **Resolution Pipeline**:
  $$\text{Analytical Model (Prior)} \xrightarrow[\text{Telemetry}]{\text{Calibration}} \text{Calibrated Expectation} \xrightarrow[\text{Contract}]{\text{Admissibility Constraint}} \text{Plan Decision}$$

---

### P2 — Objective-Aware Stage Admission Gate
```rust
pub struct ObjectiveVector {
    pub latency_weight: f32,
    pub memory_weight: f32,
    pub quality_weight: f32,
    pub throughput_weight: f32,
    pub energy_weight: f32,
}

pub fn admit_stage(
    stage: &StageProfile,
    alternatives: &[StageProfile],
    objective: &ObjectiveVector,
) -> AdmissionDecision;
```
* **Admission Invariant**: A component is removable only if it is strictly dominated over the supported objective region, or the objectives under which it wins are explicitly unsupported.

---

### P3 — Multi-Stage Funnel Tracer
```rust
pub trait TraceableStage<I, O> {
    fn execute(&self, input: I, trace: &mut StageTrace) -> O;
}

pub struct StageTrace {
    pub entered: usize,
    pub retained: usize,
    pub rejected: usize,
    pub recovered: usize,
    pub work_units: u64,
    pub duration_ns: u64,
    pub rejection_reasons: BTreeMap<ReasonCode, usize>,
}
```

---

### P4 — Counterexample Regression Corpus
```rust
pub struct RegressionCase<I, E> {
    pub id: CaseId,
    pub input: I,
    pub expected: E,
    pub environment: EnvironmentFingerprint,
    pub origin: FailureOrigin,
}
```
* **Freezing Pipeline**:
  $$\text{Newly Observed Failure} \longrightarrow \text{Minimize} \longrightarrow \text{Freeze} \longrightarrow \text{Permanent Admission Gate}$$

---

### P5 — Frozen Baseline Regression Gate
```rust
pub struct BenchmarkIdentity {
    pub workload_hash: [u8; 32],
    pub dataset_hash: Option<[u8; 32]>,
    pub configuration_hash: [u8; 32],
    pub hardware_fingerprint: HardwareFingerprint,
    pub metric_version: String,
}

pub struct RegressionPolicy {
    pub max_p50_change: f64,
    pub max_p95_change: f64,
    pub max_p99_change: f64,
    pub min_quality: Option<f64>,
}
```

---

### P6 — Contract Equivalence Harness
```rust
pub trait ContractSubject<I, O> {
    fn evaluate(&self, input: &I) -> O;
}

pub struct EquivalencePolicy {
    pub numeric_tolerance: Option<f64>,
    pub ordering_requirement: OrderingRequirement,
    pub identity_requirement: IdentityRequirement,
}
```

---

### P7 — Cross-Cutting Overhead Budget Gate
```rust
pub struct Budget {
    pub metric: MetricId,
    pub maximum: f64,
}

pub fn evaluate_budget(
    observations: &[Measurement],
    budgets: &[Budget],
) -> BudgetReport;
```

---

### P8 — Cost-Ordered Retrieval Cascade
```rust
pub struct RetrievalStage {
    pub id: StageId,
    pub estimated_cost: CostEstimate,
    pub capability: RetrievalCapability,
}

pub trait StopPolicy {
    fn sufficient(
        &self,
        accumulated: &CandidateSet,
        contract: &QualityContract,
    ) -> bool;
}
```
* **Execution Flow**:
  $$\text{Eligible Stages} \xrightarrow{\text{Sort by } \Delta \text{Utility} / \text{Cost}} \text{Stage}_1 \xrightarrow[\text{Sufficient?}]{\text{StopPolicy}} \text{Return or Advance to Stage}_{i+1}$$

---

### P9 — Lazy Snapshot Attachment
```rust
pub trait SnapshotSection {
    fn validate_metadata(&self) -> Result<()>;
    fn materialize(&self) -> Result<SectionView>;
}

pub struct AttachedSnapshot {
    pub metadata: SnapshotMetadata,
    pub sections: BTreeMap<SectionId, LazySection>,
}
```

---

### P10 — Orthogonal Verification Matrix
```rust
pub enum VerificationAxis {
    Format,
    Compile,
    Lint,
    Unit,
    Integration,
    Conformance,
    Benchmark,
    Proof,
    Fuzz,
}

pub struct VerificationMatrix {
    pub axes: BTreeMap<VerificationAxis, VerificationResult>,
}
```

---

### P11 — Required Evidence Artifact Gate
```rust
pub struct RequiredArtifact {
    pub logical_name: String,
    pub digest: [u8; 32],
    pub schema_version: String,
    pub workload_binding: Option<[u8; 32]>,
}

pub fn validate_required_artifacts(
    requirements: &[RequiredArtifact],
    available: &ArtifactCatalog,
) -> ArtifactGateResult;
```
* **Three-State Admissibility Contract**:
  - Required + Absent $\implies$ `BLOCKED`
  - Required + Incompatible / Corrupt $\implies$ `BLOCKED`
  - Required + Valid $\implies$ `EVALUATE`

---

### P12 — Mixed-Workload Concurrency Harness
```rust
pub struct WorkloadMix {
    pub operations: Vec<OperationClass>,
    pub weights: Vec<f64>,
}

pub struct ConcurrencySchedule {
    pub clients: usize,
    pub duration: Duration,
    pub seed: u64,
}

pub struct ConcurrencyReport {
    pub throughput: Throughput,
    pub latency: TailLatency,
    pub correctness_violations: Vec<Violation>,
    pub starvation: Vec<StarvationEvent>,
}
```

---

### P13 — Incremental Derived-State Parity Harness
```rust
pub trait StateBuilder<S, M> {
    fn full(&self, source: &S) -> Result<State>;
    fn incremental(
        &self,
        prior: &State,
        mutation: &M,
    ) -> Result<State>;
}
```

---

## 4. Fundamental State & Persistence Invariants

1. **Derived-State Parity Invariant**:
   $$\text{FULL}(\text{final\_source}) \equiv \text{INCREMENTAL}(\text{initial\_source} + \text{mutation\_sequence})$$
   *Logical and semantic equivalence must hold bit-for-bit across single vs. multi-threaded builds, arbitrary worker counts, renames, deletions, and crash-recovery replays.*

2. **Lazy Persistence Invariant**:
   $$\text{ATTACH}(\text{snapshot\_metadata}) \ne \text{EAGERLY TOUCH EVERYTHING}$$
   *First-touch materialization must be idempotent, metadata validation must precede exposure, and corruption must fail closed immediately.*

---

## 5. Explicitly Rejected Generalizations (Anti-Patterns)

| Rejected Abstraction | Justification for Rejection |
| :--- | :--- |
| **`UniversalAlgorithm` trait** wrapping Exact, Graph, ANN, and every execution engine | Different engines require fundamentally distinct capabilities and invariants. Forcing artificial behavioral uniformity provides zero additional use-case coverage while obscuring engine-specific constraints. Use capability descriptors and explicit dispatch instead. |
| **Generic artifact-management framework** solely for proof files | A required-artifact manifest and hash validator already completely solves the problem. Building an artifact lifecycle manager adds code complexity without expanding problem coverage. |
| **Runtime `OptimizationScheduler`** encoding "work on the biggest problem first" | Engineering prioritization is a developer policy, not a runtime software object. Encoding process rules as runtime code is a regression. |

$$\text{Harder to Use} + \text{Same Coverage} \equiv \text{Architectural Regression}$$

---

## 6. The Governing Process Rule

> **Evidence-Ordered Optimization**:
> Prioritize work strictly by **measured user-visible regret** or **violated system invariants**, never by architectural novelty or theoretical elegance.
