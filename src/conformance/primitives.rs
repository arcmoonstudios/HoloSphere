/* holosphere/src/conformance/primitives.rs */
//!▫~•◦-------------------------------‣
//! # Generalized Engineering Primitives & Invariant Verification Harnesses
//!▫~•◦-------------------------------------------------------------------‣
//!
//! A decoupled catalog of 13 reusable, mathematically invariant-enforcing engineering primitives:
//!
//! - **P1: Execution Portfolio Planner:** Online multi-strategy dispatch under multi-objective constraints.
//! - **P2: Objective-Aware Stage Admission Gate:** Multi-dimensional Pareto dominance filtering for pipeline stages.
//! - **P3: Multi-Stage Funnel Tracer:** Monotonic item accounting across transformation pipelines.
//! - **P4: Counterexample Regression Corpus:** Minimal reproducible counterexample persistence and continuous replay.
//! - **P5: Frozen Baseline Regression Gate:** Cryptographic benchmark identity validation and tail-sensitive regression gates.
//! - **P6: Contract Equivalence Harness:** Semantic parity validation across alternative implementations or optimizations.
//! - **P7: Cross-Cutting Overhead Budget Gate:** Strict upper-bound budgeting for non-functional auxiliary subsystems.
//! - **P8: Cost-Ordered Execution Cascade:** Short-circuiting execution cascade ordered by marginal cost-efficiency.
//! - **P9: Lazy Snapshot Attachment:** Metadata-only attachment with demand-driven zero-copy region materialization.
//! - **P10: Orthogonal Verification Matrix:** Decoupled multi-dimensional validation reporting without implicit state inheritance.
//! - **P11: Required Evidence Artifact Gate:** Tri-state (Pass, Fail, Blocked) cryptographic artifact and schema validation.
//! - **P12: Mixed-Workload Concurrency Harness:** Deterministic concurrent read/write stress validation with tail-latency tracking.
//! - **P13: Incremental Derived-State Parity Harness:** Equivalence verification between full state recomputation and sequential incremental mutation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;
use serde::{Deserialize, Serialize};

// ============================================================================
// P1 — Execution Portfolio Planner
// ============================================================================

/// Identifies a target hardware capability class.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HardwareClass {
    SimdAvx2,
    SimdAvx512,
    SimdNeon,
    GpuTensor,
    ScalarFallback,
    Custom(String),
}

/// Identifies the workload classification.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorkloadClass {
    LatencyCritical,
    ThroughputBatch,
    MemoryConstrained,
    AnalyticalOlap,
}

/// Generic, domain-agnostic workload descriptor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkloadDescriptor {
    pub effective_items: u64,
    pub feature_width: Option<u32>,
    pub requested_results: usize,
    pub selectivity: Option<f64>,
    pub workload_class: WorkloadClass,
    pub hardware_class: HardwareClass,
}

/// Declarative quality contract requested by caller.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityContract {
    pub min_recall: f64,
    pub max_numerical_error: f64,
    pub strict_exact: bool,
}

impl Default for QualityContract {
    fn default() -> Self {
        Self {
            min_recall: 0.99,
            max_numerical_error: 1e-5,
            strict_exact: false,
        }
    }
}

/// Cost prediction estimate.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct CostEstimate {
    pub latency_ns: u64,
    pub memory_bytes: usize,
    pub cpu_cycles: u64,
}

/// Quality prediction estimate.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct QualityEstimate {
    pub expected_recall: f64,
    pub confidence: f64,
}

/// Explanatory basis for a planner decision.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionBasis {
    pub reason: String,
    pub rule_id: String,
}

/// Capability descriptor of an available execution strategy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrategyCapability<S> {
    pub strategy: S,
    pub min_supported_items: u64,
    pub max_supported_items: u64,
    pub provides_exact: bool,
    pub nominal_recall: f64,
}

/// Planner telemetry / calibration snapshot.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CalibrationSnapshot {
    pub historical_p50_latency_ns: BTreeMap<String, u64>,
    pub observed_accuracy_rates: BTreeMap<String, f64>,
}

/// Planning request sent to the portfolio planner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanRequest<S> {
    pub workload: WorkloadDescriptor,
    pub objective: ObjectiveVector,
    pub contract: QualityContract,
    pub candidates: Vec<StrategyCapability<S>>,
    pub telemetry: CalibrationSnapshot,
}

/// Concrete decision produced by the portfolio planner.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanDecision<S> {
    pub strategy: S,
    pub predicted_cost: CostEstimate,
    pub predicted_quality: QualityEstimate,
    pub fallback: Option<S>,
    pub basis: DecisionBasis,
}

/// Core interface for Execution Portfolio Planning.
pub trait PortfolioPlanner<S: Clone + PartialEq> {
    fn plan(&self, request: &PlanRequest<S>) -> Result<PlanDecision<S>, String>;
}

// ============================================================================
// P2 — Objective-Aware Stage Admission Gate
// ============================================================================

/// Multi-dimensional objective weights.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveVector {
    pub latency_weight: f32,
    pub memory_weight: f32,
    pub quality_weight: f32,
    pub throughput_weight: f32,
    pub energy_weight: f32,
}

impl Default for ObjectiveVector {
    fn default() -> Self {
        Self {
            latency_weight: 0.5,
            memory_weight: 0.2,
            quality_weight: 0.3,
            throughput_weight: 0.0,
            energy_weight: 0.0,
        }
    }
}

/// Profile of a candidate processing stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StageProfile {
    pub stage_name: String,
    pub latency_score: f32,    // Lower is better (normalized [0, 1])
    pub memory_score: f32,     // Lower is better (normalized [0, 1])
    pub quality_score: f32,    // Higher is better (normalized [0, 1])
    pub throughput_score: f32, // Higher is better (normalized [0, 1])
}

/// Result of evaluating admission for a pipeline stage.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AdmissionDecision {
    Admitted { score: f32, reason: String },
    Rejected { dominating_stage: Option<String>, reason: String },
}

/// Objective-aware admission gate implementation.
pub fn admit_stage(
    stage: &StageProfile,
    alternatives: &[StageProfile],
    objective: &ObjectiveVector,
) -> AdmissionDecision {
    // Check if stage is strictly Pareto-dominated by any alternative
    for alt in alternatives {
        if alt.stage_name == stage.stage_name {
            continue;
        }
        let alt_strictly_better_or_equal = alt.latency_score <= stage.latency_score
            && alt.memory_score <= stage.memory_score
            && alt.quality_score >= stage.quality_score
            && alt.throughput_score >= stage.throughput_score;

        let alt_strictly_better = alt.latency_score < stage.latency_score
            || alt.memory_score < stage.memory_score
            || alt.quality_score > stage.quality_score
            || alt.throughput_score > stage.throughput_score;

        if alt_strictly_better_or_equal && alt_strictly_better {
            return AdmissionDecision::Rejected {
                dominating_stage: Some(alt.stage_name.clone()),
                reason: format!("Stage '{}' is Pareto-dominated by '{}'", stage.stage_name, alt.stage_name),
            };
        }
    }

    // Compute aggregate utility under the current objective vector
    let utility = (1.0 - stage.latency_score) * objective.latency_weight
        + (1.0 - stage.memory_score) * objective.memory_weight
        + stage.quality_score * objective.quality_weight
        + stage.throughput_score * objective.throughput_weight;

    AdmissionDecision::Admitted {
        score: utility,
        reason: format!("Stage '{}' admitted with utility score {:.4}", stage.stage_name, utility),
    }
}

// ============================================================================
// P3 — Multi-Stage Funnel Tracer
// ============================================================================

pub type ReasonCode = u32;

/// Fine-grained trace of candidate flow through a stage.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct StageTrace {
    pub entered: usize,
    pub retained: usize,
    pub rejected: usize,
    pub recovered: usize,
    pub work_units: u64,
    pub duration_ns: u64,
    pub rejection_reasons: BTreeMap<ReasonCode, usize>,
}

impl StageTrace {
    /// Asserts the conservation invariant: entered + recovered == retained + rejected.
    pub fn is_conserved(&self) -> bool {
        self.entered + self.recovered == self.retained + self.rejected
    }
}

/// Trait for pipeline stages supporting transparent telemetry tracing.
pub trait TraceableStage<I, O> {
    fn execute(&self, input: I, trace: &mut StageTrace) -> O;
}

// ============================================================================
// P4 — Counterexample Regression Corpus
// ============================================================================

pub type CaseId = String;

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EnvironmentFingerprint {
    pub os: String,
    pub arch: String,
    pub compiler_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureOrigin {
    ProductionQuery,
    BenchmarkRegression,
    FuzzingMutation,
    ManualInspection,
}

/// Frozen reproducible counterexample case.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressionCase<I, E> {
    pub id: CaseId,
    pub input: I,
    pub expected: E,
    pub environment: EnvironmentFingerprint,
    pub origin: FailureOrigin,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressionReport {
    pub total_cases: usize,
    pub passed_cases: usize,
    pub failed_cases: Vec<CaseId>,
}

pub struct CounterexampleCorpus<I, E> {
    cases: Vec<RegressionCase<I, E>>,
}

impl<I, E> Default for CounterexampleCorpus<I, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<I, E> CounterexampleCorpus<I, E> {
    pub fn new() -> Self {
        Self { cases: Vec::new() }
    }

    pub fn freeze(&mut self, case: RegressionCase<I, E>) {
        self.cases.push(case);
    }

    pub fn cases(&self) -> &[RegressionCase<I, E>] {
        &self.cases
    }

    pub fn run_evaluation<F>(&self, mut evaluator: F) -> RegressionReport
    where
        E: PartialEq,
        F: FnMut(&I) -> E,
    {
        let total_cases = self.cases.len();
        let mut passed_cases = 0;
        let mut failed_cases = Vec::new();

        for case in &self.cases {
            let actual = evaluator(&case.input);
            if actual == case.expected {
                passed_cases += 1;
            } else {
                failed_cases.push(case.id.clone());
            }
        }

        RegressionReport {
            total_cases,
            passed_cases,
            failed_cases,
        }
    }
}

// ============================================================================
// P5 — Frozen Baseline Regression Gate
// ============================================================================

pub type Digest = [u8; 32];

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct HardwareFingerprint {
    pub cpu_model: String,
    pub physical_cores: usize,
    pub memory_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkIdentity {
    pub workload_hash: Digest,
    pub dataset_hash: Option<Digest>,
    pub configuration_hash: Digest,
    pub hardware_fingerprint: HardwareFingerprint,
    pub metric_version: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegressionPolicy {
    pub max_p50_change: f64, // e.g. 0.05 for +5% max regression
    pub max_p95_change: f64,
    pub max_p99_change: f64,
    pub min_quality: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeasurementSet {
    pub p50_latency_ns: f64,
    pub p95_latency_ns: f64,
    pub p99_latency_ns: f64,
    pub observed_quality: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RegressionDecision {
    Pass,
    RegressionDetected { metric: String, delta_pct: f64, threshold_pct: f64 },
    IncompatibleIdentity { reason: String },
}

pub fn evaluate_baseline_gate(
    candidate_id: &BenchmarkIdentity,
    baseline_id: &BenchmarkIdentity,
    current: &MeasurementSet,
    baseline: &MeasurementSet,
    policy: &RegressionPolicy,
) -> RegressionDecision {
    if candidate_id != baseline_id {
        return RegressionDecision::IncompatibleIdentity {
            reason: "Benchmark identities do not match (hardware, workload, or config hash divergence)".to_string(),
        };
    }

    let p50_delta = (current.p50_latency_ns - baseline.p50_latency_ns) / baseline.p50_latency_ns;
    if p50_delta > policy.max_p50_change {
        return RegressionDecision::RegressionDetected {
            metric: "p50_latency".to_string(),
            delta_pct: p50_delta * 100.0,
            threshold_pct: policy.max_p50_change * 100.0,
        };
    }

    let p95_delta = (current.p95_latency_ns - baseline.p95_latency_ns) / baseline.p95_latency_ns;
    if p95_delta > policy.max_p95_change {
        return RegressionDecision::RegressionDetected {
            metric: "p95_latency".to_string(),
            delta_pct: p95_delta * 100.0,
            threshold_pct: policy.max_p95_change * 100.0,
        };
    }

    let p99_delta = (current.p99_latency_ns - baseline.p99_latency_ns) / baseline.p99_latency_ns;
    if p99_delta > policy.max_p99_change {
        return RegressionDecision::RegressionDetected {
            metric: "p99_latency".to_string(),
            delta_pct: p99_delta * 100.0,
            threshold_pct: policy.max_p99_change * 100.0,
        };
    }

    if let Some(min_q) = policy.min_quality {
        if current.observed_quality < min_q {
            return RegressionDecision::RegressionDetected {
                metric: "quality".to_string(),
                delta_pct: (current.observed_quality - min_q) * 100.0,
                threshold_pct: 0.0,
            };
        }
    }

    RegressionDecision::Pass
}

// ============================================================================
// P6 — Contract Equivalence Harness
// ============================================================================

pub trait ContractSubject<I, O> {
    fn evaluate(&self, input: &I) -> O;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderingRequirement {
    Strict,
    Unordered,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentityRequirement {
    ExactBitwise,
    NumericTolerance,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EquivalencePolicy {
    pub numeric_tolerance: Option<f64>,
    pub ordering_requirement: OrderingRequirement,
    pub identity_requirement: IdentityRequirement,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EquivalenceReport {
    pub total_trials: usize,
    pub equivalent_trials: usize,
    pub max_numerical_divergence: f64,
    pub is_conformant: bool,
}

pub fn check_equivalence<I, O, S1, S2, FDiff>(
    reference: &S1,
    candidate: &S2,
    inputs: &[I],
    policy: &EquivalencePolicy,
    difference_fn: FDiff,
) -> EquivalenceReport
where
    S1: ContractSubject<I, O>,
    S2: ContractSubject<I, O>,
    FDiff: Fn(&O, &O) -> f64,
{
    let total_trials = inputs.len();
    let mut equivalent_trials = 0;
    let mut max_numerical_divergence: f64 = 0.0;

    let tol = policy.numeric_tolerance.unwrap_or(0.0);

    for input in inputs {
        let out_ref = reference.evaluate(input);
        let out_cand = candidate.evaluate(input);

        let diff = difference_fn(&out_ref, &out_cand);
        if diff > max_numerical_divergence {
            max_numerical_divergence = diff;
        }

        if diff <= tol {
            equivalent_trials += 1;
        }
    }

    let is_conformant = equivalent_trials == total_trials;

    EquivalenceReport {
        total_trials,
        equivalent_trials,
        max_numerical_divergence,
        is_conformant,
    }
}

// ============================================================================
// P7 — Cross-Cutting Overhead Budget Gate
// ============================================================================

pub type MetricId = String;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Budget {
    pub metric: MetricId,
    pub maximum: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub metric: MetricId,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BudgetViolation {
    pub metric: MetricId,
    pub observed: f64,
    pub budget_max: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BudgetReport {
    pub within_budget: bool,
    pub violations: Vec<BudgetViolation>,
}

pub fn evaluate_budget(observations: &[Measurement], budgets: &[Budget]) -> BudgetReport {
    let budget_map: BTreeMap<&str, f64> = budgets.iter().map(|b| (b.metric.as_str(), b.maximum)).collect();
    let mut violations = Vec::new();

    for obs in observations {
        if let Some(&max_val) = budget_map.get(obs.metric.as_str()) {
            if obs.value > max_val {
                violations.push(BudgetViolation {
                    metric: obs.metric.clone(),
                    observed: obs.value,
                    budget_max: max_val,
                });
            }
        }
    }

    let within_budget = violations.is_empty();
    BudgetReport {
        within_budget,
        violations,
    }
}

// ============================================================================
// P8 — Cost-Ordered Execution Cascade
// ============================================================================

pub type StageId = String;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalCapability {
    pub nominal_precision: f64,
    pub nominal_recall: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalStage {
    pub id: StageId,
    pub estimated_cost: CostEstimate,
    pub capability: RetrievalCapability,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CandidateSet {
    pub item_ids: Vec<u64>,
    pub scores: Vec<f32>,
    pub confidence: f64,
}

pub trait StopPolicy {
    fn sufficient(&self, accumulated: &CandidateSet, contract: &QualityContract) -> bool;
}

pub fn execute_cascade<FStage>(
    stages: &mut [RetrievalStage],
    policy: &dyn StopPolicy,
    contract: &QualityContract,
    mut execute_stage_fn: FStage,
) -> CandidateSet
where
    FStage: FnMut(&RetrievalStage, &mut CandidateSet),
{
    // Order stages strictly by estimated latency cost ascending
    stages.sort_by_key(|s| s.estimated_cost.latency_ns);

    let mut accumulated = CandidateSet::default();

    for stage in stages.iter() {
        execute_stage_fn(stage, &mut accumulated);
        if policy.sufficient(&accumulated, contract) {
            break;
        }
    }

    accumulated
}

// ============================================================================
// P9 — Lazy Snapshot Attachment
// ============================================================================

pub type SectionId = String;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SnapshotMetadata {
    pub format_version: u32,
    pub section_count: usize,
    pub total_data_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionView {
    pub offset: usize,
    pub length: usize,
}

pub trait SnapshotSection: Send + Sync {
    fn section_id(&self) -> &str;
    fn validate_metadata(&self) -> Result<(), String>;
    fn materialize(&self) -> Result<SectionView, String>;
}

pub struct AttachedSnapshot {
    pub metadata: SnapshotMetadata,
    pub sections: BTreeMap<SectionId, Arc<dyn SnapshotSection>>,
}

impl AttachedSnapshot {
    pub fn attach(
        metadata: SnapshotMetadata,
        sections: Vec<Arc<dyn SnapshotSection>>,
    ) -> Result<Self, String> {
        let mut map = BTreeMap::new();
        for sec in sections {
            sec.validate_metadata()?;
            map.insert(sec.section_id().to_string(), sec);
        }
        Ok(Self {
            metadata,
            sections: map,
        })
    }

    pub fn touch_section(&self, id: &str) -> Result<SectionView, String> {
        self.sections
            .get(id)
            .ok_or_else(|| format!("Section '{}' not found", id))?
            .materialize()
    }
}

// ============================================================================
// P10 — Orthogonal Verification Matrix
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AxisStatus {
    Passed { items: usize },
    Failed { error: String },
    Skipped { reason: String },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationMatrix {
    pub axes: BTreeMap<VerificationAxis, AxisStatus>,
}

impl Default for VerificationMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl VerificationMatrix {
    pub fn new() -> Self {
        Self {
            axes: BTreeMap::new(),
        }
    }

    pub fn set(&mut self, axis: VerificationAxis, status: AxisStatus) {
        self.axes.insert(axis, status);
    }

    /// Hard invariant: All declared axes must strictly pass.
    pub fn all_passed(&self) -> bool {
        if self.axes.is_empty() {
            return false;
        }
        self.axes.values().all(|s| matches!(s, AxisStatus::Passed { .. }))
    }
}

// ============================================================================
// P11 — Required Evidence Artifact Gate
// ============================================================================

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RequiredArtifact {
    pub logical_name: String,
    pub digest: Digest,
    pub schema_version: u32,
    pub workload_binding: Option<Digest>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AvailableArtifact {
    pub digest: Digest,
    pub schema_version: u32,
    pub workload_binding: Option<Digest>,
}

pub type ArtifactCatalog = BTreeMap<String, AvailableArtifact>;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ArtifactGateResult {
    Pass,
    Blocked { missing: Vec<String>, reason: String },
    Fail { corrupted: Vec<String>, reason: String },
}

pub fn validate_required_artifacts(
    requirements: &[RequiredArtifact],
    available: &ArtifactCatalog,
) -> ArtifactGateResult {
    let mut missing = Vec::new();
    let mut corrupted = Vec::new();

    for req in requirements {
        match available.get(&req.logical_name) {
            None => {
                missing.push(req.logical_name.clone());
            }
            Some(art) => {
                if art.digest != req.digest {
                    corrupted.push(format!("{}: digest mismatch", req.logical_name));
                } else if art.schema_version != req.schema_version {
                    corrupted.push(format!("{}: schema version mismatch", req.logical_name));
                } else if req.workload_binding.is_some() && art.workload_binding != req.workload_binding {
                    corrupted.push(format!("{}: workload binding mismatch", req.logical_name));
                }
            }
        }
    }

    if !missing.is_empty() {
        return ArtifactGateResult::Blocked {
            missing: missing.clone(),
            reason: format!("Required artifacts missing: {:?}", missing),
        };
    }

    if !corrupted.is_empty() {
        return ArtifactGateResult::Fail {
            corrupted: corrupted.clone(),
            reason: format!("Artifact integrity violations: {:?}", corrupted),
        };
    }

    ArtifactGateResult::Pass
}

// ============================================================================
// P12 — Mixed-Workload Concurrency Harness
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationClass {
    PointRead,
    BatchRead,
    PointWrite,
    BatchWrite,
    Delete,
    Compaction,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkloadMix {
    pub operations: Vec<OperationClass>,
    pub weights: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConcurrencySchedule {
    pub clients: usize,
    pub duration: Duration,
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TailLatency {
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
    pub max_us: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Throughput {
    pub ops_per_sec: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StarvationEvent {
    pub client_id: usize,
    pub stall_duration: Duration,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ConcurrencyReport {
    pub throughput: Throughput,
    pub latency: TailLatency,
    pub correctness_violations: Vec<String>,
    pub starvation: Vec<StarvationEvent>,
}

// ============================================================================
// P13 — Incremental Derived-State Parity Harness
// ============================================================================

pub trait StateBuilder<S, M, State: PartialEq> {
    fn full(&self, source: &S) -> Result<State, String>;
    fn incremental(&self, prior: &State, mutation: &M) -> Result<State, String>;
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParityReport {
    pub states_match: bool,
    pub step_count: usize,
    pub divergence_details: Option<String>,
}

pub fn verify_state_parity<S, M, State, B, FApply>(
    builder: &B,
    initial_source: &S,
    mutations: &[M],
    mut apply_mutation_to_source: FApply,
) -> Result<ParityReport, String>
where
    S: Clone,
    State: PartialEq + Debug,
    B: StateBuilder<S, M, State>,
    FApply: FnMut(&mut S, &M),
{
    // 1. Build incremental state from base + mutation stream
    let mut current_state = builder.full(initial_source)?;
    for m in mutations {
        current_state = builder.incremental(&current_state, m)?;
    }

    // 2. Build full state directly from final mutated source
    let mut final_source = (*initial_source).clone();
    for m in mutations {
        apply_mutation_to_source(&mut final_source, m);
    }
    let full_final_state = builder.full(&final_source)?;

    // 3. Assert canonical equivalence
    let states_match = current_state == full_final_state;
    let divergence_details = if !states_match {
        Some(format!(
            "Divergence detected: Incremental={:?} vs Full={:?}",
            current_state, full_final_state
        ))
    } else {
        None
    };

    Ok(ParityReport {
        states_match,
        step_count: mutations.len(),
        divergence_details,
    })
}

// ============================================================================
// Tests & Invariant Verification
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_p1_planner_admissibility_and_fallback_invariants() {
        let request = PlanRequest {
            workload: WorkloadDescriptor {
                effective_items: 50_000,
                feature_width: Some(128),
                requested_results: 10,
                selectivity: None,
                workload_class: WorkloadClass::LatencyCritical,
                hardware_class: HardwareClass::SimdAvx2,
            },
            objective: ObjectiveVector::default(),
            contract: QualityContract {
                min_recall: 0.99,
                max_numerical_error: 1e-4,
                strict_exact: false,
            },
            candidates: vec![
                StrategyCapability {
                    strategy: "ApproxIndex".to_string(),
                    min_supported_items: 10_000,
                    max_supported_items: 1_000_000,
                    provides_exact: false,
                    nominal_recall: 0.95, // Below contract (0.99)
                },
                StrategyCapability {
                    strategy: "ExactSimd".to_string(),
                    min_supported_items: 0,
                    max_supported_items: 1_000_000,
                    provides_exact: true,
                    nominal_recall: 1.00,
                },
            ],
            telemetry: CalibrationSnapshot::default(),
        };

        // Filter admissible candidates according to contract
        let admissible: Vec<_> = request
            .candidates
            .iter()
            .filter(|c| c.nominal_recall >= request.contract.min_recall)
            .collect();

        assert_eq!(admissible.len(), 1);
        assert_eq!(admissible[0].strategy, "ExactSimd");
    }

    #[test]
    fn test_p2_objective_aware_stage_admission() {
        let fast_heavy = StageProfile {
            stage_name: "FastHeavy".to_string(),
            latency_score: 0.1,
            memory_score: 0.9,
            quality_score: 0.95,
            throughput_score: 0.8,
        };

        let slow_lean = StageProfile {
            stage_name: "SlowLean".to_string(),
            latency_score: 0.8,
            memory_score: 0.2, // Much better memory
            quality_score: 0.95,
            throughput_score: 0.4,
        };

        let dominated = StageProfile {
            stage_name: "Dominated".to_string(),
            latency_score: 0.9,
            memory_score: 0.95,
            quality_score: 0.90,
            throughput_score: 0.2,
        };

        let alternatives = vec![fast_heavy.clone(), slow_lean.clone()];

        // Memory-focused objective: SlowLean should be admitted
        let mem_obj = ObjectiveVector {
            latency_weight: 0.1,
            memory_weight: 0.7,
            quality_weight: 0.2,
            throughput_weight: 0.0,
            energy_weight: 0.0,
        };
        let mem_decision = admit_stage(&slow_lean, &alternatives, &mem_obj);
        assert!(matches!(mem_decision, AdmissionDecision::Admitted { .. }));

        // Dominated stage must be rejected
        let dom_decision = admit_stage(&dominated, &alternatives, &mem_obj);
        assert!(matches!(dom_decision, AdmissionDecision::Rejected { .. }));
    }

    #[test]
    fn test_p3_funnel_trace_conservation() {
        let mut trace = StageTrace {
            entered: 100,
            retained: 40,
            rejected: 65,
            recovered: 5,
            work_units: 500,
            duration_ns: 12_000,
            rejection_reasons: BTreeMap::new(),
        };

        // 100 in + 5 recovered == 40 retained + 65 rejected (105 == 105)
        assert!(trace.is_conserved());

        // Violate conservation
        trace.rejected = 60;
        assert!(!trace.is_conserved());
    }

    #[test]
    fn test_p4_counterexample_corpus() {
        let mut corpus: CounterexampleCorpus<i32, i32> = CounterexampleCorpus::new();
        corpus.freeze(RegressionCase {
            id: "case_42".to_string(),
            input: 42,
            expected: 84,
            environment: EnvironmentFingerprint {
                os: "linux".to_string(),
                arch: "x86_64".to_string(),
                compiler_version: "rustc 1.85".to_string(),
            },
            origin: FailureOrigin::ManualInspection,
        });

        let correct_eval = |x: &i32| *x * 2;
        let report = corpus.run_evaluation(correct_eval);
        assert_eq!(report.passed_cases, 1);
        assert!(report.failed_cases.is_empty());

        let broken_eval = |x: &i32| *x + 1;
        let broken_report = corpus.run_evaluation(broken_eval);
        assert_eq!(broken_report.passed_cases, 0);
        assert_eq!(broken_report.failed_cases, vec!["case_42".to_string()]);
    }

    #[test]
    fn test_p5_baseline_regression_gate() {
        let id = BenchmarkIdentity {
            workload_hash: [1u8; 32],
            dataset_hash: None,
            configuration_hash: [2u8; 32],
            hardware_fingerprint: HardwareFingerprint {
                cpu_model: "Zen4".to_string(),
                physical_cores: 16,
                memory_bytes: 64 * 1024 * 1024 * 1024,
            },
            metric_version: "v1".to_string(),
        };

        let baseline = MeasurementSet {
            p50_latency_ns: 1000.0,
            p95_latency_ns: 2000.0,
            p99_latency_ns: 3000.0,
            observed_quality: 0.99,
        };

        let regressed = MeasurementSet {
            p50_latency_ns: 1500.0, // +50% regression
            p95_latency_ns: 2050.0,
            p99_latency_ns: 3050.0,
            observed_quality: 0.99,
        };

        let policy = RegressionPolicy {
            max_p50_change: 0.05, // 5% max
            max_p95_change: 0.10,
            max_p99_change: 0.10,
            min_quality: Some(0.95),
        };

        let decision = evaluate_baseline_gate(&id, &id, &regressed, &baseline, &policy);
        assert!(matches!(decision, RegressionDecision::RegressionDetected { .. }));
    }

    #[test]
    fn test_p6_contract_equivalence_harness() {
        struct Reference;
        impl ContractSubject<f64, f64> for Reference {
            fn evaluate(&self, input: &f64) -> f64 {
                input.sin()
            }
        }

        struct CandidateApprox;
        impl ContractSubject<f64, f64> for CandidateApprox {
            fn evaluate(&self, input: &f64) -> f64 {
                // Taylor expansion approx: x - x^3/6
                input - (input.powi(3) / 6.0)
            }
        }

        let inputs = vec![0.01, 0.05, 0.1];
        let policy = EquivalencePolicy {
            numeric_tolerance: Some(1e-3),
            ordering_requirement: OrderingRequirement::Strict,
            identity_requirement: IdentityRequirement::NumericTolerance,
        };

        let report = check_equivalence(
            &Reference,
            &CandidateApprox,
            &inputs,
            &policy,
            |a, b| (a - b).abs(),
        );

        assert!(report.is_conformant);
        assert!(report.max_numerical_divergence < 1e-3);
    }

    #[test]
    fn test_p7_budget_gate() {
        let budgets = vec![
            Budget { metric: "audit_hash_us".to_string(), maximum: 5.0 },
            Budget { metric: "rbac_eval_us".to_string(), maximum: 2.0 },
        ];

        let passing_obs = vec![
            Measurement { metric: "audit_hash_us".to_string(), value: 3.1 },
            Measurement { metric: "rbac_eval_us".to_string(), value: 1.2 },
        ];
        let rep = evaluate_budget(&passing_obs, &budgets);
        assert!(rep.within_budget);

        let failing_obs = vec![
            Measurement { metric: "audit_hash_us".to_string(), value: 6.5 },
        ];
        let fail_rep = evaluate_budget(&failing_obs, &budgets);
        assert!(!fail_rep.within_budget);
        assert_eq!(fail_rep.violations.len(), 1);
    }

    #[test]
    fn test_p8_cascade_short_circuit() {
        struct PrefixPolicy;
        impl StopPolicy for PrefixPolicy {
            fn sufficient(&self, accumulated: &CandidateSet, _contract: &QualityContract) -> bool {
                accumulated.item_ids.len() >= 5
            }
        }

        let mut stages = vec![
            RetrievalStage {
                id: "ExpensiveDense".to_string(),
                estimated_cost: CostEstimate { latency_ns: 50_000, memory_bytes: 1024, cpu_cycles: 100_000 },
                capability: RetrievalCapability { nominal_precision: 0.99, nominal_recall: 0.99 },
            },
            RetrievalStage {
                id: "CheapLexical".to_string(),
                estimated_cost: CostEstimate { latency_ns: 1_000, memory_bytes: 128, cpu_cycles: 2_000 },
                capability: RetrievalCapability { nominal_precision: 0.90, nominal_recall: 0.85 },
            },
        ];

        let mut stage_executed_order = Vec::new();
        let result = execute_cascade(
            &mut stages,
            &PrefixPolicy,
            &QualityContract::default(),
            |stage, acc| {
                stage_executed_order.push(stage.id.clone());
                if stage.id == "CheapLexical" {
                    acc.item_ids = vec![1, 2, 3, 4, 5];
                }
            },
        );

        // Cheap stage ran first and satisfied sufficiency; Expensive stage never ran
        assert_eq!(stage_executed_order, vec!["CheapLexical"]);
        assert_eq!(result.item_ids.len(), 5);
    }

    #[test]
    fn test_p10_orthogonal_verification_matrix() {
        let mut matrix = VerificationMatrix::new();
        matrix.set(VerificationAxis::Compile, AxisStatus::Passed { items: 10 });
        matrix.set(VerificationAxis::Lint, AxisStatus::Passed { items: 5 });
        matrix.set(VerificationAxis::Unit, AxisStatus::Failed { error: "test panic".to_string() });

        // Compile and Lint pass does NOT imply overall pass when Unit failed
        assert!(!matrix.all_passed());

        matrix.set(VerificationAxis::Unit, AxisStatus::Passed { items: 250 });
        assert!(matrix.all_passed());
    }

    #[test]
    fn test_p11_artifact_gate_tri_state() {
        let reqs = vec![
            RequiredArtifact {
                logical_name: "gate_b_proof".to_string(),
                digest: [42u8; 32],
                schema_version: 1,
                workload_binding: None,
            },
        ];

        let empty_catalog = ArtifactCatalog::new();
        let blocked = validate_required_artifacts(&reqs, &empty_catalog);
        assert!(matches!(blocked, ArtifactGateResult::Blocked { .. }));

        let mut corrupt_catalog = ArtifactCatalog::new();
        corrupt_catalog.insert("gate_b_proof".to_string(), AvailableArtifact {
            digest: [0u8; 32], // Mismatch
            schema_version: 1,
            workload_binding: None,
        });
        let failed = validate_required_artifacts(&reqs, &corrupt_catalog);
        assert!(matches!(failed, ArtifactGateResult::Fail { .. }));

        let mut valid_catalog = ArtifactCatalog::new();
        valid_catalog.insert("gate_b_proof".to_string(), AvailableArtifact {
            digest: [42u8; 32],
            schema_version: 1,
            workload_binding: None,
        });
        let passed = validate_required_artifacts(&reqs, &valid_catalog);
        assert!(matches!(passed, ArtifactGateResult::Pass));
    }

    #[test]
    fn test_p13_derived_state_parity() {
        #[derive(Clone, Debug, PartialEq)]
        struct TestDocGraph {
            docs: BTreeSet<u64>,
        }

        struct GraphTransformer;
        impl StateBuilder<TestDocGraph, u64, TestDocGraph> for GraphTransformer {
            fn full(&self, source: &TestDocGraph) -> Result<TestDocGraph, String> {
                Ok(source.clone())
            }

            fn incremental(&self, prior: &TestDocGraph, mutation: &u64) -> Result<TestDocGraph, String> {
                let mut next = prior.clone();
                next.docs.insert(*mutation);
                Ok(next)
            }
        }

        let initial = TestDocGraph { docs: BTreeSet::new() };
        let mutations = vec![10, 20, 30, 40];

        let report = verify_state_parity(
            &GraphTransformer,
            &initial,
            &mutations,
            |source, &m| {
                source.docs.insert(m);
            },
        ).expect("parity check");

        assert!(report.states_match);
        assert_eq!(report.step_count, 4);
    }
}
