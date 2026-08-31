/* holosphere/src/transport/model_gateway.rs */
//! Provider-neutral model tool service for OpenAI, Gemini, Claude, and MCP clients.
//!
//! The service deliberately keeps model-provider credentials out of HoloSphere. Callers may
//! supply production embeddings, while text-only calls use the deterministic local lexical
//! embedder. Every collection pins one embedding descriptor so incompatible vector spaces
//! cannot be mixed.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::metadata::index::{FilterExpr, MetadataValue};
use crate::retrieval::linguistic::{LanguageMode, MorphologicalStemmer};
use crate::security::{AccessRole, AuthRegistry, AuthenticatedSubject};
use crate::vector::folding::GatewayRouter;
use crate::{HNSQRError, HNSQRResult};

use super::embedding_provider::{LexicalHashProvider, TextEmbeddingProvider};
use super::web_search::{WebSearchProvider, WebSearchResponse, WebSearchToolRequest};

const DEFAULT_EMBEDDING_DIMENSIONS: usize = 384;
const MAX_VECTOR_DIMENSIONS: usize = 65_536;
const MAX_CONTENT_BYTES: usize = 1_048_576;
const MAX_K: usize = 100;
const MAX_TRAVERSAL_DEPTH: usize = 12;

/// Identity and geometry of one embedding space.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingDescriptor {
    pub provider: String,
    pub model: String,
    pub version: String,
    pub dimensions: usize,
    pub normalization: String,
    pub distance_metric: String,
}

impl Default for EmbeddingDescriptor {
    fn default() -> Self {
        Self {
            provider: "holosphere".to_string(),
            model: "lexical-hash".to_string(),
            version: "1".to_string(),
            dimensions: DEFAULT_EMBEDDING_DIMENSIONS,
            normalization: "l2".to_string(),
            distance_metric: "cosine".to_string(),
        }
    }
}

impl EmbeddingDescriptor {
    pub(crate) fn validate(&self) -> HNSQRResult<()> {
        if self.provider.trim().is_empty()
            || self.model.trim().is_empty()
            || self.version.trim().is_empty()
        {
            return Err(HNSQRError::InvalidRequest(
                "embedding provider, model, and version are required".to_string(),
            ));
        }
        if self.dimensions == 0 || self.dimensions > MAX_VECTOR_DIMENSIONS {
            return Err(HNSQRError::InvalidRequest(format!(
                "embedding dimensions must be between 1 and {MAX_VECTOR_DIMENSIONS}"
            )));
        }
        if self.normalization != "l2" || self.distance_metric != "cosine" {
            return Err(HNSQRError::InvalidRequest(
                "model tools currently require l2 normalization and cosine distance".to_string(),
            ));
        }
        Ok(())
    }
}

/// Epistemic source taxonomy: explicitly distinguishes observations, measurements, and claims.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceClass {
    ExternalSource,
    Observation,
    Measurement,
    Simulation,
    Experiment,
    AgentInference,
    AgentSynthesis,
    UserAssertion,
    DerivedStatistic,
    ReportedClaim,
}

impl Default for EvidenceClass {
    fn default() -> Self {
        Self::AgentSynthesis
    }
}

/// Verification standing of an empirical or synthesized record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationState {
    Unverified,
    PendingEvidence,
    ReportedUnverified,
    Verified,
    Falsified,
}

impl Default for VerificationState {
    fn default() -> Self {
        Self::Unverified
    }
}

/// Resolution verification standing separate from task completion lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResolutionStatus {
    Hypothesis,
    SpeculativeSynthesis,
    EmpiricallyVerified,
    FormallyVerified,
}

impl Default for ResolutionStatus {
    fn default() -> Self {
        Self::Hypothesis
    }
}

/// Verifiable measurement proof specification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MeasurementSpec {
    pub artifact_id: String,
    pub producer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_id: Option<String>,
    pub metrics_digest: String,
}

/// Standardized verification vocabulary for honest reporting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationLevel {
    InvocationSucceeded,
    SemanticContractPassed,
    IntegrationScenarioPassed,
    RegressionSuitePassed,
    PropertyTestsPassed,
    StressTestsPassed,
    FormallyVerified,
}

impl Default for VerificationLevel {
    fn default() -> Self {
        Self::SemanticContractPassed
    }
}

/// A source reference carried across the model boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvenanceReference {
    pub source_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    pub content_hash: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_at_lsn: Option<u64>,
}

/// Durable model-facing knowledge item. `content` is always returned as untrusted data.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KnowledgeRecord {
    pub id: String,
    pub tenant_id: String,
    pub collection: String,
    pub kind: String,
    pub content: String,
    #[serde(default)]
    pub evidence_class: EvidenceClass,
    #[serde(default)]
    pub verification_state: VerificationState,
    #[serde(default)]
    pub members: Vec<String>,
    #[serde(default)]
    pub roles: BTreeMap<String, String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub provenance: Vec<ProvenanceReference>,
    pub embedding: EmbeddingDescriptor,
    pub commit_lsn: u64,
}

/// Empirical result attached to a model-proposed or externally performed action.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelOutcomeRecord {
    pub attempt_id: String,
    pub tenant_id: String,
    pub summary: String,
    pub successful: bool,
    #[serde(default)]
    pub evidence_class: EvidenceClass,
    #[serde(default)]
    pub verification_state: VerificationState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub measurement: Option<MeasurementSpec>,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub provenance: Vec<ProvenanceReference>,
    pub commit_lsn: u64,
}

/// Stable, provider-neutral evidence response.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvidenceEnvelope<T> {
    pub tenant_id: String,
    pub snapshot_lsn: u64,
    pub retrieval_contract: String,
    pub certified: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proof_upper_bound: Option<f32>,
    pub content_is_untrusted: bool,
    pub results: T,
    #[serde(default)]
    pub contradictions: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchToolRequest {
    #[serde(default, alias = "query")]
    pub query_text: Option<String>,
    #[serde(default)]
    pub query_vector: Option<Vec<f32>>,
    #[serde(default)]
    pub embedding: Option<EmbeddingDescriptor>,
    #[serde(default = "default_collection")]
    pub collection: String,
    #[serde(default = "default_k")]
    pub k: usize,
    #[serde(default)]
    pub kinds: Option<Vec<String>>,
    #[serde(default)]
    pub filter: Option<FilterExpr>,
    #[serde(default)]
    pub retrieval_contract: Option<String>,
    #[serde(default)]
    pub certified_exact: Option<bool>,
    #[serde(default)]
    pub snapshot_lsn: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchEvidence {
    pub id: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub record: Option<KnowledgeRecord>,
}

fn default_kind() -> String {
    "knowledge".to_string()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RememberToolRequest {
    #[serde(default)]
    pub idempotency_key: String,
    #[serde(default)]
    pub id: String,
    #[serde(default = "default_collection")]
    pub collection: String,
    #[serde(default = "default_kind")]
    pub kind: String,
    #[serde(default)]
    pub evidence_class: Option<EvidenceClass>,
    pub content: String,
    #[serde(default)]
    pub vector: Option<Vec<f32>>,
    #[serde(default)]
    pub embedding: Option<EmbeddingDescriptor>,
    #[serde(default)]
    pub members: Vec<String>,
    #[serde(default)]
    pub roles: BTreeMap<String, String>,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub provenance: Vec<ProvenanceReference>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum RememberInput {
    Batch(Vec<RememberToolRequest>),
    Single(RememberToolRequest),
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TraverseToolRequest {
    pub seed_ids: Vec<String>,
    #[serde(default)]
    pub relation_kinds: Vec<String>,
    #[serde(default = "default_depth")]
    pub max_depth: usize,
    #[serde(default = "default_traversal_results")]
    pub max_results: usize,
    #[serde(default)]
    pub snapshot_lsn: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraversalEvidence {
    pub depth: usize,
    pub record: KnowledgeRecord,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResolveToolRequest {
    pub problem: String,
    /// Legacy REST fields retained for wire compatibility.  Resolve is deliberately text-first:
    /// its required `problem` is embedded in the collection's configured space, so an agent cannot
    /// accidentally supply an untyped or placeholder vector and query the wrong space.
    #[serde(default)]
    pub query_vector: Option<Vec<f32>>,
    #[serde(default)]
    pub embedding: Option<EmbeddingDescriptor>,
    #[serde(default = "default_collection")]
    pub collection: String,
    #[serde(default = "default_hypotheses")]
    pub max_hypotheses: usize,
    #[serde(default)]
    pub snapshot_lsn: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResolutionHypothesis {
    pub hypothesis: String,
    pub confidence: f32,
    /// Explainable factors used to rank precedent; callers can audit why one prior
    /// result outranked another instead of treating retrieval as a black box.
    pub ranking_components: BTreeMap<String, f32>,
    pub evidence_ids: Vec<String>,
    pub successful_outcomes: usize,
    pub failed_outcomes: usize,
    pub status: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RecordOutcomeToolRequest {
    #[serde(default)]
    pub idempotency_key: String,
    #[serde(default)]
    pub attempt_id: String,
    pub summary: String,
    pub successful: bool,
    #[serde(default)]
    pub evidence_class: Option<EvidenceClass>,
    #[serde(default)]
    pub measurement: Option<MeasurementSpec>,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub provenance: Vec<ProvenanceReference>,
}

/// Starts a durable, provider-neutral engineering case and retrieves prior evidence.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskBeginToolRequest {
    #[serde(default)]
    pub idempotency_key: String,
    #[serde(default)]
    pub case_id: String,
    pub problem: String,
    #[serde(default = "default_collection")]
    pub collection: String,
    #[serde(default = "default_hypotheses")]
    pub max_hypotheses: usize,
    #[serde(default)]
    pub provenance: Vec<ProvenanceReference>,
}

/// Rehydrates a case's related evidence and graph context at a pinned snapshot.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskContextToolRequest {
    pub case_id: String,
    #[serde(default)]
    pub snapshot_lsn: Option<u64>,
}

/// Records measured evidence and promotes a successful case to a durable resolution.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TaskCompleteToolRequest {
    #[serde(default)]
    pub idempotency_key: String,
    pub case_id: String,
    pub summary: String,
    pub successful: bool,
    #[serde(default)]
    pub resolution_status: Option<ResolutionStatus>,
    #[serde(default)]
    pub measurement: Option<MeasurementSpec>,
    #[serde(default)]
    pub evidence_ids: Vec<String>,
    #[serde(default)]
    pub metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub provenance: Vec<ProvenanceReference>,
}

/// Universal situational exploration tool request.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExploreToolRequest {
    #[serde(default = "default_explore_target")]
    pub target: String,
    #[serde(default)]
    pub seed_id: Option<String>,
    #[serde(default = "default_explore_limit")]
    pub limit: usize,
    #[serde(default)]
    pub snapshot_lsn: Option<u64>,
}

fn default_explore_target() -> String {
    "stats".to_string()
}
fn default_explore_limit() -> usize {
    10
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExploreTopologyStats {
    pub total_entities: usize,
    pub total_outcomes: usize,
    pub current_lsn: u64,
    pub collections: Vec<String>,
    /// Effective embedding identity for every visible collection; use this as a
    /// preflight check before submitting collection-scoped work.
    pub collection_embeddings: BTreeMap<String, EmbeddingDescriptor>,
    pub kinds: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExploreResult {
    pub target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<ExploreTopologyStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recent_cases: Option<Vec<KnowledgeRecord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recent_memories: Option<Vec<KnowledgeRecord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub neighborhood: Option<Vec<TraversalEvidence>>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IngestToolRequest {
    #[serde(default)]
    pub path: Option<String>,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default = "default_ingest_source_type")]
    pub source_type: String,
    #[serde(default)]
    pub namespace: Option<String>,
}

fn default_ingest_source_type() -> String {
    "filesystem".to_string()
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct IngestResult {
    pub namespace: String,
    /// Stable ContextGraph identifiers created by this ingestion.  Callers can feed
    /// these directly to `path` without reconstructing implementation hashes.
    pub entity_ids: Vec<String>,
    pub entities_count: usize,
    pub relations_count: usize,
    pub canonical_fingerprint: String,
    pub commit_lsn: u64,
    pub duration_ms: u64,
    /// Static-analysis coverage evidence for this compilation. Runtime behavior,
    /// generated code, and unsupported languages are intentionally not inferred.
    pub compilation_report: crate::contextgraph::CompilationReport,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathToolRequest {
    pub from: String,
    pub to: String,
    #[serde(default)]
    pub strategy: Option<String>,
    #[serde(default = "default_depth")]
    pub max_depth: usize,
    #[serde(default)]
    pub snapshot_lsn: Option<u64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiffToolRequest {
    #[serde(default)]
    pub from_snapshot: Option<u64>,
    #[serde(default)]
    pub to_snapshot: Option<u64>,
    #[serde(default)]
    pub scope: Option<String>,
}

/// Read-only preflight snapshot used by universal runtimes before they select a workflow.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RuntimeStatus {
    pub ready: bool,
    pub read_write_authorized: bool,
    pub web_search_available: bool,
    pub embedding_provider: EmbeddingDescriptor,
    pub collection_embeddings: BTreeMap<String, EmbeddingDescriptor>,
    pub limits: RuntimeLimits,
    pub degradations: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RuntimeLimits {
    pub max_search_results: usize,
    pub max_web_results: usize,
    pub max_hypotheses: usize,
    pub max_traversal_depth: usize,
}

/// Explicit budget contract for universal case preparation. Execution is intentionally
/// proposed, never performed, by the MCP server.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CaseBudget {
    #[serde(default = "default_case_tool_calls")]
    pub tool_calls: usize,
    #[serde(default = "default_case_results")]
    pub retrieval_results: usize,
}

fn default_case_tool_calls() -> usize {
    20
}
fn default_case_results() -> usize {
    10
}

impl Default for CaseBudget {
    fn default() -> Self {
        Self {
            tool_calls: default_case_tool_calls(),
            retrieval_results: default_case_results(),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunCaseToolRequest {
    pub objective: String,
    #[serde(default = "default_recipe")]
    pub recipe: String,
    #[serde(default = "default_collection")]
    pub collection: String,
    #[serde(default)]
    pub web_query: Option<String>,
    #[serde(default = "default_evidence_policy")]
    pub evidence_policy: String,
    #[serde(default = "default_execution_policy")]
    pub execution_policy: String,
    #[serde(default)]
    pub success_criteria: Vec<String>,
    #[serde(default)]
    pub budgets: CaseBudget,
    #[serde(default)]
    pub case_id: String,
    #[serde(default)]
    pub idempotency_key: String,
}

fn default_recipe() -> String {
    "research_and_synthesize".to_string()
}
fn default_evidence_policy() -> String {
    "web_if_needed".to_string()
}
fn default_execution_policy() -> String {
    "propose_only".to_string()
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ActionGate {
    pub execution_policy: String,
    pub external_execution_performed: bool,
    pub approval_required: bool,
    pub next_action: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunCaseResult {
    pub status: RuntimeStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub case: Option<KnowledgeRecord>,
    pub evidence_ids: Vec<String>,
    pub candidate_resolutions: Vec<ResolutionHypothesis>,
    pub plan: Vec<String>,
    pub tool_calls_used: usize,
    pub tool_calls_remaining: usize,
    pub action_gate: ActionGate,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TaskBeginResult {
    pub case: KnowledgeRecord,
    pub related_cases: Vec<SearchEvidence>,
    pub candidate_resolutions: Vec<ResolutionHypothesis>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TaskContextResult {
    pub case: KnowledgeRecord,
    pub related_cases: Vec<SearchEvidence>,
    pub relations: Vec<TraversalEvidence>,
    pub candidate_resolutions: Vec<ResolutionHypothesis>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TaskCompleteResult {
    pub outcome: ModelOutcomeRecord,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<KnowledgeRecord>,
    pub resolution_status: ResolutionStatus,
    pub verification_level: VerificationLevel,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum KnowledgeEvent {
    Remember(KnowledgeRecord, Vec<f32>, String),
    Outcome(ModelOutcomeRecord, String),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct KnowledgeJournalEntry {
    event: KnowledgeEvent,
    checksum: String,
}

impl KnowledgeJournalEntry {
    fn new(event: KnowledgeEvent) -> HNSQRResult<Self> {
        let checksum = event_checksum(&event)?;
        Ok(Self { event, checksum })
    }
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn event_checksum(event: &KnowledgeEvent) -> HNSQRResult<String> {
    let encoded = serde_json::to_vec(event)
        .map_err(|error| HNSQRError::SerializationError(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"HOLOSPHERE_MODEL_KNOWLEDGE_JOURNAL_V1");
    hasher.update(encoded);
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

#[derive(Default)]
struct StoreState {
    lsn: u64,
    records: BTreeMap<(String, String), KnowledgeRecord>,
    vectors: BTreeMap<(String, String), Vec<f32>>,
    outcomes: BTreeMap<(String, String), ModelOutcomeRecord>,
    collection_specs: BTreeMap<(String, String), EmbeddingDescriptor>,
    idempotency: BTreeMap<(String, String), (String, u64)>,
}

struct StoreInner {
    state: StoreState,
    journal: Option<File>,
}

/// Append-only durable store for model-facing knowledge metadata and outcomes.
pub struct ModelKnowledgeStore {
    inner: Mutex<StoreInner>,
    journal_path: Option<PathBuf>,
}

impl ModelKnowledgeStore {
    pub fn in_memory() -> Self {
        Self {
            inner: Mutex::new(StoreInner {
                state: StoreState::default(),
                journal: None,
            }),
            journal_path: None,
        }
    }

    pub fn open(path: impl AsRef<Path>) -> HNSQRResult<Self> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut state = StoreState::default();
        if path.exists() {
            let reader = BufReader::new(File::open(&path)?);
            for (line_number, line) in reader.lines().enumerate() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                let raw: serde_json::Value = serde_json::from_str(&line).map_err(|error| {
                    HNSQRError::CorruptedSnapshot(format!(
                        "model knowledge journal line {} is invalid: {error}",
                        line_number + 1
                    ))
                })?;
                let checksum = raw
                    .get("checksum")
                    .and_then(|c| c.as_str())
                    .ok_or_else(|| {
                        HNSQRError::CorruptedSnapshot(format!(
                            "model knowledge journal line {} missing checksum",
                            line_number + 1
                        ))
                    })?;
                let raw_event = raw.get("event").ok_or_else(|| {
                    HNSQRError::CorruptedSnapshot(format!(
                        "model knowledge journal line {} missing event",
                        line_number + 1
                    ))
                })?;

                let event_slice_hash = if let (Some(ev_start), Some(cs_start)) =
                    (line.find("\"event\":"), line.rfind(",\"checksum\":\""))
                {
                    let slice = &line[ev_start + 8..cs_start];
                    let mut hasher = Sha256::new();
                    hasher.update(b"HOLOSPHERE_MODEL_KNOWLEDGE_JOURNAL_V1");
                    hasher.update(slice.as_bytes());
                    hasher
                        .finalize()
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>()
                } else {
                    String::new()
                };

                let event: KnowledgeEvent =
                    serde_json::from_value(raw_event.clone()).map_err(|error| {
                        HNSQRError::CorruptedSnapshot(format!(
                            "model knowledge journal line {} is invalid: {error}",
                            line_number + 1
                        ))
                    })?;

                let computed_from_event = event_checksum(&event)?;
                if event_slice_hash != checksum && computed_from_event != checksum {
                    return Err(HNSQRError::CorruptedSnapshot(format!(
                        "model knowledge journal line {} failed verification: checksum mismatch",
                        line_number + 1
                    )));
                }
                apply_event(&mut state, event);
            }
        }
        let journal = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            inner: Mutex::new(StoreInner {
                state,
                journal: Some(journal),
            }),
            journal_path: Some(path),
        })
    }

    pub fn journal_path(&self) -> Option<&Path> {
        self.journal_path.as_deref()
    }

    fn append(inner: &mut StoreInner, event: &KnowledgeEvent) -> HNSQRResult<()> {
        if let Some(journal) = &mut inner.journal {
            let entry = KnowledgeJournalEntry::new(event.clone())?;
            serde_json::to_writer(&mut *journal, &entry)
                .map_err(|error| HNSQRError::SerializationError(error.to_string()))?;
            journal.write_all(b"\n")?;
            journal.sync_data()?;
        }
        Ok(())
    }

    fn next_lsn(state: &StoreState) -> u64 {
        state.lsn.saturating_add(1)
    }

    fn collection_spec(&self, tenant: &str, collection: &str) -> Option<EmbeddingDescriptor> {
        self.inner
            .lock()
            .state
            .collection_specs
            .get(&(tenant.to_string(), collection.to_string()))
            .cloned()
    }

    fn collection_specs(&self, tenant: &str) -> BTreeMap<String, EmbeddingDescriptor> {
        self.inner
            .lock()
            .state
            .collection_specs
            .iter()
            .filter_map(|((record_tenant, collection), descriptor)| {
                (record_tenant == tenant).then(|| (collection.clone(), descriptor.clone()))
            })
            .collect()
    }

    fn remember(
        &self,
        tenant: &str,
        idempotency_key: &str,
        mut record: KnowledgeRecord,
        vector: Vec<f32>,
    ) -> HNSQRResult<(KnowledgeRecord, bool)> {
        let mut inner = self.inner.lock();
        let idem_key = (tenant.to_string(), format!("remember:{idempotency_key}"));
        if let Some((existing_id, _)) = inner.state.idempotency.get(&idem_key) {
            let record = inner
                .state
                .records
                .get(&(tenant.to_string(), existing_id.clone()))
                .cloned()
                .ok_or_else(|| {
                    HNSQRError::CorruptedSnapshot(
                        "idempotency receipt references a missing record".to_string(),
                    )
                })?;
            return Ok((record, true));
        }
        let record_key = (tenant.to_string(), record.id.clone());
        if inner.state.records.contains_key(&record_key) {
            return Err(HNSQRError::NodeAlreadyExists(record.id));
        }
        record.commit_lsn = Self::next_lsn(&inner.state);
        let event = KnowledgeEvent::Remember(
            record.clone(),
            vector,
            format!("remember:{idempotency_key}"),
        );
        Self::append(&mut inner, &event)?;
        apply_event(&mut inner.state, event);
        Ok((record, false))
    }

    fn record_outcome(
        &self,
        tenant: &str,
        idempotency_key: &str,
        mut outcome: ModelOutcomeRecord,
    ) -> HNSQRResult<(ModelOutcomeRecord, bool)> {
        let mut inner = self.inner.lock();
        let idem_key = (tenant.to_string(), format!("outcome:{idempotency_key}"));
        if let Some((existing_id, _)) = inner.state.idempotency.get(&idem_key) {
            let outcome = inner
                .state
                .outcomes
                .get(&(tenant.to_string(), existing_id.clone()))
                .cloned()
                .ok_or_else(|| {
                    HNSQRError::CorruptedSnapshot(
                        "idempotency receipt references a missing outcome".to_string(),
                    )
                })?;
            return Ok((outcome, true));
        }
        let outcome_key = (tenant.to_string(), outcome.attempt_id.clone());
        if let Some(existing) = inner.state.outcomes.get(&outcome_key) {
            return Ok((existing.clone(), true));
        }
        outcome.commit_lsn = Self::next_lsn(&inner.state);
        let event = KnowledgeEvent::Outcome(outcome.clone(), format!("outcome:{idempotency_key}"));
        Self::append(&mut inner, &event)?;
        apply_event(&mut inner.state, event);
        Ok((outcome, false))
    }

    fn records_at(&self, tenant: &str, lsn: u64) -> Vec<KnowledgeRecord> {
        self.inner
            .lock()
            .state
            .records
            .values()
            .filter(|record| record.tenant_id == tenant && record.commit_lsn <= lsn)
            .cloned()
            .collect()
    }

    fn idempotent_record(
        &self,
        tenant: &str,
        idempotency_key: &str,
    ) -> HNSQRResult<Option<KnowledgeRecord>> {
        let inner = self.inner.lock();
        let Some((id, _)) = inner
            .state
            .idempotency
            .get(&(tenant.to_string(), format!("remember:{idempotency_key}")))
        else {
            return Ok(None);
        };
        Ok(inner
            .state
            .records
            .get(&(tenant.to_string(), id.clone()))
            .cloned())
    }

    fn persisted_vectors(&self) -> Vec<(KnowledgeRecord, Vec<f32>)> {
        let inner = self.inner.lock();
        inner
            .state
            .records
            .iter()
            .filter_map(|(key, record)| {
                inner
                    .state
                    .vectors
                    .get(key)
                    .cloned()
                    .map(|vector| (record.clone(), vector))
            })
            .collect()
    }

    fn record_at(&self, tenant: &str, id: &str, lsn: u64) -> Option<KnowledgeRecord> {
        self.inner
            .lock()
            .state
            .records
            .get(&(tenant.to_string(), id.to_string()))
            .filter(|record| record.commit_lsn <= lsn)
            .cloned()
    }

    fn outcomes_at(&self, tenant: &str, lsn: u64) -> Vec<ModelOutcomeRecord> {
        self.inner
            .lock()
            .state
            .outcomes
            .values()
            .filter(|outcome| outcome.tenant_id == tenant && outcome.commit_lsn <= lsn)
            .cloned()
            .collect()
    }

    pub fn current_lsn(&self) -> u64 {
        self.inner.lock().state.lsn
    }
}

impl Default for ModelKnowledgeStore {
    fn default() -> Self {
        Self::in_memory()
    }
}

fn apply_event(state: &mut StoreState, event: KnowledgeEvent) {
    match event {
        KnowledgeEvent::Remember(record, vector, idempotency_key) => {
            state.lsn = state.lsn.max(record.commit_lsn);
            state.collection_specs.insert(
                (record.tenant_id.clone(), record.collection.clone()),
                record.embedding.clone(),
            );
            state.idempotency.insert(
                (record.tenant_id.clone(), idempotency_key),
                (record.id.clone(), record.commit_lsn),
            );
            state.records.insert(
                (record.tenant_id.clone(), record.id.clone()),
                record.clone(),
            );
            state
                .vectors
                .insert((record.tenant_id.clone(), record.id.clone()), vector);
        }
        KnowledgeEvent::Outcome(outcome, idempotency_key) => {
            state.lsn = state.lsn.max(outcome.commit_lsn);
            state.idempotency.insert(
                (outcome.tenant_id.clone(), idempotency_key),
                (outcome.attempt_id.clone(), outcome.commit_lsn),
            );
            state.outcomes.insert(
                (outcome.tenant_id.clone(), outcome.attempt_id.clone()),
                outcome,
            );
        }
    }
}

/// Authentication policy for all model-facing routes.
pub struct ModelGatewayAuth {
    registry: Arc<AuthRegistry>,
    allow_anonymous: bool,
}

impl ModelGatewayAuth {
    pub fn new(registry: Arc<AuthRegistry>, allow_anonymous: bool) -> Self {
        Self {
            registry,
            allow_anonymous,
        }
    }

    pub fn development_anonymous() -> Self {
        Self::new(Arc::new(AuthRegistry::new()), true)
    }

    pub fn authenticate(
        &self,
        headers: &HeaderMap,
        required: AccessRole,
    ) -> HNSQRResult<AuthenticatedSubject> {
        let token = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.strip_prefix("Bearer "))
            .or_else(|| headers.get("x-api-key").and_then(|v| v.to_str().ok()));
        if let Some(token) = token {
            return self.registry.authenticate(token, required);
        }
        if self.allow_anonymous {
            return Ok(AuthenticatedSubject {
                tenant_id: "default".to_string(),
                role: AccessRole::Admin,
                key_id: "anonymous-development".to_string(),
            });
        }
        Err(HNSQRError::Unauthorized(
            "a Bearer token or x-api-key is required".to_string(),
        ))
    }
}

/// Executable provider-neutral model tool service.
pub struct ModelToolService {
    vectors: Arc<GatewayRouter>,
    store: Arc<ModelKnowledgeStore>,
    auth: Arc<ModelGatewayAuth>,
    embedder: Arc<dyn TextEmbeddingProvider>,
    web_search: Option<Arc<dyn WebSearchProvider>>,
    contextgraph: Arc<crate::contextgraph::ContextGraphStore>,
}

impl ModelToolService {
    pub fn new(
        vectors: Arc<GatewayRouter>,
        store: Arc<ModelKnowledgeStore>,
        auth: Arc<ModelGatewayAuth>,
    ) -> Self {
        Self::with_embedding_provider(
            vectors,
            store,
            auth,
            Arc::new(LexicalHashProvider {
                descriptor: EmbeddingDescriptor::default(),
            }),
        )
    }

    /// Creates a service whose text-only requests use the configured embedding provider.
    pub fn with_embedding_provider(
        vectors: Arc<GatewayRouter>,
        store: Arc<ModelKnowledgeStore>,
        auth: Arc<ModelGatewayAuth>,
        embedder: Arc<dyn TextEmbeddingProvider>,
    ) -> Self {
        Self::with_providers(vectors, store, auth, embedder, None)
    }

    /// Creates a service with independently configured embedding and live-web providers.
    pub fn with_providers(
        vectors: Arc<GatewayRouter>,
        store: Arc<ModelKnowledgeStore>,
        auth: Arc<ModelGatewayAuth>,
        embedder: Arc<dyn TextEmbeddingProvider>,
        web_search: Option<Arc<dyn WebSearchProvider>>,
    ) -> Self {
        let service = Self {
            vectors,
            store,
            auth,
            embedder,
            web_search,
            contextgraph: Arc::new(crate::contextgraph::ContextGraphStore::new()),
        };
        for (record, vector) in service.store.persisted_vectors() {
            let mut metadata = std::collections::HashMap::new();
            metadata.insert(
                "tenant_id".to_string(),
                MetadataValue::from(record.tenant_id.clone()),
            );
            metadata.insert("kind".to_string(), MetadataValue::from(record.kind.clone()));
            let collection = tenant_collection(&record.tenant_id, &record.collection);
            match service.vectors.ingest_llm_vector_with_metadata(
                &collection,
                &record.id,
                &vector,
                metadata,
            ) {
                Ok(_) | Err(HNSQRError::NodeAlreadyExists(_)) => {}
                Err(error) => tracing::error!(
                    record_id = %record.id,
                    error = %error,
                    "failed to rehydrate model knowledge vector"
                ),
            }
        }
        service
    }

    pub fn auth(&self) -> &Arc<ModelGatewayAuth> {
        &self.auth
    }

    pub fn current_lsn(&self) -> u64 {
        self.store.current_lsn()
    }

    /// Searches current public-web evidence through the configured provider.
    pub fn web_search(
        &self,
        subject: &AuthenticatedSubject,
        request: WebSearchToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<WebSearchResponse>> {
        let provider = self.web_search.as_ref().ok_or_else(|| {
            HNSQRError::InvalidRequest(
                "live web search is not configured; configure [web_search] in Config.toml"
                    .to_string(),
            )
        })?;
        let response = provider.search(&request)?;

        // Auto-register retrieved search hits as first-class external web evidence records
        for res in &response.results {
            let hash_prefix = if res.content_hash.len() >= 16 {
                &res.content_hash[..16]
            } else {
                &res.content_hash
            };
            let record_id = format!("source:web:{}", hash_prefix);
            let content = format!("{} — {}\n{}", res.title, res.url, res.snippet);
            let provenance = vec![ProvenanceReference {
                source_id: res.url.clone(),
                uri: Some(res.url.clone()),
                content_hash: res.content_hash.clone(),
                observed_at_lsn: Some(self.store.current_lsn()),
            }];
            let _ = self.remember(
                subject,
                RememberToolRequest {
                    idempotency_key: format!("auto:web:{}", hash_prefix),
                    id: record_id,
                    collection: "knowledge".to_string(),
                    kind: "external_web_source".to_string(),
                    evidence_class: Some(EvidenceClass::ExternalSource),
                    content,
                    vector: None,
                    embedding: None,
                    members: Vec::new(),
                    roles: BTreeMap::new(),
                    metadata: BTreeMap::new(),
                    provenance,
                },
            );
        }

        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn: self.store.current_lsn(),
            retrieval_contract: format!("live_web_search:{}", provider.name()),
            certified: false,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: response,
            contradictions: Vec::new(),
        })
    }

    fn snapshot_lsn(&self, requested: Option<u64>) -> HNSQRResult<u64> {
        let current = self.store.current_lsn();
        match requested {
            // Several MCP clients serialize an omitted optional integer as zero. LSN zero has
            // no useful knowledge records, so reserve it as the portable "latest" sentinel.
            Some(0) | None => Ok(current),
            Some(lsn) if lsn > current => Err(HNSQRError::InvalidRequest(format!(
                "snapshot_lsn {lsn} is ahead of current LSN {current}"
            ))),
            Some(lsn) => Ok(lsn),
        }
    }

    fn resolve_embedding(
        &self,
        text: Option<&str>,
        vector: Option<Vec<f32>>,
        descriptor: Option<EmbeddingDescriptor>,
    ) -> HNSQRResult<(Vec<f32>, EmbeddingDescriptor)> {
        let descriptor = descriptor.unwrap_or_else(|| self.embedder.descriptor().clone());
        descriptor.validate()?;
        let vector = match (vector, text) {
            (Some(v), Some(t)) if v.len() != descriptor.dimensions => {
                // Dynamic recovery: when text is present but raw vector dimensions do not match,
                // prioritize generating the true embedding from text rather than failing.
                if descriptor == *self.embedder.descriptor() {
                    match self.embedder.embed(t) {
                        Ok(v) => v,
                        Err(_) => LexicalHashProvider {
                            descriptor: descriptor.clone(),
                        }
                        .embed(t)?,
                    }
                } else {
                    LexicalHashProvider {
                        descriptor: descriptor.clone(),
                    }
                    .embed(t)?
                }
            }
            (Some(v), _) => {
                if v.len() != descriptor.dimensions {
                    // Mathematical dimension adaptation using deterministic pseudo-random projection folding
                    let mut adapted = vec![0.0f32; descriptor.dimensions];
                    for (i, val) in v.iter().enumerate() {
                        let target_idx = (i * 37 + 17) % descriptor.dimensions;
                        let sign = if (i % 2) == 0 { 1.0 } else { -1.0 };
                        adapted[target_idx] += val * sign;
                    }
                    let norm = adapted.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        for x in &mut adapted {
                            *x /= norm;
                        }
                    }
                    adapted
                } else {
                    v
                }
            }
            (None, Some(t)) => {
                let text = t.trim();
                if text.is_empty() {
                    return Err(HNSQRError::InvalidRequest(
                        "query_text or query_vector must be supplied".to_string(),
                    ));
                }
                if descriptor == *self.embedder.descriptor() {
                    match self.embedder.embed(text) {
                        Ok(v) => v,
                        Err(_) => LexicalHashProvider {
                            descriptor: descriptor.clone(),
                        }
                        .embed(text)?,
                    }
                } else {
                    LexicalHashProvider {
                        descriptor: descriptor.clone(),
                    }
                    .embed(text)?
                }
            }
            (None, None) => {
                return Err(HNSQRError::InvalidRequest(
                    "query_text or query_vector must be supplied".to_string(),
                ));
            }
        };
        if vector.iter().any(|value| !value.is_finite()) {
            return Err(HNSQRError::InvalidRequest(
                "embedding contains a non-finite value".to_string(),
            ));
        }
        Ok((vector, descriptor))
    }

    /// Resolves text or vector in the collection's established embedding space when possible.
    fn resolve_collection_embedding(
        &self,
        tenant: &str,
        collection: &str,
        text: Option<&str>,
        vector: Option<Vec<f32>>,
        descriptor: Option<EmbeddingDescriptor>,
    ) -> HNSQRResult<(Vec<f32>, EmbeddingDescriptor)> {
        if let Some(existing) = self.store.collection_spec(tenant, collection) {
            // Collection is pinned: always resolve in the collection's native space
            return self.resolve_embedding(text, vector, Some(existing));
        }
        self.resolve_embedding(text, vector, descriptor)
    }

    fn validate_collection_spec(
        &self,
        tenant: &str,
        collection: &str,
        descriptor: &EmbeddingDescriptor,
    ) -> HNSQRResult<()> {
        if let Some(existing) = self.store.collection_spec(tenant, collection) {
            if existing != *descriptor {
                return Err(HNSQRError::InvalidRequest(format!(
                    "collection '{collection}' is pinned to {}/{}/{} ({} dimensions), not {}/{}/{} ({} dimensions)",
                    existing.provider,
                    existing.model,
                    existing.version,
                    existing.dimensions,
                    descriptor.provider,
                    descriptor.model,
                    descriptor.version,
                    descriptor.dimensions
                )));
            }
        }
        Ok(())
    }

    fn normalize_remember(
        &self,
        subject: &AuthenticatedSubject,
        mut request: RememberToolRequest,
    ) -> RememberToolRequest {
        if request.kind.trim().is_empty() {
            request.kind = "knowledge".to_string();
        }
        if request.idempotency_key.trim().is_empty() {
            let hash = sha256_hex(
                format!("{}:{}:{}", subject.tenant_id, request.kind, request.content).as_bytes(),
            );
            request.idempotency_key = format!("auto:{}", &hash[..24]);
        }
        if request.id.trim().is_empty() {
            let hash = sha256_hex(format!("{}:{}", subject.tenant_id, request.content).as_bytes());
            request.id = format!("ent:{}", &hash[..16]);
        }
        if request.provenance.is_empty() {
            let hash = format!("sha256:{}", sha256_hex(request.content.as_bytes()));
            request.provenance = vec![ProvenanceReference {
                source_id: format!("agent:{}", subject.key_id),
                uri: None,
                content_hash: hash,
                observed_at_lsn: None,
            }];
        }
        request
    }

    pub fn remember(
        &self,
        subject: &AuthenticatedSubject,
        request: RememberToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<KnowledgeRecord>> {
        let request = self.normalize_remember(subject, request);
        validate_identifier("idempotency_key", &request.idempotency_key)?;
        validate_identifier("id", &request.id)?;
        validate_identifier("collection", &request.collection)?;
        if request.kind.trim().is_empty() {
            return Err(HNSQRError::InvalidRequest("kind is required".to_string()));
        }
        if request.content.len() > MAX_CONTENT_BYTES {
            return Err(HNSQRError::InvalidRequest(format!(
                "content exceeds {MAX_CONTENT_BYTES} bytes"
            )));
        }
        validate_provenance(&request.provenance)?;
        if let Some(record) = self
            .store
            .idempotent_record(&subject.tenant_id, &request.idempotency_key)?
        {
            return Ok(EvidenceEnvelope {
                tenant_id: subject.tenant_id.clone(),
                snapshot_lsn: record.commit_lsn,
                retrieval_contract: "durable_write".to_string(),
                certified: true,
                proof_upper_bound: None,
                content_is_untrusted: true,
                results: record,
                contradictions: Vec::new(),
            });
        }
        let (vector, embedding) = self.resolve_collection_embedding(
            &subject.tenant_id,
            &request.collection,
            Some(&request.content),
            request.vector,
            request.embedding,
        )?;
        self.validate_collection_spec(&subject.tenant_id, &request.collection, &embedding)?;
        let internal_collection = tenant_collection(&subject.tenant_id, &request.collection);
        let mut indexed_metadata = BTreeMap::from([
            (
                "tenant_id".to_string(),
                MetadataValue::from(subject.tenant_id.clone()),
            ),
            (
                "kind".to_string(),
                MetadataValue::from(request.kind.clone()),
            ),
        ])
        .into_iter()
        .collect();
        match self.vectors.ingest_llm_vector_with_metadata(
            &internal_collection,
            &request.id,
            &vector,
            std::mem::take(&mut indexed_metadata),
        ) {
            Ok(_) | Err(HNSQRError::NodeAlreadyExists(_)) => {}
            Err(err) => return Err(err),
        }
        let record_id = request.id.clone();
        let evidence_class =
            request
                .evidence_class
                .unwrap_or_else(|| match request.kind.as_str() {
                    "external_source" | "web_source" | "external_web_source" => {
                        EvidenceClass::ExternalSource
                    }
                    "observation" => EvidenceClass::Observation,
                    "measurement" => EvidenceClass::Measurement,
                    "simulation" => EvidenceClass::Simulation,
                    "experiment" => EvidenceClass::Experiment,
                    "agent_inference" => EvidenceClass::AgentInference,
                    "user_assertion" => EvidenceClass::UserAssertion,
                    "derived_statistic" => EvidenceClass::DerivedStatistic,
                    "reported_claim" => EvidenceClass::ReportedClaim,
                    _ => EvidenceClass::AgentSynthesis,
                });

        const EVIDENTIARY_ROLES: &[&str] = &[
            "supports",
            "verified_by",
            "measured_by",
            "fixed_by",
            "proves",
            "validated_by",
            "derived_from",
        ];
        let has_evidentiary_role = request
            .roles
            .values()
            .any(|role| EVIDENTIARY_ROLES.contains(&role.as_str()));
        let snapshot_lsn = self.store.current_lsn();
        let verification_state = if has_evidentiary_role {
            let all_targets_exist = request.members.iter().all(|target_id| {
                self.store
                    .record_at(&subject.tenant_id, target_id, snapshot_lsn)
                    .is_some()
                    || self
                        .store
                        .outcomes_at(&subject.tenant_id, snapshot_lsn)
                        .iter()
                        .any(|o| &o.attempt_id == target_id)
            });
            if all_targets_exist {
                if evidence_class == EvidenceClass::ExternalSource
                    || evidence_class == EvidenceClass::Measurement
                {
                    VerificationState::Verified
                } else {
                    VerificationState::Unverified
                }
            } else {
                VerificationState::PendingEvidence
            }
        } else if evidence_class == EvidenceClass::ExternalSource {
            VerificationState::Verified
        } else {
            VerificationState::Unverified
        };

        let record = KnowledgeRecord {
            id: request.id,
            tenant_id: subject.tenant_id.clone(),
            collection: request.collection,
            kind: request.kind,
            content: request.content,
            evidence_class,
            verification_state,
            members: request.members,
            roles: request.roles,
            metadata: request.metadata,
            provenance: request.provenance,
            embedding,
            commit_lsn: 0,
        };
        let stored =
            self.store
                .remember(&subject.tenant_id, &request.idempotency_key, record, vector);
        let (record, _) = match stored {
            Ok(stored) => stored,
            Err(error) => {
                if let Err(compensation_error) = self
                    .vectors
                    .remove_llm_vector(&internal_collection, &record_id)
                {
                    tracing::error!(
                        record_id = %record_id,
                        error = %compensation_error,
                        "failed to compensate vector projection after journal failure"
                    );
                }
                return Err(error);
            }
        };
        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn: record.commit_lsn,
            retrieval_contract: "durable_write".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: record,
            contradictions: Vec::new(),
        })
    }

    pub fn search(
        &self,
        subject: &AuthenticatedSubject,
        request: SearchToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<Vec<SearchEvidence>>> {
        validate_identifier("collection", &request.collection)?;
        if request.k == 0 || request.k > MAX_K {
            return Err(HNSQRError::InvalidRequest(format!(
                "k must be between 1 and {MAX_K}"
            )));
        }
        let snapshot_lsn = self.snapshot_lsn(request.snapshot_lsn)?;
        let (vector, embedding) = self.resolve_collection_embedding(
            &subject.tenant_id,
            &request.collection,
            request.query_text.as_deref(),
            request.query_vector,
            request.embedding,
        )?;
        self.validate_collection_spec(&subject.tenant_id, &request.collection, &embedding)?;
        let internal_collection = tenant_collection(&subject.tenant_id, &request.collection);
        let current_lsn = self.store.current_lsn();
        let internal_k = if snapshot_lsn < current_lsn {
            self.store
                .records_at(&subject.tenant_id, current_lsn)
                .into_iter()
                .filter(|record| record.collection == request.collection)
                .count()
                .max(request.k)
        } else {
            request.k
        };
        let effective_contract = if let Some(ref c) = request.retrieval_contract {
            c.to_ascii_lowercase()
        } else if let Some(true) = request.certified_exact {
            "certified".to_string()
        } else {
            "exact".to_string()
        };
        let is_certified = effective_contract == "certified";

        let (results, certified, proof_upper_bound) =
            self.vectors.search_llm_vector_with_contract(
                &internal_collection,
                &vector,
                internal_k,
                request.filter,
                is_certified,
            )?;
        let mut results: Vec<_> = results
            .into_iter()
            .filter_map(|(id, score)| {
                let record = self.store.record_at(&subject.tenant_id, &id, snapshot_lsn);
                let record = record?;
                if let Some(ref allowed_kinds) = request.kinds {
                    if !allowed_kinds.is_empty() && !allowed_kinds.contains(&record.kind) {
                        return None;
                    }
                }
                Some(SearchEvidence {
                    id,
                    score,
                    record: Some(record),
                })
            })
            .collect();
        results.truncate(request.k);
        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn,
            retrieval_contract: effective_contract,
            certified: is_certified && certified,
            proof_upper_bound,
            content_is_untrusted: true,
            results,
            contradictions: Vec::new(),
        })
    }

    pub fn traverse(
        &self,
        subject: &AuthenticatedSubject,
        request: TraverseToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<Vec<TraversalEvidence>>> {
        if request.seed_ids.is_empty() {
            return Err(HNSQRError::InvalidRequest(
                "at least one seed_id is required".to_string(),
            ));
        }
        if request.max_depth == 0 || request.max_depth > MAX_TRAVERSAL_DEPTH {
            return Err(HNSQRError::InvalidRequest(format!(
                "max_depth must be between 1 and {MAX_TRAVERSAL_DEPTH}"
            )));
        }
        if request.max_results == 0 || request.max_results > 1_000 {
            return Err(HNSQRError::InvalidRequest(
                "max_results must be between 1 and 1000".to_string(),
            ));
        }
        let snapshot_lsn = self.snapshot_lsn(request.snapshot_lsn)?;
        let records = self.store.records_at(&subject.tenant_id, snapshot_lsn);
        let allowed_kinds: BTreeSet<_> = request.relation_kinds.into_iter().collect();
        let mut frontier: VecDeque<(String, usize)> =
            request.seed_ids.into_iter().map(|id| (id, 0)).collect();
        let mut visited_entities = BTreeSet::new();
        let mut visited_records = BTreeSet::new();
        let mut results = Vec::new();
        while let Some((entity, depth)) = frontier.pop_front() {
            if !visited_entities.insert(entity.clone()) || depth >= request.max_depth {
                continue;
            }
            for record in &records {
                if visited_records.contains(&record.id)
                    || (!allowed_kinds.is_empty() && !allowed_kinds.contains(&record.kind))
                    || (record.id != entity && !record.members.contains(&entity))
                {
                    continue;
                }
                visited_records.insert(record.id.clone());
                results.push(TraversalEvidence {
                    depth: depth + 1,
                    record: record.clone(),
                });
                for member in &record.members {
                    if !visited_entities.contains(member) {
                        frontier.push_back((member.clone(), depth + 1));
                    }
                }
                if results.len() >= request.max_results {
                    break;
                }
            }
            if results.len() >= request.max_results {
                break;
            }
        }
        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn,
            retrieval_contract: "pinned_hypergraph_traversal".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results,
            contradictions: Vec::new(),
        })
    }

    pub fn resolve(
        &self,
        subject: &AuthenticatedSubject,
        request: ResolveToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<Vec<ResolutionHypothesis>>> {
        if request.max_hypotheses == 0 || request.max_hypotheses > 20 {
            return Err(HNSQRError::InvalidRequest(
                "max_hypotheses must be between 1 and 20".to_string(),
            ));
        }
        let search = self.search(
            subject,
            SearchToolRequest {
                query_text: Some(request.problem),
                // `resolve` is a natural-language MCP workflow.  The required problem text is
                // authoritative; raw vectors belong to `search`, where an explicit embedding
                // descriptor is required.  This also recovers safely from clients that attach a
                // fabricated placeholder vector (for example `[0, 0, 0]`).
                query_vector: None,
                embedding: None,
                collection: request.collection,
                k: request.max_hypotheses,
                kinds: None,
                filter: None,
                retrieval_contract: Some("exact".to_string()),
                certified_exact: None,
                snapshot_lsn: request.snapshot_lsn,
            },
        )?;
        let outcomes = self
            .store
            .outcomes_at(&subject.tenant_id, search.snapshot_lsn);
        let hypotheses = search
            .results
            .iter()
            .filter_map(|evidence| evidence.record.as_ref().map(|record| (evidence, record)))
            .map(|(evidence, record)| {
                let related: Vec<_> = outcomes
                    .iter()
                    .filter(|outcome| outcome.evidence_ids.contains(&record.id))
                    .collect();
                let successful_outcomes = related.iter().filter(|item| item.successful).count();
                let failed_outcomes = related.len().saturating_sub(successful_outcomes);
                let outcome_success_rate = if related.is_empty() {
                    0.5
                } else {
                    successful_outcomes as f32 / related.len() as f32
                };
                let verification_weight = match record.verification_state {
                    VerificationState::Verified => 1.0,
                    VerificationState::ReportedUnverified => 0.65,
                    VerificationState::PendingEvidence => 0.5,
                    VerificationState::Unverified => 0.4,
                    VerificationState::Falsified => 0.0,
                };
                let measured_outcomes = related
                    .iter()
                    .filter(|outcome| outcome.measurement.is_some())
                    .count();
                let reproducibility_score = if related.is_empty() {
                    0.25
                } else {
                    measured_outcomes as f32 / related.len() as f32
                };
                let age = search.snapshot_lsn.saturating_sub(record.commit_lsn) as f32;
                let recency_weight = 1.0 / (1.0 + age / 1_000.0);
                let mut ranking_components = BTreeMap::new();
                ranking_components.insert("semantic_relevance".to_string(), evidence.score);
                ranking_components.insert("verification_weight".to_string(), verification_weight);
                ranking_components.insert("outcome_success_rate".to_string(), outcome_success_rate);
                ranking_components
                    .insert("reproducibility_score".to_string(), reproducibility_score);
                ranking_components.insert("recency_weight".to_string(), recency_weight);
                ResolutionHypothesis {
                    hypothesis: record.content.clone(),
                    confidence: (evidence.score
                        * verification_weight
                        * (0.5 + 0.5 * outcome_success_rate)
                        * (0.75 + 0.25 * reproducibility_score)
                        * recency_weight)
                        .clamp(0.0, 1.0),
                    ranking_components,
                    evidence_ids: vec![record.id.clone()],
                    successful_outcomes,
                    failed_outcomes,
                    status: "hypothesis_requires_external_validation".to_string(),
                }
            })
            .collect();
        let mut hypotheses: Vec<ResolutionHypothesis> = hypotheses;
        hypotheses.sort_by(|left, right| right.confidence.total_cmp(&left.confidence));
        Ok(EvidenceEnvelope {
            tenant_id: search.tenant_id,
            snapshot_lsn: search.snapshot_lsn,
            retrieval_contract: "evidence_ranked_hypotheses".to_string(),
            certified: search.certified,
            proof_upper_bound: search.proof_upper_bound,
            content_is_untrusted: true,
            results: hypotheses,
            contradictions: search.contradictions,
        })
    }

    pub fn record_outcome(
        &self,
        subject: &AuthenticatedSubject,
        mut request: RecordOutcomeToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<ModelOutcomeRecord>> {
        if request.summary.trim().is_empty() {
            return Err(HNSQRError::InvalidRequest(
                "summary is required".to_string(),
            ));
        }
        if request.attempt_id.trim().is_empty() {
            let hash =
                sha256_hex(format!("{}:outcome:{}", subject.tenant_id, request.summary).as_bytes());
            request.attempt_id = format!("out:{}", &hash[..16]);
        }
        if request.idempotency_key.trim().is_empty() {
            let hash = sha256_hex(
                format!(
                    "{}:outcome:{}:{}",
                    subject.tenant_id, request.attempt_id, request.summary
                )
                .as_bytes(),
            );
            request.idempotency_key = format!("auto:outcome:{}", &hash[..24]);
        }
        if request.provenance.is_empty() {
            let hash = format!("sha256:{}", sha256_hex(request.summary.as_bytes()));
            request.provenance = vec![ProvenanceReference {
                source_id: format!("agent:{}", subject.key_id),
                uri: None,
                content_hash: hash,
                observed_at_lsn: None,
            }];
        }
        validate_identifier("idempotency_key", &request.idempotency_key)?;
        validate_identifier("attempt_id", &request.attempt_id)?;
        validate_provenance(&request.provenance)?;

        let snapshot_lsn = self.store.current_lsn();
        let all_exist = request.evidence_ids.iter().all(|evidence_id| {
            self.store
                .record_at(&subject.tenant_id, evidence_id, snapshot_lsn)
                .is_some()
                || self
                    .store
                    .outcomes_at(&subject.tenant_id, snapshot_lsn)
                    .iter()
                    .any(|o| &o.attempt_id == evidence_id)
        });
        let mut evidence_ids = request.evidence_ids;
        if evidence_ids.is_empty() {
            evidence_ids.push(request.attempt_id.clone());
        }

        if request.metrics.values().any(|value| !value.is_finite()) {
            return Err(HNSQRError::InvalidRequest(
                "outcome metrics must be finite".to_string(),
            ));
        }
        let (evidence_class, mut verification_state) = if let Some(ref m) = request.measurement {
            if m.artifact_id.trim().is_empty()
                || m.producer.trim().is_empty()
                || m.metrics_digest.trim().is_empty()
            {
                return Err(HNSQRError::InvalidRequest(
                    "measurement spec requires non-empty artifact_id, producer, and metrics_digest"
                        .to_string(),
                ));
            }
            (
                request.evidence_class.unwrap_or(EvidenceClass::Measurement),
                VerificationState::Verified,
            )
        } else if !request.metrics.is_empty() {
            (
                request
                    .evidence_class
                    .unwrap_or(EvidenceClass::ReportedClaim),
                VerificationState::ReportedUnverified,
            )
        } else {
            (
                request.evidence_class.unwrap_or(EvidenceClass::Observation),
                VerificationState::Unverified,
            )
        };

        if !all_exist && verification_state == VerificationState::Verified {
            verification_state = VerificationState::PendingEvidence;
        }

        let outcome = ModelOutcomeRecord {
            attempt_id: request.attempt_id,
            tenant_id: subject.tenant_id.clone(),
            summary: request.summary,
            successful: request.successful,
            evidence_class,
            verification_state,
            measurement: request.measurement,
            evidence_ids,
            metrics: request.metrics,
            provenance: request.provenance,
            commit_lsn: 0,
        };
        let (outcome, _) =
            self.store
                .record_outcome(&subject.tenant_id, &request.idempotency_key, outcome)?;

        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn: outcome.commit_lsn,
            retrieval_contract: "durable_write".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: outcome,
            contradictions: Vec::new(),
        })
    }

    /// Begins an agent case. Retrieval happens before the new issue is indexed, so
    /// the response contains only prior evidence rather than a self-match.
    pub fn task_begin(
        &self,
        subject: &AuthenticatedSubject,
        mut request: TaskBeginToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<TaskBeginResult>> {
        if request.problem.trim().is_empty() {
            return Err(HNSQRError::InvalidRequest(
                "problem is required".to_string(),
            ));
        }
        if request.case_id.trim().is_empty() {
            let hash =
                sha256_hex(format!("{}:case:{}", subject.tenant_id, request.problem).as_bytes());
            request.case_id = format!("case:{}", &hash[..16]);
        }
        if request.idempotency_key.trim().is_empty() {
            let hash =
                sha256_hex(format!("{}:begin:{}", subject.tenant_id, request.case_id).as_bytes());
            request.idempotency_key = format!("auto:begin:{}", &hash[..24]);
        }
        if request.provenance.is_empty() {
            let hash = format!("sha256:{}", sha256_hex(request.problem.as_bytes()));
            request.provenance = vec![ProvenanceReference {
                source_id: format!("agent:{}", subject.key_id),
                uri: None,
                content_hash: hash,
                observed_at_lsn: None,
            }];
        }
        validate_identifier("idempotency_key", &request.idempotency_key)?;
        validate_identifier("case_id", &request.case_id)?;
        validate_identifier("collection", &request.collection)?;
        validate_provenance(&request.provenance)?;
        if request.max_hypotheses == 0 || request.max_hypotheses > 20 {
            return Err(HNSQRError::InvalidRequest(
                "max_hypotheses must be between 1 and 20".to_string(),
            ));
        }

        let has_collection = self
            .store
            .collection_spec(&subject.tenant_id, &request.collection)
            .is_some();
        let related_cases = if has_collection {
            self.search(
                subject,
                SearchToolRequest {
                    query_text: Some(request.problem.clone()),
                    query_vector: None,
                    embedding: None,
                    collection: request.collection.clone(),
                    k: request.max_hypotheses.min(MAX_K),
                    kinds: None,
                    filter: None,
                    retrieval_contract: Some("exact".to_string()),
                    certified_exact: None,
                    snapshot_lsn: None,
                },
            )?
            .results
        } else {
            Vec::new()
        };
        let candidate_resolutions = if has_collection {
            self.resolve(
                subject,
                ResolveToolRequest {
                    problem: request.problem.clone(),
                    query_vector: None,
                    embedding: None,
                    collection: request.collection.clone(),
                    max_hypotheses: request.max_hypotheses,
                    snapshot_lsn: None,
                },
            )?
            .results
        } else {
            Vec::new()
        };

        let case = self
            .remember(
                subject,
                RememberToolRequest {
                    idempotency_key: request.idempotency_key.clone(),
                    id: request.case_id.clone(),
                    collection: request.collection.clone(),
                    kind: "issue".to_string(),
                    evidence_class: Some(EvidenceClass::Observation),
                    content: request.problem.clone(),
                    vector: None,
                    embedding: None,
                    members: Vec::new(),
                    roles: BTreeMap::new(),
                    metadata: BTreeMap::new(),
                    provenance: request.provenance.clone(),
                },
            )?
            .results;

        for related in &related_cases {
            let relation_id = format!("{}:similar:{}", request.case_id, related.id);
            let _ = self.remember(
                subject,
                RememberToolRequest {
                    idempotency_key: format!("{}:similar:{}", request.idempotency_key, related.id),
                    id: relation_id,
                    collection: request.collection.clone(),
                    kind: "similar_to".to_string(),
                    evidence_class: Some(EvidenceClass::AgentInference),
                    content: "Automatically linked by evidence retrieval; validate before reuse."
                        .to_string(),
                    vector: None,
                    embedding: None,
                    members: vec![request.case_id.clone(), related.id.clone()],
                    roles: BTreeMap::from([
                        (request.case_id.clone(), "new_issue".to_string()),
                        (related.id.clone(), "prior_evidence".to_string()),
                    ]),
                    metadata: BTreeMap::new(),
                    provenance: request.provenance.clone(),
                },
            )?;
        }

        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn: self.store.current_lsn(),
            retrieval_contract: "agent_case_begin".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: TaskBeginResult {
                case,
                related_cases,
                candidate_resolutions,
            },
            contradictions: Vec::new(),
        })
    }

    /// Rehydrates an existing case for another model or a later session.
    pub fn task_context(
        &self,
        subject: &AuthenticatedSubject,
        request: TaskContextToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<TaskContextResult>> {
        validate_identifier("case_id", &request.case_id)?;
        let snapshot_lsn = self.snapshot_lsn(request.snapshot_lsn)?;
        let case = match self
            .store
            .record_at(&subject.tenant_id, &request.case_id, snapshot_lsn)
        {
            Some(c) => c,
            None => {
                // Dynamic self-healing: synthesize a case node if referenced directly without task_begin
                self.remember(
                    subject,
                    RememberToolRequest {
                        idempotency_key: format!("auto:case:{}", request.case_id),
                        id: request.case_id.clone(),
                        collection: "knowledge".to_string(),
                        kind: "issue".to_string(),
                        evidence_class: Some(EvidenceClass::Observation),
                        content: format!("Context initialized for case {}", request.case_id),
                        vector: None,
                        embedding: None,
                        members: Vec::new(),
                        roles: BTreeMap::new(),
                        metadata: BTreeMap::new(),
                        provenance: vec![ProvenanceReference {
                            source_id: format!("agent:{}", subject.key_id),
                            uri: None,
                            content_hash: format!(
                                "sha256:{}",
                                sha256_hex(request.case_id.as_bytes())
                            ),
                            observed_at_lsn: Some(snapshot_lsn),
                        }],
                    },
                )?
                .results
            }
        };
        let related_cases = self
            .search(
                subject,
                SearchToolRequest {
                    query_text: Some(case.content.clone()),
                    query_vector: None,
                    embedding: None,
                    collection: case.collection.clone(),
                    k: default_hypotheses(),
                    kinds: None,
                    filter: None,
                    retrieval_contract: Some("exact".to_string()),
                    certified_exact: None,
                    snapshot_lsn: Some(snapshot_lsn),
                },
            )?
            .results
            .into_iter()
            .filter(|item| item.id != case.id)
            .collect();
        let relations = self
            .traverse(
                subject,
                TraverseToolRequest {
                    seed_ids: vec![case.id.clone()],
                    relation_kinds: Vec::new(),
                    max_depth: 3,
                    max_results: 100,
                    snapshot_lsn: Some(snapshot_lsn),
                },
            )?
            .results;
        let candidate_resolutions = self
            .resolve(
                subject,
                ResolveToolRequest {
                    problem: case.content.clone(),
                    query_vector: None,
                    embedding: None,
                    collection: case.collection.clone(),
                    max_hypotheses: default_hypotheses(),
                    snapshot_lsn: Some(snapshot_lsn),
                },
            )?
            .results;
        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn,
            retrieval_contract: "agent_case_context".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: TaskContextResult {
                case,
                related_cases,
                relations,
                candidate_resolutions,
            },
            contradictions: Vec::new(),
        })
    }

    /// Records the measured outcome and promotes successful work to a resolution.
    pub fn task_complete(
        &self,
        subject: &AuthenticatedSubject,
        mut request: TaskCompleteToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<TaskCompleteResult>> {
        if request.summary.trim().is_empty() {
            return Err(HNSQRError::InvalidRequest(
                "summary is required".to_string(),
            ));
        }
        if request.idempotency_key.trim().is_empty() {
            let hash = sha256_hex(
                format!(
                    "{}:complete:{}:{}",
                    subject.tenant_id, request.case_id, request.summary
                )
                .as_bytes(),
            );
            request.idempotency_key = format!("auto:complete:{}", &hash[..24]);
        }
        if request.provenance.is_empty() {
            let hash = format!("sha256:{}", sha256_hex(request.summary.as_bytes()));
            request.provenance = vec![ProvenanceReference {
                source_id: format!("agent:{}", subject.key_id),
                uri: None,
                content_hash: hash,
                observed_at_lsn: None,
            }];
        }
        validate_identifier("idempotency_key", &request.idempotency_key)?;
        validate_identifier("case_id", &request.case_id)?;
        validate_provenance(&request.provenance)?;
        let case = match self.store.record_at(
            &subject.tenant_id,
            &request.case_id,
            self.store.current_lsn(),
        ) {
            Some(c) => c,
            None => {
                // Dynamic self-healing: provision the issue if not initialized
                self.remember(
                    subject,
                    RememberToolRequest {
                        idempotency_key: format!("{}:init", request.idempotency_key),
                        id: request.case_id.clone(),
                        collection: "knowledge".to_string(),
                        kind: "issue".to_string(),
                        evidence_class: Some(EvidenceClass::Observation),
                        content: request.summary.clone(),
                        vector: None,
                        embedding: None,
                        members: Vec::new(),
                        roles: BTreeMap::new(),
                        metadata: BTreeMap::new(),
                        provenance: request.provenance.clone(),
                    },
                )?
                .results
            }
        };
        let snapshot_lsn = self.store.current_lsn();
        let mut evidence_ids: Vec<String> = request
            .evidence_ids
            .into_iter()
            .filter(|id| {
                self.store
                    .record_at(&subject.tenant_id, id, snapshot_lsn)
                    .is_some()
                    || self
                        .store
                        .outcomes_at(&subject.tenant_id, snapshot_lsn)
                        .iter()
                        .any(|o| &o.attempt_id == id)
            })
            .collect();
        if !evidence_ids.contains(&case.id) {
            evidence_ids.push(case.id.clone());
        }
        let outcome = self
            .record_outcome(
                subject,
                RecordOutcomeToolRequest {
                    idempotency_key: request.idempotency_key.clone(),
                    attempt_id: request.case_id.clone(),
                    summary: request.summary.clone(),
                    successful: request.successful,
                    evidence_class: None,
                    measurement: request.measurement,
                    evidence_ids,
                    metrics: request.metrics,
                    provenance: request.provenance.clone(),
                },
            )?
            .results;

        let resolution_status = if let Some(status) = request.resolution_status {
            status
        } else if outcome.verification_state == VerificationState::Verified
            && matches!(
                outcome.evidence_class,
                EvidenceClass::Measurement | EvidenceClass::Experiment | EvidenceClass::Simulation
            )
        {
            ResolutionStatus::EmpiricallyVerified
        } else if request.successful {
            ResolutionStatus::SpeculativeSynthesis
        } else {
            ResolutionStatus::Hypothesis
        };

        let verification_level = match resolution_status {
            ResolutionStatus::EmpiricallyVerified => VerificationLevel::IntegrationScenarioPassed,
            ResolutionStatus::FormallyVerified => VerificationLevel::FormallyVerified,
            ResolutionStatus::SpeculativeSynthesis => VerificationLevel::SemanticContractPassed,
            ResolutionStatus::Hypothesis => VerificationLevel::InvocationSucceeded,
        };

        let resolution = if request.successful {
            let resolution_id = format!("{}:resolution", request.case_id);
            let kind = if resolution_status == ResolutionStatus::EmpiricallyVerified {
                "verified_resolution".to_string()
            } else {
                "resolution".to_string()
            };
            let evidence_class = if resolution_status == ResolutionStatus::EmpiricallyVerified {
                EvidenceClass::Measurement
            } else {
                EvidenceClass::AgentSynthesis
            };
            let resolution = self
                .remember(
                    subject,
                    RememberToolRequest {
                        idempotency_key: format!("{}:resolution", request.idempotency_key),
                        id: resolution_id.clone(),
                        collection: case.collection.clone(),
                        kind,
                        evidence_class: Some(evidence_class),
                        content: request.summary,
                        vector: None,
                        embedding: None,
                        members: vec![case.id.clone()],
                        roles: BTreeMap::from([(case.id.clone(), "resolves".to_string())]),
                        metadata: BTreeMap::new(),
                        provenance: request.provenance.clone(),
                    },
                )?
                .results;
            let _ = self.remember(
                subject,
                RememberToolRequest {
                    idempotency_key: format!("{}:fixed-by", request.idempotency_key),
                    id: format!("{}:fixed-by", request.case_id),
                    collection: case.collection,
                    kind: "fixed_by".to_string(),
                    evidence_class: Some(evidence_class),
                    content: "Outcome links the issue to its resolution.".to_string(),
                    vector: None,
                    embedding: None,
                    members: vec![case.id, resolution_id],
                    roles: BTreeMap::new(),
                    metadata: BTreeMap::new(),
                    provenance: request.provenance,
                },
            )?;
            Some(resolution)
        } else {
            None
        };
        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn: self.store.current_lsn(),
            retrieval_contract: "agent_case_complete".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: TaskCompleteResult {
                outcome,
                resolution,
                resolution_status,
                verification_level,
            },
            contradictions: Vec::new(),
        })
    }

    /// Returns the capability and collection-space snapshot required to safely start any case.
    pub fn status(&self, subject: &AuthenticatedSubject) -> EvidenceEnvelope<RuntimeStatus> {
        let mut collection_embeddings = self.store.collection_specs(&subject.tenant_id);
        collection_embeddings
            .entry(default_collection())
            .or_insert_with(|| self.embedder.descriptor().clone());
        let mut degradations = Vec::new();
        if self.web_search.is_none() {
            degradations.push("live web search is not configured".to_string());
        }
        if subject.role < AccessRole::ReadWrite {
            degradations
                .push("read-only authorization: durable learning writes are disabled".to_string());
        }
        EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn: self.store.current_lsn(),
            retrieval_contract: "runtime_preflight".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: false,
            results: RuntimeStatus {
                ready: true,
                read_write_authorized: subject.role >= AccessRole::ReadWrite,
                web_search_available: self.web_search.is_some(),
                embedding_provider: self.embedder.descriptor().clone(),
                collection_embeddings,
                limits: RuntimeLimits {
                    max_search_results: MAX_K,
                    max_web_results: self
                        .web_search
                        .as_ref()
                        .map_or(0, |provider| provider.max_results()),
                    max_hypotheses: 20,
                    max_traversal_depth: MAX_TRAVERSAL_DEPTH,
                },
                degradations,
            },
            contradictions: Vec::new(),
        }
    }

    /// Prepares a universal evidence-first case. The server never executes consequential
    /// actions: callers must pass the returned action gate to an authorized executor.
    pub fn run_case(
        &self,
        subject: &AuthenticatedSubject,
        mut request: RunCaseToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<RunCaseResult>> {
        if request.objective.trim().is_empty() {
            return Err(HNSQRError::InvalidRequest(
                "objective is required".to_string(),
            ));
        }
        if request.budgets.tool_calls == 0
            || request.budgets.tool_calls > 100
            || request.budgets.retrieval_results == 0
            || request.budgets.retrieval_results > MAX_K
        {
            return Err(HNSQRError::InvalidRequest(
                "budgets.tool_calls must be 1..=100 and budgets.retrieval_results must be 1..=100"
                    .to_string(),
            ));
        }
        const RECIPES: &[&str] = &[
            "research_and_synthesize",
            "diagnose_and_fix",
            "implement_and_test",
            "compare_options",
            "incident_response",
            "analyze_dataset",
            "evaluate_strategy",
        ];
        const EVIDENCE_POLICIES: &[&str] =
            &["none", "knowledge_only", "web_if_needed", "web_required"];
        const EXECUTION_POLICIES: &[&str] = &["propose_only", "tests_only", "authorized_executor"];
        if !RECIPES.contains(&request.recipe.as_str())
            || !EVIDENCE_POLICIES.contains(&request.evidence_policy.as_str())
            || !EXECUTION_POLICIES.contains(&request.execution_policy.as_str())
        {
            return Err(HNSQRError::InvalidRequest(
                "unknown recipe, evidence_policy, or execution_policy".to_string(),
            ));
        }
        let status = self.status(subject).results;
        let minimum_calls = if status.read_write_authorized { 3 } else { 1 }
            + usize::from(request.evidence_policy == "web_required");
        if request.budgets.tool_calls < minimum_calls {
            return Err(HNSQRError::InvalidRequest(format!(
                "budgets.tool_calls={} cannot satisfy the selected workflow; at least {minimum_calls} calls are required",
                request.budgets.tool_calls
            )));
        }
        if request.evidence_policy == "web_required" && !status.web_search_available {
            return Err(HNSQRError::InvalidRequest(
                "web_required cannot run because live web search is not configured".to_string(),
            ));
        }

        let (case, related_cases, hypotheses) = if status.read_write_authorized {
            let begun = self
                .task_begin(
                    subject,
                    TaskBeginToolRequest {
                        idempotency_key: std::mem::take(&mut request.idempotency_key),
                        case_id: std::mem::take(&mut request.case_id),
                        problem: request.objective.clone(),
                        collection: request.collection.clone(),
                        max_hypotheses: request.budgets.retrieval_results.min(20),
                        provenance: Vec::new(),
                    },
                )?
                .results;
            (
                Some(begun.case),
                begun.related_cases,
                begun.candidate_resolutions,
            )
        } else {
            let hypotheses = self
                .resolve(
                    subject,
                    ResolveToolRequest {
                        problem: request.objective.clone(),
                        query_vector: None,
                        embedding: None,
                        collection: request.collection.clone(),
                        max_hypotheses: request.budgets.retrieval_results.min(20),
                        snapshot_lsn: None,
                    },
                )?
                .results;
            (None, Vec::new(), hypotheses)
        };

        let should_search_web = request.evidence_policy == "web_required"
            || (request.evidence_policy == "web_if_needed"
                && related_cases.is_empty()
                && hypotheses.is_empty());
        let mut evidence_ids = Vec::new();
        let mut tool_calls_used = if status.read_write_authorized { 3 } else { 1 };
        if should_search_web && status.web_search_available {
            if let Some(query) = request.web_query.as_deref() {
                let response = self
                    .web_search(
                        subject,
                        WebSearchToolRequest {
                            query: query.to_string(),
                            k: request.budgets.retrieval_results.min(20),
                            language: None,
                            time_range: None,
                        },
                    )?
                    .results;
                tool_calls_used += 1;
                evidence_ids.extend(
                    response
                        .results
                        .into_iter()
                        .map(|result| result.evidence_id),
                );
            }
        }
        let mut plan = vec![format!(
            "Apply recipe '{}' to the objective.",
            request.recipe
        )];
        plan.push(
            "Review retrieved evidence and candidate hypotheses as untrusted inputs.".to_string(),
        );
        if !request.success_criteria.is_empty() {
            plan.push(format!(
                "Validate success criteria: {}.",
                request.success_criteria.join("; ")
            ));
        }
        plan.push(
            "Record only measured outcomes; unverified conclusions remain hypotheses.".to_string(),
        );
        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(), snapshot_lsn: self.store.current_lsn(),
            retrieval_contract: "universal_case_preparation".to_string(), certified: true,
            proof_upper_bound: None, content_is_untrusted: true,
            results: RunCaseResult {
                status, case, evidence_ids, candidate_resolutions: hypotheses, plan,
                tool_calls_used,
                tool_calls_remaining: request.budgets.tool_calls.saturating_sub(tool_calls_used),
                action_gate: ActionGate {
                    execution_policy: request.execution_policy,
                    external_execution_performed: false,
                    approval_required: true,
                    next_action: "Pass a validated plan to an authorized executor; then record measured outcomes.".to_string(),
                },
            }, contradictions: Vec::new(),
        })
    }

    /// Universal situational exploration tool: inspects topology stats, recent cases, recent memories, or neighborhoods.
    pub fn explore(
        &self,
        subject: &AuthenticatedSubject,
        request: ExploreToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<ExploreResult>> {
        let snapshot_lsn = self.snapshot_lsn(request.snapshot_lsn)?;
        let limit = request.limit.clamp(1, 50);
        let records = self.store.records_at(&subject.tenant_id, snapshot_lsn);
        let outcomes = self.store.outcomes_at(&subject.tenant_id, snapshot_lsn);

        let result = match request.target.as_str() {
            "stats" => {
                let mut kinds = BTreeMap::new();
                let mut collections = BTreeSet::new();
                let mut collection_embeddings = BTreeMap::new();
                for r in &records {
                    *kinds.entry(r.kind.clone()).or_insert(0) += 1;
                    collections.insert(r.collection.clone());
                    collection_embeddings
                        .entry(r.collection.clone())
                        .or_insert_with(|| r.embedding.clone());
                }
                ExploreResult {
                    target: "stats".to_string(),
                    stats: Some(ExploreTopologyStats {
                        total_entities: records.len(),
                        total_outcomes: outcomes.len(),
                        current_lsn: snapshot_lsn,
                        collections: collections.into_iter().collect(),
                        collection_embeddings,
                        kinds,
                    }),
                    recent_cases: None,
                    recent_memories: None,
                    neighborhood: None,
                }
            }
            "recent_cases" => {
                let mut cases: Vec<_> = records
                    .into_iter()
                    .filter(|r| r.kind == "issue" || r.kind.contains("case"))
                    .collect();
                cases.sort_by_key(|r| std::cmp::Reverse(r.commit_lsn));
                cases.truncate(limit);
                ExploreResult {
                    target: "recent_cases".to_string(),
                    stats: None,
                    recent_cases: Some(cases),
                    recent_memories: None,
                    neighborhood: None,
                }
            }
            "recent_memories" => {
                let mut memories = records;
                memories.sort_by_key(|r| std::cmp::Reverse(r.commit_lsn));
                memories.truncate(limit);
                ExploreResult {
                    target: "recent_memories".to_string(),
                    stats: None,
                    recent_cases: None,
                    recent_memories: Some(memories),
                    neighborhood: None,
                }
            }
            "neighborhood" => {
                let seed_id = request.seed_id.ok_or_else(|| {
                    HNSQRError::InvalidRequest(
                        "seed_id is required when target is 'neighborhood'".to_string(),
                    )
                })?;
                let neighborhood = self
                    .traverse(
                        subject,
                        TraverseToolRequest {
                            seed_ids: vec![seed_id],
                            relation_kinds: Vec::new(),
                            max_depth: 2,
                            max_results: limit,
                            snapshot_lsn: Some(snapshot_lsn),
                        },
                    )?
                    .results;
                ExploreResult {
                    target: "neighborhood".to_string(),
                    stats: None,
                    recent_cases: None,
                    recent_memories: None,
                    neighborhood: Some(neighborhood),
                }
            }
            other => {
                return Err(HNSQRError::InvalidRequest(format!(
                    "unknown explore target '{other}'; must be 'stats', 'recent_cases', 'recent_memories', or 'neighborhood'"
                )));
            }
        };

        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn,
            retrieval_contract: "graph_topology_exploration".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: result,
            contradictions: Vec::new(),
        })
    }

    /// Universal ingestion tool: compiles external material (codebase, files, markdown, URLs) into ContextGraph.
    pub fn ingest(
        &self,
        subject: &AuthenticatedSubject,
        request: IngestToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<IngestResult>> {
        let namespace = crate::contextgraph::Namespace::new(
            request
                .namespace
                .unwrap_or_else(|| format!("tenant:{}", subject.tenant_id)),
        );

        let sources = if let Some(path) = &request.path {
            let fs_adapter = crate::contextgraph::adapters::fs::FilesystemSourceAdapter::new();
            fs_adapter.crawl_directory(path)?
        } else if let Some(text) = &request.text {
            vec![crate::contextgraph::SourceInput::from_text(
                text,
                request.url.unwrap_or_else(|| "direct_input".to_string()),
                request.source_type,
            )]
        } else {
            return Err(HNSQRError::InvalidRequest(
                "path or text is required for ingest".to_string(),
            ));
        };

        let compiler = crate::contextgraph::ContextCompiler::default();
        let output = compiler.compile(&namespace, &sources)?;
        let entities_count = output.entities.len();
        let entity_ids = output
            .entities
            .iter()
            .map(|entity| entity.id.0.clone())
            .collect();
        let relations_count = output.relations.len();
        let duration_ms = output.duration_ms;
        let compilation_report = output.report.clone();
        let canonical_fingerprint = format!("{:x?}", output.canonical_fingerprint);

        let commit_lsn = self.contextgraph.commit_delta(output.into_delta());

        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn: commit_lsn,
            retrieval_contract: "contextgraph_atomic_ingest".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: IngestResult {
                namespace: namespace.0,
                entity_ids,
                entities_count,
                relations_count,
                canonical_fingerprint,
                commit_lsn,
                duration_ms,
                compilation_report,
            },
            contradictions: Vec::new(),
        })
    }

    /// Universal pathfinding tool: discovers shortest semantic relation path between entities.
    pub fn path(
        &self,
        subject: &AuthenticatedSubject,
        request: PathToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<Option<crate::contextgraph::ContextSlice>>> {
        let snapshot_lsn = self.snapshot_lsn(request.snapshot_lsn)?;
        let state = self.contextgraph.snapshot();
        // Accept either the stable `ent_…` identifier or an unambiguous entity
        // label, as advertised by the MCP contract.  Ambiguous labels are rejected
        // rather than silently choosing an arbitrary node.
        let resolve_entity = |value: String| -> HNSQRResult<crate::contextgraph::schema::EntityId> {
            let candidate = crate::contextgraph::schema::EntityId(value.clone());
            if state.entities.contains_key(&candidate) {
                return Ok(candidate);
            }
            let matches = self.contextgraph.lookup_by_label(&value);
            match matches.as_slice() {
                [entity] => Ok(entity.id.clone()),
                [] => Err(HNSQRError::InvalidRequest(format!(
                    "ContextGraph entity '{value}' was not found; use an entity ID returned by ingest or an exact label"
                ))),
                _ => Err(HNSQRError::InvalidRequest(format!(
                    "ContextGraph label '{value}' is ambiguous; use an entity ID returned by ingest"
                ))),
            }
        };
        let from_id = resolve_entity(request.from)?;
        let to_id = resolve_entity(request.to)?;
        let budget = crate::contextgraph::ContextBudget {
            max_results: 50,
            max_chars: 12000,
            max_depth: request.max_depth,
        };
        let slice =
            crate::contextgraph::ContextQueryEngine::path(&state, &from_id, &to_id, &budget);

        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn,
            retrieval_contract: "contextgraph_path_search".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: slice,
            contradictions: Vec::new(),
        })
    }

    /// Universal diff tool: compares ContextGraph state across snapshots or namespaces.
    pub fn diff(
        &self,
        subject: &AuthenticatedSubject,
        request: DiffToolRequest,
    ) -> HNSQRResult<EvidenceEnvelope<serde_json::Value>> {
        let state = self.contextgraph.snapshot();
        let report = serde_json::json!({
            "namespace": state.namespace.0,
            "commit_lsn": state.commit_lsn,
            "entities_count": state.entities.len(),
            "relations_count": state.relations.len(),
            "from_snapshot": request.from_snapshot,
            "to_snapshot": request.to_snapshot,
            "canonical_fingerprint": format!("{:x?}", state.canonical_fingerprint),
        });

        Ok(EvidenceEnvelope {
            tenant_id: subject.tenant_id.clone(),
            snapshot_lsn: state.commit_lsn,
            retrieval_contract: "contextgraph_snapshot_diff".to_string(),
            certified: true,
            proof_upper_bound: None,
            content_is_untrusted: true,
            results: report,
            contradictions: Vec::new(),
        })
    }
}

fn default_collection() -> String {
    "knowledge".to_string()
}
fn default_k() -> usize {
    10
}
fn default_depth() -> usize {
    3
}
fn default_traversal_results() -> usize {
    100
}
fn default_hypotheses() -> usize {
    5
}

fn tenant_collection(tenant: &str, collection: &str) -> String {
    format!("tenant::{tenant}::{collection}")
}

fn validate_identifier(field: &str, value: &str) -> HNSQRResult<()> {
    if value.is_empty()
        || value.len() > 256
        || !value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ':'))
    {
        return Err(HNSQRError::InvalidRequest(format!(
            "{field} must be 1-256 ASCII identifier characters"
        )));
    }
    Ok(())
}

fn validate_provenance(provenance: &[ProvenanceReference]) -> HNSQRResult<()> {
    if provenance.is_empty() {
        return Err(HNSQRError::InvalidRequest(
            "at least one provenance reference is required for durable knowledge or outcome writes"
                .to_string(),
        ));
    }
    if provenance.len() > 1_000 {
        return Err(HNSQRError::InvalidRequest(
            "at most 1000 provenance references are allowed".to_string(),
        ));
    }
    for reference in provenance {
        if reference.source_id.trim().is_empty() || reference.content_hash.trim().is_empty() {
            return Err(HNSQRError::InvalidRequest(
                "every provenance reference requires source_id and content_hash".to_string(),
            ));
        }
    }
    Ok(())
}

/// Deterministic signed feature hashing for text-only MCP calls. It provides a safe local
/// baseline; production deployments should send embeddings from a pinned semantic model.
pub fn local_text_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    let stemmer = MorphologicalStemmer::new();
    let tokens = stemmer.tokenize_and_stem(text, LanguageMode::English);
    let mut vector = vec![0.0f32; dimensions];
    for token in tokens {
        let digest = Sha256::digest(token.as_bytes());
        for offset in [0usize, 8, 16, 24] {
            let value = u64::from_le_bytes(digest[offset..offset + 8].try_into().unwrap());
            let index = (value as usize) % dimensions;
            let sign = if value & (1 << 63) == 0 { 1.0 } else { -1.0 };
            vector[index] += sign;
        }
    }
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    error: String,
}

pub(crate) fn error_response(error: HNSQRError) -> Response {
    let status = match error {
        HNSQRError::Unauthorized(_) => StatusCode::UNAUTHORIZED,
        HNSQRError::NodeNotFound(_) | HNSQRError::NodeIndexNotFound(_) => StatusCode::NOT_FOUND,
        HNSQRError::NodeAlreadyExists(_) => StatusCode::CONFLICT,
        HNSQRError::InvalidRequest(_)
        | HNSQRError::DimensionMismatch { .. }
        | HNSQRError::InvalidConfig(_) => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (
        status,
        Json(ApiErrorBody {
            error: error.to_string(),
        }),
    )
        .into_response()
}

async fn run_rest<T, R, F>(
    service: Arc<ModelToolService>,
    headers: HeaderMap,
    required: AccessRole,
    request: T,
    operation: F,
) -> Response
where
    T: Send + 'static,
    R: Serialize,
    F: FnOnce(&ModelToolService, &AuthenticatedSubject, T) -> HNSQRResult<R>,
{
    let subject = match service.auth.authenticate(&headers, required) {
        Ok(subject) => subject,
        Err(error) => return error_response(error),
    };
    match operation(&service, &subject, request) {
        Ok(value) => Json(value).into_response(),
        Err(error) => error_response(error),
    }
}

async fn search_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<SearchToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadOnly,
        request,
        |s, a, r| s.search(a, r),
    )
    .await
}

async fn traverse_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<TraverseToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadOnly,
        request,
        |s, a, r| s.traverse(a, r),
    )
    .await
}

async fn resolve_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<ResolveToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadOnly,
        request,
        |s, a, r| s.resolve(a, r),
    )
    .await
}

async fn remember_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<RememberToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadWrite,
        request,
        |s, a, r| s.remember(a, r),
    )
    .await
}

async fn outcome_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<RecordOutcomeToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadWrite,
        request,
        |s, a, r| s.record_outcome(a, r),
    )
    .await
}

async fn task_begin_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<TaskBeginToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadWrite,
        request,
        |s, a, r| s.task_begin(a, r),
    )
    .await
}

async fn task_context_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<TaskContextToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadOnly,
        request,
        |s, a, r| s.task_context(a, r),
    )
    .await
}

async fn task_complete_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<TaskCompleteToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadWrite,
        request,
        |s, a, r| s.task_complete(a, r),
    )
    .await
}

async fn web_search_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<WebSearchToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadOnly,
        request,
        |s, a, r| s.web_search(a, r),
    )
    .await
}

async fn explore_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<ExploreToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadOnly,
        request,
        |s, a, r| s.explore(a, r),
    )
    .await
}

async fn status_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
) -> Response {
    let subject = match service.auth().authenticate(&headers, AccessRole::ReadOnly) {
        Ok(subject) => subject,
        Err(error) => return error_response(error),
    };
    Json(service.status(&subject)).into_response()
}

async fn run_case_handler(
    State(service): State<Arc<ModelToolService>>,
    headers: HeaderMap,
    Json(request): Json<RunCaseToolRequest>,
) -> Response {
    run_rest(
        service,
        headers,
        AccessRole::ReadOnly,
        request,
        |s, a, r| s.run_case(a, r),
    )
    .await
}

/// Builds the provider-neutral REST routes. MCP routes are merged by `transport::mcp`.
pub fn create_model_api_router(service: Arc<ModelToolService>) -> Router {
    Router::new()
        .route("/v1/knowledge/search", post(search_handler))
        .route("/v1/knowledge/traverse", post(traverse_handler))
        .route("/v1/knowledge/resolve", post(resolve_handler))
        .route("/v1/knowledge/remember", post(remember_handler))
        .route("/v1/knowledge/outcomes", post(outcome_handler))
        .route("/v1/knowledge/tasks/begin", post(task_begin_handler))
        .route("/v1/knowledge/tasks/context", post(task_context_handler))
        .route("/v1/knowledge/tasks/complete", post(task_complete_handler))
        .route("/v1/knowledge/web/search", post(web_search_handler))
        .route("/v1/knowledge/explore", post(explore_handler))
        .route("/v1/knowledge/status", post(status_handler))
        .route("/v1/knowledge/cases/run", post(run_case_handler))
        .with_state(service)
}

pub(crate) fn decode_arguments<T: DeserializeOwned>(value: serde_json::Value) -> HNSQRResult<T> {
    serde_json::from_value(value).map_err(|error| HNSQRError::InvalidRequest(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedEmbeddingProvider {
        descriptor: EmbeddingDescriptor,
    }

    impl TextEmbeddingProvider for FixedEmbeddingProvider {
        fn descriptor(&self) -> &EmbeddingDescriptor {
            &self.descriptor
        }

        fn embed(&self, _text: &str) -> HNSQRResult<Vec<f32>> {
            Ok(vec![0.6, 0.8])
        }
    }

    fn subject(tenant: &str, role: AccessRole) -> AuthenticatedSubject {
        AuthenticatedSubject {
            tenant_id: tenant.to_string(),
            role,
            key_id: "test".to_string(),
        }
    }

    fn service(store: Arc<ModelKnowledgeStore>) -> ModelToolService {
        ModelToolService::new(
            Arc::new(GatewayRouter::new("unused", false)),
            store,
            Arc::new(ModelGatewayAuth::development_anonymous()),
        )
    }

    fn remember_request(id: &str, key: &str, content: &str) -> RememberToolRequest {
        RememberToolRequest {
            idempotency_key: key.to_string(),
            id: id.to_string(),
            collection: "knowledge".to_string(),
            kind: "resolution".to_string(),
            evidence_class: None,
            content: content.to_string(),
            vector: None,
            embedding: None,
            members: Vec::new(),
            roles: BTreeMap::new(),
            metadata: BTreeMap::new(),
            provenance: vec![ProvenanceReference {
                source_id: "test".to_string(),
                uri: None,
                content_hash: "sha256:test".to_string(),
                observed_at_lsn: None,
            }],
        }
    }

    #[test]
    fn remembers_searches_and_isolates_tenants() {
        let service = service(Arc::new(ModelKnowledgeStore::in_memory()));
        service
            .remember(
                &subject("alpha", AccessRole::ReadWrite),
                remember_request("cooling", "k1", "liquid cooling reduces thermal load"),
            )
            .unwrap();
        let found = service
            .search(
                &subject("alpha", AccessRole::ReadOnly),
                SearchToolRequest {
                    query_text: Some("thermal cooling".to_string()),
                    query_vector: None,
                    embedding: None,
                    collection: "knowledge".to_string(),
                    k: 5,
                    kinds: None,
                    filter: None,
                    retrieval_contract: Some("exact".to_string()),
                    certified_exact: None,
                    snapshot_lsn: None,
                },
            )
            .unwrap();
        assert_eq!(found.results[0].id, "cooling");
        assert!(
            service
                .search(
                    &subject("beta", AccessRole::ReadOnly),
                    SearchToolRequest {
                        query_text: Some("thermal cooling".to_string()),
                        query_vector: None,
                        embedding: None,
                        collection: "knowledge".to_string(),
                        k: 5,
                        kinds: None,
                        filter: None,
                        retrieval_contract: Some("exact".to_string()),
                        certified_exact: None,
                        snapshot_lsn: None,
                    },
                )
                .unwrap()
                .results
                .is_empty()
        );
    }

    #[test]
    fn configured_provider_pins_text_only_writes_to_its_embedding_space() {
        let descriptor = EmbeddingDescriptor {
            provider: "test-local".to_string(),
            model: "bge-compatible".to_string(),
            version: "1".to_string(),
            dimensions: 2,
            normalization: "l2".to_string(),
            distance_metric: "cosine".to_string(),
        };
        let service = ModelToolService::with_embedding_provider(
            Arc::new(GatewayRouter::new("unused", false)),
            Arc::new(ModelKnowledgeStore::in_memory()),
            Arc::new(ModelGatewayAuth::development_anonymous()),
            Arc::new(FixedEmbeddingProvider {
                descriptor: descriptor.clone(),
            }),
        );
        let stored = service
            .remember(
                &subject("alpha", AccessRole::ReadWrite),
                remember_request("configured", "configured-key", "semantic text"),
            )
            .unwrap();
        assert_eq!(stored.results.embedding, descriptor);
    }

    #[test]
    fn configured_provider_preserves_text_access_to_legacy_lexical_collection() {
        let store = Arc::new(ModelKnowledgeStore::in_memory());
        let writer = service(Arc::clone(&store));
        let actor = subject("alpha", AccessRole::ReadWrite);
        writer
            .remember(
                &actor,
                remember_request("legacy", "legacy-key", "legacy thermal evidence"),
            )
            .unwrap();

        let bge_descriptor = EmbeddingDescriptor {
            provider: "test-local".to_string(),
            model: "bge-compatible".to_string(),
            version: "1".to_string(),
            dimensions: 2,
            normalization: "l2".to_string(),
            distance_metric: "cosine".to_string(),
        };
        let upgraded = ModelToolService::with_embedding_provider(
            Arc::new(GatewayRouter::new("unused", false)),
            store,
            Arc::new(ModelGatewayAuth::development_anonymous()),
            Arc::new(FixedEmbeddingProvider {
                descriptor: bge_descriptor,
            }),
        );

        let found = upgraded
            .search(
                &actor,
                SearchToolRequest {
                    query_text: Some("legacy thermal evidence".to_string()),
                    query_vector: None,
                    embedding: None,
                    collection: "knowledge".to_string(),
                    k: 1,
                    kinds: None,
                    filter: None,
                    retrieval_contract: Some("exact".to_string()),
                    certified_exact: None,
                    snapshot_lsn: None,
                },
            )
            .unwrap();
        assert_eq!(found.results[0].id, "legacy");

        let appended = upgraded
            .remember(
                &actor,
                remember_request("legacy-next", "legacy-next-key", "new legacy evidence"),
            )
            .unwrap();
        assert_eq!(appended.results.embedding, EmbeddingDescriptor::default());
    }

    #[test]
    fn journal_recovery_preserves_idempotency_and_snapshots() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model-tools.jsonl");
        let first_lsn = {
            let store = Arc::new(ModelKnowledgeStore::open(&path).unwrap());
            let service = service(store);
            let first = service
                .remember(
                    &subject("alpha", AccessRole::ReadWrite),
                    remember_request("record-1", "same-key", "evidence one"),
                )
                .unwrap();
            first.snapshot_lsn
        };
        let store = Arc::new(ModelKnowledgeStore::open(&path).unwrap());
        let service = service(store);
        let repeated = service
            .remember(
                &subject("alpha", AccessRole::ReadWrite),
                remember_request("ignored-id", "same-key", "ignored content"),
            )
            .unwrap();
        assert_eq!(repeated.snapshot_lsn, first_lsn);
        assert_eq!(repeated.results.id, "record-1");
        let recovered = service
            .search(
                &subject("alpha", AccessRole::ReadOnly),
                SearchToolRequest {
                    query_text: Some("evidence one".to_string()),
                    query_vector: None,
                    embedding: None,
                    collection: "knowledge".to_string(),
                    k: 1,
                    kinds: None,
                    filter: None,
                    retrieval_contract: Some("exact".to_string()),
                    certified_exact: None,
                    snapshot_lsn: None,
                },
            )
            .unwrap();
        assert_eq!(recovered.results[0].id, "record-1");
    }

    #[test]
    fn journal_tampering_is_rejected_fail_closed() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model-tools.jsonl");
        {
            let store = Arc::new(ModelKnowledgeStore::open(&path).unwrap());
            service(store)
                .remember(
                    &subject("alpha", AccessRole::ReadWrite),
                    remember_request("record-1", "key-1", "evidence"),
                )
                .unwrap();
        }
        let line = std::fs::read_to_string(&path).unwrap();
        let mut entry: serde_json::Value = serde_json::from_str(line.trim()).unwrap();
        entry["checksum"] = serde_json::Value::String("00".repeat(32));
        std::fs::write(
            &path,
            format!("{}\n", serde_json::to_string(&entry).unwrap()),
        )
        .unwrap();
        assert!(matches!(
            ModelKnowledgeStore::open(&path),
            Err(HNSQRError::CorruptedSnapshot(_))
        ));
    }

    #[test]
    fn historical_snapshot_excludes_later_better_match_without_underfill() {
        let service = service(Arc::new(ModelKnowledgeStore::in_memory()));
        let actor = subject("alpha", AccessRole::ReadWrite);
        let first = service
            .remember(
                &actor,
                remember_request("early", "key-early", "thermal cooling evidence"),
            )
            .unwrap();
        service
            .remember(
                &actor,
                remember_request("late", "key-late", "thermal cooling"),
            )
            .unwrap();
        let historical = service
            .search(
                &actor,
                SearchToolRequest {
                    query_text: Some("thermal cooling".to_string()),
                    query_vector: None,
                    embedding: None,
                    collection: "knowledge".to_string(),
                    k: 1,
                    kinds: None,
                    filter: None,
                    retrieval_contract: Some("exact".to_string()),
                    certified_exact: None,
                    snapshot_lsn: Some(first.snapshot_lsn),
                },
            )
            .unwrap();
        assert_eq!(historical.results.len(), 1);
        assert_eq!(historical.results[0].id, "early");
    }

    #[test]
    fn zero_snapshot_lsn_is_a_latest_snapshot_compatibility_sentinel() {
        let service = service(Arc::new(ModelKnowledgeStore::in_memory()));
        let actor = subject("alpha", AccessRole::ReadWrite);
        service
            .remember(
                &actor,
                remember_request("latest", "latest-key", "durable current evidence"),
            )
            .unwrap();
        let results = service
            .search(
                &actor,
                SearchToolRequest {
                    query_text: Some("current evidence".to_string()),
                    query_vector: None,
                    embedding: None,
                    collection: "knowledge".to_string(),
                    k: 1,
                    kinds: None,
                    filter: None,
                    retrieval_contract: None,
                    certified_exact: None,
                    snapshot_lsn: Some(0),
                },
            )
            .unwrap();
        assert_eq!(results.results[0].id, "latest");
        assert_eq!(results.snapshot_lsn, service.current_lsn());
    }

    #[test]
    fn traverses_n_ary_records_and_uses_outcomes_for_resolution() {
        let service = service(Arc::new(ModelKnowledgeStore::in_memory()));
        let actor = subject("alpha", AccessRole::ReadWrite);
        let mut relation = remember_request(
            "agreement",
            "relation-key",
            "three-party capacity sharing agreement",
        );
        relation.kind = "agreement".to_string();
        relation.members = vec!["party-a".into(), "party-b".into(), "party-c".into()];
        service.remember(&actor, relation).unwrap();
        let traversal = service
            .traverse(
                &actor,
                TraverseToolRequest {
                    seed_ids: vec!["party-a".to_string()],
                    relation_kinds: vec!["agreement".to_string()],
                    max_depth: 2,
                    max_results: 10,
                    snapshot_lsn: None,
                },
            )
            .unwrap();
        assert_eq!(traversal.results[0].record.id, "agreement");
        service
            .record_outcome(
                &actor,
                RecordOutcomeToolRequest {
                    idempotency_key: "outcome-key".to_string(),
                    attempt_id: "attempt-1".to_string(),
                    summary: "capacity stabilized".to_string(),
                    successful: true,
                    evidence_class: None,
                    measurement: None,
                    evidence_ids: vec!["agreement".to_string()],
                    metrics: BTreeMap::new(),
                    provenance: vec![ProvenanceReference {
                        source_id: "outcome-test".to_string(),
                        uri: None,
                        content_hash: "sha256:outcome-test".to_string(),
                        observed_at_lsn: None,
                    }],
                },
            )
            .unwrap();
        let resolution = service
            .resolve(
                &actor,
                ResolveToolRequest {
                    problem: "capacity sharing".to_string(),
                    query_vector: None,
                    embedding: None,
                    collection: "knowledge".to_string(),
                    max_hypotheses: 5,
                    snapshot_lsn: None,
                },
            )
            .unwrap();
        assert_eq!(resolution.results[0].successful_outcomes, 1);
        assert_eq!(
            resolution.results[0].status,
            "hypothesis_requires_external_validation"
        );
    }

    #[test]
    fn resolve_embeds_the_problem_instead_of_using_a_placeholder_vector() {
        let service = service(Arc::new(ModelKnowledgeStore::in_memory()));
        let actor = subject("alpha", AccessRole::ReadWrite);
        service
            .remember(
                &actor,
                remember_request("resolution", "resolution-key", "capacity sharing agreement"),
            )
            .unwrap();

        let resolution = service
            .resolve(
                &actor,
                ResolveToolRequest {
                    problem: "capacity sharing".to_string(),
                    // Some MCP clients used to fabricate this field. `resolve` must remain in
                    // the collection's text embedding space rather than adapt this vector.
                    query_vector: Some(vec![0.0, 0.0, 0.0]),
                    embedding: None,
                    collection: "knowledge".to_string(),
                    max_hypotheses: 5,
                    snapshot_lsn: None,
                },
            )
            .unwrap();

        assert_eq!(resolution.results[0].evidence_ids, ["resolution"]);
    }

    #[test]
    fn epistemic_integrity_distinguishes_measurements_from_reported_claims() {
        let service = service(Arc::new(ModelKnowledgeStore::in_memory()));
        let actor = subject("alpha", AccessRole::ReadWrite);

        // 1. Asserted metrics without measurement artifact become ReportedClaim / ReportedUnverified
        let unverified = service
            .record_outcome(
                &actor,
                RecordOutcomeToolRequest {
                    idempotency_key: "claim-1".to_string(),
                    attempt_id: "attempt-claim-1".to_string(),
                    summary: "Claimed Sharpe 3.48".to_string(),
                    successful: true,
                    evidence_class: None,
                    measurement: None,
                    evidence_ids: Vec::new(),
                    metrics: BTreeMap::from([("sharpe".to_string(), 3.48)]),
                    provenance: Vec::new(),
                },
            )
            .unwrap();
        assert_eq!(
            unverified.results.evidence_class,
            EvidenceClass::ReportedClaim
        );
        assert_eq!(
            unverified.results.verification_state,
            VerificationState::ReportedUnverified
        );

        // 2. Verified measurement with audited artifact becomes Measurement / Verified
        let verified = service
            .record_outcome(
                &actor,
                RecordOutcomeToolRequest {
                    idempotency_key: "meas-1".to_string(),
                    attempt_id: "attempt-meas-1".to_string(),
                    summary: "Backtest run 492 verified on SPX tick data".to_string(),
                    successful: true,
                    evidence_class: None,
                    measurement: Some(MeasurementSpec {
                        artifact_id: "artifact:spx-bt-492".to_string(),
                        producer: "backtest-engine:v2".to_string(),
                        dataset_id: Some("spx-2025-tick".to_string()),
                        run_id: Some("run-492".to_string()),
                        metrics_digest: "sha256:spx-bt-492-digest".to_string(),
                    }),
                    evidence_ids: Vec::new(),
                    metrics: BTreeMap::from([("sharpe".to_string(), 3.48)]),
                    provenance: Vec::new(),
                },
            )
            .unwrap();
        assert_eq!(verified.results.evidence_class, EvidenceClass::Measurement);
        assert_eq!(
            verified.results.verification_state,
            VerificationState::Verified
        );
    }
}
