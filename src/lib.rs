/* holosphere/src/lib.rs */
//!▫~•◦-------------------------------‣
//! # Hierarchical Navigable Semantic Query Resolver (HNSQR)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! A classical, maximum-throughput, ultra-low-latency multimodal retrieval engine
//! backed by complex-valued isometric embeddings, lock-free concurrent arena allocation,
//! decoupled contiguous dense vector matrices, fine-grained per-layer connection arrays,
//! dual-accumulator AVX2+FMA SIMD tensor acceleration, zero-allocation thread-local scratchpads,
//! hardware cache prefetching, bounded Rivero routing, and certified Cauchy-Schwarz bounds.
//!
//! ## Mathematical Foundations
//! HNSQR is a classical retrieval system executing on conventional CPUs/GPUs. Its core algorithms
//! leverage complex-valued linear algebra, lattice routing, graph traversal, and admissible geometric bounds:
//! - **Pairwise Complex Isometric Folding:** Real-to-complex isometry $\Phi: \mathbb{R}^{2d} \to \mathbb{C}^d$ preserving Euclidean inner products $\text{Re}\langle\Phi(x),\Phi(y)\rangle = x^\top y$.
//! - **Complex Projective Overlap (CPO):** Normalized squared complex ray overlap $P(z, w) = |\langle z, w\rangle|^2 / (\|z\|^2 \|w\|^2)$.
//! - **Hierarchical Rivero Envelopes & Proof Frontier:** Provable blockwise bounds guaranteeing bit-exact Top-$K$ retrieval.
//! - **Lock-Free Concurrency & Zero-Copy Persistence:** Concurrent arenas backed by memory-mapped files (`MmapArena`).
//!
//! ### Architectural Notes
//! Works directly with `server`, `gateway`, `sparse`, `multivector`, `planner`, and `metadata_index` modules for production vector deployment.
//!
//! #### Example
//! ```rust
//! use hnsqr::{HNSQRIndex, HNSQRConfig, VectorEmbedding};
//!
//! let config = HNSQRConfig::default();
//! let index = HNSQRIndex::new(config, 64);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

#![allow(
    unsafe_op_in_unsafe_fn,
    clippy::too_many_arguments,
    clippy::collapsible_if,
    clippy::field_reassign_with_default
)]

use std::cell::{RefCell, UnsafeCell};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fmt::{self, Debug};
use std::sync::Arc;
use std::sync::atomic::{
    AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering as AtomicOrdering,
};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use num_complex::Complex32;
use parking_lot::RwLock;
use rand::{RngExt, rng as thread_rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use smallvec::{SmallVec, smallvec};
use thiserror::Error;
use tracing::{info, instrument, trace, warn};

/// Capacity Planning & Infrastructure Sizing.
pub mod capacity;
/// Distributed Cluster Control Plane & Partition Sharding.
pub mod cluster;
/// Distributed Consensus & Raft State Machine.
pub mod consensus;
/// AI Ecosystem & Framework Integrations.
pub mod ecosystem;
/// Geo-Distributed Federation Subsystem.
pub mod federation;
/// Kubernetes Operator & Cloud Native Orchestration.
pub mod kubernetes;
/// Metadata Indexing, Inverted Stores & Cardinality Governance Subsystem.
pub mod metadata;
/// Query Planning & Automated Index Calibration Subsystem.
pub mod planning;
/// Canonical Corpus-Covering Semantic Proof Hierarchy & Best-Bound Proof Engine.
pub mod proof;
/// Multi-Modal & Hybrid Retrieval Subsystem.
pub mod retrieval;
/// Rivero bounded semantic address resolution, parallel bulk builder, and reciprocal witnesses.
pub mod rivero;
use crate::rivero::witness as rivero_witness;
/// Semantic Kernel v1 Conformance & Compatibility Subsystem.
pub mod conformance;
/// Unified Entity Kernel — Universal Entity Universe, Epistemics, Provenance & Lineage.
pub mod entity;
/// Empirical Experience Subsystem — Problems, Contexts, Actions, Attempts & Outcomes.
pub mod experience;
/// Native Graph Engine — Index-Free Adjacency, GraphQuery-compatible Query, GDS Analytics.
pub mod graph;
/// Empirical Learning & Deterministic Adjudication Subsystem.
pub mod learning;
/// Native Dynamic Hypergraph Relations & N-ary Role Bindings.
pub mod relation;
/// Security, Multi-Tenancy & Authorization.
pub mod security;
/// Unified Production Service Layer.
pub mod service;
/// Storage Durability, WAL & Unified Snapshots.
pub mod storage;
/// Production Telemetry & Prometheus Observability.
pub mod telemetry;
/// Wire Transport & QIR0 Network Protocol Subsystem.
pub mod transport;
/// Vector Representation, Folding & Quantization Subsystem.
pub mod vector;

pub use capacity::{
    CapacityPlanner, CapacityRequirements, ClusterCapacityPlan, MachineTelemetryProfile,
};
pub use cluster::{
    ApplyReceipt, ClientIdentity, ClusterId, ClusterTopology, ConsistentHashRing,
    ControlPlaneReconciliationPlan, DBaaSControlPlane, DataMutation, DeduplicationHorizon,
    DesiredClusterState, DisasterRecoveryCoordinator, DisasterRecoverySla, DistributedCoordinator,
    LocalShard, MigrationPhase, MigrationTask, NodeAddress, ObservedClusterState, OrganizationId,
    ReplicatedStateMachine, RetrySemantics, ShardId, ShardReplica, ShardRole, ShardStateMachine,
    WorldStateDigest,
};
pub use conformance::{
    CANONICAL_EXPORT_VERSION, CanonicalExportArchive, ENTITY_SCHEMA_VERSION,
    EXPERIENCE_SCHEMA_VERSION, ExportedEntity, ExportedExperience, ExportedLearningRecord,
    ExportedRelation, INFERENCE_TRACE_VERSION, KernelError, LEARNING_SCHEMA_VERSION,
    RAFT_LOG_RECORD_VERSION, RELATION_SCHEMA_VERSION, SEMANTIC_KERNEL_VERSION,
    SNAPSHOT_FORMAT_VERSION, SYNTHESIS_TRACE_VERSION, WORLD_DIGEST_VERSION,
    create_v1_golden_fixture,
};
pub use consensus::{
    AdaptiveMicrobatcher, AppendEntriesArgs, AppendEntriesReply, ApplyError, CommitReceipt,
    CommitStatus, DurabilityBatchPlan, DurabilityController, DurabilityLevel, DurableRaftStorage,
    LinearizableReadMode, LogLocation, LogSegmentMeta, MembershipMutation, MemoryRaftStorage,
    MutationId, PendingProposals, ProposalId, RaftCluster, RaftCommand, RaftHardState,
    RaftLogEntry, RaftNode, RaftPersistentProgress, RaftPipelineTelemetry, RaftRole,
    RaftSnapshotMeta, RaftStorage, ReadConsistency, ReadContextId, ReadIndexConfirmation,
    ReadIndexEngine, ReadIndexRequest, ReadIndexTelemetry, RequestVoteArgs, RequestVoteReply,
    StorageHealthMetrics, StorageTelemetry, TopologyMutation,
};
pub use ecosystem::{
    ClientSearchResult, FrameworkDocument, HNSQRClientConfig, HNSQRClientRouter, HNSQRVectorStore,
    HaystackAdapter, LangChainAdapter, LlamaIndexAdapter,
};
pub use entity::{
    ContextSignature, CosineMetric, DeterministicEvidenceStats, DurableEntityVersion,
    DurableEvidenceRef, ENTITY_FLAG_HAS_INFERENCE_SIDECAR, ENTITY_FLAG_HAS_PROPERTIES,
    ENTITY_FLAG_HAS_PROVENANCE, ENTITY_FLAG_HAS_VECTOR, ENTITY_FLAG_HAS_VERSION_HISTORY,
    ENTITY_FLAG_LIVE, EligibilityError, EligibilityRepresentation, EligibilitySet, EligibilityView,
    EntityArena, EntityHeader, EntityId, EntityIndex, EntityMutation, EntityReadSnapshot,
    EntitySegment, EpistemicStatus, EpistemicTransitionError, EuclideanMetric,
    ExactEligibilityCostModel, ExactEligibilityProof, ExactRetrievalContext, ExactScanOperator,
    ExactScanPlan, ExactVectorMetric, InnerProductMetric, LifecycleStatus,
    LifecycleTransitionError, MutationApplyError, NULL_ROW_REF, ProjectiveOverlapMetric,
    ProvenanceArena, ProvenanceId, ProvenanceIndex, ProvenanceRecord, ProvenanceRow,
    ResolvedEntityVersion, SNAPSHOT_V3_MAGIC, ScoredEntity, SnapshotV3Error,
    SnapshotV3SectionHeader, VectorArena, VectorLayout, VectorLayoutId, VectorNormalization,
    VectorScalarType, VersionId, VersionIndex, VersionRelation, VersionRow, VersionTable,
    decode_snapshot_v3, encode_snapshot_v3, exact_top_k, exact_top_k_scalar, masked_dense_scan,
    resolve_metric, sparse_gather_scan, validate_epistemic_transition,
    validate_lifecycle_transition,
};
pub use experience::{
    ActionDefinition, ActionId, ActionInvocation, ActionParameterValue, AttemptId, AttemptRecord,
    AttemptState, ContextId, ContextRecord, ContextValue, DurableActionParameter,
    DurableContextDimension, DurableOutcomeObservation, EvaluationPolicyId, ExperienceMutation,
    ExperienceMutationError, ExperienceQuery, ExperienceReadSnapshot, ExperienceSegment,
    ExperienceTrace, MetricId, MetricValue, MetricValueKind, OutcomeId, OutcomeMetricSchema,
    OutcomeRecord, ProblemId, ProblemOccurrence, compute_context_fingerprint,
};
pub use federation::{
    ClusterProofResponse, ClusterRegionId, FederatedProofCoordinator, FederatedProofStatus,
    FederatedQueryResult,
};
pub use graph::{
    // Storage
    AdjacencyBlock,
    // Analytics
    BfsResult,
    // Query
    BindingColumn,
    ConnectedComponents,
    CscAdjacency,
    CsrAdjacency,
    CsrProjection,
    DegreeCentrality,
    // Stats
    DegreeStats,
    Direction,
    EdgeDelta,
    EdgeDeltaStats,
    ExecutionContext as GraphExecutionContext,
    ExplainOutput,
    GraphCardinalityStats,
    GraphGeneration,
    // Mutation
    GraphMutation,
    GraphMutationApplier,
    GraphNodeRecord,
    GraphPattern,
    GraphProjection,
    GraphProperties,
    GraphPropertyStore,
    GraphPropertyValue,
    GraphReadGeneration,
    GraphSnapshot,
    KCoreDecomposition,
    // Catalog
    LabelCatalog,
    LabelId,
    LabelResolution,
    LogicalPlan,
    LouvainEngine,
    LouvainResult,
    Morsel,
    NULL_OVERFLOW_REF,
    NeighborSlice,
    NodeArena,
    PageRankEngine,
    PathfindingEngine,
    PhysicalPlan,
    PropertyKey,
    PropertyKeyCatalog,
    QueryAst,
    QueryResult as GraphQueryResult,
    RelTypeCatalog,
    RelTypeId,
    RelTypeResolution,
    RelationshipId as GraphRelationshipId,
    ReturnClause,
    SemanticAnalyzer,
    SemanticError,
    ShortestPath,
    SymbolId,
    SymbolTable,
    TriangleCount,
    WhereClause,
};
pub use kubernetes::{
    AutoscalerMetrics, AutoscalerRecommendation, HNSQRClusterSpec, HNSQRClusterStatus,
    KubernetesOperator, NativeAutoscaler, OperatorLifecyclePhase,
};
pub use learning::{
    ActiveExperimentKind, ActiveExperimentProposal, AdjudicationDecisionCode,
    AdjudicationDisposition, AdjudicationExplanation, AdjudicationId, AdjudicationPolicy,
    AdjudicationQuery, AdjudicationRecord, AgentBelief, AgentId, AgentMeta, BIVECTOR_DIM,
    BarycentricWeightSemantics, CandidateActionStepId, CandidateEntityId, CandidateEntityRef,
    CandidateResolutionState, CandidateRoleBinding, CanonicalLearningAuditDigest,
    CausalOrientation, CircularityCheck, Cl24BasisError, Cl24Blade, Cl24CompositionArtifact,
    Cl24EntityBasis, ClosureArtifactId, ClosureCandidate, ClosureKind, ComparisonOperator,
    CompetitiveEvaluationPolicy, CompetitiveOperatorEvaluation, CompositionRule,
    CompositionRuleRegistry, CompositionSemantics, ConceptBehavior, ConceptId,
    ConceptMappingHypothesis, ConceptProfile, ConditionExpression, ConfirmedConceptMappingIndex,
    ConflictPair, ConflictResolution, ConsensusResult, ConstraintCheck, ConstraintCode,
    ConstraintResult, ContextApplicability, ContextClassId, ContextClassRegistry,
    ContextDifference, ContinuousDiscoveryEngine, ContinuousDiscoveryInput,
    ContinuousDiscoveryPolicy, ContinuousDiscoveryReport, DEFAULT_MAX_TRUNCATION_LOSS_RATIO,
    DEFAULT_TRUNCATION_TOPK, DISCOVERY_CHECKPOINT_VERSION, DeclarativeOperator,
    DerivedEntityProposal, DirectedWedgeArtifact, DirectedWedgeRequest, DiscoveredMotif,
    DiscoveredOperatorId, DiscoveryAuditAction, DiscoveryAuditEntry, DiscoveryAuditLog,
    DiscoveryCase, DiscoveryCaseId, DiscoveryCatalog, DiscoveryCatalogError,
    DiscoveryCheckpointError, DiscoveryCorpus, DiscoveryGovernance, DiscoveryOutcome,
    DiscoveryPolicy, DiscoveryReport, DiscoveryStateError, DiscoveryStateMutation,
    DiscoveryStateSnapshot, DomainId, DslEffect, EmpiricalRootId, EpistemicLineageGraph,
    EvaluationObservation, EvaluationRole, EvidenceAccumulator, EvidenceDirection, EvidenceId,
    EvidenceIndependenceReport, EvidenceKey, EvidencePartition, EvidenceRecord, EvidenceSummary,
    EvidenceSummaryId, EvolutionArtifact, EvolutionHistoryView, EvolutionProposal, EvolvedSchemaId,
    EvolvedSchemaKind, EvolvedSchemaProposal, ExperienceProjectionPolicy,
    ExperienceProjectionReport, ExperimentAuthorization, ExperimentExecutionError, ExperimentKind,
    ExperimentPlanningPolicy, ExperimentProposal, ExperimentProposalId, ExperimentStatus,
    FeatureId, FixedUtility, GovernanceAuthority, GovernedDiscoveryCheckpoint,
    GovernedDiscoveryEngine, GovernedDiscoveryState, GovernedMappingRecord, GovernedSchemaRecord,
    HyperedgeMember, HypergraphMotifId, HypergraphMotifKind, HypergraphMotifPolicy,
    HypergraphTransformation, ImmutableSafetyKernel, InducedSchemaProposal, InferenceCandidate,
    InferenceCandidateId, InferenceError, InferenceGeometryArtifact, InferenceMethod,
    InferenceMethodId, InferenceMode, InferenceProposal, InferenceProposalBundle,
    InferenceRegistry, InferenceRequest, InferenceScope, InferenceScore, InferenceSeed,
    InferenceTrace, KnowledgeProjectionPolicy, KnowledgeProjectionReport, KnowledgeSnapshot,
    LearningMutation, LearningMutationError, LearningReadSnapshot, LearningSegment,
    LineageNodeKind, MAX_OPERATOR_CHAIN, MappingHypothesisId, MappingInductionPolicy,
    MappingLifecycle, MappingTransitionPlan, MappingValidation, MappingValidationPolicy,
    MetricDirection, MetricEvaluationRule, MotifId, MotifMinerConfig, MultivectorCl24Sparse,
    NormalizationRule, NovelResolution, NumericAttributeId, NumericExpression, OperatorAssessment,
    OperatorEffect, OperatorEpistemicRecord, OperatorLifecycle, OperatorPredicate, OperatorProgram,
    OperatorRevisionProposal, OperatorSandboxError, OperatorTransitionPlan, OperatorValidation,
    OperatorValidationPolicy, PARALLEL_BLEND_THRESHOLD, PhaseShift, PhaseShiftArtifact,
    PlanAttributionMethod, PlanAttributionRecord, Precedent, PrecedentDisposition, ProgramCost,
    ProgramResult, ProjectionSkip, ProjectionSkipReason, ProposalStalenessCheck, ProposedRole,
    RUNE_ANALOGY_METHOD_ID, RUNE_ANALOGY_METHOD_VERSION, RUNE_BARYCENTRIC_METHOD_ID,
    RUNE_BARYCENTRIC_METHOD_VERSION, RUNE_CLOSURE_METHOD_ID, RUNE_CLOSURE_METHOD_VERSION,
    RUNE_DIRECTED_WEDGE_METHOD_ID, RUNE_DIRECTED_WEDGE_METHOD_VERSION, RUNE_EVOLUTION_METHOD_ID,
    RUNE_EVOLUTION_METHOD_VERSION, ReasoningContext, ReasoningOperator, ReasoningOperatorId,
    RelationProposal, ReplicatedDiscoveryAction, ResolutionCandidate, ResolutionCandidateId,
    ResolutionId, ResolutionSemanticKey, ResourceCostBounds, RiskLevel, RotorAlignmentResult,
    RuneAnalogyConfig, RuneBarycentricConfig, RuneBarycentricInsight, RuneBarycentricV1,
    RuneCl24CompositionConfig, RuneClosureEvidenceV1, RuneOperatorClass, RunePhaseEvolutionV1,
    RuneStructuralAnalogyV1, SWARM_CONSENSUS_METHOD_ID, SWARM_CONSENSUS_METHOD_VERSION,
    SafetyKernelViolation, SandboxExperimentResult, SchemaInductionPolicy, SchemaProposalId,
    SchemaProposalState, SchemaTransitionPlan, SchemaValidation, SchemaValidationPolicy,
    SemanticCandidateRegistry, SemanticFingerprint, StructuralAnalogyArtifact, StructuralRole,
    StructuralSynthesisTrace, SynthesisAttempt, SynthesisBasis, SynthesisCandidateId,
    SynthesisDependencyDigest, SynthesisGoal, SynthesisKnowledgeBase, SynthesisOccurrence,
    SynthesisPolicy, SynthesisPolicyId, SynthesisRequest, SynthesisResult, SynthesisRunId,
    SynthesisScores, TemporalHyperedge, TemporalHypergraphMotif, TemporalInterval, align_regions,
    apply_competitive_evaluation, apply_givens_rotation, apply_phase_shift, apply_rotation,
    authorize_experiment, bivector_contract, bivector_strength, blade_product_sign,
    build_directed_wedge_edge, causal_bivector, check_epistemic_circularity, compile_closure,
    complete_experiment, compose_programs, compute_audit_digest, compute_evidence_digest,
    compute_plan_attribution, compute_swarm_consensus, derive_concept_behaviors, dot8,
    epistemic_record_from_evaluation, euclidean_dist_8, evaluate_adjudication,
    evaluate_evidence_independence, evaluate_program_competitively, execute_operator_chain,
    execute_program, execute_sandbox_experiment, geometric_counterfactual_projection,
    gram_schmidt_tangent, identity_rotation, induce_evolved_schemas, induce_schemas, infer_between,
    l2_sq_8, learn_concept_mappings, leech_to_e8_f32, materialize_collective_hypothesis,
    materialize_proposed_relation_type, materialize_relation_type, mean_alignment_residual,
    mine_motifs, mine_temporal_hypergraph_motifs, normalise_weights, normalize_vector_8,
    optimal_givens_angle, parallel_centroid, plan_active_experiments, plan_experiments,
    project_experience, project_knowledge, propose_compensating_rollback, region_centroid,
    resolve_barycentric, sequential_centroid, snap_to_e8_lattice, start_experiment, synthesize,
    synthesize_program_from_motif, validate_concept_mapping, validate_evolved_schema,
    validate_operator, validate_program,
};
pub use metadata::cardinality::{
    CardinalityBudget, CardinalityGuard, PostingRepresentation, TenantCardinalityTracker,
};
pub use metadata::index::{FilterExpr, MetadataInvertedIndex, MetadataValue};
pub use metadata::store::{MetadataQuotaConfig, MetadataStore, QuotaTracker};
pub use planning::autoforge::{
    AutoForge, DerivedPhysicalConfig, OperatorIntent, OperatorIntentConfig, PlannerProfile,
};
pub use planning::planner::{
    ExecutionPlan, ExecutionProof, QueryModality, RetrievalContract, UniversalPlanner,
};
pub use proof::lutz::{
    LutzCandidateThreat, LutzCertificationDiagnostics, LutzCertifier, LutzCode,
    LutzGlobalCertified, LutzQueryTable, SemanticRerankPlan, exact_rerank_locality_sorted,
};
pub use proof::{
    DenseExactProof, GlobalExactProofSearch, ProofCentroidCode, ProofNode, ProofQuery,
    SegmentProofView, SemanticProofTree,
};
pub use relation::{
    BinaryCsrProjection, BinaryProjectionCache, BinaryProjectionSpec, DurableRelationInstance,
    DurableRoleBinding, HyperPattern, HyperPatternError, HyperPatternMatch, HyperPatternMember,
    HyperPatternSemantics, IncidenceIndex, IncidenceKey, ProjectedBinaryRelationEdge,
    ProjectionDirection, RelationArena, RelationHeader, RelationId, RelationIndex,
    RelationMutation, RelationMutationError, RelationQuery, RelationReadSnapshot, RelationSegment,
    RelationType, RelationTypeId, RelationTypeState, RelationVersionId, RelationVersionRow,
    RelationVersionTable, ResolvedRelationVersion, RoleId, RoleSchema, SchemaScope,
    SchemaValidationError, SegmentRoleBinding, compute_canonical_fingerprint,
    project_resolved_relation,
};
pub use retrieval::hybrid::{
    HybridFusionEngine, HybridFusionMethod, ModalityRankings, RRF_DEFAULT_K,
};
pub use retrieval::multivector::{MultiVectorEmbedding, MultiVectorIndex};
pub use retrieval::performance_trial::{
    AdmissionGateStatus, BenchmarkRecord, BenchmarkRunIdentity, CertifiedEvidence,
    HnswBuildDescriptor, HnswSearchDescriptor, RetrievalTrial, TrialValidationError,
    evaluate_admission_gates,
};
pub use retrieval::sparse::{InvertedPostingList, SparseInvertedIndex, SparseVector};
pub use rivero::{
    AdaptivePolicy, AdaptiveRouteState, BuiltRiveroState, BulkBuildTelemetry, LaneAssignment,
    RIVERO_BUILD_CANDIDATE_CAP, RIVERO_CELL_CAPACITY, RIVERO_DEFAULT_FOUNDATIONS,
    RIVERO_FOUNDATIONS, RIVERO_MAX_FOUNDATIONS, RIVERO_QUERY_CANDIDATE_CAP, RIVERO_SCHEMA_VERSION,
    RIVERO_WITNESS_DEFAULT_DEGREE, RIVERO_WITNESS_DEFAULT_SECOND_SEEDS,
    RIVERO_WITNESS_DEFAULT_SEEDS, RIVERO_WITNESS_INLINE_DEGREE, RIVERO_WITNESS_MAX_DEGREE,
    RIVERO_WITNESS_MAX_SEEDS, RiveroAddress, RiveroAddressConfig, RiveroBulkBuilder,
    RiveroCompiler, RiveroConfidence, RiveroConfig, RiveroProfile, RiveroProjectionMode,
    RiveroRouteDiagnostics, RiveroSearchMode, RiveroTerritoryIndex, ScoredWitness, VotedCandidate,
    select_top, witness_edge_scan_bound, witness_two_hop_edge_scan_bound,
};
pub use security::{
    AccessRole, ActiveCertificate, AuditAction, AuditLogger, AuditRecord, AuthCredential,
    AuthRegistry, AuthenticatedSubject, CertificateManager, ComplianceEvidenceGenerator,
    DEFAULT_MAX_FRAME_BYTES, KmsProvider, LocalKmsProvider, ProtocolFuzzSummary, ProtocolFuzzer,
    SecurityReportDocument, SiemExporter, SiemFormat, TenantContext, TenantManager,
    TenantNamespace, TenantQuota, TlsConfig,
};
pub use service::{
    ClusterService, DeleteRequest, HNSQRService, MutationReceipt, MutationService,
    PinnedReadSnapshot, ReadSnapshot, RequestContext, SearchResponse, SearchService,
    StandaloneService, UpsertRequest,
};
pub use storage::mmap_arena::{MmapArena, MmapHeader};
pub use storage::predictive_warming::{PredictiveWarmer, ProofHeatMap};
pub use storage::remote_cache::{CachedChunk, ChunkId, RemoteRangeCache};
pub use storage::remote_layout::{
    ProofAwareLayoutBuilder, ProofLeafBlockMapping, RemoteAmplificationMetrics, RemoteChunkSize,
};
pub use storage::segment::{
    ImmutableSegment, MutableSegment, SegmentId, SegmentState, SegmentStats, SegmentedEngine,
};
pub use storage::segment_store::{
    ImmutableSegmentStore, LocalSegmentStore, S3SegmentStore, SegmentObjectId,
    SegmentObjectMetadata,
};
pub use storage::snapshot::{
    PrefaultMode, SectionDescriptor, SectionKind, SnapshotAttachBreakdown, SnapshotHeaderV2,
    SnapshotOpenOptions, SnapshotStats, VerificationMode,
};
pub use storage::two_tier_cache::{CacheBlockId, CachedVectorBlock, TwoTierCache};
pub use storage::wal::{
    DurabilityPolicy, WalFrameHeader, WalManager, WalMetrics, WalMutation, WalRecordType,
    WalRecoverySummary,
};
pub use telemetry::metrics::{EngineMetrics, PrometheusExporter};
pub use telemetry::slo::{SloAlertSeverity, SloManager, SloReport, SloTargetConfig};
pub use telemetry::tracing::{ExecutionSpan, SpanRecord, TraceContext};
pub use transport::qir0::{
    HNSQRClient, HNSQRServer, MessageHeader, OpCode, PROTOCOL_MAGIC, Qir0SearchResponse,
};
pub use transport::{
    ConfiguredProviders, EmbeddingBackend, EmbeddingDescriptor, EmbeddingProviderConfig,
    EvidenceEnvelope, HoloSphereConfig, KnowledgeRecord, MCP_PROTOCOL_VERSION, ModelGatewayAuth,
    ModelKnowledgeStore, ModelOutcomeRecord, ModelToolService, RecordOutcomeToolRequest,
    RememberToolRequest, ResolveToolRequest, SearchToolRequest, TextEmbeddingProvider,
    TraverseToolRequest, WebSearchBackend, WebSearchConfig, WebSearchProvider, WebSearchResponse,
    WebSearchResult, WebSearchToolRequest, create_mcp_router, create_model_api_router, load_config,
    process_mcp_payload, provider_from_config, provider_from_file_if_exists,
    providers_from_file_if_exists,
};
pub use vector::folding::{ComplexWeaver, GatewayRouter, create_http_router, run_http_server};
pub use vector::quantization::PolarQuantizedVector;

/// Helper macro for RAII-style cleanup using `defer`.
macro_rules! defer {
    ($($e:tt)*) => {
        struct Defer<F: FnOnce()> {
            f: Option<F>,
        }
        impl<F: FnOnce()> Drop for Defer<F> {
            fn drop(&mut self) {
                if let Some(f) = self.f.take() {
                    f();
                }
            }
        }
        let _deferred = Defer {
            f: Some(|| {
                $($e)*
            }),
        };
    };
}

/// Type alias for the dimensionality of complex vector embeddings.
pub type Dimension = usize;
/// Type alias for internal integer node identifiers in the contiguous arena.
pub type NodeIndex = u32;
/// Type alias for similarity score values returned during searches (e.g. Projective Overlap in [0, 1]).
pub type SimilarityScore = f32;
/// Type alias for user-facing external node identifiers.
/// `Arc<str>` instead of `String`: cloning a search result ID is an O(1) atomic
/// refcount increment rather than a heap copy, regardless of string length.
pub type NodeId = Arc<str>;

/// Result type for HNSQR operations, using [`HNSQRError`] for errors.
pub type HNSQRResult<T> = std::result::Result<T, HNSQRError>;

/// Errors that can occur during HNSQR index operations.
#[derive(Error, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HNSQRError {
    /// Vector dimension does not match the configured index dimension.
    #[error("Vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected dimension
        expected: Dimension,
        /// Actual dimension provided
        actual: Dimension,
    },

    /// A node with the specified external ID was not found in the index.
    #[error("Node not found: {0}")]
    NodeNotFound(String),

    /// A node with the specified internal index was not found.
    #[error("Node index out of bounds: {0}")]
    NodeIndexNotFound(NodeIndex),

    /// A node with the specified external ID already exists.
    #[error("Node already exists: {0}")]
    NodeAlreadyExists(String),

    /// The index has reached its maximum capacity.
    #[error("Index is full: current capacity {0}")]
    IndexFull(usize),

    /// Search failed due to an internal inconsistency or missing entry point.
    #[error("Search error: {0}")]
    SearchError(String),

    /// Concurrency violation or conflicting state during index maintenance.
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),

    /// An invalid configuration parameter was supplied.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// An invalid request or query parameter was supplied.
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Serialization or deserialization failure.
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// File I/O failure during snapshot or persistence operations.
    #[error("I/O error: {0}")]
    IoError(String),

    /// Incompatible snapshot header, schema, or structural mismatch.
    #[error("Snapshot incompatible: {0}")]
    SnapshotIncompatible(String),

    /// Corrupted snapshot or log frame checksum mismatch.
    #[error("Corrupted snapshot/log: {0}")]
    CorruptedSnapshot(String),

    /// Internal engine error or admission rejection.
    #[error("Internal error: {0}")]
    Internal(String),

    /// Stale topology epoch error.
    #[error("Stale topology epoch: {0}")]
    StaleEpoch(String),

    /// Unauthorized access or permission denial.
    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    /// Unsupported feature or operation.
    #[error("Unsupported feature: {0}")]
    UnsupportedFeature(String),

    /// A `Certified` proof search was aborted because the configured query deadline
    /// expired before the proof frontier was exhausted.
    ///
    /// The partial results and full telemetry are preserved in the payload so that
    /// `hnsqr_doctor` and application metrics can diagnose the cause:
    ///   - Large `frontier_nodes_remaining` → pathological proof geometry (isotropic corpus)
    ///   - Small `frontier_nodes_remaining` + low `region_prune_ratio_ppm` → CPU overload
    ///   - Large `elapsed_us` → remote storage latency or enormous mutable segment
    ///
    /// Callers who want best-effort partial results instead of an error should use
    /// [`HNSQRIndex::certified_search`] which returns a [`CertifiedSearchOutcome`].
    #[error(
        "Certified search deadline exceeded: elapsed {elapsed_us}µs > budget {budget_us}µs \
         ({frontier_nodes_remaining} frontier nodes remaining, \
         prune ratio {region_prune_ratio_ppm}/1000000)"
    )]
    CertifiedDeadlineExceeded {
        /// Microseconds elapsed when the deadline fired.
        elapsed_us: u64,
        /// Configured budget in microseconds.
        budget_us: u64,
        /// Frontier entries still pending when aborted.
        frontier_nodes_remaining: usize,
        /// Region prune ratio in parts-per-million (0–1_000_000).
        /// Divide by 1_000_000.0 to recover the float ratio.
        region_prune_ratio_ppm: u32,
    },
}

impl From<std::io::Error> for HNSQRError {
    fn from(err: std::io::Error) -> Self {
        HNSQRError::IoError(err.to_string())
    }
}

/// Typed outcome of a [`HNSQRIndex::certified_search`] call.
///
/// This enum is the correct return type for callers that requested the `Certified`
/// contract.  It makes the incomplete-proof case structurally impossible to confuse
/// with a successful exact result — no boolean flag needs to be checked.
///
/// ```text
/// match index.certified_search(&query, k, None) {
///     CertifiedSearchOutcome::Exact { results, proof } => { /* 100% exact */ }
///     CertifiedSearchOutcome::DeadlineExceeded { partial_results, proof } => {
///         // Budget expired — partial_results are best-effort, not certified.
///         // proof.frontier_nodes_remaining indicates why the deadline fired.
///     }
/// }
/// ```
#[derive(Debug)]
pub enum CertifiedSearchOutcome {
    /// The proof frontier was fully exhausted.  `results` are guaranteed
    /// 100.000% exact Top-K for the pinned snapshot.
    Exact {
        results: Vec<(NodeIndex, SimilarityScore)>,
        proof: DenseExactProof,
    },
    /// The configured query deadline expired before the proof was complete.
    /// `partial_results` are the best-known candidates at abort time; they are
    /// **not** certified exact.  `proof` carries full telemetry including
    /// `frontier_nodes_remaining` and `region_prune_ratio` to diagnose the cause.
    DeadlineExceeded {
        partial_results: Vec<(NodeIndex, SimilarityScore)>,
        proof: DenseExactProof,
    },
}

/// Detailed forensic tracing of Ground Truth (GT) neighbor survival through the routing pipeline.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GtPipelineTrace {
    pub gt_count: usize,
    pub gt_in_raw_route: usize,
    pub gt_after_vote_selection: usize,
    pub gt_after_witness: usize,
    pub gt_in_final_results: usize,
    pub missing_gt_ranks: Vec<Option<usize>>,
    pub top1_recalled: bool,
    pub recall_at_10: f64,
}

/// Returns the current Unix timestamp in seconds.
fn current_unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ════════════════════════════════════════════════════════════════════════════════
// 0. ZERO-ALLOCATION THREAD-LOCAL BUFFERS
// ════════════════════════════════════════════════════════════════════════════════

thread_local! {
    static THREAD_VISITED_POOL: RefCell<VisitedPool> = RefCell::new(VisitedPool::new(65536));
    static THREAD_SEARCH_SCRATCHPAD: RefCell<SearchScratchpad> = RefCell::new(SearchScratchpad::new());
}

/// A zero-allocation thread-local visited tracking pool using query epochs.
pub struct VisitedPool {
    tags: Vec<u32>,
    epoch: u32,
}

impl VisitedPool {
    /// Creates a new visited pool with pre-allocated capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            tags: vec![0; capacity],
            epoch: 1,
        }
    }

    /// Advances the query epoch, clearing tags only when epoch wraps around ($2^{32}$ searches).
    #[inline(always)]
    pub fn next_epoch(&mut self, capacity_needed: usize) -> u32 {
        if self.tags.len() < capacity_needed {
            self.tags
                .resize(capacity_needed.max(self.tags.len() * 2), 0);
        }
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.tags.fill(0);
            self.epoch = 1;
        }
        self.epoch
    }

    /// Checks whether `index` has been visited during `epoch`.
    #[inline(always)]
    pub fn is_visited(&self, index: NodeIndex, epoch: u32) -> bool {
        let idx = index as usize;
        if idx < self.tags.len() {
            unsafe { *self.tags.get_unchecked(idx) == epoch }
        } else {
            false
        }
    }

    /// Marks `index` as visited for `epoch`.
    #[inline(always)]
    pub fn mark_visited(&mut self, index: NodeIndex, epoch: u32) {
        let idx = index as usize;
        if idx >= self.tags.len() {
            self.tags.resize((idx + 1).max(self.tags.len() * 2), 0);
        }
        unsafe {
            *self.tags.get_unchecked_mut(idx) = epoch;
        }
    }
}

/// Reusable scratchpad avoiding all inner-loop dynamic allocations during searches.
pub struct SearchScratchpad {
    candidate_queue: BinaryHeap<Candidate>,
    results_heap: BinaryHeap<WorstResultCandidate>,
    current_batch: Vec<Candidate>,
    neighbor_candidates: Vec<(NodeIndex, usize)>,
    scored_neighbors: Vec<(NodeIndex, f32, f32)>,
    beam: Vec<(NodeIndex, f32)>,
    next_beam: Vec<(NodeIndex, f32)>,
    exps: Vec<f32>,
    temp_conns: SmallVec<[NodeIndex; 64]>,
    psi_beam_data: Vec<Complex32>,
}

impl SearchScratchpad {
    fn new() -> Self {
        Self {
            candidate_queue: BinaryHeap::with_capacity(512),
            results_heap: BinaryHeap::with_capacity(512),
            current_batch: Vec::with_capacity(64),
            neighbor_candidates: Vec::with_capacity(256),
            scored_neighbors: Vec::with_capacity(256),
            beam: Vec::with_capacity(32),
            next_beam: Vec::with_capacity(32),
            exps: Vec::with_capacity(256),
            temp_conns: SmallVec::with_capacity(64),
            psi_beam_data: Vec::with_capacity(128),
        }
    }

    #[inline(always)]
    fn reset(&mut self, dim: usize) {
        self.candidate_queue.clear();
        self.results_heap.clear();
        self.current_batch.clear();
        self.neighbor_candidates.clear();
        self.scored_neighbors.clear();
        self.beam.clear();
        self.next_beam.clear();
        self.exps.clear();
        self.temp_conns.clear();
        if self.psi_beam_data.len() != dim {
            self.psi_beam_data.resize(dim, Complex32::new(0.0, 0.0));
        }
        self.psi_beam_data.fill(Complex32::new(0.0, 0.0));
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 1. PHASE-ENCODED EMBEDDINGS & DUAL-ACCUMULATOR SIMD KERNELS
// ════════════════════════════════════════════════════════════════════════════════

/// Computes the complex conjugate inner product $\langle\psi|\phi\rangle = \sum_j \psi_j^* \phi_j$
/// accelerated with dual-accumulator AVX2+FMA SIMD vectorization.
#[inline(always)]
pub fn dot_product_complex_simd(a: &[Complex32], b: &[Complex32]) -> Complex32 {
    #[cfg(target_arch = "x86_64")]
    {
        unsafe { dot_product_complex_avx2_dual(a, b) }
    }
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { dot_product_complex_neon_dual(a, b) }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_product_complex_scalar_unrolled(a, b)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_complex_neon_dual(a: &[Complex32], b: &[Complex32]) -> Complex32 {
    use core::arch::aarch64::*;

    let len = a.len().min(b.len());
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr() as *const f32;

    let mut acc_re0 = vdupq_n_f32(0.0);
    let mut acc_im0 = vdupq_n_f32(0.0);
    let mut acc_re1 = vdupq_n_f32(0.0);
    let mut acc_im1 = vdupq_n_f32(0.0);

    let chunks8 = len / 8;
    let mut offset = 0;

    for _ in 0..chunks8 {
        // Load 4 complex numbers (8 interleaved floats) per chunk into separate real and imag vectors
        let va0 = vld2q_f32(a_ptr.add(offset * 2));
        let vb0 = vld2q_f32(b_ptr.add(offset * 2));
        let va1 = vld2q_f32(a_ptr.add(offset * 2 + 8));
        let vb1 = vld2q_f32(b_ptr.add(offset * 2 + 8));

        // Chunk 0: Re<a, b> = a.re * b.re + a.im * b.im
        acc_re0 = vfmaq_f32(acc_re0, va0.val[0], vb0.val[0]);
        acc_re0 = vfmaq_f32(acc_re0, va0.val[1], vb0.val[1]);
        // Im<a, b> = a.re * b.im - a.im * b.re
        acc_im0 = vfmaq_f32(acc_im0, va0.val[0], vb0.val[1]);
        acc_im0 = vfmsq_f32(acc_im0, va0.val[1], vb0.val[0]);

        // Chunk 1:
        acc_re1 = vfmaq_f32(acc_re1, va1.val[0], vb1.val[0]);
        acc_re1 = vfmaq_f32(acc_re1, va1.val[1], vb1.val[1]);
        acc_im1 = vfmaq_f32(acc_im1, va1.val[0], vb1.val[1]);
        acc_im1 = vfmsq_f32(acc_im1, va1.val[1], vb1.val[0]);

        offset += 8;
    }

    let chunks4 = (len - offset) / 4;
    for _ in 0..chunks4 {
        let va0 = vld2q_f32(a_ptr.add(offset * 2));
        let vb0 = vld2q_f32(b_ptr.add(offset * 2));

        acc_re0 = vfmaq_f32(acc_re0, va0.val[0], vb0.val[0]);
        acc_re0 = vfmaq_f32(acc_re0, va0.val[1], vb0.val[1]);
        acc_im0 = vfmaq_f32(acc_im0, va0.val[0], vb0.val[1]);
        acc_im0 = vfmsq_f32(acc_im0, va0.val[1], vb0.val[0]);

        offset += 4;
    }

    let sum_re = vaddvq_f32(vaddq_f32(acc_re0, acc_re1));
    let sum_im = vaddvq_f32(vaddq_f32(acc_im0, acc_im1));

    let mut result_re = sum_re;
    let mut result_im = sum_im;

    for i in offset..len {
        let a_z = a[i];
        let b_z = b[i];
        result_re += a_z.re * b_z.re + a_z.im * b_z.im;
        result_im += a_z.re * b_z.im - a_z.im * b_z.re;
    }

    Complex32::new(result_re, result_im)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_complex_avx2_dual(a: &[Complex32], b: &[Complex32]) -> Complex32 {
    use core::arch::x86_64::*;

    let len = a.len().min(b.len());
    let a_ptr = a.as_ptr() as *const f32;
    let b_ptr = b.as_ptr() as *const f32;
    let float_len = len * 2;

    let mut acc_re0 = _mm256_setzero_ps();
    let mut acc_im0 = _mm256_setzero_ps();
    let mut acc_re1 = _mm256_setzero_ps();
    let mut acc_im1 = _mm256_setzero_ps();
    let mut acc_re2 = _mm256_setzero_ps();
    let mut acc_im2 = _mm256_setzero_ps();
    let mut acc_re3 = _mm256_setzero_ps();
    let mut acc_im3 = _mm256_setzero_ps();

    let chunks32 = float_len / 32;
    let mut offset = 0;

    for _ in 0..chunks32 {
        let va0 = _mm256_loadu_ps(a_ptr.add(offset));
        let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
        let va1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(offset + 8));
        let va2 = _mm256_loadu_ps(a_ptr.add(offset + 16));
        let vb2 = _mm256_loadu_ps(b_ptr.add(offset + 16));
        let va3 = _mm256_loadu_ps(a_ptr.add(offset + 24));
        let vb3 = _mm256_loadu_ps(b_ptr.add(offset + 24));

        acc_re0 = _mm256_fmadd_ps(va0, vb0, acc_re0);
        let vb0_s = _mm256_permute_ps(vb0, 0b10_11_00_01);
        acc_im0 = _mm256_fmadd_ps(va0, vb0_s, acc_im0);

        acc_re1 = _mm256_fmadd_ps(va1, vb1, acc_re1);
        let vb1_s = _mm256_permute_ps(vb1, 0b10_11_00_01);
        acc_im1 = _mm256_fmadd_ps(va1, vb1_s, acc_im1);

        acc_re2 = _mm256_fmadd_ps(va2, vb2, acc_re2);
        let vb2_s = _mm256_permute_ps(vb2, 0b10_11_00_01);
        acc_im2 = _mm256_fmadd_ps(va2, vb2_s, acc_im2);

        acc_re3 = _mm256_fmadd_ps(va3, vb3, acc_re3);
        let vb3_s = _mm256_permute_ps(vb3, 0b10_11_00_01);
        acc_im3 = _mm256_fmadd_ps(va3, vb3_s, acc_im3);

        offset += 32;
    }

    let chunks16 = (float_len - offset) / 16;
    for _ in 0..chunks16 {
        let va0 = _mm256_loadu_ps(a_ptr.add(offset));
        let vb0 = _mm256_loadu_ps(b_ptr.add(offset));
        let va1 = _mm256_loadu_ps(a_ptr.add(offset + 8));
        let vb1 = _mm256_loadu_ps(b_ptr.add(offset + 8));

        acc_re0 = _mm256_fmadd_ps(va0, vb0, acc_re0);
        let vb0_s = _mm256_permute_ps(vb0, 0b10_11_00_01);
        acc_im0 = _mm256_fmadd_ps(va0, vb0_s, acc_im0);

        acc_re1 = _mm256_fmadd_ps(va1, vb1, acc_re1);
        let vb1_s = _mm256_permute_ps(vb1, 0b10_11_00_01);
        acc_im1 = _mm256_fmadd_ps(va1, vb1_s, acc_im1);

        offset += 16;
    }

    let acc_re_a = _mm256_add_ps(acc_re0, acc_re1);
    let acc_re_b = _mm256_add_ps(acc_re2, acc_re3);
    let acc_re = _mm256_add_ps(acc_re_a, acc_re_b);

    let acc_im_a = _mm256_add_ps(acc_im0, acc_im1);
    let acc_im_b = _mm256_add_ps(acc_im2, acc_im3);
    let acc_im = _mm256_add_ps(acc_im_a, acc_im_b);

    let mut re_arr = [0.0f32; 8];
    let mut im_arr = [0.0f32; 8];
    _mm256_storeu_ps(re_arr.as_mut_ptr(), acc_re);
    _mm256_storeu_ps(im_arr.as_mut_ptr(), acc_im);

    let mut sum_re = (re_arr[0] + re_arr[1])
        + (re_arr[2] + re_arr[3])
        + (re_arr[4] + re_arr[5])
        + (re_arr[6] + re_arr[7]);
    let mut sum_im = (im_arr[0] - im_arr[1])
        + (im_arr[2] - im_arr[3])
        + (im_arr[4] - im_arr[5])
        + (im_arr[6] - im_arr[7]);

    for i in (offset / 2)..len {
        let za = *a.get_unchecked(i);
        let zb = *b.get_unchecked(i);
        sum_re += za.re * zb.re + za.im * zb.im;
        sum_im += za.re * zb.im - za.im * zb.re;
    }

    Complex32::new(sum_re, sum_im)
}

#[inline]
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dot_product_complex_scalar_unrolled(a: &[Complex32], b: &[Complex32]) -> Complex32 {
    let len = a.len().min(b.len());
    let a_slice = &a[..len];
    let b_slice = &b[..len];

    let mut acc_re = 0.0f32;
    let mut acc_im = 0.0f32;

    let chunks_a = a_slice.chunks_exact(4);
    let chunks_b = b_slice.chunks_exact(4);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        acc_re += ca[0].re * cb[0].re + ca[0].im * cb[0].im;
        acc_im += ca[0].re * cb[0].im - ca[0].im * cb[0].re;

        acc_re += ca[1].re * cb[1].re + ca[1].im * cb[1].im;
        acc_im += ca[1].re * cb[1].im - ca[1].im * cb[1].re;

        acc_re += ca[2].re * cb[2].re + ca[2].im * cb[2].im;
        acc_im += ca[2].re * cb[2].im - ca[2].im * cb[2].re;

        acc_re += ca[3].re * cb[3].re + ca[3].im * cb[3].im;
        acc_im += ca[3].re * cb[3].im - ca[3].im * cb[3].re;
    }

    for (za, zb) in rem_a.iter().zip(rem_b.iter()) {
        acc_re += za.re * zb.re + za.im * zb.im;
        acc_im += za.re * zb.im - za.im * zb.re;
    }

    Complex32::new(acc_re, acc_im)
}

/// Issues a hardware cache-line prefetch for vector data.
#[inline(always)]
pub fn prefetch_vector(data: &[Complex32]) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
        let ptr = data.as_ptr() as *const i8;
        let bytes = data.len() * std::mem::size_of::<Complex32>();
        let lines = (bytes + 63) / 64;
        for i in 0..lines.min(8) {
            _mm_prefetch(ptr.add(i * 64), _MM_HINT_T0);
        }
    }
}

/// A complex-valued, phase-encoded vector embedding.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorEmbedding {
    data: Vec<Complex32>,
}

impl Debug for VectorEmbedding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorEmbedding")
            .field("dim", &self.data.len())
            .field("norm", &self.norm())
            .field(
                "preview_amplitudes",
                &self
                    .data
                    .iter()
                    .take(4)
                    .map(|c| c.norm())
                    .collect::<Vec<_>>(),
            )
            .field(
                "preview_phases",
                &self
                    .data
                    .iter()
                    .take(4)
                    .map(|c| c.arg())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl VectorEmbedding {
    /// Creates a complex vector embedding from a raw slice of real floats (zero phase).
    pub fn new(data: Vec<f32>) -> Self {
        Self::from_reals(&data)
    }

    /// Constructs a complex vector embedding from real float values ($z_j = x_j + 0i$).
    pub fn from_reals(data: &[f32]) -> Self {
        let complex_data = data.iter().map(|&x| Complex32::new(x, 0.0)).collect();
        Self { data: complex_data }
    }

    /// Constructs a vector embedding directly from complex numbers.
    pub fn from_complex(data: Vec<Complex32>) -> Self {
        Self { data }
    }

    /// Constructs a phase-encoded embedding from separate amplitude and phase arrays.
    pub fn from_amplitudes_and_phases(amplitudes: &[f32], phases: &[f32]) -> Self {
        let dim = amplitudes.len().min(phases.len());
        let data: Vec<Complex32> = (0..dim)
            .map(|i| {
                let r = amplitudes[i];
                let theta = phases[i];
                Complex32::from_polar(r, theta)
            })
            .collect();
        Self { data }
    }

    /// Returns the dimensionality $d$ of the embedding.
    #[inline(always)]
    pub fn dimension(&self) -> usize {
        self.data.len()
    }

    /// Returns a slice of the internal complex components.
    #[inline(always)]
    pub fn complex_data(&self) -> &[Complex32] {
        &self.data
    }

    /// Extracts the amplitudes ($r_j = |z_j|$) of all components.
    pub fn amplitudes(&self) -> Vec<f32> {
        self.data.iter().map(|z| z.norm()).collect()
    }

    /// Extracts the phase angles ($\theta_j = \text{arg}(z_j) \in [-\pi, \pi]$) of all components.
    pub fn phases(&self) -> Vec<f32> {
        self.data.iter().map(|z| z.arg()).collect()
    }

    /// Computes the squared $L_2$ norm $\sum_j |z_j|^2$.
    #[inline]
    pub fn norm_squared(&self) -> f32 {
        let mut sum = 0.0f32;
        let chunks = self.data.chunks_exact(4);
        let remainder = chunks.remainder();

        for chunk in chunks {
            sum += chunk[0].norm_sqr()
                + chunk[1].norm_sqr()
                + chunk[2].norm_sqr()
                + chunk[3].norm_sqr();
        }
        for z in remainder {
            sum += z.norm_sqr();
        }
        sum
    }

    /// Computes the Euclidean norm $||\psi|| = \sqrt{\langle\psi|\psi\rangle}$.
    #[inline(always)]
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Returns a normalized complex vector such that $||\psi|| = 1$.
    pub fn normalize(&self) -> Self {
        Self::from_complex(self.data.clone()).into_normalized()
    }

    /// Normalizes an owned embedding in place, retaining its existing allocation.
    ///
    /// Zero-copy: scales the owned buffer directly; no replacement vector is allocated.
    #[inline]
    pub fn into_normalized(mut self) -> Self {
        for z in &mut self.data {
            if !z.re.is_finite() {
                z.re = 0.0;
            }
            if !z.im.is_finite() {
                z.im = 0.0;
            }
        }
        let n = self.norm();
        if n < 1e-9 || !n.is_finite() {
            self.data.fill(Complex32::new(0.0, 0.0));
            return self;
        }

        let inv_n = 1.0 / n;
        for z in &mut self.data {
            *z *= inv_n;
            if !z.re.is_finite() {
                z.re = 0.0;
            }
            if !z.im.is_finite() {
                z.im = 0.0;
            }
        }
        self
    }

    /// Computes the complex inner product $\langle\psi|\phi\rangle = \sum_j \psi_j^* \phi_j$ with SIMD acceleration.
    #[inline(always)]
    pub fn dot_product_complex(&self, other: &Self) -> Complex32 {
        dot_product_complex_simd(&self.data, &other.data)
    }

    /// Computes the Complex Projective Overlap (CPO) between two complex vectors:
    ///
    /// $$P(z, w) = \frac{|\langle z, w\rangle|^2}{\|z\|^2 \|w\|^2} \in [0, 1]$$
    ///
    /// Preserves global-phase invariance ($P(z, e^{i\theta}w) = P(z, w)$).
    #[inline]
    pub fn projective_overlap(&self, other: &Self) -> SimilarityScore {
        let ip = self.dot_product_complex(other);
        let num = ip.norm_sqr();
        let denom = (self.norm_squared() * other.norm_squared()).max(1e-12);
        (num / denom).clamp(0.0, 1.0)
    }

    /// Computes the Projective Sine Distance:
    ///
    /// $$D(z, w) = \sqrt{1 - P(z, w)} \in [0, 1]$$
    #[inline]
    pub fn projective_sine_distance(&self, other: &Self) -> f32 {
        let p = self.projective_overlap(other);
        (1.0 - p).max(0.0).sqrt()
    }

    /// Computes the Phase-Aligned Chordal Distance:
    ///
    /// $$D(z, w) = \sqrt{2(1 - \sqrt{P(z, w)})}$$
    ///
    /// Note: Mathematically equivalent to the pure-state Bures metric evaluated classically.
    #[inline]
    pub fn phase_aligned_chordal_distance(&self, other: &Self) -> f32 {
        let p = self.projective_overlap(other);
        (2.0 * (1.0 - p.sqrt())).max(0.0).sqrt()
    }

    /// Computes the Euclidean distance over complex vector components.
    #[inline]
    pub fn euclidean_distance(&self, other: &Self) -> f32 {
        let len = self.data.len().min(other.data.len());
        let mut sum = 0.0f32;
        for i in 0..len {
            let diff = self.data[i] - other.data[i];
            sum += diff.norm_sqr();
        }
        sum.sqrt()
    }

    /// Computes the classical Cosine distance $1 - \text{Re}(\langle\psi|\phi\rangle) / (||\psi|| \cdot ||\phi||)$.
    #[inline]
    pub fn cosine_distance(&self, other: &Self) -> f32 {
        let ip = self.dot_product_complex(other);
        let n1 = self.norm();
        let n2 = other.norm();
        if n1 > 1e-9 && n2 > 1e-9 {
            1.0 - (ip.re / (n1 * n2)).clamp(-1.0, 1.0)
        } else {
            1.0
        }
    }
}

/// High-level query execution plan and automatic small-corpus crossover strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchPlan {
    /// Force exact scan across all live nodes (production authoritative default).
    #[default]
    Exact,
    /// Automatically choose between Exact scan and candidate routing based on live corpus size (explicit experimental).
    Auto,
    /// Force Rivero candidate routing (Fast/Balanced/Strict).
    Rivero,
    /// Classical HNSW graph traversal (conventional candidate/visited heaps, no superposition).
    HnswClassical,
    /// HoloGraph Superposition graph traversal.
    GraphSuperposition,
    /// Legacy alias for GraphSuperposition / HnswClassical.
    #[serde(alias = "GraphOnly")]
    GraphOnly,
}

/// Detailed execution work diagnostics for graph search and traversal telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphSearchDiagnostics {
    pub visited_nodes: u64,
    pub distance_evaluations: u64,
    pub edges_examined: u64,
    pub candidate_pushes: u64,
    pub candidate_pops: u64,
    pub exact_reranks: u64,
}

/// Distance metric used for vector similarity comparison in the HNSQR graph.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceFunction {
    /// Complex Cosine / Folded Hermitian Similarity: $\text{Re}(\langle\psi|\phi\rangle) / (||\psi|| \cdot ||\phi||)$.
    #[default]
    Cosine,
    /// Complex Projective Overlap (CPO): $P(\psi, \phi) = |\langle\psi|\phi\rangle|^2 / (||\psi||^2 ||\phi||^2)$ (Similarity $\in [0, 1]$).
    ProjectiveOverlap,
    /// Projective Sine Distance: $D(\psi, \phi) = \sqrt{1 - P(\psi, \phi)}$ (Distance $\in [0, 1]$).
    ProjectiveSineDistance,
    /// Phase-Aligned Chordal Distance: $D(\psi, \phi) = \sqrt{2(1 - \sqrt{P(\psi, \phi)})}$.
    /// Note: Chordal projective metric on the complex projective manifold.
    PhaseAlignedChordalDistance,
    /// Complex Euclidean Distance.
    Euclidean,
}

// ════════════════════════════════════════════════════════════════════════════════
// 2. CONFIGURATION & SEARCH INTENT
// ════════════════════════════════════════════════════════════════════════════════

/// Configuration parameters for the HNSQR index.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HNSQRConfig {
    /// Maximum connections per node at layer 0 (base graph). Default: 64.
    pub m0: usize,
    /// Maximum connections per node at layers > 0. Default: 32.
    pub m: usize,
    /// Beam size of the dynamic candidate list during graph construction. Default: 128.
    pub ef_construction: usize,
    /// Beam size of the candidate queue during nearest neighbor search. Default: 64.
    pub ef_search: usize,
    /// Exponential level multiplier $m_L = 1 / \ln(M)$. Default: $\approx 0.36067$.
    pub level_multiplier: f32,
    /// Maximum number of nodes the index arena is bounded to. Default: 100,000.
    pub max_elements: usize,
    /// Distance and similarity metric. Default: ProjectiveOverlap.
    pub distance_function: DistanceFunction,
    /// Execution plan and automatic small-corpus crossover strategy. Default: Auto.
    pub search_plan: SearchPlan,
    /// Maximum live corpus size where exact scan is chosen automatically under `SearchPlan::Auto`. Default: 2000.
    pub exact_scan_threshold: usize,
    /// Neural attention beam width for superposition traversal. Default: 8.
    pub superposition_beam_width: usize,
    /// Softmax temperature $\tau$ for attention weight distribution. Default: 0.15.
    pub attention_temperature: f32,
    /// Weight $\lambda \in [0, 1]$ given to beam interference vs query projective overlap. Default: 0.35.
    pub interference_weight: f32,
    /// Maximum search queue exploration limit. Default: 2,000.
    pub search_queue_size: usize,
    /// Candidate over-sampling factor $\alpha$ for exact rescoring. Default: 3.0.
    pub oversample_factor: f32,
    /// Enables Heuristic Diverse Edge Selection (Algorithm 4). Default: true.
    pub heuristic_edge_selection: bool,
    /// Maximum number of diverse roots in the top-layer ensemble. Default: 4.
    pub multi_root_ensemble_size: usize,
    /// Whether to consider 2-hop neighbors during heuristic edge selection. Default: true.
    pub extend_candidates: bool,
    /// Whether to retain pruned connections if heuristic selection produces fewer than M edges. Default: true.
    pub keep_pruned_connections: bool,
    /// Use Rayon parallel construction in batch operations. Default: true.
    pub use_parallel_construction: bool,
    /// Enables 8-bit Phase Quantization (PQ-C) reducing vector memory footprint by 4x. Default: false.
    pub quantization_enabled: bool,
    /// Optional file path for the memory-mapped quantized vector mirror. Default: None.
    pub mmap_path: Option<String>,
    /// Enables fixed-budget Rivero address resolution as the primary search path.
    pub rivero_enabled: bool,
    /// Operational search mode (Strict, Adaptive, or GraphOnly). Default: Adaptive.
    pub rivero_mode: RiveroSearchMode,
    /// Fallback policy for adaptive queries (RiveroOnly or AllowGraphFallback). Default: RiveroOnly.
    pub adaptive_policy: AdaptivePolicy,
    /// Maximum residents read from each Rivero territory cell. Default: 16.
    pub rivero_cell_budget: usize,
    /// Full configuration profile for Rivero candidate routing and Pareto optimization.
    pub rivero_config: RiveroConfig,
    /// Maximum deterministic layer-zero witnesses retained per strict node. Default: 64.
    pub rivero_witness_degree: usize,
    /// Maximum exactly-ranked Rivero seeds expanded during strict search. Default: 48.
    pub rivero_witness_seeds: usize,
    /// Maximum new first-hop witnesses expanded for one bounded second hop. Default: 16.
    pub rivero_witness_second_seeds: usize,
    /// Address geometry and multi-lane projection architecture. Default: 24-foundation GlobalMix.
    pub rivero_address_config: RiveroAddressConfig,
    /// Falls back to graph traversal when Rivero cannot fill the requested `k`. Default: true.
    #[serde(alias = "rivero_fallback_on_empty")]
    pub rivero_fallback_on_underfill: bool,
    /// Optional hard deadline for `Certified` proof search in milliseconds.
    ///
    /// When set, `GlobalExactProofSearch` will abort the branch-and-bound loop if elapsed
    /// time exceeds this budget and return the best-known Top-K candidates found so far,
    /// with `DenseExactProof::globally_exact = false` to signal an incomplete proof.
    ///
    /// `None` (default) means no deadline: the engine always produces a complete proof.
    /// Setting this is the recommended mitigation for P99 tail-latency expansion on
    /// high-entropy isotropic query distributions.
    pub certified_query_timeout_ms: Option<u64>,
    /// Optional deterministic construction seed for reproducible graph level generation.
    pub construction_seed: Option<u64>,
}

impl Default for HNSQRConfig {
    fn default() -> Self {
        let m = 32;
        Self {
            m0: 64,
            m,
            ef_construction: 128,
            ef_search: 64,
            level_multiplier: 1.0 / (m as f32).ln().max(1.0),
            max_elements: 100_000,
            distance_function: DistanceFunction::Cosine,
            search_plan: SearchPlan::Exact,
            exact_scan_threshold: 0,
            superposition_beam_width: 8,
            attention_temperature: 0.15,
            interference_weight: 0.35,
            search_queue_size: 2000,
            oversample_factor: 3.0,
            heuristic_edge_selection: true,
            multi_root_ensemble_size: 4,
            extend_candidates: true,
            keep_pruned_connections: true,
            use_parallel_construction: true,
            quantization_enabled: false,
            mmap_path: None,
            rivero_enabled: false,
            rivero_mode: RiveroSearchMode::Adaptive,
            adaptive_policy: AdaptivePolicy::RiveroOnly,
            rivero_cell_budget: 16,
            rivero_config: RiveroConfig::strict_default(),
            rivero_witness_degree: RIVERO_WITNESS_DEFAULT_DEGREE,
            rivero_witness_seeds: RIVERO_WITNESS_DEFAULT_SEEDS,
            rivero_witness_second_seeds: RIVERO_WITNESS_DEFAULT_SECOND_SEEDS,
            rivero_address_config: RiveroAddressConfig::default(),
            rivero_fallback_on_underfill: true,
            certified_query_timeout_ms: None,
            construction_seed: None,
        }
    }
}

impl HNSQRConfig {
    /// Constructs an adaptive configuration optimized for dimensionality and hardware efficiency.
    pub fn adaptive_for_dim(dimension: usize) -> Self {
        let mut config = Self::default();
        if dimension >= 512 {
            config.ef_construction = 64;
            config.ef_search = 32;
            config.extend_candidates = false;
            config.m0 = 32;
            config.m = 16;
            config.superposition_beam_width = 4;
        } else if dimension >= 128 {
            config.ef_construction = 96;
            config.ef_search = 48;
            config.extend_candidates = false;
            config.m0 = 48;
            config.m = 24;
            config.superposition_beam_width = 6;
        }
        config
    }

    /// Constructs a graphless strict Rivero configuration for fixed-budget serving.
    ///
    /// Searches in this mode never fall back to HNSW. Resolution is constant in
    /// corpus size; address compilation and bounded exact reranking remain linear
    /// in the embedding dimension.
    pub fn strict_rivero_for_dim(dimension: usize) -> Self {
        let mut config = Self::adaptive_for_dim(dimension);
        config.rivero_enabled = true;
        config.rivero_fallback_on_underfill = false;
        config
    }
}

/// Dynamic contextual intent applied during search and complex phase re-ranking.
#[derive(Clone, Default)]
pub struct SearchIntent {
    /// Recency bias factor $[0, 1]$ prioritizing freshly inserted nodes. Default: 0.0.
    pub recency_bias: f32,
    /// Diversity penalty $[0, 1]$ repelling candidates similar to already selected results. Default: 0.0.
    pub diversity: f32,
    /// Precision vs Recall trade-off modulation $[0, 1]$. Default: 0.5.
    pub recall_precision_balance: f32,
    /// Multiplier scaling the phase alignment influence $[0, 1]$. Default: 0.0.
    pub phase_alignment_weight: f32,
    /// Focus width for softmax attention distribution. Default: 0.5.
    pub attention_width: f32,
    /// Relative compute budget multiplier (e.g. 1.0 = normal, 2.0 = double ef). Default: 1.0.
    pub compute_budget: f32,
    /// Structured boolean filter expression compiled to a RoaringBitmap.
    pub filter: Option<FilterExpr>,
    /// Pre-compiled roaring bitmap mask for fast graph pruning.
    pub filter_mask: Option<Arc<roaring::RoaringBitmap>>,
    /// Exact match key-value attributes compiled to a RoaringBitmap mask before search.
    pub exact_matches: HashMap<String, String>,
    /// Optional XyCo 8D Affective State Tensor for somatic and blast-radius gated planning.
    pub affect: Option<crate::planning::affect::AffectiveStateTensor8D>,
}

// ════════════════════════════════════════════════════════════════════════════════
// 3. LOCK-FREE ARENA & FINE-GRAINED NODE CONNECTIONS
// ════════════════════════════════════════════════════════════════════════════════

/// A graph vertex stored in the concurrent arena.
pub struct Node {
    /// Internal contiguous integer index in the arena.
    pub index: NodeIndex,
    /// External identifier provided by the caller (UUID / string key).
    /// `Arc<str>` is shared with `id_to_index` — one allocation, two O(1) references.
    pub external_id: Arc<str>,
    /// Maximum level assigned to this node in the hierarchy.
    pub level: usize,
    /// Creation timestamp (seconds since Unix Epoch).
    pub created_at: u64,
    /// Timestamp of last read/search access.
    pub last_accessed: AtomicU64,
    /// Number of times this node has been traversed or retrieved.
    pub access_count: AtomicUsize,
    /// Optional arbitrary JSON metadata.
    pub metadata: Option<serde_json::Value>,
    /// Fine-grained connection arrays per layer.
    pub layers: Box<[RwLock<SmallVec<[NodeIndex; 64]>>]>,
    /// Exact-sorted bounded witnesses used only by strict Rivero routing.
    rivero_witnesses:
        RwLock<SmallVec<[rivero_witness::ScoredWitness; RIVERO_WITNESS_INLINE_DEGREE]>>,
}

unsafe impl Sync for Node {}
unsafe impl Send for Node {}

impl Node {
    /// Creates a new node instance for insertion into the arena.
    pub fn new(index: NodeIndex, external_id: Arc<str>, level: usize) -> Self {
        let mut layers = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            layers.push(RwLock::new(SmallVec::new()));
        }
        let now = current_unix_timestamp();
        Self {
            index,
            external_id,
            level,
            created_at: now,
            last_accessed: AtomicU64::new(now),
            access_count: AtomicUsize::new(0),
            metadata: None,
            layers: layers.into_boxed_slice(),
            rivero_witnesses: RwLock::new(SmallVec::new()),
        }
    }

    /// Appends an edge to `neighbor` at `level` under fine-grained per-layer lock.
    #[inline(always)]
    pub fn add_connection(&self, neighbor: NodeIndex, level: usize) {
        if level < self.layers.len() {
            let mut conns = self.layers[level].write();
            if !conns.contains(&neighbor) {
                conns.push(neighbor);
            }
        }
    }

    /// Reads connections at `level` into a stack buffer.
    #[inline(always)]
    pub fn connections_at(&self, level: usize, out: &mut SmallVec<[NodeIndex; 64]>) {
        if level < self.layers.len() {
            let conns = self.layers[level].read();
            out.extend_from_slice(&conns);
        }
    }

    /// Clones connections at `level`.
    #[inline(always)]
    pub fn get_connections_clone(&self, level: usize) -> SmallVec<[NodeIndex; 64]> {
        if level < self.layers.len() {
            self.layers[level].read().clone()
        } else {
            SmallVec::new()
        }
    }

    /// Sets connections at `level` atomically under layer lock.
    #[inline(always)]
    pub fn set_connections(&self, level: usize, new_conns: SmallVec<[NodeIndex; 64]>) {
        if level < self.layers.len() {
            *self.layers[level].write() = new_conns;
        }
    }
}

/// True Lock-Free Pre-Allocated Concurrent Arena.
pub struct ConcurrentArena {
    len: AtomicUsize,
    live_count: AtomicUsize,
    max_capacity: usize,
    dimension: usize,
    nodes: Box<[UnsafeCell<Option<Node>>]>,
    vectors: Box<[UnsafeCell<Complex32>]>,
    norms_sq: Box<[AtomicU32]>,
    slot_states: Box<[AtomicU32]>,
}

const SLOT_EMPTY: u32 = 0;
const SLOT_WRITING: u32 = 1;
const SLOT_LIVE: u32 = 2;
const SLOT_DELETED: u32 = 3;

unsafe impl Sync for ConcurrentArena {}
unsafe impl Send for ConcurrentArena {}

impl ConcurrentArena {
    /// Pre-allocates the concurrent arena for `max_capacity` elements.
    pub fn new(max_capacity: usize, dimension: usize) -> Self {
        let mut nodes = Vec::with_capacity(max_capacity);
        let mut norms = Vec::with_capacity(max_capacity);
        let mut slot_states = Vec::with_capacity(max_capacity);
        for _ in 0..max_capacity {
            nodes.push(UnsafeCell::new(None));
            norms.push(AtomicU32::new(0));
            slot_states.push(AtomicU32::new(SLOT_EMPTY));
        }

        let total_complex_elements = max_capacity * dimension;
        let mut vectors = Vec::with_capacity(total_complex_elements);
        for _ in 0..total_complex_elements {
            vectors.push(UnsafeCell::new(Complex32::new(0.0, 0.0)));
        }

        Self {
            len: AtomicUsize::new(0),
            live_count: AtomicUsize::new(0),
            max_capacity,
            dimension,
            nodes: nodes.into_boxed_slice(),
            vectors: vectors.into_boxed_slice(),
            norms_sq: norms.into_boxed_slice(),
            slot_states: slot_states.into_boxed_slice(),
        }
    }

    /// Claims a new slot in O(1) lock-free.
    #[inline(always)]
    pub fn claim_slot(&self) -> HNSQRResult<NodeIndex> {
        let index = self.len.fetch_add(1, AtomicOrdering::Relaxed);
        if index >= self.max_capacity {
            self.len.fetch_sub(1, AtomicOrdering::Relaxed);
            return Err(HNSQRError::IndexFull(self.max_capacity));
        }
        self.slot_states[index].store(SLOT_WRITING, AtomicOrdering::Release);
        Ok(index as NodeIndex)
    }

    /// Publishes a fully initialized slot to concurrent readers.
    #[inline(always)]
    pub fn publish_slot(&self, index: NodeIndex) {
        let previous = self.slot_states[index as usize].swap(SLOT_LIVE, AtomicOrdering::Release);
        if previous != SLOT_LIVE {
            self.live_count.fetch_add(1, AtomicOrdering::Relaxed);
        }
    }

    /// Marks a slot deleted before its routing entries are removed.
    #[inline(always)]
    pub fn delete_slot(&self, index: NodeIndex) {
        let previous = self.slot_states[index as usize].swap(SLOT_DELETED, AtomicOrdering::Release);
        if previous == SLOT_LIVE {
            self.live_count.fetch_sub(1, AtomicOrdering::Relaxed);
        }
    }

    /// Returns whether a slot is fully initialized and visible.
    #[inline(always)]
    pub fn is_live(&self, index: NodeIndex) -> bool {
        let idx = index as usize;
        idx < self.len() && self.slot_states[idx].load(AtomicOrdering::Acquire) == SLOT_LIVE
    }

    /// Writes vector data and cached norm into the claimed slot.
    #[inline(always)]
    pub fn write_vector(&self, index: NodeIndex, data: &[Complex32]) {
        let idx = index as usize;
        let offset = idx * self.dimension;
        let len = data.len().min(self.dimension);

        let mut sum_norm_sq = 0.0f32;
        unsafe {
            for i in 0..len {
                let val = *data.get_unchecked(i);
                sum_norm_sq += val.norm_sqr();
                *self.vectors.get_unchecked(offset + i).get() = val;
            }
        }
        self.norms_sq[idx].store(sum_norm_sq.to_bits(), AtomicOrdering::Release);
    }

    /// Writes node struct into the claimed slot.
    #[inline(always)]
    pub fn write_node(&self, index: NodeIndex, node: Node) {
        unsafe {
            *self.nodes[index as usize].get() = Some(node);
        }
    }

    /// Returns the active number of allocated nodes.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len.load(AtomicOrdering::Acquire)
    }

    /// Returns the number of published, non-deleted nodes.
    #[inline(always)]
    pub fn live_len(&self) -> usize {
        self.live_count.load(AtomicOrdering::Acquire)
    }

    /// Returns true if the arena contains no allocated nodes.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.live_len() == 0
    }

    /// Returns a lock-free direct slice to the vector data of node `index`.
    #[inline(always)]
    pub fn get_vector_slice(&self, index: NodeIndex) -> &[Complex32] {
        let idx = index as usize;
        let offset = idx * self.dimension;
        unsafe {
            let ptr = self.vectors.get_unchecked(offset).get() as *const Complex32;
            std::slice::from_raw_parts(ptr, self.dimension)
        }
    }

    /// Returns the pre-computed squared norm for node `index`.
    #[inline(always)]
    pub fn get_norm_squared(&self, index: NodeIndex) -> f32 {
        f32::from_bits(self.norms_sq[index as usize].load(AtomicOrdering::Relaxed))
    }

    /// Returns a direct lock-free reference to the node at `index`.
    #[inline(always)]
    pub fn get_node(&self, index: NodeIndex) -> Option<&Node> {
        let idx = index as usize;
        if self.is_live(index) {
            unsafe { (*self.nodes.get_unchecked(idx).get()).as_ref() }
        } else {
            None
        }
    }

    /// Returns an initialized claimed slot even before publication. This is only
    /// for completing insertion-time reciprocal wiring; query paths must use
    /// `get_node`, which enforces liveness/publication.
    #[inline(always)]
    fn get_claimed_node(&self, index: NodeIndex) -> Option<&Node> {
        let idx = index as usize;
        if idx < self.len() {
            unsafe { (*self.nodes.get_unchecked(idx).get()).as_ref() }
        } else {
            None
        }
    }

    /// Clears all slots in the arena.
    pub fn clear(&self) {
        let count = self.len.swap(0, AtomicOrdering::SeqCst);
        self.live_count.store(0, AtomicOrdering::Release);
        for i in 0..count.min(self.max_capacity) {
            unsafe {
                *self.nodes[i].get() = None;
            }
            self.norms_sq[i].store(0, AtomicOrdering::Relaxed);
            self.slot_states[i].store(SLOT_EMPTY, AtomicOrdering::Release);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 4. PRIORITY QUEUE CANDIDATES FOR TRAVERSAL
// ════════════════════════════════════════════════════════════════════════════════

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    index: NodeIndex,
    similarity: SimilarityScore,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.similarity
            .partial_cmp(&other.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Min-heap candidate wrapper (lowest similarity has highest priority for eviction).
#[derive(Clone, Copy, Debug, PartialEq)]
struct WorstResultCandidate(Candidate);

impl Eq for WorstResultCandidate {}

impl Ord for WorstResultCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .similarity
            .partial_cmp(&self.0.similarity)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for WorstResultCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 5. INDEX METRICS & INSTRUMENTATION
// ════════════════════════════════════════════════════════════════════════════════

/// Operational and performance statistics for the index.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    /// Total number of nodes successfully inserted.
    pub insertions: usize,
    /// Total number of search queries processed.
    pub searches: usize,
    /// Total intent search operations executed.
    pub intent_searches: usize,
    /// Average query search latency in microseconds.
    pub avg_search_latency_us: f64,
    /// Peak concurrent search requests observed.
    pub peak_concurrent_searches: usize,
    /// Searches served by the fixed-budget Rivero path.
    pub rivero_searches: usize,
    /// Rivero searches that had to fall back to graph traversal.
    pub rivero_fallbacks: usize,
    /// Most candidates exactly reranked by any Rivero query.
    pub rivero_peak_candidates: usize,
    /// Total fixed territory-cell probes performed by Rivero searches.
    pub rivero_cells_probed: u64,
    /// Total bounded cell residents read before candidate deduplication.
    pub rivero_resident_reads: u64,
    /// Total compact resident records inspected before query-adaptive admission.
    pub rivero_resident_scans: u64,
    /// Total fixed-degree witness edges inspected by Rivero searches.
    pub rivero_witness_edges_scanned: u64,
    /// Total previously unseen candidates admitted through witness expansion.
    pub rivero_witness_candidates_added: u64,
    /// Total full-dimensional exact score evaluations after routing and filters.
    pub rivero_exact_score_evaluations: u64,
    /// Rivero routes that produced no eligible candidate.
    pub rivero_empty_routes: usize,
    /// Current number of populated territorial cells.
    pub rivero_cell_count: usize,
    /// Cells that have reached the configured fixed resident capacity.
    pub rivero_overflowed_cells: u64,
}

/// Per-query proof counters for strict Rivero resolution.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RiveroSearchDiagnostics {
    /// Exact number of fixed territory cells inspected.
    pub cells_probed: usize,
    /// Total cell residents read before deduplication.
    pub resident_reads: usize,
    /// Compact resident codes inspected before query-adaptive admission.
    pub resident_scans: usize,
    /// Hard configured ceiling on resident reads.
    pub candidate_read_bound: usize,
    /// Hard configured ceiling on compact resident scans.
    pub resident_scan_bound: usize,
    /// Distinct arena slots returned by the route.
    pub unique_candidates: usize,
    /// Distinct route slots before global collision-vote selection.
    pub raw_unique_candidates: usize,
    /// Vote-selected route candidates before witness expansion.
    pub route_candidates_selected: usize,
    /// Hard ceiling on raw distinct route candidates.
    pub raw_unique_candidate_bound: usize,
    /// Hard ceiling on vote-selected route candidates.
    pub selected_candidate_bound: usize,
    /// Deleted or unpublished slots rejected before vector access.
    pub non_live_rejections: usize,
    /// Live candidates rejected by the precompiled metadata mask.
    pub filter_rejections: usize,
    /// Full-dimensional exact similarity evaluations performed.
    pub exact_score_evaluations: usize,
    /// Rivero candidates used as fixed witness-expansion seeds.
    pub witness_seeds: usize,
    /// Newly discovered first-hop candidates used as bounded second-hop seeds.
    pub witness_second_hop_seeds: usize,
    /// Fixed-degree witness edges inspected from those seeds.
    pub witness_edges_scanned: usize,
    /// Previously unseen candidates admitted through witness expansion.
    pub witness_candidates_added: usize,
    /// Hard configured ceiling on witness edges inspected.
    pub witness_edge_scan_bound: usize,
    /// Results returned after bounded ranking.
    pub results_returned: usize,
    /// Whether an unbounded graph fallback was used.
    pub fallback_used: bool,
}

/// Comprehensive telemetry and diagnostics for staged confidence-adaptive queries.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq)]
pub struct AdaptiveSearchDiagnostics {
    /// Initial profile attempted (e.g. Fast).
    pub initial_profile: RiveroProfile,
    /// Final profile reached upon acceptance or exhaustion.
    pub final_profile: RiveroProfile,
    /// Number of distinct Rivero routing stages executed.
    pub stages_executed: u8,
    /// Composite confidence score at stage 0.
    pub confidence_initial: f32,
    /// Composite confidence score upon acceptance.
    pub confidence_final: f32,
    /// Whether stage escalation occurred.
    pub escalated: bool,
    /// Whether unbounded graph fallback was triggered.
    pub graph_fallback_used: bool,
    /// Cumulative compact resident codes scanned across all stages.
    pub cumulative_resident_scans: usize,
    /// Cumulative exact vector score evaluations performed by Rivero stages before any graph fallback.
    pub cumulative_exact_scores: usize,
    /// Detailed confidence sub-metrics at final stage.
    pub confidence: RiveroConfidence,
    /// Final Rivero search diagnostics.
    pub rivero: RiveroSearchDiagnostics,
}

// ════════════════════════════════════════════════════════════════════════════════
// 6. MAIN HNSQR INDEX IMPLEMENTATION
// ════════════════════════════════════════════════════════════════════════════════

/// Hierarchical Navigable Semantic Query Resolver (HNSQR) Vector Index.
pub struct HNSQRIndex {
    config: RwLock<HNSQRConfig>,
    dimension: Dimension,
    arena: ConcurrentArena,
    mmap_arena: Option<Arc<MmapArena>>,
    metadata_index: MetadataInvertedIndex,
    rivero_index: RiveroTerritoryIndex,
    rivero_compiler: RwLock<rivero::RiveroCompiler>,
    /// String-to-index lookup. Key is `Arc<str>` shared with `Node::external_id`:
    /// `HashMap<Arc<str>, NodeIndex>` supports zero-copy `&str` lookups via `Borrow<str>`.
    id_to_index: RwLock<HashMap<Arc<str>, NodeIndex>>,
    layers: Box<[RwLock<Vec<NodeIndex>>]>,
    entry_points: RwLock<SmallVec<[NodeIndex; 8]>>,
    max_level: AtomicUsize,
    is_optimizing: AtomicBool,
    active_searches: AtomicUsize,
    peak_active_searches: AtomicUsize,
    stats: RwLock<IndexStats>,
    lifecycle: RwLock<()>,
    lutz_codes: RwLock<Vec<Option<crate::proof::lutz::LutzCode>>>,
    proof_tree: RwLock<Option<Arc<SemanticProofTree>>>,
    wal: RwLock<Option<Arc<crate::storage::wal::WalManager>>>,
    wal_durability: RwLock<crate::storage::wal::DurabilityPolicy>,
}

impl HNSQRIndex {
    /// Creates a new in-memory HNSQR index configured for the given dimensionality.
    pub fn new(config: HNSQRConfig, dimension: Dimension) -> Self {
        let max_capacity = config.max_elements.max(1000);
        let mut layers = Vec::with_capacity(32);
        for _ in 0..32 {
            layers.push(RwLock::new(Vec::new()));
        }

        let rivero_compiler =
            rivero::RiveroCompiler::with_config(dimension, config.rivero_address_config);

        Self {
            config: RwLock::new(config),
            dimension,
            arena: ConcurrentArena::new(max_capacity, dimension),
            mmap_arena: None,
            metadata_index: MetadataInvertedIndex::new(),
            rivero_index: RiveroTerritoryIndex::new(),
            rivero_compiler: RwLock::new(rivero_compiler),
            id_to_index: RwLock::new(HashMap::new()),
            layers: layers.into_boxed_slice(),
            entry_points: RwLock::new(SmallVec::new()),
            max_level: AtomicUsize::new(0),
            is_optimizing: AtomicBool::new(false),
            active_searches: AtomicUsize::new(0),
            peak_active_searches: AtomicUsize::new(0),
            stats: RwLock::new(IndexStats::default()),
            lifecycle: RwLock::new(()),
            lutz_codes: RwLock::new(Vec::with_capacity(max_capacity)),
            proof_tree: RwLock::new(None),
            wal: RwLock::new(None),
            wal_durability: RwLock::new(crate::storage::wal::DurabilityPolicy::WalSync),
        }
    }

    /// Attaches a crash-safe Write-Ahead Log (WAL) to the index.
    pub fn with_wal<P: AsRef<std::path::Path>>(
        self,
        wal_dir: P,
        durability: crate::storage::wal::DurabilityPolicy,
    ) -> HNSQRResult<Self> {
        let wal = crate::storage::wal::WalManager::open(wal_dir)?;
        *self.wal.write() = Some(Arc::new(wal));
        *self.wal_durability.write() = durability;
        Ok(self)
    }

    /// Replays uncommitted WAL records from disk to restore state after crash.
    pub fn recover_from_wal(&self) -> HNSQRResult<usize> {
        let wal_opt = self.wal.read().clone();
        if let Some(wal) = wal_opt {
            let count = std::sync::atomic::AtomicUsize::new(0);
            wal.replay(0, |_lsn, mutation| {
                match mutation {
                    crate::storage::wal::WalMutation::Upsert {
                        external_id,
                        vector,
                        metadata,
                    } => {
                        let _ = self.remove(&external_id);
                        if let Some(meta) = metadata {
                            let _ = self.insert_with_metadata(external_id.as_str(), vector, meta);
                        } else {
                            let _ = self.insert(external_id.as_str(), vector);
                        }
                    }
                    crate::storage::wal::WalMutation::Delete { external_id } => {
                        let _ = self.remove(&external_id);
                    }
                    _ => {}
                }
                count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok(())
            })?;
            Ok(count.load(std::sync::atomic::Ordering::Relaxed))
        } else {
            Ok(0)
        }
    }

    /// Creates an HNSQR index with a memory-mapped quantized vector mirror.
    ///
    /// External IDs, metadata, graph state, and Rivero routes are not yet persisted.
    pub fn create_mmap<P: AsRef<std::path::Path>>(
        path: P,
        mut config: HNSQRConfig,
        dimension: Dimension,
    ) -> HNSQRResult<Self> {
        let max_capacity = config.max_elements.max(1000);
        let mmap = MmapArena::create(&path, max_capacity, dimension)?;
        config.mmap_path = Some(path.as_ref().to_string_lossy().into_owned());
        config.quantization_enabled = true;

        let mut layers = Vec::with_capacity(32);
        for _ in 0..32 {
            layers.push(RwLock::new(Vec::new()));
        }

        let rivero_compiler =
            rivero::RiveroCompiler::with_config(dimension, config.rivero_address_config);

        Ok(Self {
            config: RwLock::new(config),
            dimension,
            arena: ConcurrentArena::new(max_capacity, dimension),
            mmap_arena: Some(Arc::new(mmap)),
            metadata_index: MetadataInvertedIndex::new(),
            rivero_index: RiveroTerritoryIndex::new(),
            rivero_compiler: RwLock::new(rivero_compiler),
            id_to_index: RwLock::new(HashMap::new()),
            layers: layers.into_boxed_slice(),
            entry_points: RwLock::new(SmallVec::new()),
            max_level: AtomicUsize::new(0),
            is_optimizing: AtomicBool::new(false),
            active_searches: AtomicUsize::new(0),
            peak_active_searches: AtomicUsize::new(0),
            stats: RwLock::new(IndexStats::default()),
            lifecycle: RwLock::new(()),
            lutz_codes: RwLock::new(Vec::with_capacity(max_capacity)),
            proof_tree: RwLock::new(None),
            wal: RwLock::new(None),
            wal_durability: RwLock::new(crate::storage::wal::DurabilityPolicy::WalSync),
        })
    }

    /// Attaches an existing quantized mmap file to a new empty in-memory routing index.
    ///
    /// This does not yet reconstruct external IDs, metadata, graph state, or Rivero routes.
    pub fn open_mmap<P: AsRef<std::path::Path>>(path: P) -> HNSQRResult<Self> {
        let mmap = MmapArena::open(&path)?;
        let dim = mmap.dimension();
        let max_cap = mmap.max_capacity();

        let mut config = HNSQRConfig::default();
        config.max_elements = max_cap;
        config.quantization_enabled = true;
        config.mmap_path = Some(path.as_ref().to_string_lossy().into_owned());

        let mut layers = Vec::with_capacity(32);
        for _ in 0..32 {
            layers.push(RwLock::new(Vec::new()));
        }

        let rivero_compiler =
            rivero::RiveroCompiler::with_config(dim, config.rivero_address_config);

        Ok(Self {
            config: RwLock::new(config),
            dimension: dim,
            arena: ConcurrentArena::new(max_cap, dim),
            mmap_arena: Some(Arc::new(mmap)),
            metadata_index: MetadataInvertedIndex::new(),
            rivero_index: RiveroTerritoryIndex::new(),
            rivero_compiler: RwLock::new(rivero_compiler),
            id_to_index: RwLock::new(HashMap::new()),
            layers: layers.into_boxed_slice(),
            entry_points: RwLock::new(SmallVec::new()),
            max_level: AtomicUsize::new(0),
            is_optimizing: AtomicBool::new(false),
            active_searches: AtomicUsize::new(0),
            peak_active_searches: AtomicUsize::new(0),
            stats: RwLock::new(IndexStats::default()),
            lifecycle: RwLock::new(()),
            lutz_codes: RwLock::new(Vec::with_capacity(max_cap)),
            proof_tree: RwLock::new(None),
            wal: RwLock::new(None),
            wal_durability: RwLock::new(crate::storage::wal::DurabilityPolicy::WalSync),
        })
    }

    /// Flushes memory-mapped storage to disk if backed by mmap.
    pub fn flush(&self) -> HNSQRResult<()> {
        if let Some(ref mmap) = self.mmap_arena {
            mmap.flush()?;
        }
        Ok(())
    }

    /// Returns the embedding dimension configured for this index.
    #[inline(always)]
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the number of elements indexed.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.arena.live_len()
    }

    /// Returns `true` if the index contains no nodes.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }

    /// Returns a copy of the index configuration.
    pub fn config(&self) -> HNSQRConfig {
        self.config.read().clone()
    }

    /// Updates the dynamic `ef_search` parameter for subsequent searches.
    pub fn set_ef_search(&self, ef_search: usize) -> HNSQRResult<()> {
        if ef_search == 0 {
            return Err(HNSQRError::InvalidConfig(
                "ef_search must be > 0".to_string(),
            ));
        }
        self.config.write().ef_search = ef_search;
        Ok(())
    }

    /// Returns current operational statistics.
    pub fn stats(&self) -> IndexStats {
        let mut snapshot = self.stats.read().clone();
        snapshot.peak_concurrent_searches = self.peak_active_searches.load(AtomicOrdering::Relaxed);
        snapshot.rivero_cell_count = self.rivero_index.cell_count();
        snapshot.rivero_overflowed_cells = self.rivero_index.overflow_count();
        snapshot
    }

    /// Checks whether an external string ID is present in the index.
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_index.read().contains_key(id)
    }

    /// Returns the distribution of nodes across graph layers.
    pub fn level_distribution(&self) -> Vec<usize> {
        let max_l = self.max_level.load(AtomicOrdering::Acquire);
        (0..=max_l).map(|l| self.layers[l].read().len()).collect()
    }

    // ────────────────────────────────────────────────────────────────────────
    // DISTANCE & FIDELITY CALCULATION
    // ────────────────────────────────────────────────────────────────────────

    #[inline(always)]
    fn similarity_score_slices_with_metric(
        &self,
        q: &[Complex32],
        v: &[Complex32],
        q_norm_sq: f32,
        v_norm_sq: f32,
        dist_fn: DistanceFunction,
    ) -> SimilarityScore {
        let ip = dot_product_complex_simd(q, v);
        let denom = (q_norm_sq * v_norm_sq).max(1e-12);
        match dist_fn {
            DistanceFunction::Cosine => (ip.re / denom.sqrt()).clamp(-1.0, 1.0),
            DistanceFunction::ProjectiveOverlap => (ip.norm_sqr() / denom).clamp(0.0, 1.0),
            DistanceFunction::ProjectiveSineDistance => {
                let p = (ip.norm_sqr() / denom).clamp(0.0, 1.0);
                1.0 - (1.0 - p).max(0.0).sqrt()
            }
            DistanceFunction::PhaseAlignedChordalDistance => {
                let p = (ip.norm_sqr() / denom).clamp(0.0, 1.0);
                2.0 - (2.0 * (1.0 - p.sqrt())).max(0.0).sqrt()
            }
            DistanceFunction::Euclidean => {
                let dist_sq = (q_norm_sq + v_norm_sq - 2.0 * ip.re).max(0.0);
                -dist_sq.sqrt()
            }
        }
    }

    #[inline(always)]
    fn similarity_score_slices(
        &self,
        q: &[Complex32],
        v: &[Complex32],
        q_norm_sq: f32,
        v_norm_sq: f32,
    ) -> SimilarityScore {
        let dist_fn = self.config.read().distance_function;
        self.similarity_score_slices_with_metric(q, v, q_norm_sq, v_norm_sq, dist_fn)
    }

    fn generate_random_level(&self, config: &HNSQRConfig, seed_entropy: u64) -> usize {
        let r: f32 = if let Some(seed) = config.construction_seed {
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            seed.hash(&mut hasher);
            seed_entropy.hash(&mut hasher);
            let h = hasher.finish();
            ((h as f64) / (u64::MAX as f64)).max(1e-9) as f32
        } else {
            let mut rng = thread_rng();
            rng.random::<f32>().max(1e-9)
        };
        let ml = if config.level_multiplier > 0.0 && config.level_multiplier < 10.0 {
            config.level_multiplier
        } else {
            1.0 / (config.m as f32).ln().max(1.0)
        };
        let level = (-r.ln() * ml).floor() as usize;
        level.min(31)
    }

    // ────────────────────────────────────────────────────────────────────────
    // HEURISTIC NEIGHBOR SELECTION (ALGORITHM 4)
    // ────────────────────────────────────────────────────────────────────────

    fn select_neighbors_heuristic(
        &self,
        base_slice: &[Complex32],
        base_norm_sq: f32,
        candidates: Vec<Candidate>,
        m: usize,
        level: usize,
        extend_candidates: bool,
        keep_pruned_connections: bool,
    ) -> Vec<NodeIndex> {
        if candidates.len() <= m && !extend_candidates {
            return candidates.into_iter().map(|c| c.index).collect();
        }

        let mut candidate_pool = candidates;

        if extend_candidates {
            let total_nodes = self.arena.len();
            let mut added = Vec::new();
            let mut temp_conns = SmallVec::<[NodeIndex; 64]>::new();

            THREAD_VISITED_POOL.with(|pool_cell| {
                let mut pool = pool_cell.borrow_mut();
                let epoch = pool.next_epoch(total_nodes);

                for cand in &candidate_pool {
                    pool.mark_visited(cand.index, epoch);
                }

                // Only extend from top candidates to prevent exponential neighbor exploration
                for cand in candidate_pool.iter().take(m) {
                    if let Some(cand_node) = self.arena.get_node(cand.index) {
                        temp_conns.clear();
                        cand_node.connections_at(level, &mut temp_conns);
                        for &neighbor_idx in &temp_conns {
                            if !pool.is_visited(neighbor_idx, epoch) {
                                pool.mark_visited(neighbor_idx, epoch);
                                let n_slice = self.arena.get_vector_slice(neighbor_idx);
                                let n_norm_sq = self.arena.get_norm_squared(neighbor_idx);
                                let sim = self.similarity_score_slices(
                                    base_slice,
                                    n_slice,
                                    base_norm_sq,
                                    n_norm_sq,
                                );
                                added.push(Candidate {
                                    index: neighbor_idx,
                                    similarity: sim,
                                });
                            }
                        }
                    }
                }
            });
            candidate_pool.extend(added);
        }

        candidate_pool.sort_by(|a, b| {
            b.similarity
                .partial_cmp(&a.similarity)
                .unwrap_or(Ordering::Equal)
        });
        if candidate_pool.len() > m * 2 {
            candidate_pool.truncate(m * 2);
        }

        let mut selected: Vec<NodeIndex> = Vec::with_capacity(m);
        let mut discarded: Vec<NodeIndex> = Vec::new();

        for cand in candidate_pool {
            let cand_slice = self.arena.get_vector_slice(cand.index);
            let cand_norm_sq = self.arena.get_norm_squared(cand.index);
            let cand_base_sim = cand.similarity;

            let mut is_diverse = true;
            // Cap diversity comparisons to top 16 selected edges for ultra-fast throughput
            for &sel_idx in selected.iter().take(16) {
                let sel_slice = self.arena.get_vector_slice(sel_idx);
                let sel_norm_sq = self.arena.get_norm_squared(sel_idx);
                let cand_sel_sim =
                    self.similarity_score_slices(cand_slice, sel_slice, cand_norm_sq, sel_norm_sq);

                if cand_sel_sim >= cand_base_sim {
                    is_diverse = false;
                    break;
                }
            }

            if is_diverse {
                selected.push(cand.index);
                if selected.len() >= m {
                    break;
                }
            } else if keep_pruned_connections {
                discarded.push(cand.index);
            }
        }

        if keep_pruned_connections && selected.len() < m {
            for disc_idx in discarded {
                if !selected.contains(&disc_idx) {
                    selected.push(disc_idx);
                    if selected.len() >= m {
                        break;
                    }
                }
            }
        }

        selected
    }

    // ────────────────────────────────────────────────────────────────────────
    // CONCURRENT INSERTION ENGINE
    // ────────────────────────────────────────────────────────────────────────

    /// Inserts a vector embedding with an external string ID into the index.
    pub fn insert(&self, id: impl Into<NodeId>, vector: VectorEmbedding) -> HNSQRResult<NodeIndex> {
        self.insert_slice(id, vector.complex_data(), vector.norm_squared())
    }

    /// Inserts a vector embedding with structured metadata attributes for inverted index filtering.
    pub fn insert_with_metadata(
        &self,
        id: impl Into<NodeId>,
        vector: VectorEmbedding,
        metadata: HashMap<String, MetadataValue>,
    ) -> HNSQRResult<NodeIndex> {
        self.insert_slice_with_metadata(id, vector.complex_data(), vector.norm_squared(), metadata)
    }

    /// Concurrent batch insertion of vectors using Rayon multi-threading.
    pub fn batch_insert(&self, items: &[(String, VectorEmbedding)]) -> HNSQRResult<Vec<NodeIndex>> {
        items
            .par_iter()
            .map(|(id, vector)| {
                self.insert_slice(id.as_str(), vector.complex_data(), vector.norm_squared())
            })
            .collect()
    }

    /// Concurrent batch insertion of vectors with structured metadata using Rayon multi-threading.
    pub fn batch_insert_with_metadata(
        &self,
        items: &[(String, VectorEmbedding, HashMap<String, MetadataValue>)],
    ) -> HNSQRResult<Vec<NodeIndex>> {
        items
            .par_iter()
            .map(|(id, vector, metadata)| {
                self.insert_internal(
                    id.as_str().into(),
                    vector.complex_data(),
                    vector.norm_squared(),
                    None,
                    Some(metadata),
                    None,
                )
            })
            .collect()
    }

    /// Inserts a pre-constructed [`Node`] preserving timestamps and metadata.
    pub fn insert_node(&self, node: Node) -> HNSQRResult<NodeIndex> {
        let vec_slice = self.arena.get_vector_slice(node.index);
        let norm_sq = self.arena.get_norm_squared(node.index);
        let ext_id = node.external_id.clone();
        self.insert_internal(ext_id, vec_slice, norm_sq, Some(node), None, None)
    }

    /// Evaluates a structured [`FilterExpr`] against the inverted index, returning a [`RoaringBitmap`] mask.
    pub fn evaluate_filter(&self, expr: &FilterExpr) -> roaring::RoaringBitmap {
        self.metadata_index.evaluate_filter(expr, self.arena.len())
    }

    /// Freezes Rivero territory stripes into a zero-lock flat open-addressed directory for double-digit microsecond serving.
    pub fn freeze_rivero_routing(&self) {
        self.rivero_index.freeze_flat_table();
    }

    fn select_rivero_witnesses(
        &self,
        data: &[Complex32],
        norm_sq: f32,
        address: &RiveroAddress,
        per_cell_budget: usize,
        degree: usize,
    ) -> SmallVec<[rivero_witness::ScoredWitness; RIVERO_WITNESS_INLINE_DEGREE]> {
        self.rivero_index
            .with_candidates_for_build(address, per_cell_budget, |candidates, _| {
                let mut scored = Vec::with_capacity(candidates.len());
                for &index in candidates {
                    if !self.arena.is_live(index) {
                        continue;
                    }
                    let vector = self.arena.get_vector_slice(index);
                    let vector_norm_sq = self.arena.get_norm_squared(index);
                    scored.push(rivero_witness::ScoredWitness {
                        index,
                        similarity: self.similarity_score_slices(
                            data,
                            vector,
                            norm_sq,
                            vector_norm_sq,
                        ),
                    });
                }
                let direct_seeds = rivero_witness::select_top(&mut scored, degree);
                let mut expanded: SmallVec<
                    [NodeIndex; RIVERO_WITNESS_MAX_DEGREE * RIVERO_WITNESS_MAX_DEGREE],
                > = SmallVec::new();
                let mut edges_scanned = 0usize;
                for seed in &direct_seeds {
                    let Some(seed_node) = self.arena.get_node(seed.index) else {
                        continue;
                    };
                    let connections = seed_node.rivero_witnesses.read();
                    for witness in connections.iter().take(degree) {
                        let candidate = witness.index;
                        edges_scanned += 1;
                        if candidates.binary_search(&candidate).is_ok()
                            || expanded.contains(&candidate)
                        {
                            continue;
                        }
                        expanded.push(candidate);
                        if !self.arena.is_live(candidate) {
                            continue;
                        }
                        scored.push(rivero_witness::ScoredWitness {
                            index: candidate,
                            similarity: self.similarity_score_slices(
                                data,
                                self.arena.get_vector_slice(candidate),
                                norm_sq,
                                self.arena.get_norm_squared(candidate),
                            ),
                        });
                    }
                }
                debug_assert!(edges_scanned <= degree.saturating_mul(degree));
                rivero_witness::select_top(&mut scored, degree)
            })
    }

    fn add_reciprocal_rivero_witnesses(
        &self,
        node_index: NodeIndex,
        witnesses: &[rivero_witness::ScoredWitness],
        degree: usize,
    ) {
        let mut rejected = SmallVec::<[NodeIndex; RIVERO_WITNESS_INLINE_DEGREE]>::new();
        for &witness in witnesses {
            let Some(owner) = self.arena.get_node(witness.index) else {
                rejected.push(witness.index);
                continue;
            };
            let outcome = {
                let mut connections = owner.rivero_witnesses.write();
                rivero_witness::insert_reciprocal(
                    &mut connections,
                    rivero_witness::ScoredWitness {
                        index: node_index,
                        similarity: witness.similarity,
                    },
                    degree,
                )
            };

            if !outcome.retained {
                rejected.push(witness.index);
            }
            if let Some(evicted) = outcome.evicted {
                if evicted.index != node_index {
                    if let Some(evicted_node) = self.arena.get_node(evicted.index) {
                        evicted_node
                            .rivero_witnesses
                            .write()
                            .retain(|edge| edge.index != witness.index);
                    }
                }
            }
        }

        // Publish only mutually retained edges. The strict two-hop resolver must
        // never observe a directed proposal masquerading as a reciprocal witness.
        if !rejected.is_empty() {
            if let Some(node) = self.arena.get_claimed_node(node_index) {
                node.rivero_witnesses
                    .write()
                    .retain(|edge| !rejected.contains(&edge.index));
            }
        }

        debug_assert!(witnesses.len() <= rivero_witness::bounded_degree(degree));
    }

    fn copy_rivero_witness_connections(
        &self,
        seed: NodeIndex,
        _strict_rivero: bool,
        degree: usize,
        out: &mut SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_DEGREE]>,
    ) {
        out.clear();
        let Some(node) = self.arena.get_node(seed) else {
            return;
        };
        let witnesses = node.rivero_witnesses.read();
        if !witnesses.is_empty() {
            out.extend(witnesses.iter().take(degree).map(|witness| witness.index));
        } else {
            let connections = node.layers[0].read();
            out.extend(connections.iter().take(degree).copied());
        }
    }

    fn insert_internal(
        &self,
        external_id: Arc<str>, // shared Arc: zero-copy between Node and id_to_index
        data: &[Complex32],    // borrowed slice: no VectorEmbedding consumed/cloned
        norm_sq: f32,          // precomputed by caller from the same slice
        custom_node: Option<Node>,
        metadata: Option<&HashMap<String, MetadataValue>>,
        json_metadata: Option<&serde_json::Value>,
    ) -> HNSQRResult<NodeIndex> {
        let _lifecycle = self.lifecycle.read();
        if data.len() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: data.len(),
            });
        }

        if self.is_optimizing.load(AtomicOrdering::Acquire) {
            return Err(HNSQRError::ConcurrencyError(
                "Cannot insert while index optimization is active".to_string(),
            ));
        }

        {
            let mut ids = self.id_to_index.write();
            if ids.contains_key(&external_id) {
                return Err(HNSQRError::NodeAlreadyExists(external_id.to_string()));
            }
            // Reserve the ID atomically so concurrent duplicate inserts cannot both
            // claim arena slots. The sentinel is never exposed as a live node.
            ids.insert(external_id.clone(), NodeIndex::MAX);
        }

        // Write mutation to WAL if attached
        if let Some(wal) = self.wal.read().as_ref() {
            let durability = *self.wal_durability.read();
            let mutation = crate::storage::wal::WalMutation::Upsert {
                external_id: external_id.to_string(),
                vector: VectorEmbedding::from_complex(data.to_vec()),
                metadata: metadata.cloned(),
            };
            if let Err(e) = wal.append(&mutation, durability) {
                self.id_to_index.write().remove(&external_id);
                return Err(e);
            }
        }

        // Extract all needed config values under a single short-lived read guard.
        // This replaces self.config() which cloned the entire 19-field HNSQRConfig struct.
        let (
            new_node_level,
            ef_construction,
            m0,
            m,
            heuristic_edge_selection,
            extend_candidates,
            keep_pruned_connections,
            multi_root_ensemble_size,
            rivero_enabled,
            rivero_graph_fallback,
            rivero_cell_budget,
            rivero_witness_degree,
        ) = {
            let cfg = self.config.read();
            let rivero_graph_fallback = cfg.rivero_fallback_on_underfill;
            let new_node_level = if cfg.rivero_enabled && !rivero_graph_fallback {
                0
            } else {
                let slot_entropy = self.arena.live_len() as u64;
                self.generate_random_level(&cfg, slot_entropy)
            };
            (
                new_node_level,
                cfg.ef_construction,
                cfg.m0,
                cfg.m,
                cfg.heuristic_edge_selection,
                cfg.extend_candidates,
                cfg.keep_pruned_connections,
                cfg.multi_root_ensemble_size,
                cfg.rivero_enabled,
                rivero_graph_fallback,
                cfg.rivero_cell_budget,
                rivero_witness::bounded_degree(cfg.rivero_witness_degree),
            )
        };
        let start_time = Instant::now();

        // 1. Claim slot in O(1) lock-free atomic bump allocator
        let node_index = match self.arena.claim_slot() {
            Ok(index) => index,
            Err(error) => {
                self.id_to_index.write().remove(&external_id);
                return Err(error);
            }
        };

        // 2. Write vector and norm into contiguous flat memory.
        // `data` is borrowed from the caller; the arena is its sole vector owner.
        self.arena.write_vector(node_index, data);

        if let Some(ref mmap) = self.mmap_arena {
            mmap.write_vector(node_index, data, norm_sq);
        }

        let build_graph = !rivero_enabled || rivero_graph_fallback;
        let rivero_address = rivero_enabled.then(|| self.rivero_compiler.read().compile(data));
        let witnesses = if rivero_enabled && rivero_witness_degree > 0 {
            self.select_rivero_witnesses(
                data,
                norm_sq,
                rivero_address.as_ref().expect("Rivero address is present"),
                rivero_cell_budget,
                rivero_witness_degree,
            )
        } else {
            SmallVec::new()
        };

        // 3. Construct and write node struct to its slot
        let node = if let Some(mut n) = custom_node {
            n.index = node_index;
            n.external_id = external_id.clone();
            n.level = new_node_level;
            n
        } else {
            Node::new(node_index, external_id.clone(), new_node_level)
        };
        *node.rivero_witnesses.write() = witnesses.clone();
        self.arena.write_node(node_index, node);

        // 4. Map string ID to integer index and index metadata
        self.id_to_index.write().insert(external_id, node_index); // moved: no third clone
        if let Some(meta) = metadata {
            self.metadata_index.index_node(node_index, meta);
        }
        if let Some(meta) = json_metadata {
            self.metadata_index.insert_metadata(node_index, meta);
        }

        if build_graph {
            for l in 0..=new_node_level {
                self.layers[l].write().push(node_index);
            }
        }

        // Publish into the bounded resolver only after vector, node, ID, metadata,
        // and optional graph layer membership are initialized. Readers cannot observe a
        // partially written slot through the Rivero path.
        if rivero_enabled {
            self.add_reciprocal_rivero_witnesses(node_index, &witnesses, rivero_witness_degree);
            self.rivero_index.insert(
                rivero_address.as_ref().expect("Rivero address is present"),
                node_index,
            );
        }
        if rivero_enabled {
            let lutz_code = crate::proof::lutz::LutzCode::encode(
                &VectorEmbedding::from_complex(data.to_vec()),
                true,
            );
            let mut lutz_guard = self.lutz_codes.write();
            if lutz_guard.len() <= node_index as usize {
                lutz_guard.resize(node_index as usize + 1, None);
            }
            lutz_guard[node_index as usize] = Some(lutz_code);
        }

        self.arena.publish_slot(node_index);
        {
            let mut stats = self.stats.write();
            stats.insertions = stats.insertions.saturating_add(1);
        }

        // Strict Rivero mode intentionally omits all probabilistic graph
        // construction. This makes insertion routing bounded as well as search
        // routing, and avoids paying for a fallback that the mode forbids.
        if !build_graph {
            let elapsed = start_time.elapsed().as_micros() as u64;
            trace!(target: "hnsqr::index", index = node_index, elapsed_us = elapsed, "Node inserted into strict Rivero index");
            return Ok(node_index);
        }

        // Read entry points under a short-lived guard; clone only the small SmallVec
        // if we actually need to iterate beyond the guard scope.
        let eps = self.entry_points.read().clone();

        // If this is the root node:
        if eps.is_empty() || node_index == 0 {
            *self.entry_points.write() = smallvec![node_index];
            self.max_level
                .store(new_node_level, AtomicOrdering::Release);
            return Ok(node_index);
        }

        let current_max_level = self.max_level.load(AtomicOrdering::Acquire);

        // 6. Multi-Root Superposition Seeding
        let mut current_ep = eps[0];
        let ep_slice = self.arena.get_vector_slice(current_ep);
        let ep_norm_sq = self.arena.get_norm_squared(current_ep);
        let mut current_ep_sim = self.similarity_score_slices(data, ep_slice, norm_sq, ep_norm_sq);

        for &ep_idx in &eps[1..] {
            let slice = self.arena.get_vector_slice(ep_idx);
            let n_norm_sq = self.arena.get_norm_squared(ep_idx);
            let sim = self.similarity_score_slices(data, slice, norm_sq, n_norm_sq);
            if sim > current_ep_sim {
                current_ep = ep_idx;
                current_ep_sim = sim;
            }
        }

        // 7. Top-down traversal through upper layers
        for l in ((new_node_level + 1)..=current_max_level).rev() {
            let candidates = self.superposition_search_layer_raw(
                data,
                norm_sq,
                current_ep,
                current_ep_sim,
                l,
                1,
                None,
                None,
            );
            if let Some(best) = candidates.first() {
                current_ep = best.index;
                current_ep_sim = best.similarity;
            }
        }

        // 8. Connect at layers new_node_level down to 0
        let mut connections_to_make: Vec<(usize, Vec<NodeIndex>)> =
            Vec::with_capacity(new_node_level + 1);

        for l in (0..=new_node_level).rev() {
            let found_candidates = self.superposition_search_layer_raw(
                data,
                norm_sq,
                current_ep,
                current_ep_sim,
                l,
                ef_construction,
                None,
                None,
            );

            if !found_candidates.is_empty() {
                current_ep = found_candidates[0].index;
                current_ep_sim = found_candidates[0].similarity;

                let max_m = if l == 0 { m0 } else { m };

                let filtered_candidates: Vec<Candidate> = found_candidates
                    .into_iter()
                    .filter(|c| c.index != node_index)
                    .collect();

                let selected = if heuristic_edge_selection {
                    self.select_neighbors_heuristic(
                        data,
                        norm_sq,
                        filtered_candidates,
                        max_m,
                        l,
                        extend_candidates,
                        keep_pruned_connections,
                    )
                } else {
                    filtered_candidates
                        .into_iter()
                        .take(max_m)
                        .map(|c| c.index)
                        .collect()
                };

                connections_to_make.push((l, selected));
            } else {
                connections_to_make.push((l, Vec::new()));
            }
        }

        // 9. Edge Wiring & Heuristic Diversity Pruning
        if let Some(new_node) = self.arena.get_node(node_index) {
            for (level, neighbors) in connections_to_make {
                let max_m = if level == 0 { m0 } else { m };

                for &neighbor_idx in &neighbors {
                    new_node.add_connection(neighbor_idx, level);

                    if let Some(neighbor_ref) = self.arena.get_node(neighbor_idx) {
                        neighbor_ref.add_connection(node_index, level);

                        let cur_conns = neighbor_ref.get_connections_clone(level);
                        if cur_conns.len() > max_m {
                            let n_slice = self.arena.get_vector_slice(neighbor_idx);
                            let n_norm_sq = self.arena.get_norm_squared(neighbor_idx);

                            let candidates: Vec<Candidate> = cur_conns
                                .into_iter()
                                .map(|idx| {
                                    let other_slice = self.arena.get_vector_slice(idx);
                                    let other_norm_sq = self.arena.get_norm_squared(idx);
                                    let sim = self.similarity_score_slices(
                                        n_slice,
                                        other_slice,
                                        n_norm_sq,
                                        other_norm_sq,
                                    );
                                    Candidate {
                                        index: idx,
                                        similarity: sim,
                                    }
                                })
                                .collect();

                            let pruned = if heuristic_edge_selection && level == 0 {
                                self.select_neighbors_heuristic(
                                    n_slice,
                                    n_norm_sq,
                                    candidates,
                                    max_m,
                                    level,
                                    false,
                                    keep_pruned_connections,
                                )
                            } else {
                                let mut sorted = candidates;
                                sorted.sort_by(|a, b| {
                                    b.similarity
                                        .partial_cmp(&a.similarity)
                                        .unwrap_or(Ordering::Equal)
                                });
                                sorted.into_iter().take(max_m).map(|c| c.index).collect()
                            };

                            let mut new_conns = SmallVec::new();
                            new_conns.extend(pruned);
                            neighbor_ref.set_connections(level, new_conns);
                        }
                    }
                }
            }
        }

        // 10. Update Multi-Root Ensemble at Top Level
        if new_node_level > current_max_level {
            *self.entry_points.write() = smallvec![node_index];
            self.max_level
                .store(new_node_level, AtomicOrdering::Release);
        } else if new_node_level == current_max_level {
            let mut eps_write = self.entry_points.write();
            if eps_write.len() < multi_root_ensemble_size && !eps_write.contains(&node_index) {
                eps_write.push(node_index);
            }
        }

        let elapsed = start_time.elapsed().as_micros() as u64;
        trace!(target: "hnsqr::index", index = node_index, elapsed_us = elapsed, "Node inserted successfully");
        Ok(node_index)
    }

    /// Inserts a borrowed vector slice without cloning its backing allocation.
    #[inline]
    fn insert_slice(
        &self,
        id: impl Into<NodeId>,
        data: &[Complex32],
        norm_sq: f32,
    ) -> HNSQRResult<NodeIndex> {
        self.insert_internal(id.into(), data, norm_sq, None, None, None)
    }

    /// Borrowed-slice variant used by metadata-bearing ingestion paths.
    #[inline]
    fn insert_slice_with_metadata(
        &self,
        id: impl Into<NodeId>,
        data: &[Complex32],
        norm_sq: f32,
        metadata: HashMap<String, MetadataValue>,
    ) -> HNSQRResult<NodeIndex> {
        self.insert_internal(id.into(), data, norm_sq, None, Some(&metadata), None)
    }

    /// Inserts an owned embedding and indexes borrowed metadata without cloning it.
    ///
    /// The metadata index materializes only its own durable keys and bitmaps.
    pub(crate) fn insert_with_metadata_ref(
        &self,
        id: impl Into<NodeId>,
        vector: VectorEmbedding,
        metadata: &HashMap<String, MetadataValue>,
    ) -> HNSQRResult<NodeIndex> {
        self.insert_internal(
            id.into(),
            vector.complex_data(),
            vector.norm_squared(),
            None,
            Some(metadata),
            None,
        )
    }

    /// Inserts a vector embedding with associated JSON metadata for fast inverted-index filtering.
    pub fn insert_with_json_metadata(
        &self,
        id: impl Into<NodeId>,
        embedding: &VectorEmbedding,
        metadata: &serde_json::Value,
    ) -> HNSQRResult<NodeIndex> {
        self.insert_internal(
            id.into(),
            embedding.complex_data(),
            embedding.norm_squared(),
            None,
            None,
            Some(metadata),
        )
    }

    // ────────────────────────────────────────────────────────────────────────
    // 3. ZERO-ALLOCATION SUPERPOSITION SEARCH
    // ────────────────────────────────────────────────────────────────────────

    fn superposition_search_layer_raw(
        &self,
        query_data: &[Complex32],
        query_norm_sq: f32,
        entry_point: NodeIndex,
        entry_sim: SimilarityScore,
        level: usize,
        ef: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        mut diagnostics: Option<&mut GraphSearchDiagnostics>,
    ) -> Vec<Candidate> {
        let total_nodes = self.arena.len();
        if ef == 0 || total_nodes == 0 {
            return Vec::new();
        }

        let config = self.config.read();
        let base_beam_width = config.superposition_beam_width.max(1);
        let temp = config.attention_temperature.max(1e-4);
        let lambda = config.interference_weight.clamp(0.0, 1.0);

        THREAD_SEARCH_SCRATCHPAD.with(|pad_cell| {
            let mut pad_guard = pad_cell.borrow_mut();
            let pad: &mut SearchScratchpad = &mut pad_guard;
            pad.reset(self.dimension);
            let SearchScratchpad {
                candidate_queue,
                results_heap,
                current_batch,
                neighbor_candidates,
                scored_neighbors,
                beam,
                next_beam,
                exps,
                temp_conns,
                psi_beam_data,
            } = pad;

            candidate_queue.push(Candidate {
                index: entry_point,
                similarity: entry_sim,
            });
            if let Some(d) = diagnostics.as_deref_mut() {
                d.candidate_pushes += 1;
            }

            results_heap.push(WorstResultCandidate(Candidate {
                index: entry_point,
                similarity: entry_sim,
            }));

            beam.push((entry_point, 1.0));

            THREAD_VISITED_POOL.with(|pool_cell| {
                let mut pool = pool_cell.borrow_mut();
                let epoch = pool.next_epoch(total_nodes);
                pool.mark_visited(entry_point, epoch);
                if let Some(d) = diagnostics.as_deref_mut() {
                    d.visited_nodes += 1;
                }

                while !candidate_queue.is_empty() {
                    if results_heap.len() >= ef {
                        if let (Some(best_cand), Some(WorstResultCandidate(worst_res))) =
                            (candidate_queue.peek(), results_heap.peek())
                        {
                            if best_cand.similarity < worst_res.similarity {
                                break;
                            }
                        }
                    }

                    current_batch.clear();
                    for _ in 0..base_beam_width {
                        if let Some(cand) = candidate_queue.pop() {
                            if let Some(d) = diagnostics.as_deref_mut() {
                                d.candidate_pops += 1;
                            }
                            current_batch.push(cand);
                        } else {
                            break;
                        }
                    }
                    if current_batch.is_empty() {
                        break;
                    }

                    // 1. Construct superposition beam state vector directly in scratchpad buffer (only at base layer if lambda > 0)
                    let use_superposition = level == 0 && lambda > 0.0;
                    if use_superposition {
                        psi_beam_data.fill(Complex32::new(0.0, 0.0));
                        for (idx, w) in beam.iter() {
                            let slice = self.arena.get_vector_slice(*idx);
                            let weight = *w;
                            for (dst, src) in psi_beam_data.iter_mut().zip(slice.iter()) {
                                *dst += *src * weight;
                            }
                        }
                    }

                    // 2. Collect unvisited neighbors with hardware prefetch and a Roaring bitmap mask.
                    neighbor_candidates.clear();
                    for cand in current_batch.iter() {
                        if let Some(cand_node) = self.arena.get_node(cand.index) {
                            temp_conns.clear();
                            cand_node.connections_at(level, temp_conns);
                            for &neighbor_idx in temp_conns.iter() {
                                if let Some(d) = diagnostics.as_deref_mut() {
                                    d.edges_examined += 1;
                                }
                                if !pool.is_visited(neighbor_idx, epoch) {
                                    if let Some(mask) = filter_mask {
                                        if !mask.contains(neighbor_idx) {
                                            continue;
                                        }
                                    }
                                    pool.mark_visited(neighbor_idx, epoch);
                                    if let Some(d) = diagnostics.as_deref_mut() {
                                        d.visited_nodes += 1;
                                    }
                                    let n_slice = self.arena.get_vector_slice(neighbor_idx);
                                    prefetch_vector(n_slice);
                                    neighbor_candidates.push((neighbor_idx, neighbor_idx as usize));
                                }
                            }
                        }
                    }

                    if neighbor_candidates.is_empty() {
                        continue;
                    }

                    // 3. Evaluate Phase Interference & Total Superposition Scores with Single-Pass SIMD
                    scored_neighbors.clear();
                    for &(n_idx, _) in neighbor_candidates.iter() {
                        let n_slice = self.arena.get_vector_slice(n_idx);
                        let n_norm_sq = self.arena.get_norm_squared(n_idx);

                        let ip_query = dot_product_complex_simd(query_data, n_slice);
                        let ip_query_norm_sq = ip_query.norm_sqr();
                        let fidelity_query = self.similarity_score_slices(
                            query_data,
                            n_slice,
                            query_norm_sq,
                            n_norm_sq,
                        );
                        if let Some(d) = diagnostics.as_deref_mut() {
                            d.distance_evaluations += 1;
                        }

                        let total_score = if use_superposition {
                            let ip_beam = dot_product_complex_simd(psi_beam_data, n_slice);
                            let ip_beam_norm_sq = ip_beam.norm_sqr();
                            let interference_mag = ip_beam_norm_sq.clamp(0.0, 1.0);
                            let denom = (ip_query_norm_sq * ip_beam_norm_sq + 1e-12).sqrt();
                            let phase_diff =
                                (ip_query.re * ip_beam.re + ip_query.im * ip_beam.im) / denom;
                            let constructive_bonus = interference_mag * phase_diff.max(-0.5);
                            (1.0 - lambda) * fidelity_query + lambda * constructive_bonus
                        } else {
                            fidelity_query
                        };

                        scored_neighbors.push((n_idx, total_score, fidelity_query));
                    }

                    // 4. Softmax Attention Distribution
                    let max_score = scored_neighbors
                        .iter()
                        .map(|(_, s, _)| *s)
                        .fold(f32::NEG_INFINITY, f32::max);

                    exps.clear();
                    let mut sum_exp = 0.0f32;
                    for (_, score, _) in scored_neighbors.iter() {
                        let exp_val = ((score - max_score) / temp).exp();
                        exps.push(exp_val);
                        sum_exp += exp_val;
                    }

                    let inv_sum = if sum_exp > 1e-9 { 1.0 / sum_exp } else { 1.0 };
                    next_beam.clear();

                    // 5. Update Candidate Queue and Results Heap
                    for (i, &(n_idx, _, fidelity)) in scored_neighbors.iter().enumerate() {
                        let attention_weight = exps[i] * inv_sum;

                        let worst_sim = results_heap
                            .peek()
                            .map(|w| w.0.similarity)
                            .unwrap_or(f32::NEG_INFINITY);

                        if results_heap.len() < ef || fidelity > worst_sim {
                            // Candidate is Copy ({u32, f32} = 8 bytes) — no heap allocation.
                            let cand = Candidate {
                                index: n_idx,
                                similarity: fidelity,
                            };
                            candidate_queue.push(cand);
                            if let Some(d) = diagnostics.as_deref_mut() {
                                d.candidate_pushes += 1;
                            }
                            results_heap.push(WorstResultCandidate(cand));
                            if results_heap.len() > ef {
                                results_heap.pop();
                            }
                        }

                        if next_beam.len() < base_beam_width {
                            next_beam.push((n_idx, attention_weight));
                        }
                    }

                    if !next_beam.is_empty() {
                        std::mem::swap(beam, next_beam);
                    }
                }
            });

            let mut final_results: Vec<Candidate> = results_heap.drain().map(|w| w.0).collect();
            final_results.sort_by(|a, b| {
                b.similarity
                    .total_cmp(&a.similarity)
                    .then_with(|| a.index.cmp(&b.index))
            });
            final_results
        })
    }

    /// Zero-allocation classical HNSW graph traversal without phase superposition or attention beams.
    fn hnsw_classical_search_layer_raw(
        &self,
        query_data: &[Complex32],
        query_norm_sq: f32,
        entry_point: NodeIndex,
        entry_sim: SimilarityScore,
        level: usize,
        ef: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        mut diagnostics: Option<&mut GraphSearchDiagnostics>,
    ) -> Vec<Candidate> {
        let total_nodes = self.arena.len();
        if ef == 0 || total_nodes == 0 {
            return Vec::new();
        }

        THREAD_SEARCH_SCRATCHPAD.with(|pad_cell| {
            let mut pad_guard = pad_cell.borrow_mut();
            let pad: &mut SearchScratchpad = &mut pad_guard;
            pad.reset(self.dimension);
            let SearchScratchpad {
                candidate_queue,
                results_heap,
                temp_conns,
                ..
            } = pad;

            let entry_cand = Candidate {
                index: entry_point,
                similarity: entry_sim,
            };
            candidate_queue.push(entry_cand);
            if let Some(d) = diagnostics.as_deref_mut() {
                d.candidate_pushes += 1;
            }
            results_heap.push(WorstResultCandidate(entry_cand));

            THREAD_VISITED_POOL.with(|pool_cell| {
                let mut pool = pool_cell.borrow_mut();
                let epoch = pool.next_epoch(total_nodes);
                pool.mark_visited(entry_point, epoch);
                if let Some(d) = diagnostics.as_deref_mut() {
                    d.visited_nodes += 1;
                }

                while let Some(current_candidate) = candidate_queue.pop() {
                    if let Some(d) = diagnostics.as_deref_mut() {
                        d.candidate_pops += 1;
                    }

                    let worst_sim = if results_heap.len() >= ef {
                        results_heap
                            .peek()
                            .map(|w| w.0.similarity)
                            .unwrap_or(f32::NEG_INFINITY)
                    } else {
                        f32::NEG_INFINITY
                    };

                    if results_heap.len() >= ef && current_candidate.similarity < worst_sim {
                        break;
                    }

                    if let Some(cand_node) = self.arena.get_node(current_candidate.index) {
                        temp_conns.clear();
                        cand_node.connections_at(level, temp_conns);

                        for &neighbor_idx in temp_conns.iter() {
                            if let Some(d) = diagnostics.as_deref_mut() {
                                d.edges_examined += 1;
                            }
                            if !pool.is_visited(neighbor_idx, epoch) {
                                pool.mark_visited(neighbor_idx, epoch);
                                if let Some(d) = diagnostics.as_deref_mut() {
                                    d.visited_nodes += 1;
                                }

                                if let Some(mask) = filter_mask {
                                    if !mask.contains(neighbor_idx) {
                                        continue;
                                    }
                                }

                                let n_slice = self.arena.get_vector_slice(neighbor_idx);
                                let n_norm_sq = self.arena.get_norm_squared(neighbor_idx);
                                prefetch_vector(n_slice);

                                let sim = self.similarity_score_slices(
                                    query_data,
                                    n_slice,
                                    query_norm_sq,
                                    n_norm_sq,
                                );
                                if let Some(d) = diagnostics.as_deref_mut() {
                                    d.distance_evaluations += 1;
                                }

                                let current_worst = results_heap
                                    .peek()
                                    .map(|w| w.0.similarity)
                                    .unwrap_or(f32::NEG_INFINITY);

                                if results_heap.len() < ef || sim > current_worst {
                                    let cand = Candidate {
                                        index: neighbor_idx,
                                        similarity: sim,
                                    };
                                    candidate_queue.push(cand);
                                    if let Some(d) = diagnostics.as_deref_mut() {
                                        d.candidate_pushes += 1;
                                    }
                                    results_heap.push(WorstResultCandidate(cand));
                                    if results_heap.len() > ef {
                                        results_heap.pop();
                                    }
                                }
                            }
                        }
                    }
                }
            });

            let mut final_results: Vec<Candidate> = results_heap.drain().map(|w| w.0).collect();
            final_results.sort_by(|a, b| {
                b.similarity
                    .total_cmp(&a.similarity)
                    .then_with(|| a.index.cmp(&b.index))
            });
            final_results
        })
    }

    // ────────────────────────────────────────────────────────────────────────
    // 4. SEARCH API WITH OVER-SAMPLING & EXACT RESCORING
    // ────────────────────────────────────────────────────────────────────────

    /// Searches for the $k$ nearest neighbors to the query vector.
    #[instrument(skip(self, query), level = "debug", target = "hnsqr::search")]
    pub fn search(
        &self,
        query: &VectorEmbedding,
        k: usize,
    ) -> HNSQRResult<Vec<(NodeId, SimilarityScore)>> {
        let indices = self.search_indices(query, k)?;

        let results = indices
            .into_iter()
            .filter_map(|(idx, score)| {
                self.arena
                    .get_node(idx)
                    .map(|n| (n.external_id.clone(), score))
            })
            .collect();

        Ok(results)
    }

    /// Computes the canonical empirical exact-scan crossover threshold for a complex vector dimension.
    ///
    /// Delegates to [`crate::planning::planner::ExactScanCrossoverModel`] so the index `Auto`
    /// path and [`crate::planning::planner::UniversalPlanner`] cannot diverge.
    #[inline]
    #[must_use]
    pub fn default_exact_scan_threshold(dimension: usize) -> usize {
        crate::planning::planner::UniversalPlanner::compute_crossover(dimension)
    }

    /// Searches for the $k$ nearest neighbor internal arena indices with an optional Roaring Bitmap filter mask.
    pub fn search_indices_filtered(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<Vec<(NodeIndex, SimilarityScore)>> {
        let _lifecycle = self.lifecycle.read();
        if query.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let (search_plan, exact_threshold, rivero_enabled, rivero_mode, adaptive_policy) = {
            let cfg = self.config.read();
            (
                cfg.search_plan,
                cfg.exact_scan_threshold,
                cfg.rivero_enabled,
                cfg.rivero_mode,
                cfg.adaptive_policy,
            )
        };

        let effective_threshold = if exact_threshold > 0 {
            exact_threshold
        } else {
            Self::default_exact_scan_threshold(self.dimension)
        };

        if search_plan == SearchPlan::Exact
            || (search_plan == SearchPlan::Auto && self.arena.live_len() <= effective_threshold)
        {
            return self.search_indices_exact(query, k, filter_mask);
        }

        if search_plan == SearchPlan::HnswClassical {
            return self.search_indices_hnsw_classical(query, k, filter_mask);
        }

        if search_plan != SearchPlan::GraphOnly
            && search_plan != SearchPlan::GraphSuperposition
            && rivero_enabled
        {
            match rivero_mode {
                RiveroSearchMode::Strict => {
                    return self
                        .search_indices_strict(query, k, filter_mask)
                        .map(|(res, _)| res);
                }
                RiveroSearchMode::Adaptive => {
                    return self
                        .search_indices_adaptive(query, k, filter_mask, adaptive_policy)
                        .map(|(res, _)| res);
                }
                RiveroSearchMode::GraphOnly => {}
            }
        }

        let start_time = Instant::now();
        let active = self.active_searches.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.peak_active_searches
            .fetch_max(active, AtomicOrdering::Relaxed);
        defer! { self.active_searches.fetch_sub(1, AtomicOrdering::Relaxed); }

        let result = self.search_indices_graph_internal(query, k, filter_mask);
        self.record_search_latency(start_time);
        result
    }

    /// Builds or rebuilds the canonical corpus-covering semantic proof hierarchy.
    pub fn build_proof_tree(&self) -> Arc<SemanticProofTree> {
        let live_slots: Vec<NodeIndex> = (0..self.arena.len() as NodeIndex)
            .filter(|&slot| self.arena.is_live(slot))
            .collect();

        let mut slot_to_vec = HashMap::with_capacity(live_slots.len());
        for &slot in &live_slots {
            let slice = self.arena.get_vector_slice(slot);
            slot_to_vec.insert(
                slot,
                VectorEmbedding::from_complex(slice.to_vec()).into_normalized(),
            );
        }

        let normalized_vectors: Vec<VectorEmbedding> = (0..self.arena.len() as NodeIndex)
            .map(|slot| {
                if let Some(v) = slot_to_vec.get(&slot) {
                    v.clone()
                } else {
                    VectorEmbedding::from_complex(vec![
                        num_complex::Complex32::default();
                        self.dimension
                    ])
                }
            })
            .collect();

        let tree = Arc::new(SemanticProofTree::build(
            &normalized_vectors,
            &live_slots,
            self.dimension,
        ));

        *self.proof_tree.write() = Some(tree.clone());
        tree
    }

    /// Gets or constructs the cached proof tree.
    pub fn get_or_build_proof_tree(&self) -> Arc<SemanticProofTree> {
        if let Some(tree) = self.proof_tree.read().as_ref() {
            return tree.clone();
        }
        self.build_proof_tree()
    }

    /// Executes a `Certified` proof search with a typed outcome.
    ///
    /// This is the preferred API for callers that explicitly require the `Certified`
    /// contract.  Unlike `search_indices_with_proof`, the deadline-abort case is
    /// structurally distinct: the caller pattern-matches on `CertifiedSearchOutcome`
    /// and cannot accidentally treat an incomplete proof as a successful exact result.
    ///
    /// - `CertifiedSearchOutcome::Exact`           → 100.000% exact, complete proof.
    /// - `CertifiedSearchOutcome::DeadlineExceeded` → budget expired; partial results only.
    ///
    /// The deadline is derived from `HNSQRConfig::certified_query_timeout_ms` when set.
    /// If no budget is configured, the result is always `Exact`.
    pub fn certified_search(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<CertifiedSearchOutcome> {
        let _lifecycle = self.lifecycle.read();
        if query.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        if k == 0 || self.is_empty() {
            return Ok(CertifiedSearchOutcome::Exact {
                results: Vec::new(),
                proof: DenseExactProof::default(),
            });
        }

        // The spherical proof hierarchy currently certifies normalized cosine
        // similarity. Other declared metrics remain certifiable via exhaustive
        // evaluation; routing or proof bounds from a different metric must never
        // be presented as a proof for the requested metric.
        if self.config.read().distance_function != DistanceFunction::Cosine {
            let started = Instant::now();
            let results = self.search_indices_exact(query, k, filter_mask)?;
            let live = self.arena.live_len();
            let eligible = match filter_mask {
                Some(mask) => mask.iter().filter(|&slot| self.arena.is_live(slot)).count(),
                None => live,
            };
            let proof = DenseExactProof {
                corpus_size: live,
                exact_evaluations: eligible,
                filtered_or_tombstoned: live.saturating_sub(eligible),
                exact_bytes_touched: eligible
                    .saturating_mul(self.dimension)
                    .saturating_mul(std::mem::size_of::<Complex32>()),
                kth_score: results
                    .last()
                    .map_or(f32::NEG_INFINITY, |(_, score)| *score),
                globally_exact: true,
                elapsed_us: started.elapsed().as_micros().max(1) as u64,
                ..DenseExactProof::default()
            };
            debug_assert!(proof.is_accounting_exact());
            return Ok(CertifiedSearchOutcome::Exact { results, proof });
        }

        let q_norm = query.clone().into_normalized();
        let tree = self.get_or_build_proof_tree();

        let rivero_cfg = RiveroProfile::Strict.config();
        let addr = self.rivero_compiler.read().compile(q_norm.complex_data());
        let mut seed_slots = Vec::new();
        self.rivero_index
            .with_candidates_config(&addr, &rivero_cfg, |cands, _| {
                seed_slots.extend_from_slice(cands);
            });

        let normalized_vectors: Vec<VectorEmbedding> = (0..self.arena.len() as NodeIndex)
            .map(|slot| {
                if self.arena.is_live(slot) {
                    let slice = self.arena.get_vector_slice(slot);
                    VectorEmbedding::from_complex(slice.to_vec()).into_normalized()
                } else {
                    VectorEmbedding::from_complex(vec![Complex32::default(); self.dimension])
                }
            })
            .collect();

        let seg_view = SegmentProofView {
            tree: &tree,
            vectors: &normalized_vectors,
            lutz_codes: None,
            tombstones: None,
        };

        let deadline = {
            let cfg = self.config.read();
            cfg.certified_query_timeout_ms
                .map(|m| Instant::now() + std::time::Duration::from_millis(m))
        };

        let (results, proof) = GlobalExactProofSearch::search_with_deadline(
            &q_norm,
            k,
            &[seg_view],
            &[],
            &seed_slots,
            filter_mask,
            deadline,
        );

        if proof.globally_exact && proof.leaf_vectors_considered > 0 {
            let warmer = crate::storage::predictive_warming::PredictiveWarmer::default();
            let threat_slots: Vec<NodeIndex> = results.iter().map(|(s, _)| *s).collect();
            warmer.record_proof_access(proof.proof_regions_popped, &threat_slots);
        }

        if proof.deadline_exceeded {
            Ok(CertifiedSearchOutcome::DeadlineExceeded {
                partial_results: results,
                proof,
            })
        } else {
            // Sanity: if somehow globally_exact is false without deadline_exceeded,
            // surface it as an internal error rather than silently returning as Exact.
            if !proof.globally_exact {
                return Err(HNSQRError::Internal(
                    "Proof search completed without deadline but globally_exact is false".into(),
                ));
            }
            Ok(CertifiedSearchOutcome::Exact { results, proof })
        }
    }

    /// Searches the index returning finalists alongside detailed mathematical proof telemetry.
    ///
    /// Returns `Ok((results, proof))` in both the exact and deadline-aborted cases.
    /// Callers **must** inspect `proof.globally_exact` and `proof.deadline_exceeded`
    /// to distinguish complete certification from a best-effort partial result.
    ///
    /// Prefer [`certified_search`][HNSQRIndex::certified_search] when you need the
    /// `Certified` contract enforced at the type level.
    pub fn search_indices_with_proof(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, DenseExactProof)> {
        // Delegate through certified_search so both methods share one code path,
        // then flatten the typed outcome back to the legacy tuple form.
        match self.certified_search(query, k, filter_mask)? {
            CertifiedSearchOutcome::Exact { results, proof } => Ok((results, proof)),
            CertifiedSearchOutcome::DeadlineExceeded {
                partial_results,
                proof,
            } => Ok((partial_results, proof)),
        }
    }

    /// Searches the index enforcing a declared retrieval contract (Certified, Exact, HighRecall, or Budget).
    pub fn search_indices_with_contract(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        contract: crate::planning::planner::RetrievalContract,
    ) -> HNSQRResult<Vec<(NodeIndex, SimilarityScore)>> {
        let _lifecycle = self.lifecycle.read();
        if query.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let n = self.arena.live_len();
        let default_affect = crate::planning::affect::AffectiveStateTensor8D::default();
        let plan = crate::planning::planner::UniversalPlanner::plan_with_affect(
            n,
            self.dimension,
            filter_mask.map(|m| m.len() as usize),
            contract,
            self.mmap_arena.is_some(),
            &default_affect,
        );

        match plan {
            crate::planning::planner::ExecutionPlan::ExactScan { .. } => {
                self.search_indices_exact(query, k, filter_mask)
            }
            crate::planning::planner::ExecutionPlan::LutzGlobalCertified {
                initial_seed_cap: _,
            } => match self.certified_search(query, k, filter_mask)? {
                CertifiedSearchOutcome::Exact { results, .. } => Ok(results),
                CertifiedSearchOutcome::DeadlineExceeded { proof, .. } => {
                    let total_regions = proof.proof_regions_pruned + proof.proof_regions_expanded;
                    let ppm = if total_regions > 0 {
                        ((proof.proof_regions_pruned as f64 / total_regions as f64) * 1_000_000.0)
                            as u32
                    } else {
                        0
                    };
                    let budget_us =
                        self.config.read().certified_query_timeout_ms.unwrap_or(0) * 1000;
                    Err(HNSQRError::CertifiedDeadlineExceeded {
                        elapsed_us: proof.elapsed_us,
                        budget_us,
                        frontier_nodes_remaining: proof.frontier_nodes_remaining,
                        region_prune_ratio_ppm: ppm,
                    })
                }
            },
            crate::planning::planner::ExecutionPlan::LutzPacRelaxed {
                epsilon,
                delta,
                initial_seed_cap: _,
            } => {
                let q_norm = query.clone().into_normalized();
                let tree = self.build_proof_tree();
                let seed_slots: Vec<NodeIndex> =
                    (0..self.arena.live_len() as NodeIndex).take(256).collect();
                let normalized_vectors: Vec<VectorEmbedding> = (0..self.arena.len())
                    .map(|i| {
                        let slice = self.arena.get_vector_slice(i as NodeIndex);
                        VectorEmbedding::from_complex(slice.to_vec()).into_normalized()
                    })
                    .collect();
                let seg_view = crate::proof::SegmentProofView {
                    tree: &tree,
                    vectors: &normalized_vectors,
                    lutz_codes: None,
                    tombstones: None,
                };
                let (results, _) = crate::proof::GlobalPacProofSearch::search(
                    &q_norm,
                    k,
                    &[seg_view],
                    &[],
                    &seed_slots,
                    filter_mask,
                    epsilon,
                    delta,
                );
                Ok(results)
            }
            crate::planning::planner::ExecutionPlan::RiveroRetrieval {
                profile,
                candidate_cap,
                ..
            } => {
                let mut rivero_cfg = profile.config();
                rivero_cfg.query_candidate_cap = candidate_cap;
                self.search_indices_o1_with_config(query, k, filter_mask, &rivero_cfg)
                    .map(|(res, _)| res)
            }
            _ => self.search_indices_filtered(query, k, filter_mask),
        }
    }

    /// Internal helper executing classical multi-root superposition graph traversal.
    /// Internal helper executing classical multi-root superposition graph traversal.
    fn search_indices_graph_internal(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<Vec<(NodeIndex, SimilarityScore)>> {
        let (results, _) = self.search_indices_superposition_internal(query, k, filter_mask)?;
        Ok(results)
    }

    /// Internal helper executing superposition graph traversal with work diagnostics telemetry.
    fn search_indices_superposition_internal(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, GraphSearchDiagnostics)> {
        let mut diagnostics = GraphSearchDiagnostics::default();
        let eps = self.entry_points.read().clone();
        if eps.is_empty() {
            return Ok((Vec::new(), diagnostics));
        }

        let ef_search = {
            let cfg = self.config.read();
            let oversample_k = ((k as f32) * cfg.oversample_factor).ceil() as usize;
            cfg.ef_search.max(oversample_k)
        };

        let query_data = query.complex_data();
        let query_norm_sq = query.norm_squared();

        // 1. Multi-Root Superposition Seeding
        let mut current_ep = eps[0];
        let ep_slice = self.arena.get_vector_slice(current_ep);
        let ep_norm_sq = self.arena.get_norm_squared(current_ep);
        let mut current_ep_sim =
            self.similarity_score_slices(query_data, ep_slice, query_norm_sq, ep_norm_sq);
        diagnostics.distance_evaluations += 1;
        diagnostics.visited_nodes += 1;

        for &ep_idx in &eps[1..] {
            let slice = self.arena.get_vector_slice(ep_idx);
            let norm_sq = self.arena.get_norm_squared(ep_idx);
            let sim = self.similarity_score_slices(query_data, slice, query_norm_sq, norm_sq);
            diagnostics.distance_evaluations += 1;
            diagnostics.visited_nodes += 1;
            if sim > current_ep_sim {
                current_ep = ep_idx;
                current_ep_sim = sim;
            }
        }

        let max_level = self.max_level.load(AtomicOrdering::Acquire);

        // 2. Superposition traversal through upper layers
        for l in (1..=max_level).rev() {
            let candidates = self.superposition_search_layer_raw(
                query_data,
                query_norm_sq,
                current_ep,
                current_ep_sim,
                l,
                1,
                None,
                Some(&mut diagnostics),
            );
            if let Some(best) = candidates.first() {
                current_ep = best.index;
                current_ep_sim = best.similarity;
            }
        }

        // 3. Base layer superposition search with a precompiled bitmap mask.
        let candidate_pool = self.superposition_search_layer_raw(
            query_data,
            query_norm_sq,
            current_ep,
            current_ep_sim,
            0,
            ef_search,
            filter_mask,
            Some(&mut diagnostics),
        );

        let rescored: Vec<(NodeIndex, SimilarityScore)> = candidate_pool
            .into_iter()
            .filter(|candidate| self.arena.is_live(candidate.index))
            .take(k)
            .map(|c| (c.index, c.similarity))
            .collect();
        diagnostics.exact_reranks = rescored.len() as u64;

        Ok((rescored, diagnostics))
    }

    /// Internal helper executing conventional HNSW traversal without superposition interference or attention beams.
    fn search_indices_hnsw_classical_internal(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, GraphSearchDiagnostics)> {
        let mut diagnostics = GraphSearchDiagnostics::default();
        let eps = self.entry_points.read().clone();
        if eps.is_empty() {
            return Ok((Vec::new(), diagnostics));
        }

        let ef_search = {
            let cfg = self.config.read();
            let oversample_k = ((k as f32) * cfg.oversample_factor).ceil() as usize;
            cfg.ef_search.max(oversample_k)
        };

        let query_data = query.complex_data();
        let query_norm_sq = query.norm_squared();

        // 1. Classical Entry Point Selection
        let mut current_ep = eps[0];
        let ep_slice = self.arena.get_vector_slice(current_ep);
        let ep_norm_sq = self.arena.get_norm_squared(current_ep);
        let mut current_ep_sim =
            self.similarity_score_slices(query_data, ep_slice, query_norm_sq, ep_norm_sq);
        diagnostics.distance_evaluations += 1;
        diagnostics.visited_nodes += 1;

        for &ep_idx in &eps[1..] {
            let slice = self.arena.get_vector_slice(ep_idx);
            let norm_sq = self.arena.get_norm_squared(ep_idx);
            let sim = self.similarity_score_slices(query_data, slice, query_norm_sq, norm_sq);
            diagnostics.distance_evaluations += 1;
            diagnostics.visited_nodes += 1;
            if sim > current_ep_sim {
                current_ep = ep_idx;
                current_ep_sim = sim;
            }
        }

        let max_level = self.max_level.load(AtomicOrdering::Acquire);

        // 2. Greedy search through upper layers (ef = 1)
        for l in (1..=max_level).rev() {
            let candidates = self.hnsw_classical_search_layer_raw(
                query_data,
                query_norm_sq,
                current_ep,
                current_ep_sim,
                l,
                1,
                None,
                Some(&mut diagnostics),
            );
            if let Some(best) = candidates.first() {
                current_ep = best.index;
                current_ep_sim = best.similarity;
            }
        }

        // 3. Base layer search with ef_search
        let candidate_pool = self.hnsw_classical_search_layer_raw(
            query_data,
            query_norm_sq,
            current_ep,
            current_ep_sim,
            0,
            ef_search,
            filter_mask,
            Some(&mut diagnostics),
        );

        let rescored: Vec<(NodeIndex, SimilarityScore)> = candidate_pool
            .into_iter()
            .filter(|candidate| self.arena.is_live(candidate.index))
            .take(k)
            .map(|c| (c.index, c.similarity))
            .collect();
        diagnostics.exact_reranks = rescored.len() as u64;

        Ok((rescored, diagnostics))
    }

    /// Executes multimodal hybrid search combining dense vector similarity with sparse lexical terms,
    /// allowing dynamic choice of fusion method (RRF vs Weighted Linear).
    pub fn search_hybrid(
        &self,
        dense_query: &VectorEmbedding,
        sparse_index: &crate::retrieval::sparse::SparseInvertedIndex,
        sparse_query_terms: &[u32],
        k: usize,
        method: crate::retrieval::hybrid::HybridFusionMethod,
    ) -> HNSQRResult<Vec<(Arc<str>, SimilarityScore)>> {
        use crate::retrieval::hybrid::{HybridFusionEngine, ModalityRankings};
        use std::sync::Arc;

        let dense_results = self.search(dense_query, k * 2)?;

        let sparse_raw = sparse_index.search_bm25(sparse_query_terms, k * 2);
        let sparse_results: Vec<(Arc<str>, SimilarityScore)> = sparse_raw
            .into_iter()
            .map(|(idx, score)| (Arc::from(format!("node_{idx}")) as Arc<str>, score))
            .collect();

        let dense_modality = ModalityRankings {
            name: "dense".to_string(),
            weight: 1.0,
            results: dense_results,
        };
        let sparse_modality = ModalityRankings {
            name: "sparse".to_string(),
            weight: 1.0,
            results: sparse_results,
        };
        let modalities = [dense_modality, sparse_modality];

        Ok(match method {
            crate::retrieval::hybrid::HybridFusionMethod::Rrf { k: rrf_k } => {
                HybridFusionEngine::fuse_rrf(&modalities, rrf_k, k)
            }
            crate::retrieval::hybrid::HybridFusionMethod::WeightedLinear => {
                HybridFusionEngine::fuse_weighted(&modalities, k)
            }
        })
    }

    /// Executes multimodal hybrid search combining dense vector similarity with sparse BM25
    /// lexical rankings via Reciprocal Rank Fusion (RRF).
    ///
    /// Retrieves `k * 2` candidates from each modality to maximise fusion quality before
    /// truncating the merged list to `k`.  The `rrf_k` constant (default `60.0`) controls
    /// the smoothing of rank-based scores; higher values reduce the advantage of top ranks.
    pub fn search_hybrid_rrf(
        &self,
        dense_query: &VectorEmbedding,
        sparse_index: &crate::retrieval::sparse::SparseInvertedIndex,
        sparse_query_terms: &[u32],
        k: usize,
        rrf_k: f32,
    ) -> HNSQRResult<Vec<(Arc<str>, SimilarityScore)>> {
        self.search_hybrid(
            dense_query,
            sparse_index,
            sparse_query_terms,
            k,
            crate::retrieval::hybrid::HybridFusionMethod::Rrf { k: rrf_k },
        )
    }

    /// Executes a brute-force exact scan across all live nodes with distance_function scoring.
    ///
    /// Highly optimized for small corpora (e.g. N <= 2,000) where exact vector scanning
    /// outperforms candidate generation and graph traversal in both throughput and exactness.
    pub fn search_indices_exact(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<Vec<(NodeIndex, SimilarityScore)>> {
        let _lifecycle = self.lifecycle.read();
        if query.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let start_time = Instant::now();
        let active = self.active_searches.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.peak_active_searches
            .fetch_max(active, AtomicOrdering::Relaxed);
        defer! { self.active_searches.fetch_sub(1, AtomicOrdering::Relaxed); }

        let query_data = query.complex_data();
        let query_norm_sq = query.norm_squared();
        let n = self.arena.len();

        let mut scored: Vec<(NodeIndex, SimilarityScore)> = Vec::with_capacity(n.min(k * 4));
        for i in 0..n as NodeIndex {
            if !self.arena.is_live(i) {
                continue;
            }
            if filter_mask.is_some_and(|m| !m.contains(i)) {
                continue;
            }
            let v = self.arena.get_vector_slice(i);
            let norm_sq = self.arena.get_norm_squared(i);
            let score = self.similarity_score_slices(query_data, v, query_norm_sq, norm_sq);
            scored.push((i, score));
        }

        if scored.len() > k {
            scored.select_nth_unstable_by(k - 1, |a, b| {
                b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0))
            });
            scored.truncate(k);
        }
        scored.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

        self.record_search_latency(start_time);
        Ok(scored)
    }

    /// Searches strictly using graph traversal without invoking Rivero routing.
    pub fn search_indices_graph(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<Vec<(NodeIndex, SimilarityScore)>> {
        let _lifecycle = self.lifecycle.read();
        if query.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        if k == 0 || self.is_empty() {
            return Ok(Vec::new());
        }

        let start_time = Instant::now();
        let active = self.active_searches.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.peak_active_searches
            .fetch_max(active, AtomicOrdering::Relaxed);
        defer! { self.active_searches.fetch_sub(1, AtomicOrdering::Relaxed); }

        let result = self.search_indices_graph_internal(query, k, filter_mask);
        self.record_search_latency(start_time);
        result
    }

    /// Searches using classical HNSW graph traversal without superposition or attention beams.
    pub fn search_indices_hnsw_classical(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<Vec<(NodeIndex, SimilarityScore)>> {
        let (results, _) = self.search_indices_hnsw_classical_diagnostics(query, k, filter_mask)?;
        Ok(results)
    }

    /// Searches using classical HNSW graph traversal and returns detailed execution work diagnostics.
    pub fn search_indices_hnsw_classical_diagnostics(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, GraphSearchDiagnostics)> {
        let _lifecycle = self.lifecycle.read();
        if query.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        if k == 0 || self.is_empty() {
            return Ok((Vec::new(), GraphSearchDiagnostics::default()));
        }

        let start_time = Instant::now();
        let active = self.active_searches.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.peak_active_searches
            .fetch_max(active, AtomicOrdering::Relaxed);
        defer! { self.active_searches.fetch_sub(1, AtomicOrdering::Relaxed); }

        let result = self.search_indices_hnsw_classical_internal(query, k, filter_mask);
        self.record_search_latency(start_time);
        result
    }

    /// Searches using HoloSphere superposition graph traversal and returns detailed execution work diagnostics.
    pub fn search_indices_superposition_diagnostics(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, GraphSearchDiagnostics)> {
        let _lifecycle = self.lifecycle.read();
        if query.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        if k == 0 || self.is_empty() {
            return Ok((Vec::new(), GraphSearchDiagnostics::default()));
        }

        let start_time = Instant::now();
        let active = self.active_searches.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.peak_active_searches
            .fetch_max(active, AtomicOrdering::Relaxed);
        defer! { self.active_searches.fetch_sub(1, AtomicOrdering::Relaxed); }

        let result = self.search_indices_superposition_internal(query, k, filter_mask);
        self.record_search_latency(start_time);
        result
    }

    /// Strict Rivero search with guaranteed mathematically bounded corpus-independent ceiling.
    pub fn search_indices_strict(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, RiveroSearchDiagnostics)> {
        let config = RiveroProfile::Strict.config();
        self.search_indices_o1_with_config(query, k, filter_mask, &config)
    }

    /// Staged confidence-adaptive Rivero search with progressive state reuse.
    ///
    /// Executes progressive Rivero stages (`Fast` -> `Balanced` -> `Strict`) with zero-redundancy
    /// work reuse (preserving probed cell buffers and exact vector evaluations).
    /// Escalates dynamically only when routing confidence falls below threshold.
    #[allow(unused_assignments)]
    pub fn search_indices_adaptive(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        policy: AdaptivePolicy,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, AdaptiveSearchDiagnostics)> {
        let _lifecycle = self.lifecycle.read();
        let address = self.compile_rivero_address(query)?;
        if k == 0 || self.is_empty() {
            return Ok((Vec::new(), AdaptiveSearchDiagnostics::default()));
        }

        let start_time = Instant::now();
        let active = self.active_searches.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.peak_active_searches
            .fetch_max(active, AtomicOrdering::Relaxed);
        defer! { self.active_searches.fetch_sub(1, AtomicOrdering::Relaxed); }

        let query_data = query.complex_data();
        let query_norm_sq = query.norm_squared();
        let (witness_degree, witness_seed_limit, witness_second_seed_limit, dist_fn) = {
            let cfg = self.config.read();
            (
                rivero_witness::bounded_degree(cfg.rivero_witness_degree),
                rivero_witness::bounded_seeds(cfg.rivero_witness_seeds),
                rivero_witness::bounded_seeds(cfg.rivero_witness_second_seeds),
                cfg.distance_function,
            )
        };

        let mut route_state = AdaptiveRouteState::new();
        let mut all_scored: Vec<(NodeIndex, SimilarityScore)> = Vec::with_capacity(1024);
        let mut previous_topk: Vec<(NodeIndex, SimilarityScore)> = Vec::new();
        let mut current_profile = RiveroProfile::Fast;
        let initial_profile = current_profile;
        let mut stages_executed = 0u8;
        let mut confidence_initial = 0.0f32;
        let mut latest_confidence = RiveroConfidence::default();
        let mut latest_rivero_diag = RiveroSearchDiagnostics::default();
        let mut latest_results: Vec<(NodeIndex, SimilarityScore)> = Vec::new();

        THREAD_VISITED_POOL.with(|pool| {
            let mut visited = pool.borrow_mut();
            let epoch = visited.next_epoch(self.arena.len());

            loop {
                stages_executed += 1;
                route_state.expand_to_profile(&self.rivero_index, &address, current_profile);
                let target_config = current_profile.config();
                let selected_cap = target_config.query_candidate_cap;

                let candidates: Vec<NodeIndex> = route_state
                    .current_voted
                    .iter()
                    .take(selected_cap)
                    .map(|candidate| candidate.slot)
                    .collect();

                let mut non_live_rejections = 0usize;
                let mut filter_rejections = 0usize;

                let initial_eval_limit = if filter_mask.is_some() {
                    // Under metadata filtering, dynamically evaluate across the full candidate pool to replenish rejected candidates
                    candidates.len()
                } else {
                    match current_profile {
                        RiveroProfile::Fast => candidates.len().min(512),
                        RiveroProfile::Balanced => candidates.len().min(1024),
                        RiveroProfile::Strict => candidates.len(),
                    }
                };
                for (cand_idx, &cand) in candidates[..initial_eval_limit].iter().enumerate() {
                    if cand_idx + 4 < initial_eval_limit {
                        let next_cand = candidates[cand_idx + 4];
                        if self.arena.is_live(next_cand) {
                            let next_v = self.arena.get_vector_slice(next_cand);
                            prefetch_vector(next_v);
                        }
                    }

                    if visited.is_visited(cand, epoch) {
                        continue;
                    }
                    visited.mark_visited(cand, epoch);
                    if !self.arena.is_live(cand) {
                        non_live_rejections += 1;
                        continue;
                    }
                    if filter_mask.is_some_and(|m| !m.contains(cand)) {
                        filter_rejections += 1;
                        continue;
                    }
                    let v = self.arena.get_vector_slice(cand);
                    let norm_sq = self.arena.get_norm_squared(cand);
                    let score = self.similarity_score_slices_with_metric(
                        query_data,
                        v,
                        query_norm_sq,
                        norm_sq,
                        dist_fn,
                    );
                    all_scored.push((cand, score));
                }

                // Filter-Aware Adaptive Expansion: If high-selectivity filter starved candidates, expand probe budget
                if filter_mask.is_some()
                    && all_scored.len() < k
                    && current_profile != RiveroProfile::Strict
                {
                    let expanded_budget = target_config.cell_budget * 2;
                    let mut expanded_config = target_config;
                    expanded_config.cell_budget = expanded_budget;
                    expanded_config.simhash_query_probes =
                        (expanded_config.simhash_query_probes * 2).min(64);
                    route_state.expand_to_config(&self.rivero_index, &address, expanded_config);
                    for extra in route_state
                        .current_voted
                        .iter()
                        .skip(selected_cap)
                        .take(selected_cap)
                    {
                        let cand = extra.slot;
                        if !visited.is_visited(cand, epoch)
                            && self.arena.is_live(cand)
                            && filter_mask.unwrap().contains(cand)
                        {
                            visited.mark_visited(cand, epoch);
                            let v = self.arena.get_vector_slice(cand);
                            let norm_sq = self.arena.get_norm_squared(cand);
                            let score = self.similarity_score_slices_with_metric(
                                query_data,
                                v,
                                query_norm_sq,
                                norm_sq,
                                dist_fn,
                            );
                            all_scored.push((cand, score));
                        }
                    }
                }

                // Witness expansion dynamically scaled to stage profile

                let (stage_seeds, stage_second_seeds, stage_degree) = match current_profile {
                    RiveroProfile::Fast => (
                        witness_seed_limit.min(12),
                        witness_second_seed_limit.min(2),
                        witness_degree.min(16),
                    ),
                    RiveroProfile::Balanced => (
                        witness_seed_limit.min(24),
                        witness_second_seed_limit.min(4),
                        witness_degree.min(32),
                    ),
                    RiveroProfile::Strict => (
                        witness_seed_limit,
                        witness_second_seed_limit,
                        witness_degree,
                    ),
                };

                all_scored.sort_unstable_by(|lhs, rhs| {
                    rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                });
                let mut seeds: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_SEEDS]> = SmallVec::new();
                seeds.extend(all_scored.iter().take(stage_seeds).map(|c| c.0));

                let mut witness_candidates_added = 0usize;
                let mut first_hop_scored: SmallVec<
                    [(NodeIndex, SimilarityScore);
                        RIVERO_WITNESS_MAX_DEGREE * RIVERO_WITNESS_MAX_SEEDS],
                > = SmallVec::new();
                let mut connections: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_DEGREE]> =
                    SmallVec::new();
                let mut witness_edges_scanned = 0usize;

                for &seed in &seeds {
                    self.copy_rivero_witness_connections(
                        seed,
                        true,
                        stage_degree,
                        &mut connections,
                    );
                    for &index in &connections {
                        witness_edges_scanned += 1;
                        if visited.is_visited(index, epoch) {
                            continue;
                        }
                        visited.mark_visited(index, epoch);
                        witness_candidates_added += 1;
                        if !self.arena.is_live(index) {
                            non_live_rejections += 1;
                            continue;
                        }
                        if filter_mask.is_some_and(|m| !m.contains(index)) {
                            filter_rejections += 1;
                            continue;
                        }
                        let v = self.arena.get_vector_slice(index);
                        let norm_sq = self.arena.get_norm_squared(index);
                        let score = self.similarity_score_slices_with_metric(
                            query_data,
                            v,
                            query_norm_sq,
                            norm_sq,
                            dist_fn,
                        );
                        let candidate = (index, score);
                        all_scored.push(candidate);
                        first_hop_scored.push(candidate);
                    }
                }

                first_hop_scored.sort_unstable_by(|lhs, rhs| {
                    rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                });

                let mut second_seeds: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_SEEDS]> =
                    SmallVec::new();
                second_seeds.extend(
                    first_hop_scored
                        .iter()
                        .take(stage_second_seeds)
                        .map(|c| c.0),
                );

                for &seed in &second_seeds {
                    self.copy_rivero_witness_connections(
                        seed,
                        true,
                        stage_degree,
                        &mut connections,
                    );
                    for &index in &connections {
                        witness_edges_scanned += 1;
                        if visited.is_visited(index, epoch) {
                            continue;
                        }
                        visited.mark_visited(index, epoch);
                        witness_candidates_added += 1;
                        if !self.arena.is_live(index) {
                            non_live_rejections += 1;
                            continue;
                        }
                        if filter_mask.is_some_and(|m| !m.contains(index)) {
                            filter_rejections += 1;
                            continue;
                        }
                        let v = self.arena.get_vector_slice(index);
                        let norm_sq = self.arena.get_norm_squared(index);
                        let score = self.similarity_score_slices_with_metric(
                            query_data,
                            v,
                            query_norm_sq,
                            norm_sq,
                            dist_fn,
                        );
                        all_scored.push((index, score));
                    }
                }

                all_scored.sort_unstable_by(|lhs, rhs| {
                    rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                });

                let returned_limit = k.min(all_scored.len());
                let current_topk: Vec<(NodeIndex, SimilarityScore)> =
                    all_scored[..returned_limit].to_vec();

                // Cross-stage stability calculation
                let cross_stage_stability = if !previous_topk.is_empty() && k > 0 {
                    let current_set: HashSet<NodeIndex> =
                        current_topk.iter().take(k).map(|c| c.0).collect();
                    let hits = previous_topk
                        .iter()
                        .take(k)
                        .filter(|c| current_set.contains(&c.0))
                        .count();
                    Some((hits as f32) / (k as f32))
                } else {
                    None
                };

                let conf = RiveroConfidence::evaluate(
                    &route_state.current_voted,
                    &route_state.current_diagnostics,
                    &all_scored,
                    witness_candidates_added,
                    k,
                    cross_stage_stability,
                    current_profile,
                );

                if stages_executed == 1 {
                    confidence_initial = conf.score;
                }
                latest_confidence = conf;
                latest_results = current_topk.clone();

                latest_rivero_diag = RiveroSearchDiagnostics {
                    cells_probed: route_state.cells_visited,
                    resident_reads: route_state.cumulative_reads,
                    resident_scans: route_state.cumulative_scans,
                    candidate_read_bound: target_config.candidate_read_bound(),
                    resident_scan_bound: target_config.resident_scan_bound(),
                    unique_candidates: all_scored.len(),
                    raw_unique_candidates: route_state.current_diagnostics.raw_unique_candidates,
                    route_candidates_selected: candidates.len(),
                    raw_unique_candidate_bound: target_config.candidate_read_bound(),
                    selected_candidate_bound: selected_cap,
                    non_live_rejections,
                    filter_rejections,
                    exact_score_evaluations: all_scored.len(),
                    witness_seeds: seeds.len(),
                    witness_second_hop_seeds: second_seeds.len(),
                    witness_edges_scanned,
                    witness_candidates_added,
                    witness_edge_scan_bound: rivero_witness::witness_two_hop_edge_scan_bound(
                        witness_seed_limit,
                        witness_second_seed_limit,
                        witness_degree,
                    ),
                    results_returned: current_topk.len(),
                    fallback_used: false,
                };

                // Check stopping condition
                if !conf.escalation_recommended || current_profile == RiveroProfile::Strict {
                    break;
                }

                previous_topk = current_topk;
                if let Some(next_prof) = current_profile.next_escalation() {
                    current_profile = next_prof;
                } else {
                    break;
                }
            }
        });

        // Preserve Rivero-stage exact work before optional fallback consumes `all_scored`.
        let cumulative_exact_scores = all_scored.len();
        let mut graph_fallback_used = false;
        if latest_confidence.escalation_recommended
            && policy == AdaptivePolicy::AllowGraphFallback
            && current_profile == RiveroProfile::Strict
        {
            let graph_results = self.search_indices_graph_internal(query, k, filter_mask)?;
            let mut combined_map: std::collections::HashMap<NodeIndex, SimilarityScore> =
                std::collections::HashMap::with_capacity(all_scored.len() + graph_results.len());
            for (idx, score) in all_scored.drain(..) {
                combined_map.insert(idx, score);
            }
            for (idx, score) in graph_results {
                combined_map.entry(idx).or_insert(score);
            }
            let mut combined: Vec<(NodeIndex, SimilarityScore)> =
                combined_map.into_iter().collect();
            combined.sort_unstable_by(|lhs, rhs| {
                rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
            });
            combined.truncate(k);
            latest_results = combined;
            graph_fallback_used = true;
            latest_rivero_diag.fallback_used = true;
        }

        self.record_rivero_search(start_time, latest_rivero_diag, graph_fallback_used);

        let diag = AdaptiveSearchDiagnostics {
            initial_profile,
            final_profile: current_profile,
            stages_executed,
            confidence_initial,
            confidence_final: latest_confidence.score,
            escalated: stages_executed > 1,
            graph_fallback_used,
            cumulative_resident_scans: route_state.cumulative_scans,
            cumulative_exact_scores,
            confidence: latest_confidence,
            rivero: latest_rivero_diag,
        };

        Ok((latest_results, diag))
    }

    /// Strict Rivero search with fixed corpus-independent work and no graph fallback.
    ///
    /// Address compilation and exact reranking remain linear in vector dimension.
    /// Territory probes and resident reads are bounded independently of corpus size.
    pub fn search_indices_o1_filtered(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<Vec<(NodeIndex, SimilarityScore)>> {
        self.search_indices_o1_with_diagnostics(query, k, filter_mask)
            .map(|(results, _)| results)
    }

    /// Strict Rivero search returning counters that mechanically prove the
    /// configured probe and candidate-read bounds for this query.
    pub fn search_indices_o1_with_diagnostics(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, RiveroSearchDiagnostics)> {
        let address = self.compile_rivero_address(query)?;
        self.search_indices_with_rivero_address_and_diagnostics(query, &address, k, filter_mask)
    }

    /// Strict Rivero search using a precompiled fixed-size address.
    ///
    /// This avoids dimension-dependent address compilation inside the resolver. Exact
    /// complex-metric reranking still reads the bounded candidates' full vectors.
    pub fn search_indices_with_rivero_address(
        &self,
        query: &VectorEmbedding,
        address: &RiveroAddress,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<Vec<(NodeIndex, SimilarityScore)>> {
        self.search_indices_with_rivero_address_and_diagnostics(query, address, k, filter_mask)
            .map(|(results, _)| results)
    }

    /// Precompiled-address strict search with fixed-work diagnostics.
    pub fn search_indices_with_rivero_address_and_diagnostics(
        &self,
        query: &VectorEmbedding,
        address: &RiveroAddress,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, RiveroSearchDiagnostics)> {
        let _lifecycle = self.lifecycle.read();
        self.validate_rivero_query(query, address)?;
        if k == 0 || self.is_empty() {
            return Ok((Vec::new(), RiveroSearchDiagnostics::default()));
        }
        if !self.config.read().rivero_enabled {
            return Err(HNSQRError::InvalidConfig(
                "Rivero routing is disabled for this index".to_string(),
            ));
        }

        let start_time = Instant::now();
        let active = self.active_searches.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.peak_active_searches
            .fetch_max(active, AtomicOrdering::Relaxed);
        defer! { self.active_searches.fetch_sub(1, AtomicOrdering::Relaxed); }

        let budget = self.config.read().rivero_cell_budget;
        let (resolved, diagnostics) =
            self.resolve_rivero_candidates(query, address, k, filter_mask, budget)?;
        self.record_rivero_search(start_time, diagnostics, false);
        Ok((resolved, diagnostics))
    }

    /// Strict Rivero search using a custom [`RiveroConfig`] for Pareto sweep and tuning.
    pub fn search_indices_o1_with_config(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        rivero_config: &RiveroConfig,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, RiveroSearchDiagnostics)> {
        let address = self.compile_rivero_address(query)?;
        let start_time = Instant::now();
        let active = self.active_searches.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.peak_active_searches
            .fetch_max(active, AtomicOrdering::Relaxed);
        defer! { self.active_searches.fetch_sub(1, AtomicOrdering::Relaxed); }

        let (resolved, diagnostics) =
            self.resolve_rivero_candidates_config(query, &address, k, filter_mask, rivero_config)?;
        self.record_rivero_search(start_time, diagnostics, false);
        Ok((resolved, diagnostics))
    }

    /// Rivero search executed strictly under a specific profile's budget and witness parameters.
    pub fn search_indices_profile(
        &self,
        query: &VectorEmbedding,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        profile: RiveroProfile,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, RiveroSearchDiagnostics)> {
        let address = self.compile_rivero_address(query)?;
        let start_time = Instant::now();
        let active = self.active_searches.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        self.peak_active_searches
            .fetch_max(active, AtomicOrdering::Relaxed);
        defer! { self.active_searches.fetch_sub(1, AtomicOrdering::Relaxed); }

        let (resolved, diagnostics) =
            self.resolve_rivero_candidates_profile(query, &address, k, filter_mask, profile)?;
        self.record_rivero_search(start_time, diagnostics, false);
        Ok((resolved, diagnostics))
    }

    /// Compiles an embedding into its fixed-size Rivero routing address.
    pub fn compile_rivero_address(&self, query: &VectorEmbedding) -> HNSQRResult<RiveroAddress> {
        if query.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        Ok(self.rivero_compiler.read().compile(query.complex_data()))
    }

    /// Evaluates a structured boolean metadata filter expression into a Roaring Bitmap mask.
    pub fn compile_filter_mask(
        &self,
        expr: &crate::metadata::index::FilterExpr,
    ) -> HNSQRResult<roaring::RoaringBitmap> {
        Ok(self.metadata_index.evaluate_filter(expr, self.arena.len()))
    }

    /// Traces every Ground Truth (GT) neighbor through the candidate routing, voting, witness expansion, and reranking pipeline.
    pub fn trace_gt_coverage(
        &self,
        query: &VectorEmbedding,
        gt_ids: &[NodeIndex],
        rivero_config: rivero::RiveroConfig,
    ) -> HNSQRResult<GtPipelineTrace> {
        let address = self.compile_rivero_address(query)?;
        let mut route_state = rivero::AdaptiveRouteState::new();
        route_state.expand_to_config(&self.rivero_index, &address, rivero_config);

        let raw_voted = &route_state.current_voted;
        let mut missing_gt_ranks = Vec::with_capacity(gt_ids.len());
        let mut gt_in_raw_route = 0usize;

        for &gt in gt_ids {
            if let Some(pos) = raw_voted.iter().position(|c| c.slot == gt) {
                gt_in_raw_route += 1;
                missing_gt_ranks.push(Some(pos));
            } else {
                missing_gt_ranks.push(None);
            }
        }

        let selected_cap = rivero_config.query_candidate_cap.min(raw_voted.len());
        let selected_cands: Vec<NodeIndex> = raw_voted
            .iter()
            .take(selected_cap)
            .map(|c| c.slot)
            .collect();
        let gt_after_vote_selection = gt_ids
            .iter()
            .filter(|id| selected_cands.contains(id))
            .count();

        let (witness_degree, witness_seed_limit, witness_second_seed_limit) = {
            let cfg = self.config.read();
            (
                cfg.rivero_witness_degree,
                cfg.rivero_witness_seeds,
                cfg.rivero_witness_second_seeds,
            )
        };

        let mut exact_scores: std::collections::HashMap<NodeIndex, SimilarityScore> =
            std::collections::HashMap::with_capacity(selected_cands.len() * 2);
        let query_data = query.complex_data();
        let query_norm_sq = query.norm_squared();

        for &cand in &selected_cands {
            if self.arena.is_live(cand) {
                let v = self.arena.get_vector_slice(cand);
                let norm_sq = self.arena.get_norm_squared(cand);
                let score = self.similarity_score_slices(query_data, v, query_norm_sq, norm_sq);
                exact_scores.insert(cand, score);
            }
        }

        let mut scored: Vec<(NodeIndex, SimilarityScore)> = selected_cands
            .iter()
            .filter_map(|&cand| exact_scores.get(&cand).map(|&score| (cand, score)))
            .collect();
        scored.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));

        let mut seeds: smallvec::SmallVec<[NodeIndex; rivero_witness::RIVERO_WITNESS_MAX_SEEDS]> =
            smallvec::SmallVec::new();
        seeds.extend(scored.iter().take(witness_seed_limit).map(|c| c.0));

        let mut witness_candidates: smallvec::SmallVec<
            [NodeIndex;
                rivero_witness::RIVERO_WITNESS_MAX_DEGREE
                    * rivero_witness::RIVERO_WITNESS_MAX_SEEDS
                    * 2],
        > = smallvec::SmallVec::new();
        let mut first_hop_scored: smallvec::SmallVec<
            [(NodeIndex, SimilarityScore);
                rivero_witness::RIVERO_WITNESS_MAX_DEGREE
                    * rivero_witness::RIVERO_WITNESS_MAX_SEEDS],
        > = smallvec::SmallVec::new();
        let mut connections: smallvec::SmallVec<
            [NodeIndex; rivero_witness::RIVERO_WITNESS_MAX_DEGREE],
        > = smallvec::SmallVec::new();

        for &seed in &seeds {
            self.copy_rivero_witness_connections(seed, true, witness_degree, &mut connections);
            for &index in &connections {
                if exact_scores.contains_key(&index) || witness_candidates.contains(&index) {
                    continue;
                }
                witness_candidates.push(index);
                if !self.arena.is_live(index) {
                    continue;
                }
                let v = self.arena.get_vector_slice(index);
                let norm_sq = self.arena.get_norm_squared(index);
                let score = self.similarity_score_slices(query_data, v, query_norm_sq, norm_sq);
                exact_scores.insert(index, score);
                scored.push((index, score));
                first_hop_scored.push((index, score));
            }
        }

        first_hop_scored
            .sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));
        let mut second_seeds: smallvec::SmallVec<
            [NodeIndex; rivero_witness::RIVERO_WITNESS_MAX_SEEDS],
        > = smallvec::SmallVec::new();
        second_seeds.extend(
            first_hop_scored
                .iter()
                .take(witness_second_seed_limit)
                .map(|c| c.0),
        );

        for &seed in &second_seeds {
            self.copy_rivero_witness_connections(seed, true, witness_degree, &mut connections);
            for &index in &connections {
                if exact_scores.contains_key(&index) || witness_candidates.contains(&index) {
                    continue;
                }
                witness_candidates.push(index);
                if !self.arena.is_live(index) {
                    continue;
                }
                let v = self.arena.get_vector_slice(index);
                let norm_sq = self.arena.get_norm_squared(index);
                let score = self.similarity_score_slices(query_data, v, query_norm_sq, norm_sq);
                exact_scores.insert(index, score);
                scored.push((index, score));
            }
        }

        scored.sort_unstable_by(|lhs, rhs| rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0)));

        let all_post_witness_cands: std::collections::HashSet<NodeIndex> =
            exact_scores.keys().copied().collect();
        let gt_after_witness = gt_ids
            .iter()
            .filter(|id| all_post_witness_cands.contains(id))
            .count();

        let final_top10: Vec<NodeIndex> = scored.iter().take(10).map(|s| s.0).collect();
        let gt_in_final_results = gt_ids.iter().filter(|id| final_top10.contains(id)).count();

        let top1_recalled =
            !final_top10.is_empty() && !gt_ids.is_empty() && final_top10[0] == gt_ids[0];
        let recall_at_10 = gt_in_final_results as f64 / gt_ids.len().max(1) as f64;

        Ok(GtPipelineTrace {
            gt_count: gt_ids.len(),
            gt_in_raw_route,
            gt_after_vote_selection,
            gt_after_witness,
            gt_in_final_results,
            missing_gt_ranks,
            top1_recalled,
            recall_at_10,
        })
    }

    /// Returns a reference to the internal metadata inverted index.
    #[must_use]
    pub fn metadata_index(&self) -> &MetadataInvertedIndex {
        &self.metadata_index
    }

    /// Executes parallel deterministic bulk construction of Rivero routing territories and witnesses.
    pub fn bulk_build_rivero(
        &self,
        vectors: &[VectorEmbedding],
        profile: RiveroProfile,
    ) -> HNSQRResult<BulkBuildTelemetry> {
        let (degree, seeds, second_seeds, addr_cfg, dist_fn) = {
            let cfg = self.config.read();
            (
                cfg.rivero_witness_degree,
                cfg.rivero_witness_seeds,
                cfg.rivero_witness_second_seeds,
                cfg.rivero_address_config,
                cfg.distance_function,
            )
        };

        let builder = RiveroBulkBuilder::with_profile(profile)
            .with_address_config(addr_cfg)
            .with_distance_function(dist_fn)
            .with_witness_params(degree, seeds, second_seeds);

        let built = builder.build(vectors)?;
        let telemetry = built.telemetry.clone();
        self.install_rivero_state(built)?;
        Ok(telemetry)
    }

    /// Transactionally installs a fully built Rivero state into the live index.
    pub fn install_rivero_state(&self, built: BuiltRiveroState) -> HNSQRResult<()> {
        let _lifecycle = self.lifecycle.write();
        let BuiltRiveroState {
            territory,
            witnesses,
            descriptor,
            ..
        } = built;

        if descriptor.dimension != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: descriptor.dimension,
            });
        }
        if descriptor.schema_version != crate::rivero::RIVERO_SCHEMA_VERSION {
            return Err(HNSQRError::InvalidConfig(format!(
                "BuiltRiveroState schema version mismatch: expected {}, found {}",
                crate::rivero::RIVERO_SCHEMA_VERSION,
                descriptor.schema_version
            )));
        }

        {
            let cfg = self.config.read();
            if descriptor.distance_function != cfg.distance_function {
                return Err(HNSQRError::InvalidConfig(format!(
                    "Rivero state metric mismatch: index={:?}, built={:?}",
                    cfg.distance_function, descriptor.distance_function
                )));
            }
            if descriptor.address_config != cfg.rivero_address_config {
                return Err(HNSQRError::InvalidConfig(
                    "Rivero state address schema does not match live index".into(),
                ));
            }
        }

        // Synchronize live Rivero execution settings
        {
            let mut cfg = self.config.write();
            cfg.rivero_config = descriptor.rivero_config;
            cfg.rivero_witness_degree = descriptor.witness_degree;
            cfg.rivero_witness_seeds = descriptor.witness_seeds;
            cfg.rivero_witness_second_seeds = descriptor.witness_second_seeds;
            cfg.rivero_enabled = true;
        }
        *self.rivero_compiler.write() =
            rivero::RiveroCompiler::with_config(self.dimension, descriptor.address_config);

        // Install witnesses into arena nodes
        for (slot, witness_list) in witnesses.into_iter().enumerate() {
            if let Some(node) = self.arena.get_node(slot as NodeIndex) {
                *node.rivero_witnesses.write() = witness_list;
            }
        }

        // Install territory index
        self.rivero_index.replace_from(territory);

        Ok(())
    }

    /// Computes a canonical cryptographic fingerprint over all populated Rivero cells and witness lists.
    #[must_use]
    pub fn structural_fingerprint(&self) -> [u8; 32] {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();

        let territory_fp = self.rivero_index.structural_fingerprint();
        hasher.update(territory_fp);

        let n = self.arena.len();
        for slot in 0..n as NodeIndex {
            if let Some(node) = self.arena.get_node(slot) {
                let witnesses = node.rivero_witnesses.read();
                hasher.update(slot.to_le_bytes());
                hasher.update((witnesses.len() as u32).to_le_bytes());
                for w in witnesses.iter() {
                    hasher.update(w.index.to_le_bytes());
                    hasher.update(w.similarity.to_bits().to_le_bytes());
                }
            }
        }

        hasher.finalize().into()
    }

    fn validate_rivero_query(
        &self,
        query: &VectorEmbedding,
        address: &RiveroAddress,
    ) -> HNSQRResult<()> {
        if query.dimension() != self.dimension {
            return Err(HNSQRError::DimensionMismatch {
                expected: self.dimension,
                actual: query.dimension(),
            });
        }
        if address.schema_version != RIVERO_SCHEMA_VERSION
            || address.source_dimension as usize != self.dimension
        {
            return Err(HNSQRError::InvalidConfig(format!(
                "Rivero address schema/dimension mismatch: schema={}, dimension={}",
                address.schema_version, address.source_dimension
            )));
        }
        Ok(())
    }

    fn resolve_rivero_candidates(
        &self,
        query: &VectorEmbedding,
        address: &RiveroAddress,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        per_cell_budget: usize,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, RiveroSearchDiagnostics)> {
        let mut config = self.config.read().rivero_config;
        config.cell_budget = per_cell_budget;
        self.resolve_rivero_candidates_config(query, address, k, filter_mask, &config)
    }

    fn resolve_rivero_candidates_profile(
        &self,
        query: &VectorEmbedding,
        address: &RiveroAddress,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        profile: RiveroProfile,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, RiveroSearchDiagnostics)> {
        self.validate_rivero_query(query, address)?;
        let query_data = query.complex_data();
        let query_norm_sq = query.norm_squared();
        let rivero_config = profile.config();
        let (witness_degree, witness_seed_limit, witness_second_seed_limit, strict_rivero, dist_fn) = {
            let config = self.config.read();
            (
                rivero_witness::bounded_degree(config.rivero_witness_degree)
                    .min(profile.witness_degree()),
                rivero_witness::bounded_seeds(config.rivero_witness_seeds)
                    .min(profile.witness_seeds()),
                rivero_witness::bounded_seeds(config.rivero_witness_second_seeds)
                    .min(profile.witness_second_seeds()),
                !config.rivero_fallback_on_underfill,
                config.distance_function,
            )
        };

        Ok(self.rivero_index.with_candidates_config(
            address,
            &rivero_config,
            |candidates, route| {
                let mut scored =
                    Vec::with_capacity(candidates.len() + witness_seed_limit * witness_degree * 2);
                let mut diagnostics = RiveroSearchDiagnostics {
                    cells_probed: route.cells_probed,
                    resident_reads: route.resident_reads,
                    resident_scans: route.resident_scans,
                    candidate_read_bound: route.candidate_read_bound,
                    resident_scan_bound: route.resident_scan_bound,
                    unique_candidates: route.unique_candidates,
                    raw_unique_candidates: route.raw_unique_candidates,
                    route_candidates_selected: route.unique_candidates,
                    raw_unique_candidate_bound: route.raw_unique_candidate_bound,
                    selected_candidate_bound: route.selected_candidate_bound,
                    witness_edge_scan_bound: rivero_witness::witness_two_hop_edge_scan_bound(
                        witness_seed_limit,
                        witness_second_seed_limit,
                        witness_degree,
                    ),
                    ..RiveroSearchDiagnostics::default()
                };

                THREAD_VISITED_POOL.with(|pool| {
                    let mut visited = pool.borrow_mut();
                    let epoch = visited.next_epoch(self.arena.len());

                    let eval_limit = if filter_mask.is_some() {
                        candidates.len()
                    } else {
                        candidates.len().min(profile.candidate_eval_limit())
                    };
                    for (cand_idx, &index) in candidates[..eval_limit].iter().enumerate() {
                        if cand_idx + 4 < eval_limit {
                            let next_cand = candidates[cand_idx + 4];
                            if self.arena.is_live(next_cand) {
                                let next_v = self.arena.get_vector_slice(next_cand);
                                prefetch_vector(next_v);
                            }
                        }

                        visited.mark_visited(index, epoch);
                        if !self.arena.is_live(index) {
                            diagnostics.non_live_rejections += 1;
                            continue;
                        }
                        if filter_mask.is_some_and(|mask| !mask.contains(index)) {
                            diagnostics.filter_rejections += 1;
                            continue;
                        }
                        let vector = self.arena.get_vector_slice(index);
                        let norm_sq = self.arena.get_norm_squared(index);
                        diagnostics.exact_score_evaluations += 1;
                        scored.push((
                            index,
                            self.similarity_score_slices_with_metric(
                                query_data,
                                vector,
                                query_norm_sq,
                                norm_sq,
                                dist_fn,
                            ),
                        ));
                    }

                    scored.sort_unstable_by(|lhs, rhs| {
                        rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                    });
                    let mut seeds: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_SEEDS]> =
                        SmallVec::new();
                    seeds.extend(
                        scored
                            .iter()
                            .take(witness_seed_limit)
                            .map(|candidate| candidate.0),
                    );
                    diagnostics.witness_seeds = seeds.len();

                    let mut first_hop_scored: SmallVec<
                        [(NodeIndex, SimilarityScore);
                            RIVERO_WITNESS_MAX_DEGREE * RIVERO_WITNESS_MAX_SEEDS],
                    > = SmallVec::new();
                    let mut connections: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_DEGREE]> =
                        SmallVec::new();
                    for seed in seeds {
                        self.copy_rivero_witness_connections(
                            seed,
                            strict_rivero,
                            witness_degree,
                            &mut connections,
                        );
                        for &index in &connections {
                            diagnostics.witness_edges_scanned += 1;
                            if visited.is_visited(index, epoch) {
                                continue;
                            }
                            visited.mark_visited(index, epoch);
                            diagnostics.witness_candidates_added += 1;
                            diagnostics.unique_candidates += 1;
                            if !self.arena.is_live(index) {
                                diagnostics.non_live_rejections += 1;
                                continue;
                            }
                            if filter_mask.is_some_and(|mask| !mask.contains(index)) {
                                diagnostics.filter_rejections += 1;
                                continue;
                            }
                            let vector = self.arena.get_vector_slice(index);
                            let norm_sq = self.arena.get_norm_squared(index);
                            diagnostics.exact_score_evaluations += 1;
                            let score = self.similarity_score_slices_with_metric(
                                query_data,
                                vector,
                                query_norm_sq,
                                norm_sq,
                                dist_fn,
                            );
                            let candidate = (index, score);
                            scored.push(candidate);
                            first_hop_scored.push(candidate);
                        }
                    }

                    first_hop_scored.sort_unstable_by(|lhs, rhs| {
                        rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                    });
                    let mut second_seeds: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_SEEDS]> =
                        SmallVec::new();
                    second_seeds.extend(
                        first_hop_scored
                            .iter()
                            .take(witness_second_seed_limit)
                            .map(|candidate| candidate.0),
                    );
                    diagnostics.witness_second_hop_seeds = second_seeds.len();

                    for seed in second_seeds {
                        self.copy_rivero_witness_connections(
                            seed,
                            strict_rivero,
                            witness_degree,
                            &mut connections,
                        );
                        for &index in &connections {
                            diagnostics.witness_edges_scanned += 1;
                            if visited.is_visited(index, epoch) {
                                continue;
                            }
                            visited.mark_visited(index, epoch);
                            diagnostics.witness_candidates_added += 1;
                            diagnostics.unique_candidates += 1;
                            if !self.arena.is_live(index) {
                                diagnostics.non_live_rejections += 1;
                                continue;
                            }
                            if filter_mask.is_some_and(|mask| !mask.contains(index)) {
                                diagnostics.filter_rejections += 1;
                                continue;
                            }
                            let vector = self.arena.get_vector_slice(index);
                            let norm_sq = self.arena.get_norm_squared(index);
                            diagnostics.exact_score_evaluations += 1;
                            scored.push((
                                index,
                                self.similarity_score_slices_with_metric(
                                    query_data,
                                    vector,
                                    query_norm_sq,
                                    norm_sq,
                                    dist_fn,
                                ),
                            ));
                        }
                    }

                    scored.sort_unstable_by(|lhs, rhs| {
                        rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                    });
                    scored.truncate(k);
                    diagnostics.results_returned = scored.len();
                    (scored, diagnostics)
                })
            },
        ))
    }

    fn resolve_rivero_candidates_config(
        &self,
        query: &VectorEmbedding,
        address: &RiveroAddress,
        k: usize,
        filter_mask: Option<&roaring::RoaringBitmap>,
        rivero_config: &RiveroConfig,
    ) -> HNSQRResult<(Vec<(NodeIndex, SimilarityScore)>, RiveroSearchDiagnostics)> {
        self.validate_rivero_query(query, address)?;
        let query_data = query.complex_data();
        let query_norm_sq = query.norm_squared();
        let (witness_degree, witness_seed_limit, witness_second_seed_limit, strict_rivero, dist_fn) = {
            let config = self.config.read();
            (
                rivero_witness::bounded_degree(config.rivero_witness_degree),
                rivero_witness::bounded_seeds(config.rivero_witness_seeds),
                rivero_witness::bounded_seeds(config.rivero_witness_second_seeds),
                !config.rivero_fallback_on_underfill,
                config.distance_function,
            )
        };

        Ok(self
            .rivero_index
            .with_candidates_config(address, rivero_config, |candidates, route| {
                let mut scored =
                    Vec::with_capacity(candidates.len() + witness_seed_limit * witness_degree * 2);
                let mut diagnostics = RiveroSearchDiagnostics {
                    cells_probed: route.cells_probed,
                    resident_reads: route.resident_reads,
                    resident_scans: route.resident_scans,
                    candidate_read_bound: route.candidate_read_bound,
                    resident_scan_bound: route.resident_scan_bound,
                    unique_candidates: route.unique_candidates,
                    raw_unique_candidates: route.raw_unique_candidates,
                    route_candidates_selected: route.unique_candidates,
                    raw_unique_candidate_bound: route.raw_unique_candidate_bound,
                    selected_candidate_bound: route.selected_candidate_bound,
                    witness_edge_scan_bound: rivero_witness::witness_two_hop_edge_scan_bound(
                        witness_seed_limit,
                        witness_second_seed_limit,
                        witness_degree,
                    ),
                    ..RiveroSearchDiagnostics::default()
                };

                THREAD_VISITED_POOL.with(|pool| {
                    let mut visited = pool.borrow_mut();
                    let epoch = visited.next_epoch(self.arena.len());

                    let eval_limit = candidates.len();
                    for (cand_idx, &index) in candidates[..eval_limit].iter().enumerate() {
                        if cand_idx + 4 < eval_limit {
                            let next_cand = candidates[cand_idx + 4];
                            if self.arena.is_live(next_cand) {
                                let next_v = self.arena.get_vector_slice(next_cand);
                                prefetch_vector(next_v);
                            }
                        }

                        visited.mark_visited(index, epoch);
                        if !self.arena.is_live(index) {
                            diagnostics.non_live_rejections += 1;
                            continue;
                        }
                        if filter_mask.is_some_and(|mask| !mask.contains(index)) {
                            diagnostics.filter_rejections += 1;
                            continue;
                        }
                        let vector = self.arena.get_vector_slice(index);
                        let norm_sq = self.arena.get_norm_squared(index);
                        diagnostics.exact_score_evaluations += 1;
                        scored.push((
                            index,
                            self.similarity_score_slices_with_metric(
                                query_data,
                                vector,
                                query_norm_sq,
                                norm_sq,
                                dist_fn,
                            ),
                        ));
                    }

                    scored.sort_unstable_by(|lhs, rhs| {
                        rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                    });
                    let mut seeds: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_SEEDS]> =
                        SmallVec::new();
                    seeds.extend(
                        scored
                            .iter()
                            .take(witness_seed_limit)
                            .map(|candidate| candidate.0),
                    );
                    diagnostics.witness_seeds = seeds.len();

                    let mut first_hop_scored: SmallVec<
                        [(NodeIndex, SimilarityScore);
                            RIVERO_WITNESS_MAX_DEGREE * RIVERO_WITNESS_MAX_SEEDS],
                    > = SmallVec::new();
                    let mut connections: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_DEGREE]> =
                        SmallVec::new();
                    for seed in seeds {
                        self.copy_rivero_witness_connections(
                            seed,
                            strict_rivero,
                            witness_degree,
                            &mut connections,
                        );
                        for &index in &connections {
                            diagnostics.witness_edges_scanned += 1;
                            if visited.is_visited(index, epoch) {
                                continue;
                            }
                            visited.mark_visited(index, epoch);
                            diagnostics.witness_candidates_added += 1;
                            diagnostics.unique_candidates += 1;
                            if !self.arena.is_live(index) {
                                diagnostics.non_live_rejections += 1;
                                continue;
                            }
                            if filter_mask.is_some_and(|mask| !mask.contains(index)) {
                                diagnostics.filter_rejections += 1;
                                continue;
                            }
                            let vector = self.arena.get_vector_slice(index);
                            let norm_sq = self.arena.get_norm_squared(index);
                            diagnostics.exact_score_evaluations += 1;
                            let score = self.similarity_score_slices_with_metric(
                                query_data,
                                vector,
                                query_norm_sq,
                                norm_sq,
                                dist_fn,
                            );
                            let candidate = (index, score);
                            scored.push(candidate);
                            first_hop_scored.push(candidate);
                        }
                    }

                    first_hop_scored.sort_unstable_by(|lhs, rhs| {
                        rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                    });
                    let mut second_seeds: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_SEEDS]> =
                        SmallVec::new();
                    second_seeds.extend(
                        first_hop_scored
                            .iter()
                            .take(witness_second_seed_limit)
                            .map(|candidate| candidate.0),
                    );
                    diagnostics.witness_second_hop_seeds = second_seeds.len();

                    for seed in second_seeds {
                        self.copy_rivero_witness_connections(
                            seed,
                            strict_rivero,
                            witness_degree,
                            &mut connections,
                        );
                        for &index in &connections {
                            diagnostics.witness_edges_scanned += 1;
                            if visited.is_visited(index, epoch) {
                                continue;
                            }
                            visited.mark_visited(index, epoch);
                            diagnostics.witness_candidates_added += 1;
                            diagnostics.unique_candidates += 1;
                            if !self.arena.is_live(index) {
                                diagnostics.non_live_rejections += 1;
                                continue;
                            }
                            if filter_mask.is_some_and(|mask| !mask.contains(index)) {
                                diagnostics.filter_rejections += 1;
                                continue;
                            }
                            let vector = self.arena.get_vector_slice(index);
                            let norm_sq = self.arena.get_norm_squared(index);
                            diagnostics.exact_score_evaluations += 1;
                            scored.push((
                                index,
                                self.similarity_score_slices_with_metric(
                                    query_data,
                                    vector,
                                    query_norm_sq,
                                    norm_sq,
                                    dist_fn,
                                ),
                            ));
                        }
                    }

                    debug_assert!(
                        diagnostics.witness_edges_scanned <= diagnostics.witness_edge_scan_bound
                    );
                    debug_assert!(
                        diagnostics.witness_candidates_added <= diagnostics.witness_edges_scanned
                    );
                    if scored.len() > k {
                        scored.select_nth_unstable_by(k - 1, |lhs, rhs| {
                            rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                        });
                        scored.truncate(k);
                    }
                    scored.sort_unstable_by(|lhs, rhs| {
                        rhs.1.total_cmp(&lhs.1).then_with(|| lhs.0.cmp(&rhs.0))
                    });
                    diagnostics.results_returned = scored.len();
                    diagnostics.unique_candidates = diagnostics.exact_score_evaluations
                        + diagnostics.non_live_rejections
                        + diagnostics.filter_rejections;
                    (scored, diagnostics)
                })
            }))
    }

    fn record_rivero_attempt(&self, diagnostics: RiveroSearchDiagnostics, fallback: bool) {
        let mut stats = self.stats.write();
        stats.rivero_searches = stats.rivero_searches.saturating_add(1);
        stats.rivero_peak_candidates = stats
            .rivero_peak_candidates
            .max(diagnostics.exact_score_evaluations);
        stats.rivero_cells_probed = stats
            .rivero_cells_probed
            .saturating_add(diagnostics.cells_probed as u64);
        stats.rivero_resident_reads = stats
            .rivero_resident_reads
            .saturating_add(diagnostics.resident_reads as u64);
        stats.rivero_resident_scans = stats
            .rivero_resident_scans
            .saturating_add(diagnostics.resident_scans as u64);
        stats.rivero_witness_edges_scanned = stats
            .rivero_witness_edges_scanned
            .saturating_add(diagnostics.witness_edges_scanned as u64);
        stats.rivero_witness_candidates_added = stats
            .rivero_witness_candidates_added
            .saturating_add(diagnostics.witness_candidates_added as u64);
        stats.rivero_exact_score_evaluations = stats
            .rivero_exact_score_evaluations
            .saturating_add(diagnostics.exact_score_evaluations as u64);
        if diagnostics.exact_score_evaluations == 0 {
            stats.rivero_empty_routes = stats.rivero_empty_routes.saturating_add(1);
        }
        if fallback {
            stats.rivero_fallbacks = stats.rivero_fallbacks.saturating_add(1);
        }
    }

    fn record_rivero_search(
        &self,
        start_time: Instant,
        diagnostics: RiveroSearchDiagnostics,
        fallback: bool,
    ) {
        self.record_rivero_attempt(diagnostics, fallback);
        let mut stats = self.stats.write();
        Self::update_search_latency(&mut stats, start_time.elapsed().as_micros() as f64);
    }

    fn record_search_latency(&self, start_time: Instant) {
        let mut stats = self.stats.write();
        Self::update_search_latency(&mut stats, start_time.elapsed().as_micros() as f64);
    }

    fn update_search_latency(stats: &mut IndexStats, elapsed_us: f64) {
        stats.searches = stats.searches.saturating_add(1);
        if stats.searches == 1 {
            stats.avg_search_latency_us = elapsed_us;
        } else {
            stats.avg_search_latency_us +=
                (elapsed_us - stats.avg_search_latency_us) / stats.searches as f64;
        }
    }

    /// Searches for the $k$ nearest neighbor internal arena indices.
    pub fn search_indices(
        &self,
        query: &VectorEmbedding,
        k: usize,
    ) -> HNSQRResult<Vec<(NodeIndex, SimilarityScore)>> {
        self.search_indices_filtered(query, k, None)
    }

    /// Performs intent-aware search with algebraic phase alignment, recency bias, and metadata filtering.
    pub fn intent_rerank_search(
        &self,
        query: &VectorEmbedding,
        k: usize,
        intent: &SearchIntent,
    ) -> HNSQRResult<Vec<(NodeId, SimilarityScore, SimilarityScore)>> {
        let compiled_mask = if !intent.exact_matches.is_empty() {
            self.metadata_index
                .compile_filter_mask(&intent.exact_matches)
        } else if let Some(ref expr) = intent.filter {
            Some(self.evaluate_filter(expr))
        } else {
            intent.filter_mask.as_deref().cloned()
        };

        let base_results = if let Some(ref affect) = intent.affect {
            let contract = crate::planning::planner::RetrievalContract::HighRecall(0.99);
            let n = self.arena.live_len();
            let plan = crate::planning::planner::UniversalPlanner::plan_with_affect(
                n,
                self.dimension,
                compiled_mask.as_ref().map(|m| m.len() as usize),
                contract,
                self.mmap_arena.is_some(),
                affect,
            );
            match plan {
                crate::planning::planner::ExecutionPlan::LutzGlobalCertified { .. } => self
                    .search_indices_with_contract(
                        query,
                        k * 3,
                        compiled_mask.as_ref(),
                        crate::planning::planner::RetrievalContract::Certified,
                    )?,
                crate::planning::planner::ExecutionPlan::LutzPacRelaxed {
                    epsilon, delta, ..
                } => self.search_indices_with_contract(
                    query,
                    k * 3,
                    compiled_mask.as_ref(),
                    crate::planning::planner::RetrievalContract::PacRelaxed { epsilon, delta },
                )?,
                _ => self.search_indices_filtered(query, k * 3, compiled_mask.as_ref())?,
            }
        } else {
            self.search_indices_filtered(query, k * 3, compiled_mask.as_ref())?
        };

        let mut reranked = Vec::with_capacity(base_results.len());
        let now = current_unix_timestamp();
        let phase_weight = intent.phase_alignment_weight;

        for (idx, base_fidelity) in base_results {
            if let Some(node) = self.arena.get_node(idx) {
                let mut final_score = base_fidelity;

                if intent.recency_bias > 0.0 {
                    let age_days = (now.saturating_sub(node.created_at) as f32) / 86400.0;
                    let recency_factor = (-0.1 * age_days).exp();
                    final_score += recency_factor * intent.recency_bias * 0.25;
                }

                if phase_weight > 0.0 {
                    let vec_slice = self.arena.get_vector_slice(idx);
                    let ip = dot_product_complex_simd(query.complex_data(), vec_slice);
                    // Algebraic phase alignment: cos(arg(z)) = Re(z)/|z|.
                    // Eliminates atan2 + cos (two transcendentals) with one division.
                    let ip_norm = ip.norm();
                    let phase_alignment = if ip_norm > 1e-9 { ip.re / ip_norm } else { 0.0 };
                    final_score += phase_alignment * phase_weight * 0.15;
                }

                reranked.push((node.external_id.clone(), base_fidelity, final_score));
            }
        }

        reranked.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        reranked.truncate(k);

        {
            let mut stats = self.stats.write();
            stats.intent_searches = stats.intent_searches.saturating_add(1);
        }

        Ok(reranked)
    }

    /// Alias for [`intent_rerank_search`].
    #[inline]
    pub fn phase_aware_search(
        &self,
        query: &VectorEmbedding,
        k: usize,
        intent: &SearchIntent,
    ) -> HNSQRResult<Vec<(NodeId, SimilarityScore, SimilarityScore)>> {
        self.intent_rerank_search(query, k, intent)
    }

    /// Searches with dynamic contextual intent and diversity beam re-ranking.
    pub fn search_with_intent(
        &self,
        query: &VectorEmbedding,
        k: usize,
        intent: &SearchIntent,
    ) -> HNSQRResult<Vec<(NodeId, SimilarityScore)>> {
        let intent_res = self.intent_rerank_search(query, k, intent)?;
        Ok(intent_res
            .into_iter()
            .map(|(id, _, score)| (id, score))
            .collect())
    }

    /// Parallel batch search over multiple query vectors using Rayon.
    pub fn batch_search(
        &self,
        queries: &[VectorEmbedding],
        k: usize,
    ) -> HNSQRResult<Vec<Vec<(NodeId, SimilarityScore)>>> {
        queries.par_iter().map(|q| self.search(q, k)).collect()
    }

    /// Parallel batch search with contextual intent.
    pub fn batch_search_with_intent(
        &self,
        queries: &[VectorEmbedding],
        k: usize,
        intent: &SearchIntent,
    ) -> HNSQRResult<Vec<Vec<(NodeId, SimilarityScore)>>> {
        queries
            .par_iter()
            .map(|q| self.search_with_intent(q, k, intent))
            .collect()
    }

    // ────────────────────────────────────────────────────────────────────────
    // 5. MAINTENANCE & RETRIEVAL
    // ────────────────────────────────────────────────────────────────────────

    /// Retrieves a node by its external string identifier.
    pub fn get_node(&self, id: &str) -> HNSQRResult<Node> {
        let index = self
            .id_to_index
            .read()
            .get(id)
            .copied()
            .ok_or_else(|| HNSQRError::NodeNotFound(id.to_string()))?;

        self.get_node_by_index(index)
    }

    /// Retrieves a node directly by its internal arena index.
    pub fn get_node_by_index(&self, index: NodeIndex) -> HNSQRResult<Node> {
        self.arena
            .get_node(index)
            .map(|n| {
                let node_clone = Node::new(n.index, n.external_id.clone(), n.level);
                for l in 0..=n.level {
                    let conns = n.get_connections_clone(l);
                    node_clone.set_connections(l, conns);
                }
                *node_clone.rivero_witnesses.write() = n.rivero_witnesses.read().clone();
                node_clone
            })
            .ok_or(HNSQRError::NodeIndexNotFound(index))
    }

    /// Removes a node by its external identifier.
    pub fn remove(&self, id: &str) -> HNSQRResult<bool> {
        let _lifecycle = self.lifecycle.write();
        let maybe_index = self.id_to_index.write().remove(id);
        let index = match maybe_index {
            Some(idx) => idx,
            None => return Ok(false),
        };

        if let Some(wal) = self.wal.read().as_ref() {
            let durability = *self.wal_durability.read();
            let mutation = crate::storage::wal::WalMutation::Delete {
                external_id: id.to_string(),
            };
            let _ = wal.append(&mutation, durability);
        }

        let (strict_rivero, witness_degree) = {
            let config = self.config.read();
            (
                config.rivero_enabled && !config.rivero_fallback_on_underfill,
                rivero_witness::bounded_degree(config.rivero_witness_degree),
            )
        };
        let mut witness_neighbors: SmallVec<[NodeIndex; RIVERO_WITNESS_MAX_DEGREE]> =
            SmallVec::new();
        if strict_rivero {
            if let Some(node) = self.arena.get_node(index) {
                let witnesses = node.rivero_witnesses.read();
                witness_neighbors.extend(
                    witnesses
                        .iter()
                        .take(witness_degree)
                        .map(|witness| witness.index),
                );
            }
        }
        let address = self
            .rivero_compiler
            .read()
            .compile(self.arena.get_vector_slice(index));
        self.arena.delete_slot(index);
        self.rivero_index.evict(&address, index);
        for neighbor in witness_neighbors {
            if let Some(node) = self.arena.get_node(neighbor) {
                node.rivero_witnesses
                    .write()
                    .retain(|candidate| candidate.index != index);
            }
        }
        self.metadata_index.remove_node_index(index);

        // Invalidate cached proof tree and clear quantized code slot
        {
            let mut lutz_guard = self.lutz_codes.write();
            if (index as usize) < lutz_guard.len() {
                lutz_guard[index as usize] = None;
            }
        }
        *self.proof_tree.write() = None;

        let mut eps_write = self.entry_points.write();
        if let Some(pos) = eps_write.iter().position(|&i| i == index) {
            eps_write.remove(pos);
            if eps_write.is_empty() {
                let layer0 = self.layers[0].read();
                if let Some(&new_ep) = layer0
                    .iter()
                    .find(|&&i| i != index && self.arena.is_live(i))
                {
                    eps_write.push(new_ep);
                }
            }
        }

        Ok(true)
    }

    /// Clears the entire index and resets all layers and state.
    pub fn clear(&self) -> HNSQRResult<()> {
        let _lifecycle = self.lifecycle.write();
        self.arena.clear();
        self.id_to_index.write().clear();
        self.metadata_index.clear();
        self.rivero_index.clear();
        for l in 0..32 {
            self.layers[l].write().clear();
        }
        self.entry_points.write().clear();
        self.max_level.store(0, AtomicOrdering::Release);
        self.peak_active_searches.store(0, AtomicOrdering::Release);
        *self.stats.write() = IndexStats::default();
        info!(target: "hnsqr::index", "Index cleared successfully");
        Ok(())
    }

    /// Optimizes graph connectivity by performing 2-hop self-healing and heuristic edge rewiring.
    pub fn optimize(&self) -> HNSQRResult<()> {
        let _lifecycle = self.lifecycle.write();
        if self
            .is_optimizing
            .compare_exchange(false, true, AtomicOrdering::SeqCst, AtomicOrdering::SeqCst)
            .is_err()
        {
            return Err(HNSQRError::ConcurrencyError(
                "Optimization already in progress".to_string(),
            ));
        }
        defer! { self.is_optimizing.store(false, AtomicOrdering::Release); }

        let (total_nodes, m0, m, keep_pruned_connections) = {
            let cfg = self.config.read();
            (self.arena.len(), cfg.m0, cfg.m, cfg.keep_pruned_connections)
        };

        for i in 0..total_nodes {
            let node_idx = i as NodeIndex;
            if let Some(node) = self.arena.get_node(node_idx) {
                let v_slice = self.arena.get_vector_slice(node_idx);
                let v_norm_sq = self.arena.get_norm_squared(node_idx);

                for l in 0..=node.level {
                    let max_m = if l == 0 { m0 } else { m };
                    let conns = node.get_connections_clone(l);

                    let candidates: Vec<Candidate> = conns
                        .into_iter()
                        .map(|idx| {
                            let other_slice = self.arena.get_vector_slice(idx);
                            let other_norm_sq = self.arena.get_norm_squared(idx);
                            let sim = self.similarity_score_slices(
                                v_slice,
                                other_slice,
                                v_norm_sq,
                                other_norm_sq,
                            );
                            Candidate {
                                index: idx,
                                similarity: sim,
                            }
                        })
                        .collect();

                    let rewired = self.select_neighbors_heuristic(
                        v_slice,
                        v_norm_sq,
                        candidates,
                        max_m,
                        l,
                        true,
                        keep_pruned_connections,
                    );

                    let mut new_conns = SmallVec::new();
                    new_conns.extend(rewired);
                    node.set_connections(l, new_conns);
                }
            }
        }

        info!(target: "hnsqr::index", "Index graph optimization & self-healing complete");
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════════════
// 7. UNIT TESTS
// ════════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_encoded_vector_embeddings() {
        let v1 = VectorEmbedding::new(vec![1.0, 0.0]);
        let v2 = VectorEmbedding::new(vec![0.0, 1.0]);
        let v3 = VectorEmbedding::new(vec![1.0, 0.0]);

        assert_eq!(v1.projective_overlap(&v2), 0.0);
        assert!((v1.projective_overlap(&v3) - 1.0).abs() < 1e-5);
        assert!((v1.projective_sine_distance(&v3) - 0.0).abs() < 1e-5);
        assert!((v1.projective_sine_distance(&v2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_polar_phase_construction() {
        let amps = vec![1.0, 1.0];
        let phases = vec![0.0, std::f32::consts::PI / 2.0];
        let v = VectorEmbedding::from_amplitudes_and_phases(&amps, &phases);

        assert_eq!(v.dimension(), 2);
        let cdata = v.complex_data();
        assert!((cdata[0].re - 1.0).abs() < 1e-5);
        assert!((cdata[0].im - 0.0).abs() < 1e-5);
        assert!((cdata[1].re - 0.0).abs() < 1e-5);
        assert!((cdata[1].im - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_basic_insertion_and_search() {
        let config = HNSQRConfig::default();
        let index = HNSQRIndex::new(config, 3);

        let v1 = VectorEmbedding::new(vec![1.0, 0.0, 0.0]);
        let v2 = VectorEmbedding::new(vec![0.0, 1.0, 0.0]);
        let v3 = VectorEmbedding::new(vec![0.9, 0.1, 0.0]);

        index.insert("doc_1", v1.clone()).unwrap();
        index.insert("doc_2", v2.clone()).unwrap();
        index.insert("doc_3", v3.clone()).unwrap();

        assert_eq!(index.size(), 3);
        assert!(index.contains("doc_1"));

        let results = index.search(&v1, 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.as_ref(), "doc_1");
        assert_eq!(results[1].0.as_ref(), "doc_3");
    }

    #[test]
    fn test_intent_rerank_search() {
        let config = HNSQRConfig::default();
        let index = HNSQRIndex::new(config, 2);

        index
            .insert("node_a", VectorEmbedding::new(vec![1.0, 0.0]))
            .unwrap();
        index
            .insert("node_b", VectorEmbedding::new(vec![0.0, 1.0]))
            .unwrap();

        let query = VectorEmbedding::new(vec![0.8, 0.2]);
        let intent = SearchIntent {
            phase_alignment_weight: 0.5,
            recency_bias: 0.2,
            ..Default::default()
        };

        let results = index.intent_rerank_search(&query, 2, &intent).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.as_ref(), "node_a");
    }

    #[test]
    fn test_node_removal() {
        let config = HNSQRConfig::default();
        let index = HNSQRIndex::new(config, 2);

        index
            .insert("node_1", VectorEmbedding::new(vec![1.0, 0.0]))
            .unwrap();
        index
            .insert("node_2", VectorEmbedding::new(vec![0.0, 1.0]))
            .unwrap();

        assert_eq!(index.size(), 2);
        assert!(index.remove("node_1").unwrap());
        assert_eq!(index.size(), 1);
        assert!(index.contains("node_2"));
    }

    #[test]
    fn test_concurrent_batch_search() {
        let config = HNSQRConfig::default();
        let index = HNSQRIndex::new(config, 4);

        for i in 0..50 {
            let vec = VectorEmbedding::new(vec![i as f32, 1.0, 0.5, 0.2]).normalize();
            index.insert(format!("item_{}", i), vec).unwrap();
        }

        let queries: Vec<VectorEmbedding> = (0..10)
            .map(|i| VectorEmbedding::new(vec![i as f32, 1.0, 0.5, 0.2]).normalize())
            .collect();

        let batch_results = index.batch_search(&queries, 5).unwrap();
        assert_eq!(batch_results.len(), 10);
        for res in batch_results {
            assert_eq!(res.len(), 5);
        }
    }

    #[test]
    fn test_mmap_index_persistence() {
        let temp_dir = std::env::temp_dir();
        let mmap_path = temp_dir.join("test_index_mmap.bin");

        let v1 = VectorEmbedding::new(vec![1.0, 0.0, 0.0]);
        let v2 = VectorEmbedding::new(vec![0.0, 1.0, 0.0]);

        {
            let config = HNSQRConfig::default();
            let index = HNSQRIndex::create_mmap(&mmap_path, config, 3).unwrap();
            index.insert("node_1", v1.clone()).unwrap();
            index.insert("node_2", v2.clone()).unwrap();
            index.flush().unwrap();
            assert_eq!(index.size(), 2);
        }

        {
            let index = HNSQRIndex::open_mmap(&mmap_path).unwrap();
            assert_eq!(index.size(), 0); // Re-attached mmap with 0 parsing
        }

        let _ = std::fs::remove_file(mmap_path);
    }

    #[test]
    fn test_roaring_bitmap_filtered_search() {
        let config = HNSQRConfig::default();
        let index = HNSQRIndex::new(config, 2);

        let mut m1 = HashMap::new();
        m1.insert("category".to_string(), "gpu".into());
        index
            .insert_with_metadata("node_gpu_1", VectorEmbedding::new(vec![1.0, 0.0]), m1)
            .unwrap();

        let mut m2 = HashMap::new();
        m2.insert("category".to_string(), "cpu".into());
        index
            .insert_with_metadata("node_cpu_1", VectorEmbedding::new(vec![0.9, 0.1]), m2)
            .unwrap();

        let query = VectorEmbedding::new(vec![1.0, 0.0]);
        let intent = SearchIntent {
            filter: Some(FilterExpr::eq("category", "gpu")),
            ..Default::default()
        };

        let results = index.intent_rerank_search(&query, 2, &intent).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.as_ref(), "node_gpu_1");
    }

    #[test]
    fn test_exact_matches_json_search() {
        let config = HNSQRConfig::default();
        let index = HNSQRIndex::new(config, 2);

        let v_eng = VectorEmbedding::new(vec![1.0, 0.0]);
        let v_fin = VectorEmbedding::new(vec![0.9, 0.1]);

        index
            .insert_with_json_metadata(
                "e1",
                &v_eng,
                &serde_json::json!({ "dept": "engineering", "active": true }),
            )
            .unwrap();
        index
            .insert_with_json_metadata(
                "f1",
                &v_fin,
                &serde_json::json!({ "dept": "finance", "active": true }),
            )
            .unwrap();

        let mut exact = HashMap::new();
        exact.insert("dept".to_string(), "engineering".to_string());

        let intent = SearchIntent {
            exact_matches: exact,
            ..Default::default()
        };

        let results = index.intent_rerank_search(&v_eng, 2, &intent).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0.as_ref(), "e1");
    }

    #[test]
    fn strict_rivero_search_proves_fixed_work_and_global_phase_recall() {
        let config = HNSQRConfig::strict_rivero_for_dim(4);
        let index = HNSQRIndex::new(config, 4);
        let source = VectorEmbedding::from_complex(vec![
            Complex32::new(0.7, -0.1),
            Complex32::new(0.2, 0.4),
            Complex32::new(-0.3, 0.1),
            Complex32::new(0.1, -0.2),
        ])
        .normalize();
        let phase = Complex32::from_polar(1.0, 1.23);
        let rotated = VectorEmbedding::from_complex(
            source
                .complex_data()
                .iter()
                .map(|value| *value * phase)
                .collect(),
        );

        let inserted = index.insert("phase-state", source).unwrap();
        let (results, diagnostics) = index
            .search_indices_o1_with_diagnostics(&rotated, 1, None)
            .unwrap();

        assert_eq!(results.first().map(|result| result.0), Some(inserted));
        assert_eq!(diagnostics.cells_probed, RiveroAddress::cell_probe_count());
        assert!(diagnostics.resident_reads <= diagnostics.candidate_read_bound);
        assert!(diagnostics.exact_score_evaluations <= diagnostics.unique_candidates);
        assert!(!diagnostics.fallback_used);
        assert_eq!(index.level_distribution(), vec![0]);
    }

    #[test]
    fn strict_rivero_remove_cannot_return_tombstone() {
        let config = HNSQRConfig::strict_rivero_for_dim(2);
        let index = HNSQRIndex::new(config, 2);
        let removed = VectorEmbedding::new(vec![1.0, 0.0]);
        let live = VectorEmbedding::new(vec![0.9, 0.1]);
        let removed_index = index.insert("removed", removed.clone()).unwrap();
        index.insert("live", live).unwrap();

        assert!(index.remove("removed").unwrap());
        let (results, diagnostics) = index
            .search_indices_o1_with_diagnostics(&removed, 2, None)
            .unwrap();
        assert!(results.iter().all(|result| result.0 != removed_index));
        assert_eq!(index.size(), 1);
        assert_eq!(diagnostics.non_live_rejections, 0);
    }

    #[test]
    fn strict_rivero_witness_work_and_degree_are_bounded() {
        let mut config = HNSQRConfig::strict_rivero_for_dim(8);
        config.rivero_witness_degree = 4;
        config.rivero_witness_seeds = 2;
        config.rivero_witness_second_seeds = 2;
        let index = HNSQRIndex::new(config, 8);

        let vectors: Vec<_> = (0..16)
            .map(|slot| {
                VectorEmbedding::new(
                    (0..8)
                        .map(|lane| ((slot * 17 + lane * 11 + 3) % 31) as f32 - 15.0)
                        .collect(),
                )
                .normalize()
            })
            .collect();
        for (slot, vector) in vectors.iter().cloned().enumerate() {
            index.insert(format!("witness-{slot}"), vector).unwrap();
        }

        for slot in 0..vectors.len() as NodeIndex {
            let node = index.arena.get_node(slot).unwrap();
            let witnesses = node.rivero_witnesses.read();
            assert!(witnesses.len() <= 4);
            for witness in witnesses.iter() {
                let reverse = index.arena.get_node(witness.index).unwrap();
                let reverse_indices: Vec<_> = reverse
                    .rivero_witnesses
                    .read()
                    .iter()
                    .map(|edge| edge.index)
                    .collect();
                assert!(
                    reverse_indices.contains(&slot),
                    "incremental witness {slot} -> {} is not reciprocal; reverse list={reverse_indices:?}",
                    witness.index,
                );
            }
        }

        let (results, diagnostics) = index
            .search_indices_o1_with_diagnostics(&vectors[0], 10, None)
            .unwrap();
        assert_eq!(results.len(), 10);
        assert!(diagnostics.route_candidates_selected <= RIVERO_QUERY_CANDIDATE_CAP);
        assert!(diagnostics.raw_unique_candidates <= diagnostics.raw_unique_candidate_bound);
        assert!(diagnostics.witness_seeds <= 2);
        assert!(diagnostics.witness_second_hop_seeds <= 2);
        assert_eq!(diagnostics.witness_edge_scan_bound, 16);
        assert!(diagnostics.witness_edges_scanned <= diagnostics.witness_edge_scan_bound);
        assert!(diagnostics.witness_candidates_added <= diagnostics.witness_edges_scanned);
        assert_eq!(
            diagnostics.exact_score_evaluations
                + diagnostics.non_live_rejections
                + diagnostics.filter_rejections,
            diagnostics.unique_candidates
        );
    }

    #[test]
    fn certified_non_cosine_contract_uses_metric_exact_scan() {
        let mut config = HNSQRConfig::default();
        config.distance_function = DistanceFunction::Euclidean;
        let index = HNSQRIndex::new(config, 2);
        index
            .insert("far-collinear", VectorEmbedding::new(vec![100.0, 0.0]))
            .unwrap();
        let nearest = index
            .insert("near", VectorEmbedding::new(vec![1.0, 1.0]))
            .unwrap();
        let query = VectorEmbedding::new(vec![1.0, 0.0]);

        let exact = index.search_indices_exact(&query, 1, None).unwrap();
        let outcome = index.certified_search(&query, 1, None).unwrap();
        let CertifiedSearchOutcome::Exact { results, proof } = outcome else {
            panic!("exhaustive non-cosine certification cannot time out");
        };
        assert_eq!(results, exact);
        assert_eq!(results[0].0, nearest);
        assert!(proof.globally_exact);
        assert!(proof.is_accounting_exact());
        assert_eq!(proof.exact_evaluations, 2);
    }

    #[test]
    fn rivero_state_metric_mismatch_is_rejected_at_install() {
        let vectors = vec![
            VectorEmbedding::new(vec![1.0, 0.0]),
            VectorEmbedding::new(vec![0.0, 1.0]),
        ];
        let built = RiveroBulkBuilder::with_profile(RiveroProfile::Strict)
            .with_distance_function(DistanceFunction::Euclidean)
            .build(&vectors)
            .unwrap();
        let index = HNSQRIndex::new(HNSQRConfig::default(), 2);
        let error = match index.install_rivero_state(built) {
            Ok(()) => panic!("mismatched witness/rerank metrics must be rejected"),
            Err(error) => error,
        };
        assert!(
            matches!(error, HNSQRError::InvalidConfig(message) if message.contains("metric mismatch"))
        );
    }

    #[test]
    fn adaptive_rivero_search_progresses_and_reuses_state() {
        let mut config = HNSQRConfig::strict_rivero_for_dim(8);
        config.rivero_mode = RiveroSearchMode::Adaptive;
        let index = HNSQRIndex::new(config, 8);

        for i in 0..50 {
            let vec = VectorEmbedding::new(
                (0..8)
                    .map(|lane| ((i * 13 + lane * 7 + 5) % 19) as f32 - 9.0)
                    .collect(),
            )
            .normalize();
            index.insert(format!("node-{i}"), vec).unwrap();
        }

        let query =
            VectorEmbedding::new(vec![1.0, 0.5, -0.2, 0.8, -0.4, 0.3, 0.1, -0.7]).normalize();
        let (results, diag) = index
            .search_indices_adaptive(&query, 10, None, AdaptivePolicy::RiveroOnly)
            .unwrap();

        assert!(!results.is_empty());
        assert!(diag.stages_executed >= 1 && diag.stages_executed <= 3);
        assert!(diag.confidence_final >= 0.0 && diag.confidence_final <= 1.0);
        assert!(diag.cumulative_resident_scans > 0);
        assert!(diag.cumulative_exact_scores > 0);
        assert!(!diag.graph_fallback_used);
    }

    #[test]
    fn adaptive_graph_fallback_preserves_telemetry_and_counts_one_search() {
        let mut config = HNSQRConfig::default();
        config.rivero_enabled = true;
        config.rivero_mode = RiveroSearchMode::Adaptive;
        config.adaptive_policy = AdaptivePolicy::AllowGraphFallback;
        let index = HNSQRIndex::new(config, 8);

        // Fewer live vectors than k deterministically forces every Rivero stage to
        // recommend escalation, while still producing scored Rivero candidates.
        for i in 0..5 {
            let vector = VectorEmbedding::new(
                (0..8)
                    .map(|lane| ((i * 19 + lane * 7 + 3) % 29) as f32 - 14.0)
                    .collect(),
            )
            .normalize();
            index.insert(format!("fallback-{i}"), vector).unwrap();
        }

        let query =
            VectorEmbedding::new(vec![0.2, -0.4, 0.7, 0.1, -0.8, 0.3, 0.6, -0.5]).normalize();
        let searches_before = index.stats().searches;
        let (_, diag) = index
            .search_indices_adaptive(&query, 10, None, AdaptivePolicy::AllowGraphFallback)
            .unwrap();
        let searches_after = index.stats().searches;

        assert!(diag.graph_fallback_used);
        assert!(diag.cumulative_exact_scores > 0);
        assert_eq!(searches_after, searches_before + 1);
    }

    #[test]
    fn index_and_universal_planner_share_one_crossover_primitive() {
        for dimension in [32, 64, 128, 384, 768, 1_536, 2_048] {
            assert_eq!(
                HNSQRIndex::default_exact_scan_threshold(dimension),
                crate::planning::planner::UniversalPlanner::compute_crossover(dimension),
            );
        }
    }

    #[test]
    fn strict_vs_adaptive_vs_graph_mode_consistency() {
        let mut config = HNSQRConfig::default();
        config.rivero_enabled = true;
        config.rivero_fallback_on_underfill = true;
        config.rivero_witness_degree = 8;
        let index = HNSQRIndex::new(config, 8);

        for i in 0..30 {
            let vec = VectorEmbedding::new(
                (0..8)
                    .map(|lane| ((i * 17 + lane * 5 + 3) % 23) as f32 - 11.0)
                    .collect(),
            )
            .normalize();
            index.insert(format!("vec-{i}"), vec).unwrap();
        }

        let query =
            VectorEmbedding::new(vec![0.5, 0.2, -0.1, 0.4, -0.8, 0.3, 0.0, 0.6]).normalize();

        let (strict_res, strict_diag) = index.search_indices_strict(&query, 5, None).unwrap();
        let (adapt_res, adapt_diag) = index
            .search_indices_adaptive(&query, 5, None, AdaptivePolicy::RiveroOnly)
            .unwrap();
        let graph_res = index.search_indices_graph(&query, 5, None).unwrap();

        assert_eq!(strict_res.len(), 5);
        assert_eq!(adapt_res.len(), 5);
        assert_eq!(graph_res.len(), 5);
        assert_eq!(strict_diag.results_returned, 5);
        assert!(!adapt_diag.graph_fallback_used);
    }

    #[test]
    fn test_adaptive_never_regresses_from_completed_stage() {
        let mut config = HNSQRConfig::default();
        config.rivero_enabled = true;
        config.distance_function = DistanceFunction::Cosine;
        let index = HNSQRIndex::new(config, 8);

        let mut corpus_vectors = Vec::new();
        for i in 0..100 {
            let vec = VectorEmbedding::new(
                (0..8)
                    .map(|lane| ((i * 31 + lane * 11 + 7) % 37) as f32 - 18.0)
                    .collect(),
            )
            .normalize();
            corpus_vectors.push(vec.clone());
            index.insert(format!("node-{i}"), vec).unwrap();
        }

        let query =
            VectorEmbedding::new(vec![0.3, -0.6, 0.2, 0.7, -0.1, 0.4, -0.5, 0.8]).normalize();

        // Exact Ground Truth
        let mut gt: Vec<(NodeIndex, f32)> = corpus_vectors
            .iter()
            .enumerate()
            .map(|(idx, doc)| (idx as NodeIndex, query.dot_product_complex(doc).re))
            .collect();
        gt.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        let gt_top10: Vec<NodeIndex> = gt.iter().take(10).map(|s| s.0).collect();

        // 1. Fast stage trace
        let fast_cfg = RiveroProfile::Fast.config();
        let (fast_res, _) = index
            .search_indices_o1_with_config(&query, 10, None, &fast_cfg)
            .unwrap();
        let fast_ids: Vec<NodeIndex> = fast_res.iter().map(|s| s.0).collect();
        let fast_recall = fast_ids.iter().filter(|id| gt_top10.contains(id)).count() as f64 / 10.0;

        // 2. Adaptive search
        let (adapt_res, _) = index
            .search_indices_adaptive(&query, 10, None, AdaptivePolicy::RiveroOnly)
            .unwrap();
        let adapt_ids: Vec<NodeIndex> = adapt_res.iter().map(|s| s.0).collect();
        let adapt_recall =
            adapt_ids.iter().filter(|id| gt_top10.contains(id)).count() as f64 / 10.0;

        // Monotonicity assertion: Adaptive recall must be >= Fast recall
        assert!(
            adapt_recall >= fast_recall,
            "Adaptive recall ({adapt_recall}) regressed below Fast recall ({fast_recall})!"
        );
    }
}
