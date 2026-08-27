/* holosphere/src/contextgraph/mod.rs */
//!▫~•◦-------------------------------‣
//! # HoloSphere ContextGraph — Universal Context Compiler & Graph Reasoning Substrate
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Universal multi-domain context compiler and reasoning substrate supporting code ASTs,
//! documents, runtime architectures, Git history, datasets, and organizational knowledge.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod adapter;
pub mod adapters;
pub mod analytics;
pub mod community;
pub mod compiler;
pub mod fingerprint;
pub mod invalidation;
pub mod ir;
pub mod manifest;
pub mod planner;
pub mod query;
pub mod resolver;
pub mod schema;
pub mod store;
pub mod views;
pub mod watcher;

// Public re-exports
pub use adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
pub use adapters::AdapterRegistry;
pub use analytics::{ContextAnalytics, HubEntityInfo, UniversalCycle};
pub use community::{ScopeClustering, ScopeSummary};
pub use compiler::{ContextCompilationOutput, ContextCompiler};
pub use fingerprint::GraphFingerprinter;
pub use invalidation::InvalidationGraph;
pub use ir::{
    Diagnostic, DiagnosticSeverity, ExtractedArtifact, ExtractedEntity, ExtractedRelation,
    ExtractionBatch, SourceDescriptor, UnresolvedReference,
};
pub use manifest::{ContextGraphManifest, SourceDiff, SourceManifestEntry};
pub use planner::{ContextBudget, ContextQueryRequest, QueryPlan, QueryPlanner};
pub use query::{ContextGraphDiff, ContextQueryEngine, ContextSlice};
pub use resolver::{ReferenceResolver, UniversalReferenceResolver};
pub use schema::{
    ContextGraphDelta, Entity, EntityId, EntityKind, Namespace, ProvenanceRef, Relation,
    RelationId, RelationKind, RelationOrigin, RelationParticipant, ResourceLocator,
};
pub use store::{ContextGraphStore, ContextGraphStoreState};
pub use views::{
    GraphView, html::HtmlVisualizerView, json::JsonExportView, markdown::MarkdownReportView,
};
pub use watcher::ContextGraphWatcher;
