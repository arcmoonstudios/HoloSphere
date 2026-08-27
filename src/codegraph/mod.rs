/* holosphere/src/codegraph/mod.rs */
//!▫~•◦-------------------------------‣
//! # HoloSphere CodeGraph — Deterministic Codebase Compiler & Topology Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides deterministic repository-level AST extraction (tree-sitter), multi-pass
//! cross-file symbol resolution, position-independent stable hashing, atomic delta ingestion,
//! topological community detection, architectural analytics, and hybrid navigation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod analysis;
pub mod community;
pub mod compiler;
pub mod diff;
pub mod export;
pub mod incremental;
pub mod ingest;
pub mod languages;
pub mod manifest;
pub mod parser;
pub mod path;
pub mod query;
pub mod registry;
pub mod report;
pub mod resolver;
pub mod scanner;
pub mod schema;
pub mod watcher;

// Re-exports for clean `use hnsqr::codegraph::*` access
pub use analysis::{BlastRadiusReport, CodeGraphAnalyzer, DependencyCycle, GodNodeInfo};
pub use community::{CommunityDetector, CommunitySummary};
pub use compiler::{CodeGraphCompiler, CompilationOutput};
pub use diff::{CodeGraphDiffEngine, CodeGraphDiffReport, ModifiedSymbolPair};
pub use export::{CodeGraphExportPayload, CodeGraphExporter, ExportEdge, ExportNode};
pub use incremental::IncrementalCompiler;
pub use ingest::{CodeGraphStore, CodeGraphStoreState};
pub use languages::LanguageRegistry;
pub use manifest::{FileManifest, ManifestDiff, WorkspaceManifest};
pub use parser::{
    ExtractionContext, ExtractionResult, ImportItem, LanguageExtractor, UnresolvedCall,
    UnresolvedTypeRef,
};
pub use path::{CodePath, CodePathfinder, PathStep};
pub use query::{CodeExplainResult, CodeQueryEngine, CodeQueryResult};
pub use registry::{FileSymbolTable, SymbolEntry, WorkspaceSymbolTable};
pub use report::CodeGraphReportGenerator;
pub use resolver::CodeGraphResolver;
pub use scanner::{ScannedFile, ScannerConfig, WorkspaceScanner};
pub use schema::{
    CodeEdge, CodeEdgeId, CodeGraphDelta, CodeNode, CodeNodeId, CodeNodeKind, CodeRelation,
    Language, RelationOrigin, SourceSpan,
};
pub use watcher::CodeGraphWatcher;
