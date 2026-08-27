/* holosphere/src/codegraph/compiler.rs */
//!▫~•◦-------------------------------‣
//! # End-to-End Deterministic CodeGraph Compiler Pipeline
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Orchestrates workspace scanning, parallel AST extraction, hierarchical symbol
//! table generation, cross-file resolution, and manifest creation with strict determinism.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;

use super::languages::LanguageRegistry;
use super::manifest::{FileManifest, WorkspaceManifest};
use super::parser::ExtractionContext;
use super::registry::WorkspaceSymbolTable;
use super::resolver::CodeGraphResolver;
use super::scanner::{ScannedFile, WorkspaceScanner};
use super::schema::{CodeEdge, CodeGraphDelta, CodeNode, CodeNodeId};
use crate::{HNSQRError, HNSQRResult};

/// Output produced by a full or partial compilation pass.
#[derive(Clone, Debug)]
pub struct CompilationOutput {
    pub workspace_id: String,
    pub nodes: Vec<CodeNode>,
    pub edges: Vec<CodeEdge>,
    pub symbol_table: WorkspaceSymbolTable,
    pub manifest: WorkspaceManifest,
    pub duration_ms: u64,
}

impl CompilationOutput {
    /// Converts compilation output into an atomic CodeGraphDelta for storage ingestion.
    #[must_use]
    pub fn into_delta(self) -> CodeGraphDelta {
        let touched_files = self.manifest.files.keys().cloned().collect();

        CodeGraphDelta {
            workspace_id: self.workspace_id,
            insert_nodes: self.nodes,
            delete_nodes: Vec::new(),
            insert_edges: self.edges,
            delete_edges: Vec::new(),
            touched_files,
        }
    }
}

pub struct CodeGraphCompiler {
    registry: Arc<LanguageRegistry>,
    scanner: WorkspaceScanner,
}

impl Default for CodeGraphCompiler {
    fn default() -> Self {
        Self {
            registry: Arc::new(LanguageRegistry::default()),
            scanner: WorkspaceScanner::with_default_config(),
        }
    }
}

impl CodeGraphCompiler {
    #[must_use]
    pub fn new(registry: Arc<LanguageRegistry>, scanner: WorkspaceScanner) -> Self {
        Self { registry, scanner }
    }

    /// Performs full compilation of a repository directory with bit-exact determinism.
    pub fn compile_full(
        &self,
        workspace_id: &str,
        workspace_root: impl AsRef<Path>,
    ) -> HNSQRResult<CompilationOutput> {
        let start = Instant::now();
        let root = workspace_root.as_ref();
        let scanned = self.scanner.scan_workspace(root)?;

        let scanned_files: Vec<ScannedFile> = scanned.into_values().collect();
        self.compile_files_internal(workspace_id, &scanned_files, start)
    }

    /// Compiles a specific set of scanned files.
    pub fn compile_files(
        &self,
        workspace_id: &str,
        scanned_files: &[ScannedFile],
    ) -> HNSQRResult<CompilationOutput> {
        let start = Instant::now();
        self.compile_files_internal(workspace_id, scanned_files, start)
    }

    fn compile_files_internal(
        &self,
        workspace_id: &str,
        scanned_files: &[ScannedFile],
        start: Instant,
    ) -> HNSQRResult<CompilationOutput> {
        // Step 1: Parallel AST Extraction
        let registry = self.registry.clone();
        let extraction_results: Vec<HNSQRResult<_>> = scanned_files
            .par_iter()
            .map(|scanned| {
                let extractor = registry.get(scanned.language).ok_or_else(|| {
                    HNSQRError::InvalidRequest(format!("No extractor for {}", scanned.language))
                })?;

                let source = fs::read_to_string(&scanned.absolute_path)?;
                let ctx = ExtractionContext {
                    workspace_id,
                    relative_path: &scanned.relative_path,
                    source_code: &source,
                    content_hash: scanned.content_hash,
                };

                let res = extractor.extract(&ctx)?;
                Ok((scanned, res))
            })
            .collect();

        // Step 2: Deterministic Assembly & Symbol Table Population
        let mut symbol_table = WorkspaceSymbolTable::new();
        let mut all_nodes = Vec::new();
        let mut direct_edges = Vec::new();
        let mut all_unresolved_calls = Vec::new();
        let mut all_unresolved_types = Vec::new();
        let mut file_manifests = Vec::new();

        // Sort results by relative path to guarantee order-independent processing
        let mut sorted_extractions = Vec::new();
        for res in extraction_results {
            let (scanned, ext) = res?;
            sorted_extractions.push((scanned, ext));
        }
        sorted_extractions.sort_by(|a, b| a.0.relative_path.cmp(&b.0.relative_path));

        for (scanned, mut ext) in sorted_extractions {
            let mut file_node_ids = Vec::new();
            let mut file_edge_ids = Vec::new();

            for node in &ext.nodes {
                symbol_table.insert_node(node);
                file_node_ids.push(node.id.clone());
            }
            for import in &ext.imports {
                symbol_table.register_import(&scanned.relative_path, import);
            }
            for edge in &ext.edges {
                file_edge_ids.push(edge.id.clone());
            }

            all_nodes.append(&mut ext.nodes);
            direct_edges.append(&mut ext.edges);
            all_unresolved_calls.append(&mut ext.unresolved_calls);
            all_unresolved_types.append(&mut ext.unresolved_types);

            file_manifests.push(FileManifest::new(
                scanned.relative_path.clone(),
                scanned.content_hash,
                scanned.language,
                file_node_ids,
                file_edge_ids,
                scanned.modified_timestamp_secs,
            ));
        }

        // Step 3: Multi-Pass Resolution
        let resolver = CodeGraphResolver::new(&symbol_table);
        let resolved_edges = resolver.resolve_all(&all_unresolved_calls, &all_unresolved_types);

        // Step 4: Combine & Deterministically Sort Nodes and Edges
        let mut all_edges = direct_edges;
        all_edges.extend(resolved_edges);

        // Deduplicate and sort
        all_nodes.sort_by(|a, b| a.id.cmp(&b.id));
        all_nodes.dedup_by(|a, b| a.id == b.id);

        all_edges.sort_by(|a, b| a.id.cmp(&b.id));
        all_edges.dedup_by(|a, b| a.id == b.id);

        let mut manifest = WorkspaceManifest::new(workspace_id);
        manifest.apply_update(file_manifests, &[]);

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(CompilationOutput {
            workspace_id: workspace_id.to_string(),
            nodes: all_nodes,
            edges: all_edges,
            symbol_table,
            manifest,
            duration_ms,
        })
    }
}
