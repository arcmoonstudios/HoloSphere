/* holosphere/src/codegraph/incremental.rs */
//!▫~•◦-------------------------------‣
//! # Incremental Compilation & Delta Computation Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Recompiles only modified and added source files, prunes stale nodes/edges for deleted
//! and modified files, re-resolves affected cross-file call scopes, and produces an atomic CodeGraphDelta.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::compiler::{CodeGraphCompiler, CompilationOutput};
use super::languages::LanguageRegistry;
use super::manifest::{FileManifest, ManifestDiff, WorkspaceManifest};
use super::scanner::{ScannedFile, WorkspaceScanner};
use super::schema::{CodeEdge, CodeEdgeId, CodeGraphDelta, CodeNode, CodeNodeId};
use crate::HNSQRResult;

pub struct IncrementalCompiler {
    compiler: CodeGraphCompiler,
    scanner: WorkspaceScanner,
}

impl Default for IncrementalCompiler {
    fn default() -> Self {
        Self {
            compiler: CodeGraphCompiler::default(),
            scanner: WorkspaceScanner::with_default_config(),
        }
    }
}

impl IncrementalCompiler {
    #[must_use]
    pub fn new(compiler: CodeGraphCompiler, scanner: WorkspaceScanner) -> Self {
        Self { compiler, scanner }
    }

    /// Computes incremental delta against recorded workspace manifest.
    pub fn compile_incremental(
        &self,
        workspace_id: &str,
        workspace_root: impl AsRef<Path>,
        manifest: &mut WorkspaceManifest,
    ) -> HNSQRResult<Option<CodeGraphDelta>> {
        let root = workspace_root.as_ref();
        let scanned = self.scanner.scan_workspace(root)?;

        let current_hashes: BTreeMap<PathBuf, [u8; 32]> = scanned
            .iter()
            .map(|(path, file)| (path.clone(), file.content_hash))
            .collect();

        let diff = manifest.diff(&current_hashes);
        if diff.is_empty() {
            return Ok(None);
        }

        let mut delete_nodes = Vec::new();
        let mut delete_edges = Vec::new();

        // 1. Collect stale nodes & edges from deleted and modified files
        for deleted_path in &diff.deleted {
            if let Some(file_manifest) = manifest.files.get(deleted_path) {
                delete_nodes.extend(file_manifest.node_ids.clone());
                delete_edges.extend(file_manifest.edge_ids.clone());
            }
        }
        for modified_path in &diff.modified {
            if let Some(file_manifest) = manifest.files.get(modified_path) {
                delete_nodes.extend(file_manifest.node_ids.clone());
                delete_edges.extend(file_manifest.edge_ids.clone());
            }
        }

        // 2. Filter files requiring compilation (added + modified)
        let files_to_compile: Vec<ScannedFile> = scanned
            .into_values()
            .filter(|f| {
                diff.added.contains(&f.relative_path) || diff.modified.contains(&f.relative_path)
            })
            .collect();

        // 3. Compile changed files
        let partial_output = self
            .compiler
            .compile_files(workspace_id, &files_to_compile)?;

        // 4. Update manifest
        let new_file_manifests: Vec<FileManifest> =
            partial_output.manifest.files.into_values().collect();
        manifest.apply_update(new_file_manifests, &diff.deleted);

        let mut touched_files = diff.added;
        touched_files.extend(diff.modified);
        touched_files.extend(diff.deleted);
        touched_files.sort();
        touched_files.dedup();

        Ok(Some(CodeGraphDelta {
            workspace_id: workspace_id.to_string(),
            insert_nodes: partial_output.nodes,
            delete_nodes,
            insert_edges: partial_output.edges,
            delete_edges,
            touched_files,
        }))
    }
}
