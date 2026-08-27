/* holosphere/src/codegraph/manifest.rs */
//!▫~•◦-------------------------------‣
//! # Workspace & File Content Manifests & Incremental Diff Tracking
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Tracks file content hashes (SHA-256), emitted symbol & edge IDs, and computes
//! incremental diffs (added, modified, deleted, unchanged) across compilation passes.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::schema::{CodeEdgeId, CodeNodeId, Language};
use crate::HNSQRResult;

/// Fingerprint and emitted graph identities for a single source file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileManifest {
    pub relative_path: PathBuf,
    pub content_hash: [u8; 32],
    pub language: Language,
    pub node_ids: Vec<CodeNodeId>,
    pub edge_ids: Vec<CodeEdgeId>,
    pub modified_timestamp_secs: u64,
}

impl FileManifest {
    #[must_use]
    pub fn new(
        relative_path: PathBuf,
        content_hash: [u8; 32],
        language: Language,
        node_ids: Vec<CodeNodeId>,
        edge_ids: Vec<CodeEdgeId>,
        modified_timestamp_secs: u64,
    ) -> Self {
        Self {
            relative_path,
            content_hash,
            language,
            node_ids,
            edge_ids,
            modified_timestamp_secs,
        }
    }
}

/// Workspace-wide manifest maintaining state for incremental compilations.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct WorkspaceManifest {
    pub workspace_id: String,
    pub git_head: Option<String>,
    pub files: BTreeMap<PathBuf, FileManifest>,
    pub commit_lsn: u64,
}

/// Categorized diff between current workspace files and the recorded manifest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ManifestDiff {
    pub added: Vec<PathBuf>,
    pub modified: Vec<PathBuf>,
    pub deleted: Vec<PathBuf>,
    pub unchanged: Vec<PathBuf>,
}

impl ManifestDiff {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.deleted.is_empty()
    }

    #[must_use]
    pub fn total_changed(&self) -> usize {
        self.added.len() + self.modified.len() + self.deleted.len()
    }
}

impl WorkspaceManifest {
    #[must_use]
    pub fn new(workspace_id: impl Into<String>) -> Self {
        Self {
            workspace_id: workspace_id.into(),
            git_head: None,
            files: BTreeMap::new(),
            commit_lsn: 0,
        }
    }

    /// Computes incremental file changes by comparing current file hashes against manifest state.
    #[must_use]
    pub fn diff(&self, current_file_hashes: &BTreeMap<PathBuf, [u8; 32]>) -> ManifestDiff {
        let mut diff = ManifestDiff::default();
        let current_paths: BTreeSet<&PathBuf> = current_file_hashes.keys().collect();

        // Detect deleted files
        for (recorded_path, _) in &self.files {
            if !current_paths.contains(recorded_path) {
                diff.deleted.push(recorded_path.clone());
            }
        }

        // Detect added, modified, unchanged files
        for (current_path, current_hash) in current_file_hashes {
            if let Some(recorded) = self.files.get(current_path) {
                if &recorded.content_hash == current_hash {
                    diff.unchanged.push(current_path.clone());
                } else {
                    diff.modified.push(current_path.clone());
                }
            } else {
                diff.added.push(current_path.clone());
            }
        }

        diff.added.sort();
        diff.modified.sort();
        diff.deleted.sort();
        diff.unchanged.sort();
        diff
    }

    /// Updates file records and prunes deleted files.
    pub fn apply_update(&mut self, updated_files: Vec<FileManifest>, deleted_paths: &[PathBuf]) {
        for path in deleted_paths {
            self.files.remove(path);
        }
        for file in updated_files {
            self.files.insert(file.relative_path.clone(), file);
        }
    }

    /// Computes SHA-256 hash of a file's byte contents.
    #[must_use]
    pub fn hash_bytes(bytes: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hasher.finalize().into()
    }

    /// Saves manifest to disk.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> HNSQRResult<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    /// Loads manifest from disk.
    pub fn load_from_file(path: impl AsRef<Path>) -> HNSQRResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let manifest = serde_json::from_str(&content)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))?;
        Ok(manifest)
    }
}
