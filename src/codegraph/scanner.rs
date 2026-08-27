/* holosphere/src/codegraph/scanner.rs */
//!▫~•◦-------------------------------‣
//! # Workspace Filesystem Scanner & Language Identification
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Traverses repository directory trees, enforces standard ignore boundaries (.git, target,
//! node_modules), classifies source languages, and hashes source contents in parallel.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use rayon::prelude::*;

use super::manifest::WorkspaceManifest;
use super::schema::Language;
use crate::HNSQRResult;

const MAX_FILE_SIZE_BYTES: u64 = 8 * 1024 * 1024; // 8MB limit for single source files

/// Representation of a discovered source file on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScannedFile {
    pub relative_path: PathBuf,
    pub absolute_path: PathBuf,
    pub language: Language,
    pub content_hash: [u8; 32],
    pub size_bytes: usize,
    pub modified_timestamp_secs: u64,
}

/// Scanner configuration controlling ignored paths and supported extensions.
#[derive(Clone, Debug)]
pub struct ScannerConfig {
    pub ignored_directories: Vec<String>,
    pub max_file_size_bytes: u64,
}

impl Default for ScannerConfig {
    fn default() -> Self {
        Self {
            ignored_directories: vec![
                ".git".to_string(),
                ".github".to_string(),
                ".vscode".to_string(),
                "target".to_string(),
                "node_modules".to_string(),
                "vendor".to_string(),
                "build".to_string(),
                "dist".to_string(),
                ".holosphere".to_string(),
                ".gemini".to_string(),
                "target-benchmark-integrity-validation".to_string(),
                "target-correctness-audit".to_string(),
                "target-embedding-validation".to_string(),
                "target-gate-b-validation".to_string(),
                "target-policy-integrity-validation".to_string(),
                "target-semantics-validation".to_string(),
                "target-universal-certification-validation".to_string(),
                "target-web-release".to_string(),
                "target-web-release-fix".to_string(),
                "target-web-validation".to_string(),
            ],
            max_file_size_bytes: MAX_FILE_SIZE_BYTES,
        }
    }
}

pub struct WorkspaceScanner {
    config: ScannerConfig,
}

impl WorkspaceScanner {
    #[must_use]
    pub fn new(config: ScannerConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(ScannerConfig::default())
    }

    /// Scans root path and returns all candidate source files sorted deterministically.
    pub fn scan_workspace(
        &self,
        root_dir: impl AsRef<Path>,
    ) -> HNSQRResult<BTreeMap<PathBuf, ScannedFile>> {
        let root = root_dir
            .as_ref()
            .canonicalize()
            .unwrap_or_else(|_| root_dir.as_ref().to_path_buf());
        let mut raw_paths = Vec::new();
        self.collect_paths_recursive(&root, &root, &mut raw_paths)?;

        // Parallel read & hash
        let scanned_files: Vec<ScannedFile> = raw_paths
            .into_par_iter()
            .filter_map(|(rel_path, abs_path)| {
                let metadata = fs::metadata(&abs_path).ok()?;
                if metadata.len() > self.config.max_file_size_bytes {
                    return None;
                }
                let ext = rel_path.extension()?.to_str()?;
                let language = Language::from_extension(ext);
                if language == Language::Unknown {
                    return None;
                }
                let bytes = fs::read(&abs_path).ok()?;
                let content_hash = WorkspaceManifest::hash_bytes(&bytes);
                let modified_timestamp_secs = metadata
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);

                Some(ScannedFile {
                    relative_path: rel_path,
                    absolute_path: abs_path,
                    language,
                    content_hash,
                    size_bytes: bytes.len(),
                    modified_timestamp_secs,
                })
            })
            .collect();

        let mut results = BTreeMap::new();
        for file in scanned_files {
            results.insert(file.relative_path.clone(), file);
        }
        Ok(results)
    }

    fn collect_paths_recursive(
        &self,
        current_dir: &Path,
        root_dir: &Path,
        collector: &mut Vec<(PathBuf, PathBuf)>,
    ) -> HNSQRResult<()> {
        let entries = match fs::read_dir(current_dir) {
            Ok(e) => e,
            Err(_) => return Ok(()),
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = match path.file_name().and_then(|s| s.to_str()) {
                Some(name) => name,
                None => continue,
            };

            if path.is_dir() {
                if self
                    .config
                    .ignored_directories
                    .iter()
                    .any(|ignored| ignored == file_name)
                {
                    continue;
                }
                self.collect_paths_recursive(&path, root_dir, collector)?;
            } else if path.is_file() {
                if let Ok(rel_path) = path.strip_prefix(root_dir) {
                    collector.push((rel_path.to_path_buf(), path));
                }
            }
        }
        Ok(())
    }
}
