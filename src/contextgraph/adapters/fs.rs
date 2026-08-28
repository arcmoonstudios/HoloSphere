/* holosphere/src/contextgraph/adapters/fs.rs */
//!▫~•◦-------------------------------‣
//! # Filesystem Workspace Discovery & Crawler Adapter
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Crawls repository and folder trees, respects ignore boundaries, and yields files as inputs.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::super::adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
use super::super::ir::{ExtractedEntity, ExtractedRelation, ExtractionBatch, SourceDescriptor};
use super::super::schema::{EntityKind, Namespace, RelationKind, RelationOrigin, ResourceLocator};
use crate::HNSQRResult;
use crate::transport::model_gateway::{EvidenceClass, VerificationState};

pub struct FilesystemSourceAdapter;

impl Default for FilesystemSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl FilesystemSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Crawls a directory and returns candidate SourceInputs.
    pub fn crawl_directory(&self, root_dir: impl AsRef<Path>) -> HNSQRResult<Vec<SourceInput>> {
        let root = root_dir.as_ref();
        let mut inputs = Vec::new();
        self.crawl_recursive(root, root, &mut inputs)?;
        Ok(inputs)
    }

    fn crawl_recursive(
        &self,
        current_dir: &Path,
        root_dir: &Path,
        collector: &mut Vec<SourceInput>,
    ) -> HNSQRResult<()> {
        let ignored = [
            ".git",
            ".github",
            ".vscode",
            "target",
            "node_modules",
            "vendor",
            "build",
            "dist",
            ".holosphere",
            ".gemini",
        ];

        let entries = match fs::read_dir(current_dir) {
            Ok(e) => e,
            Err(_) => return Ok(()),
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let name = match path.file_name().and_then(|s| s.to_str()) {
                Some(n) => n,
                None => continue,
            };

            if ignored.iter().any(|ig| *ig == name) || name.starts_with("target-") {
                continue;
            }

            if path.is_dir() {
                self.crawl_recursive(&path, root_dir, collector)?;
            } else if path.is_file() {
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                let source_type = match ext {
                    "rs" => "rust",
                    "md" | "markdown" => "markdown",
                    "ts" => "typescript",
                    "tsx" => "tsx",
                    "js" => "javascript",
                    "jsx" => "jsx",
                    "go" => "go",
                    "py" => "python",
                    "json" => "json",
                    "toml" => "toml",
                    _ => "text",
                };

                if let Ok(input) = SourceInput::from_file(&path, source_type) {
                    collector.push(input);
                }
            }
        }
        Ok(())
    }
}

impl SourceAdapter for FilesystemSourceAdapter {
    fn name(&self) -> &'static str {
        "fs_crawler_adapter"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            deterministic: true,
            supports_incremental: true,
            supports_structural_relations: true,
            supports_semantic_extraction: false,
            supports_streaming: false,
        }
    }

    fn detect(&self, source: &SourceInput) -> bool {
        source.source_type == "filesystem" || source.source_type == "directory"
    }

    fn extract(
        &self,
        source: &SourceInput,
        _namespace: &Namespace,
    ) -> HNSQRResult<ExtractionBatch> {
        let content_hash = source.compute_fingerprint();
        let desc = SourceDescriptor {
            source_type: "filesystem".to_string(),
            locator: source.locator.clone(),
            content_hash,
            metadata: source.metadata.clone(),
        };

        let mut batch = ExtractionBatch::new(desc);
        batch.entities.push(ExtractedEntity {
            temp_id: format!("dir_{}", source.locator),
            kind: EntityKind::new("fs:directory"),
            label: source.locator.clone(),
            locator: Some(ResourceLocator::uri(source.locator.clone())),
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: content_hash,
        });

        Ok(batch)
    }
}
