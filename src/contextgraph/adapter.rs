/* holosphere/src/contextgraph/adapter.rs */
//!▫~•◦-------------------------------‣
//! # Universal SourceAdapter Trait & Capability Declarations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Pluggable adapter interface for extracting standardized ExtractionBatch IR from
//! arbitrary sources (Rust, TypeScript, Markdown, PDFs, Git, logs, APIs, databases).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::ir::ExtractionBatch;
use super::schema::Namespace;
use crate::HNSQRResult;

/// Declared capabilities of a source adapter to guide compiler orchestration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AdapterCapabilities {
    pub deterministic: bool,
    pub supports_incremental: bool,
    pub supports_structural_relations: bool,
    pub supports_semantic_extraction: bool,
    pub supports_streaming: bool,
}

impl Default for AdapterCapabilities {
    fn default() -> Self {
        Self {
            deterministic: true,
            supports_incremental: true,
            supports_structural_relations: true,
            supports_semantic_extraction: false,
            supports_streaming: false,
        }
    }
}

/// Raw input passed into a SourceAdapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceInput {
    pub source_type: String,
    pub locator: String,
    pub text_content: Option<String>,
    pub raw_bytes: Option<Vec<u8>>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

impl SourceInput {
    #[must_use]
    pub fn from_file(path: impl AsRef<Path>, source_type: impl Into<String>) -> HNSQRResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path)?;
        let text = String::from_utf8(bytes.clone()).ok();
        let locator = format!("file:///{}", path.to_string_lossy().replace('\\', "/"));

        Ok(Self {
            source_type: source_type.into(),
            locator,
            text_content: text,
            raw_bytes: Some(bytes),
            metadata: BTreeMap::new(),
        })
    }

    #[must_use]
    pub fn from_text(
        text: impl Into<String>,
        locator: impl Into<String>,
        source_type: impl Into<String>,
    ) -> Self {
        let text = text.into();
        let bytes = text.as_bytes().to_vec();
        Self {
            source_type: source_type.into(),
            locator: locator.into(),
            text_content: Some(text),
            raw_bytes: Some(bytes),
            metadata: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn compute_fingerprint(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        if let Some(bytes) = &self.raw_bytes {
            hasher.update(bytes);
        } else if let Some(text) = &self.text_content {
            hasher.update(text.as_bytes());
        }
        hasher.finalize().into()
    }
}

/// Domain-neutral source adapter contract.
pub trait SourceAdapter: Send + Sync {
    fn name(&self) -> &'static str;
    fn version(&self) -> &'static str;
    fn capabilities(&self) -> AdapterCapabilities;

    /// Evaluates whether this adapter handles the given source input.
    fn detect(&self, source: &SourceInput) -> bool;

    /// Computes content fingerprint.
    fn fingerprint(&self, source: &SourceInput) -> HNSQRResult<[u8; 32]> {
        Ok(source.compute_fingerprint())
    }

    /// Extracts universal IR batch from the source.
    fn extract(&self, source: &SourceInput, namespace: &Namespace) -> HNSQRResult<ExtractionBatch>;
}
