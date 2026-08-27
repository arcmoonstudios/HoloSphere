/* holosphere/src/contextgraph/ir.rs */
//!▫~•◦-------------------------------‣
//! # Universal Context Compiler Intermediate Representation (IR)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Standardized intermediate representation emitted by all source adapters (code,
//! markdown, PDFs, Git, runtime APIs) before multi-pass resolution and epistemic validation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::schema::{EntityKind, RelationKind, RelationOrigin, ResourceLocator};
use crate::transport::model_gateway::{EvidenceClass, VerificationState};

/// Identity and metadata describing the ingested raw source input.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceDescriptor {
    pub source_type: String,
    pub locator: String,
    pub content_hash: [u8; 32],
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Raw entity extracted by a domain adapter before global resolution.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub temp_id: String,
    pub kind: EntityKind,
    pub label: String,
    pub locator: Option<ResourceLocator>,
    pub attributes: BTreeMap<String, serde_json::Value>,
    pub evidence_class: EvidenceClass,
    pub verification_state: VerificationState,
    pub fingerprint: [u8; 32],
}

/// Raw relation emitted by a domain adapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractedRelation {
    pub kind: RelationKind,
    /// Participants as pairs of `(entity_temp_id_or_ref, semantic_role)`
    pub participants: Vec<(String, String)>,
    pub origin: RelationOrigin,
    pub confidence: f32,
    pub attributes: BTreeMap<String, serde_json::Value>,
}

/// Extracted binary or textual artifact (e.g. diagrams, images, tables).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExtractedArtifact {
    pub id: String,
    pub mime_type: String,
    pub content: Vec<u8>,
    pub metadata: BTreeMap<String, serde_json::Value>,
}

/// Unresolved cross-entity reference (e.g. "crate::foo::Bar", "RFC 7231", "run-492").
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnresolvedReference {
    pub source_temp_id: String,
    pub target_ref: String,
    pub expected_kind: Option<String>,
    pub relation_kind: RelationKind,
    pub role: String,
    pub locator: Option<ResourceLocator>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticSeverity {
    Info,
    Warning,
    Error,
}

/// Non-fatal diagnostic warning or informational notice during extraction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub severity: DiagnosticSeverity,
    pub message: String,
    pub locator: Option<ResourceLocator>,
    pub recoverable: bool,
}

/// Complete universal extraction payload emitted by any SourceAdapter.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractionBatch {
    pub source: SourceDescriptor,
    pub entities: Vec<ExtractedEntity>,
    pub relations: Vec<ExtractedRelation>,
    pub artifacts: Vec<ExtractedArtifact>,
    pub unresolved: Vec<UnresolvedReference>,
    pub diagnostics: Vec<Diagnostic>,
}

impl ExtractionBatch {
    #[must_use]
    pub fn new(source: SourceDescriptor) -> Self {
        Self {
            source,
            entities: Vec::new(),
            relations: Vec::new(),
            artifacts: Vec::new(),
            unresolved: Vec::new(),
            diagnostics: Vec::new(),
        }
    }
}
