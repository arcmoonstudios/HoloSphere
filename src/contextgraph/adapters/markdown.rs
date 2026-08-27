/* holosphere/src/contextgraph/adapters/markdown.rs */
//!▫~•◦-------------------------------‣
//! # Markdown & Documentation Context Adapter
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Maps Markdown documents, ADRs, RFCs, and documentation sections into universal
//! document entities and contextual reference relations.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;

use super::super::adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
use super::super::ir::{
    ExtractedEntity, ExtractedRelation, ExtractionBatch, SourceDescriptor, UnresolvedReference,
};
use super::super::schema::{EntityKind, Namespace, RelationKind, RelationOrigin, ResourceLocator};
use crate::transport::model_gateway::{EvidenceClass, VerificationState};
use crate::{HNSQRError, HNSQRResult};

pub struct MarkdownSourceAdapter;

impl Default for MarkdownSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl SourceAdapter for MarkdownSourceAdapter {
    fn name(&self) -> &'static str {
        "markdown_context_adapter"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities {
            deterministic: true,
            supports_incremental: true,
            supports_structural_relations: true,
            supports_semantic_extraction: true,
            supports_streaming: false,
        }
    }

    fn detect(&self, source: &SourceInput) -> bool {
        source.locator.ends_with(".md")
            || source.locator.ends_with(".markdown")
            || source.source_type == "markdown"
    }

    fn extract(
        &self,
        source: &SourceInput,
        _namespace: &Namespace,
    ) -> HNSQRResult<ExtractionBatch> {
        let text = match &source.text_content {
            Some(t) => t,
            None => {
                return Err(HNSQRError::InvalidRequest(
                    "source text_content required".to_string(),
                ));
            }
        };

        let content_hash = source.compute_fingerprint();
        let desc = SourceDescriptor {
            source_type: "markdown".to_string(),
            locator: source.locator.clone(),
            content_hash,
            metadata: source.metadata.clone(),
        };

        let mut batch = ExtractionBatch::new(desc);

        // Document entity
        let doc_temp_id = format!("doc_{}", source.locator);
        batch.entities.push(ExtractedEntity {
            temp_id: doc_temp_id.clone(),
            kind: EntityKind::new("document:article"),
            label: source.locator.clone(),
            locator: Some(ResourceLocator::uri(source.locator.clone())),
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::ExternalSource,
            verification_state: VerificationState::Verified,
            fingerprint: content_hash,
        });

        // Scan headings and sections
        let mut current_section_id = doc_temp_id.clone();
        for (line_idx, line) in text.lines().enumerate() {
            let line_num = line_idx + 1;
            let trimmed = line.trim();

            if trimmed.starts_with('#') {
                let level = trimmed.chars().take_while(|c| *c == '#').count();
                let heading_text = trimmed.trim_start_matches('#').trim().to_string();
                let sec_id = format!("sec_{}_{line_num}", source.locator);

                let mut attributes = BTreeMap::new();
                attributes.insert("heading_level".to_string(), serde_json::json!(level));

                batch.entities.push(ExtractedEntity {
                    temp_id: sec_id.clone(),
                    kind: EntityKind::document_section(),
                    label: heading_text.clone(),
                    locator: Some(ResourceLocator::file(
                        source.locator.clone(),
                        line_num,
                        line_num,
                    )),
                    attributes,
                    evidence_class: EvidenceClass::ExternalSource,
                    verification_state: VerificationState::Verified,
                    fingerprint: content_hash,
                });

                batch.relations.push(ExtractedRelation {
                    kind: RelationKind::contains(),
                    participants: vec![
                        (doc_temp_id.clone(), "source".to_string()),
                        (sec_id.clone(), "target".to_string()),
                    ],
                    origin: RelationOrigin::Extracted,
                    confidence: 1.0,
                    attributes: BTreeMap::new(),
                });

                current_section_id = sec_id;
            } else if trimmed.to_uppercase().starts_with("NOTE:")
                || trimmed.to_uppercase().starts_with("WARNING:")
                || trimmed.to_uppercase().starts_with("IMPORTANT:")
            {
                let claim_id = format!("claim_{}_{line_num}", source.locator);
                batch.entities.push(ExtractedEntity {
                    temp_id: claim_id.clone(),
                    kind: EntityKind::document_claim(),
                    label: trimmed.to_string(),
                    locator: Some(ResourceLocator::file(
                        source.locator.clone(),
                        line_num,
                        line_num,
                    )),
                    attributes: BTreeMap::new(),
                    evidence_class: EvidenceClass::ReportedClaim,
                    verification_state: VerificationState::ReportedUnverified,
                    fingerprint: content_hash,
                });

                batch.relations.push(ExtractedRelation {
                    kind: RelationKind::contains(),
                    participants: vec![
                        (current_section_id.clone(), "source".to_string()),
                        (claim_id, "target".to_string()),
                    ],
                    origin: RelationOrigin::Extracted,
                    confidence: 1.0,
                    attributes: BTreeMap::new(),
                });
            }
        }

        Ok(batch)
    }
}
