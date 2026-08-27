/* holosphere/src/contextgraph/adapters/data_toml.rs */
//!▫~•◦-------------------------------‣
//! # TOML Structural Context Adapter
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Parses TOML documents into deterministic document/key entities and containment relations.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;

use super::super::adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
use super::super::ir::{ExtractedEntity, ExtractedRelation, ExtractionBatch, SourceDescriptor};
use super::super::schema::{EntityKind, Namespace, RelationKind, RelationOrigin, ResourceLocator};
use crate::transport::model_gateway::{EvidenceClass, VerificationState};
use crate::{HNSQRError, HNSQRResult};

pub struct TomlSourceAdapter;

impl Default for TomlSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl TomlSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl SourceAdapter for TomlSourceAdapter {
    fn name(&self) -> &'static str {
        "toml_structural_adapter"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities::default()
    }

    fn detect(&self, source: &SourceInput) -> bool {
        source.locator.ends_with(".toml") || source.source_type == "toml"
    }

    fn extract(
        &self,
        source: &SourceInput,
        _namespace: &Namespace,
    ) -> HNSQRResult<ExtractionBatch> {
        let text = source.text_content.as_deref().ok_or_else(|| {
            HNSQRError::InvalidRequest("source text_content required".to_string())
        })?;
        let value: toml::Value = toml::from_str(text).map_err(|error| {
            HNSQRError::InvalidRequest(format!("Invalid TOML in {}: {error}", source.locator))
        })?;
        let content_hash = source.compute_fingerprint();
        let mut batch = ExtractionBatch::new(SourceDescriptor {
            source_type: "toml".to_string(),
            locator: source.locator.clone(),
            content_hash,
            metadata: source.metadata.clone(),
        });
        let document_id = format!("toml_{}", source.locator);
        batch.entities.push(ExtractedEntity {
            temp_id: document_id.clone(),
            kind: EntityKind::new("data:toml_document"),
            label: source.locator.clone(),
            locator: Some(ResourceLocator::uri(source.locator.clone())),
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: content_hash,
        });
        if let toml::Value::Table(entries) = value {
            for (key, value) in entries {
                let key_id = format!("toml_key_{}_{}", source.locator, key);
                let mut attributes = BTreeMap::new();
                attributes.insert(
                    "value_type".to_string(),
                    serde_json::json!(toml_value_type(&value)),
                );
                attributes.insert(
                    "value".to_string(),
                    serde_json::to_value(value).map_err(|error| {
                        HNSQRError::Internal(format!("Failed to serialize TOML value: {error}"))
                    })?,
                );
                batch.entities.push(ExtractedEntity {
                    temp_id: key_id.clone(),
                    kind: EntityKind::new("data:toml_key"),
                    label: key,
                    locator: Some(ResourceLocator::uri(source.locator.clone())),
                    attributes,
                    evidence_class: EvidenceClass::Observation,
                    verification_state: VerificationState::Verified,
                    fingerprint: content_hash,
                });
                batch.relations.push(ExtractedRelation {
                    kind: RelationKind::contains(),
                    participants: vec![
                        (document_id.clone(), "source".to_string()),
                        (key_id, "target".to_string()),
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

fn toml_value_type(value: &toml::Value) -> &'static str {
    match value {
        toml::Value::String(_) => "string",
        toml::Value::Integer(_) => "integer",
        toml::Value::Float(_) => "float",
        toml::Value::Boolean(_) => "boolean",
        toml::Value::Datetime(_) => "datetime",
        toml::Value::Array(_) => "array",
        toml::Value::Table(_) => "table",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_top_level_toml_keys() {
        let input =
            SourceInput::from_text("name = 'holo'\nenabled = true", "file:///a.toml", "toml");
        let batch = TomlSourceAdapter::new()
            .extract(&input, &Namespace::new("test"))
            .unwrap();
        assert_eq!(batch.entities.len(), 3);
        assert_eq!(batch.relations.len(), 2);
    }
}
