/* holosphere/src/contextgraph/adapters/data_json.rs */
//!▫~•◦-------------------------------‣
//! # JSON Structural Context Adapter
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Parses JSON documents into deterministic document/key entities and containment relations.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;

use super::super::adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
use super::super::ir::{ExtractedEntity, ExtractedRelation, ExtractionBatch, SourceDescriptor};
use super::super::schema::{EntityKind, Namespace, RelationKind, RelationOrigin, ResourceLocator};
use crate::transport::model_gateway::{EvidenceClass, VerificationState};
use crate::{HNSQRError, HNSQRResult};

pub struct JsonSourceAdapter;

impl Default for JsonSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl SourceAdapter for JsonSourceAdapter {
    fn name(&self) -> &'static str {
        "json_structural_adapter"
    }

    fn version(&self) -> &'static str {
        "1.0.0"
    }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities::default()
    }

    fn detect(&self, source: &SourceInput) -> bool {
        source.locator.ends_with(".json") || source.source_type == "json"
    }

    fn extract(
        &self,
        source: &SourceInput,
        _namespace: &Namespace,
    ) -> HNSQRResult<ExtractionBatch> {
        let text = source.text_content.as_deref().ok_or_else(|| {
            HNSQRError::InvalidRequest("source text_content required".to_string())
        })?;
        let value: serde_json::Value = serde_json::from_str(text).map_err(|error| {
            HNSQRError::InvalidRequest(format!("Invalid JSON in {}: {error}", source.locator))
        })?;
        let content_hash = source.compute_fingerprint();
        let mut batch = ExtractionBatch::new(SourceDescriptor {
            source_type: "json".to_string(),
            locator: source.locator.clone(),
            content_hash,
            metadata: source.metadata.clone(),
        });
        let document_id = format!("json_{}", source.locator);
        let mut document_attributes = BTreeMap::new();
        document_attributes.insert(
            "value_type".to_string(),
            serde_json::json!(value_type(&value)),
        );
        batch.entities.push(ExtractedEntity {
            temp_id: document_id.clone(),
            kind: EntityKind::new("data:json_document"),
            label: source.locator.clone(),
            locator: Some(ResourceLocator::uri(source.locator.clone())),
            attributes: document_attributes,
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: content_hash,
        });
        if let serde_json::Value::Object(entries) = value {
            for (key, value) in entries {
                let key_id = format!("json_key_{}_{}", source.locator, key);
                let mut attributes = BTreeMap::new();
                attributes.insert(
                    "value_type".to_string(),
                    serde_json::json!(value_type(&value)),
                );
                attributes.insert("value".to_string(), value);
                batch.entities.push(ExtractedEntity {
                    temp_id: key_id.clone(),
                    kind: EntityKind::new("data:json_key"),
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

fn value_type(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_top_level_json_keys() {
        let input = SourceInput::from_text(
            r#"{"name":"holo","enabled":true}"#,
            "file:///a.json",
            "json",
        );
        let batch = JsonSourceAdapter::new()
            .extract(&input, &Namespace::new("test"))
            .unwrap();
        assert_eq!(batch.entities.len(), 3);
        assert_eq!(batch.relations.len(), 2);
    }
}
