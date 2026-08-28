/* holosphere/src/contextgraph/adapters/code_go.rs */
//!▫~•◦-------------------------------‣
//! # Go AST Context Adapter (tree-sitter-go).
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Extracts files, named Go types, functions, test functions, rationale comments, and
//! unresolved call edges into the shared ContextGraph IR.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;

use tree_sitter::{Node, Parser};

use super::super::adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
use super::super::ir::{
    ExtractedEntity, ExtractedRelation, ExtractionBatch, SourceDescriptor, UnresolvedReference,
};
use super::super::schema::{EntityKind, Namespace, RelationKind, RelationOrigin, ResourceLocator};
use crate::transport::model_gateway::{EvidenceClass, VerificationState};
use crate::{HNSQRError, HNSQRResult};

/// Deterministic structural extractor for Go source files.
pub struct GoSourceAdapter;

impl Default for GoSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl GoSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn init_parser() -> HNSQRResult<Parser> {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_go::LANGUAGE.into())
            .map_err(|error| {
                HNSQRError::Internal(format!("tree-sitter-go init error: {error:?}"))
            })?;
        Ok(parser)
    }

    fn text_of<'a>(node: &Node, source: &'a str) -> &'a str {
        node.utf8_text(source.as_bytes()).unwrap_or("")
    }
}

impl SourceAdapter for GoSourceAdapter {
    fn name(&self) -> &'static str {
        "go_treesitter_adapter"
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
        source.locator.ends_with(".go") || source.source_type == "go"
    }

    fn extract(
        &self,
        source: &SourceInput,
        _namespace: &Namespace,
    ) -> HNSQRResult<ExtractionBatch> {
        let text = source.text_content.as_ref().ok_or_else(|| {
            HNSQRError::InvalidRequest("source text_content required".to_string())
        })?;
        let mut parser = Self::init_parser()?;
        let tree = parser.parse(text, None).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Failed to parse {}", source.locator))
        })?;
        let content_hash = source.compute_fingerprint();
        let descriptor = SourceDescriptor {
            source_type: "go".to_string(),
            locator: source.locator.clone(),
            content_hash,
            metadata: source.metadata.clone(),
        };
        let mut batch = ExtractionBatch::new(descriptor);
        let file_temp_id = format!("file_{}", source.locator);
        batch.entities.push(ExtractedEntity {
            temp_id: file_temp_id.clone(),
            kind: EntityKind::code_file(),
            label: source.locator.clone(),
            locator: Some(ResourceLocator::uri(source.locator.clone())),
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: content_hash,
        });
        GoAstVisitor {
            source: text,
            source_locator: &source.locator,
            file_temp_id: &file_temp_id,
            content_hash,
            batch: &mut batch,
            pending_rationale: Vec::new(),
        }
        .walk(&tree.root_node());
        Ok(batch)
    }
}

struct GoAstVisitor<'a, 'b> {
    source: &'a str,
    source_locator: &'a str,
    file_temp_id: &'a str,
    content_hash: [u8; 32],
    batch: &'b mut ExtractionBatch,
    pending_rationale: Vec<(String, String, usize, usize)>,
}

impl<'a, 'b> GoAstVisitor<'a, 'b> {
    fn locator(&self, node: &Node) -> ResourceLocator {
        ResourceLocator {
            uri: self.source_locator.to_string(),
            start_line: Some(node.start_position().row + 1),
            end_line: Some(node.end_position().row + 1),
            start_byte: Some(node.start_byte()),
            end_byte: Some(node.end_byte()),
        }
    }

    fn walk_children(&mut self, node: &Node) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }
    }

    fn walk(&mut self, node: &Node) {
        match node.kind() {
            "comment" => self.consume_comment(node),
            "function_declaration" | "method_declaration" => self.visit_function(node),
            "type_declaration" => self.walk_children(node),
            "type_spec" => self.visit_type(node),
            _ => self.walk_children(node),
        }
    }

    fn consume_comment(&mut self, node: &Node) {
        let text = GoSourceAdapter::text_of(node, self.source).trim();
        let stripped = text
            .trim_start_matches("//")
            .trim_start_matches("/*")
            .trim_end_matches("*/")
            .trim();
        let upper = stripped.to_ascii_uppercase();
        for marker in [
            "SAFETY:",
            "WHY:",
            "NOTE:",
            "INVARIANT:",
            "HACK:",
            "TODO:",
            "BUG:",
        ] {
            if upper.starts_with(marker) {
                self.pending_rationale.push((
                    marker.trim_end_matches(':').to_ascii_lowercase(),
                    stripped.to_string(),
                    node.start_position().row + 1,
                    node.end_position().row + 1,
                ));
                break;
            }
        }
    }

    fn attach_rationale(&mut self, target: &str) {
        for (tag, text, start, end) in self.pending_rationale.drain(..) {
            let id = format!("rat_{start}_{tag}");
            let mut attributes = BTreeMap::new();
            attributes.insert("tag".to_string(), serde_json::json!(tag));
            attributes.insert("text".to_string(), serde_json::json!(text));
            self.batch.entities.push(ExtractedEntity {
                temp_id: id.clone(),
                kind: EntityKind::code_rationale(),
                label: format!("{tag}: {}", text.chars().take(40).collect::<String>()),
                locator: Some(ResourceLocator {
                    uri: self.source_locator.to_string(),
                    start_line: Some(start),
                    end_line: Some(end),
                    start_byte: None,
                    end_byte: None,
                }),
                attributes,
                evidence_class: EvidenceClass::Observation,
                verification_state: VerificationState::Verified,
                fingerprint: self.content_hash,
            });
            self.batch.relations.push(ExtractedRelation {
                kind: if tag == "safety" {
                    RelationKind::justifies()
                } else {
                    RelationKind::explains()
                },
                participants: vec![
                    (id, "source".to_string()),
                    (target.to_string(), "target".to_string()),
                ],
                origin: RelationOrigin::Extracted,
                confidence: 1.0,
                attributes: BTreeMap::new(),
            });
        }
    }

    fn define(&mut self, id: String, kind: EntityKind, label: String, node: &Node) {
        self.batch.entities.push(ExtractedEntity {
            temp_id: id.clone(),
            kind,
            label,
            locator: Some(self.locator(node)),
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: self.content_hash,
        });
        self.batch.relations.push(ExtractedRelation {
            kind: RelationKind::defines(),
            participants: vec![
                (self.file_temp_id.to_string(), "source".to_string()),
                (id.clone(), "target".to_string()),
            ],
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            attributes: BTreeMap::new(),
        });
        self.attach_rationale(&id);
    }

    fn visit_type(&mut self, node: &Node) {
        let name = node
            .child_by_field_name("name")
            .map(|child| GoSourceAdapter::text_of(&child, self.source))
            .unwrap_or("AnonymousType");
        let kind = match node.child_by_field_name("type").map(|child| child.kind()) {
            Some("struct_type") => EntityKind::new("code:struct"),
            Some("interface_type") => EntityKind::new("code:interface"),
            _ => EntityKind::new("code:type"),
        };
        self.define(format!("type_{name}"), kind, name.to_string(), node);
    }

    fn visit_function(&mut self, node: &Node) {
        let name = node
            .child_by_field_name("name")
            .map(|child| GoSourceAdapter::text_of(&child, self.source))
            .unwrap_or("anonymous");
        let id = format!("fn_{name}_{}", node.start_byte());
        let kind = if name.starts_with("Test") {
            EntityKind::test_case()
        } else {
            EntityKind::code_function()
        };
        self.define(id.clone(), kind, name.to_string(), node);
        if let Some(body) = node.child_by_field_name("body") {
            GoBodyVisitor {
                source: self.source,
                source_locator: self.source_locator,
                owner_id: &id,
                batch: self.batch,
            }
            .walk(&body);
        }
    }
}

struct GoBodyVisitor<'a, 'b> {
    source: &'a str,
    source_locator: &'a str,
    owner_id: &'a str,
    batch: &'b mut ExtractionBatch,
}

impl<'a, 'b> GoBodyVisitor<'a, 'b> {
    fn walk(&mut self, node: &Node) {
        if node.kind() == "call_expression" {
            if let Some(function) = node.child_by_field_name("function") {
                let target_ref = GoSourceAdapter::text_of(&function, self.source)
                    .trim()
                    .to_string();
                if !target_ref.is_empty() {
                    self.batch.unresolved.push(UnresolvedReference {
                        source_temp_id: self.owner_id.to_string(),
                        target_ref,
                        expected_kind: Some("code:function".to_string()),
                        relation_kind: RelationKind::calls(),
                        role: "callee".to_string(),
                        locator: Some(ResourceLocator {
                            uri: self.source_locator.to_string(),
                            start_line: Some(node.start_position().row + 1),
                            end_line: Some(node.end_position().row + 1),
                            start_byte: Some(node.start_byte()),
                            end_byte: Some(node.end_byte()),
                        }),
                    });
                }
            }
        }
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_go_structure_rationale_and_call_edges() {
        let input = SourceInput::from_text(
            r#"
// SAFETY: state remains scoped to the request.
type Service struct{}

func TestService(t *testing.T) {
    processRequest()
}
"#,
            "file:///service.go",
            "go",
        );
        let batch = GoSourceAdapter::new()
            .extract(&input, &Namespace::new("test"))
            .unwrap();
        assert!(
            batch
                .entities
                .iter()
                .any(|entity| entity.kind == EntityKind::new("code:struct"))
        );
        assert!(
            batch
                .entities
                .iter()
                .any(|entity| entity.kind == EntityKind::test_case())
        );
        assert!(
            batch
                .entities
                .iter()
                .any(|entity| entity.kind == EntityKind::code_rationale())
        );
        assert!(
            batch
                .unresolved
                .iter()
                .any(|reference| reference.relation_kind == RelationKind::calls())
        );
    }
}
