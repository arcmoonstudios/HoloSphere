/* holosphere/src/contextgraph/adapters/code_python.rs */
//!▫~•◦-------------------------------‣
//! # Python AST Context Adapter (tree-sitter-python).
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Extracts files, classes, functions, test functions, rationale comments, inheritance, and
//! unresolved call edges into the same ContextGraph IR used by the Rust/JS/TS adapters.
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

pub struct PythonSourceAdapter;

impl Default for PythonSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn init_parser() -> HNSQRResult<Parser> {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_python::LANGUAGE.into())
            .map_err(|error| {
                HNSQRError::Internal(format!("tree-sitter-python init error: {error:?}"))
            })?;
        Ok(parser)
    }

    fn text_of<'a>(node: &Node, source: &'a str) -> &'a str {
        node.utf8_text(source.as_bytes()).unwrap_or("")
    }
}

impl SourceAdapter for PythonSourceAdapter {
    fn name(&self) -> &'static str {
        "python_treesitter_adapter"
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
        source.locator.ends_with(".py")
            || source.locator.ends_with(".pyi")
            || source.source_type == "python"
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
            source_type: "python".to_string(),
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
        let mut visitor = PythonAstVisitor {
            source: text,
            source_locator: &source.locator,
            file_temp_id: &file_temp_id,
            content_hash,
            batch: &mut batch,
            scope: Vec::new(),
            current_class: None,
            pending_rationale: Vec::new(),
        };
        visitor.walk(&tree.root_node());
        Ok(batch)
    }
}

struct PythonAstVisitor<'a, 'b> {
    source: &'a str,
    source_locator: &'a str,
    file_temp_id: &'a str,
    content_hash: [u8; 32],
    batch: &'b mut ExtractionBatch,
    scope: Vec<String>,
    current_class: Option<String>,
    pending_rationale: Vec<(String, String, usize, usize)>,
}

impl<'a, 'b> PythonAstVisitor<'a, 'b> {
    fn qualified_name(&self, name: &str) -> String {
        let mut parts = self.scope.clone();
        if let Some(class) = &self.current_class {
            parts.push(class.clone());
        }
        parts.push(name.to_string());
        parts.join(".")
    }

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
            "class_definition" => self.visit_class(node),
            "function_definition" => self.visit_function(node),
            "call" => self.visit_call(node),
            _ => self.walk_children(node),
        }
    }

    fn consume_comment(&mut self, node: &Node) {
        let text = PythonSourceAdapter::text_of(node, self.source).trim();
        let stripped = text.trim_start_matches('#').trim();
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

    fn docstring(&self, body: Node) -> Option<String> {
        let first = body.named_child(0)?;
        if first.kind() != "expression_statement" {
            return None;
        }
        let value = first.named_child(0)?;
        (value.kind() == "string").then(|| {
            PythonSourceAdapter::text_of(&value, self.source)
                .trim_matches(['\'', '"'])
                .to_string()
        })
    }

    fn define(
        &mut self,
        id: &str,
        kind: EntityKind,
        label: String,
        node: &Node,
        doc: Option<String>,
    ) {
        let mut attributes = BTreeMap::new();
        if let Some(docstring) = doc {
            attributes.insert("docstring".to_string(), serde_json::json!(docstring));
        }
        self.batch.entities.push(ExtractedEntity {
            temp_id: id.to_string(),
            kind,
            label,
            locator: Some(self.locator(node)),
            attributes,
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: self.content_hash,
        });
        self.batch.relations.push(ExtractedRelation {
            kind: RelationKind::defines(),
            participants: vec![
                (self.file_temp_id.to_string(), "source".to_string()),
                (id.to_string(), "target".to_string()),
            ],
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            attributes: BTreeMap::new(),
        });
        self.attach_rationale(id);
    }

    fn visit_class(&mut self, node: &Node) {
        let name = node
            .child_by_field_name("name")
            .map(|n| PythonSourceAdapter::text_of(&n, self.source))
            .unwrap_or("AnonClass");
        let qual = self.qualified_name(name);
        let id = format!("class_{qual}");
        let body = node.child_by_field_name("body");
        self.define(
            &id,
            EntityKind::new("code:class"),
            qual,
            node,
            body.and_then(|b| self.docstring(b)),
        );
        if let Some(superclasses) = node.child_by_field_name("superclasses") {
            let mut cursor = superclasses.walk();
            for child in superclasses.named_children(&mut cursor) {
                let reference = PythonSourceAdapter::text_of(&child, self.source)
                    .trim()
                    .to_string();
                if !reference.is_empty() {
                    self.batch.unresolved.push(UnresolvedReference {
                        source_temp_id: id.clone(),
                        target_ref: reference,
                        expected_kind: Some("code:class".to_string()),
                        relation_kind: RelationKind::new("extends"),
                        role: "extends_target".to_string(),
                        locator: Some(self.locator(&child)),
                    });
                }
            }
        }
        let previous = self.current_class.replace(name.to_string());
        if let Some(body) = body {
            self.walk_children(&body);
        }
        self.current_class = previous;
    }

    fn visit_function(&mut self, node: &Node) {
        let name = node
            .child_by_field_name("name")
            .map(|n| PythonSourceAdapter::text_of(&n, self.source))
            .unwrap_or("anonymous");
        let qual = self.qualified_name(name);
        let id = format!("fn_{qual}");
        let body = node.child_by_field_name("body");
        let kind = if name.starts_with("test_") {
            EntityKind::test_case()
        } else {
            EntityKind::code_function()
        };
        self.define(&id, kind, qual, node, body.and_then(|b| self.docstring(b)));
        if let Some(body) = body {
            let mut visitor = PythonBodyVisitor {
                source: self.source,
                source_locator: self.source_locator,
                owner_id: &id,
                batch: self.batch,
            };
            visitor.walk(&body);
        }
    }

    fn visit_call(&mut self, node: &Node) {
        self.walk_children(node);
    }
}

struct PythonBodyVisitor<'a, 'b> {
    source: &'a str,
    source_locator: &'a str,
    owner_id: &'a str,
    batch: &'b mut ExtractionBatch,
}

impl<'a, 'b> PythonBodyVisitor<'a, 'b> {
    fn walk(&mut self, node: &Node) {
        if node.kind() == "call" {
            if let Some(function) = node.child_by_field_name("function") {
                self.batch.unresolved.push(UnresolvedReference {
                    source_temp_id: self.owner_id.to_string(),
                    target_ref: PythonSourceAdapter::text_of(&function, self.source)
                        .trim()
                        .to_string(),
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
    fn extracts_python_structure_rationale_and_call_edges() {
        let input = SourceInput::from_text(
            r#"
# SAFETY: database handles remain scoped to the request.
class Service(BaseService):
    \"\"\"Coordinates a request.\"\"\"
    def test_handles_request(self):
        return process_request()
"#,
            "file:///service.py",
            "python",
        );
        let batch = PythonSourceAdapter::new()
            .extract(&input, &Namespace::new("test"))
            .unwrap();
        assert!(batch
            .entities
            .iter()
            .any(|entity| entity.kind == EntityKind::new("code:class")));
        assert!(batch
            .entities
            .iter()
            .any(|entity| entity.kind == EntityKind::test_case()));
        assert!(batch
            .entities
            .iter()
            .any(|entity| entity.kind == EntityKind::code_rationale()));
        assert!(batch
            .unresolved
            .iter()
            .any(|reference| reference.relation_kind == RelationKind::calls()));
        assert!(batch
            .unresolved
            .iter()
            .any(|reference| reference.relation_kind == RelationKind::new("extends")));
    }
}
