/* holosphere/src/contextgraph/adapters/code_rust.rs */
//!▫~•◦-------------------------------‣
//! # Rust AST Context Adapter (tree-sitter-rust)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Maps Rust source code into universal ExtractedEntity and ExtractedRelation IR tokens.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::path::Path;

use tree_sitter::{Node, Parser};

use super::super::adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
use super::super::ir::{
    ExtractedEntity, ExtractedRelation, ExtractionBatch, SourceDescriptor, UnresolvedReference,
};
use super::super::schema::{EntityKind, Namespace, RelationKind, RelationOrigin, ResourceLocator};
use crate::transport::model_gateway::{EvidenceClass, VerificationState};
use crate::{HNSQRError, HNSQRResult};

pub struct RustSourceAdapter;

impl Default for RustSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl RustSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn init_parser() -> HNSQRResult<Parser> {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .map_err(|e| HNSQRError::Internal(format!("tree-sitter-rust init error: {e:?}")))?;
        Ok(parser)
    }

    fn text_of<'a>(node: &Node, source: &'a str) -> &'a str {
        node.utf8_text(source.as_bytes()).unwrap_or("")
    }
}

impl SourceAdapter for RustSourceAdapter {
    fn name(&self) -> &'static str {
        "rust_treesitter_adapter"
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
        source.locator.ends_with(".rs") || source.source_type == "rust"
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

        let mut parser = Self::init_parser()?;
        let tree = parser.parse(text, None).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Failed to parse {}", source.locator))
        })?;

        let root_node = tree.root_node();
        let content_hash = source.compute_fingerprint();

        let desc = SourceDescriptor {
            source_type: "rust".to_string(),
            locator: source.locator.clone(),
            content_hash,
            metadata: source.metadata.clone(),
        };

        let mut batch = ExtractionBatch::new(desc);

        // 1. File Entity
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

        // 2. Walk items
        let mut visitor = RustAstVisitor {
            source: text,
            source_locator: &source.locator,
            file_temp_id: &file_temp_id,
            content_hash,
            batch: &mut batch,
            scope: Vec::new(),
            current_impl: None,
            pending_doc: Vec::new(),
            pending_rationale: Vec::new(),
        };

        visitor.walk(&root_node);

        Ok(batch)
    }
}

struct RustAstVisitor<'a, 'b> {
    source: &'a str,
    source_locator: &'a str,
    file_temp_id: &'a str,
    content_hash: [u8; 32],
    batch: &'b mut ExtractionBatch,
    scope: Vec<String>,
    current_impl: Option<String>,
    pending_doc: Vec<String>,
    pending_rationale: Vec<(String, String, usize, usize)>, // (tag, text, start_line, end_line)
}

impl<'a, 'b> RustAstVisitor<'a, 'b> {
    fn current_qual(&self, name: &str) -> String {
        if let Some(impl_tgt) = &self.current_impl {
            if self.scope.is_empty() {
                format!("{impl_tgt}::{name}")
            } else {
                format!("{}::{impl_tgt}::{name}", self.scope.join("::"))
            }
        } else if self.scope.is_empty() {
            name.to_string()
        } else {
            format!("{}::{name}", self.scope.join("::"))
        }
    }

    fn walk(&mut self, node: &Node) {
        let kind = node.kind();

        if kind == "line_comment" || kind == "block_comment" {
            let text = RustSourceAdapter::text_of(node, self.source).trim();
            let start = node.start_position().row + 1;
            let end = node.end_position().row + 1;

            if text.starts_with("///") || text.starts_with("//!") {
                let cleaned = text
                    .trim_start_matches("///")
                    .trim_start_matches("//!")
                    .trim()
                    .to_string();
                self.pending_doc.push(cleaned);
            } else if let Some(stripped) = text.strip_prefix("//") {
                let trimmed = stripped.trim();
                let upper = trimmed.to_uppercase();
                for marker in &[
                    "SAFETY:",
                    "WHY:",
                    "NOTE:",
                    "INVARIANT:",
                    "HACK:",
                    "TODO:",
                    "BUG:",
                ] {
                    if upper.starts_with(marker) {
                        let tag = marker.trim_end_matches(':').to_lowercase();
                        self.pending_rationale
                            .push((tag, trimmed.to_string(), start, end));
                        break;
                    }
                }
            }
            return;
        }

        match kind {
            "mod_item" => self.visit_mod(node),
            "struct_item" => self.visit_struct(node),
            "enum_item" => self.visit_enum(node),
            "trait_item" => self.visit_trait(node),
            "function_item" => self.visit_function(node),
            "impl_item" => self.visit_impl(node),
            "call_expression" => self.visit_call(node),
            _ => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.walk(&child);
                }
            }
        }
    }

    fn take_doc(&mut self) -> Option<String> {
        if self.pending_doc.is_empty() {
            None
        } else {
            let docs: Vec<String> = self.pending_doc.drain(..).collect();
            Some(docs.join("\n"))
        }
    }

    fn attach_rationale(&mut self, target_temp_id: &str) {
        for (tag, text, start, end) in self.pending_rationale.drain(..) {
            let rat_id = format!("rat_{}_{}", start, tag);
            let locator = ResourceLocator {
                uri: self.source_locator.to_string(),
                start_line: Some(start),
                end_line: Some(end),
                start_byte: None,
                end_byte: None,
            };

            let mut attributes = BTreeMap::new();
            attributes.insert("tag".to_string(), serde_json::json!(tag));
            attributes.insert("text".to_string(), serde_json::json!(text));

            self.batch.entities.push(ExtractedEntity {
                temp_id: rat_id.clone(),
                kind: EntityKind::code_rationale(),
                label: format!("{tag}: {}", text.chars().take(40).collect::<String>()),
                locator: Some(locator),
                attributes,
                evidence_class: EvidenceClass::Observation,
                verification_state: VerificationState::Verified,
                fingerprint: self.content_hash,
            });

            let rel_kind = if tag == "safety" {
                RelationKind::justifies()
            } else {
                RelationKind::explains()
            };

            self.batch.relations.push(ExtractedRelation {
                kind: rel_kind,
                participants: vec![
                    (rat_id, "source".to_string()),
                    (target_temp_id.to_string(), "target".to_string()),
                ],
                origin: RelationOrigin::Extracted,
                confidence: 1.0,
                attributes: BTreeMap::new(),
            });
        }
    }

    fn visit_mod(&mut self, node: &Node) {
        let mod_name = node
            .child_by_field_name("name")
            .map(|n| RustSourceAdapter::text_of(&n, self.source))
            .unwrap_or("anon_mod");

        let qual = self.current_qual(mod_name);
        let start = node.start_position().row + 1;
        let end = node.end_position().row + 1;
        let temp_id = format!("mod_{qual}");

        let doc = self.take_doc();
        let mut attributes = BTreeMap::new();
        if let Some(d) = doc {
            attributes.insert("docstring".to_string(), serde_json::json!(d));
        }

        self.batch.entities.push(ExtractedEntity {
            temp_id: temp_id.clone(),
            kind: EntityKind::code_module(),
            label: qual.clone(),
            locator: Some(ResourceLocator {
                uri: self.source_locator.to_string(),
                start_line: Some(start),
                end_line: Some(end),
                start_byte: Some(node.start_byte()),
                end_byte: Some(node.end_byte()),
            }),
            attributes,
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: self.content_hash,
        });

        // File defines Module
        self.batch.relations.push(ExtractedRelation {
            kind: RelationKind::defines(),
            participants: vec![
                (self.file_temp_id.to_string(), "source".to_string()),
                (temp_id.clone(), "target".to_string()),
            ],
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            attributes: BTreeMap::new(),
        });

        self.attach_rationale(&temp_id);

        self.scope.push(mod_name.to_string());
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }
        self.scope.pop();
    }

    fn visit_struct(&mut self, node: &Node) {
        let struct_name = node
            .child_by_field_name("name")
            .map(|n| RustSourceAdapter::text_of(&n, self.source))
            .unwrap_or("AnonStruct");

        let qual = self.current_qual(struct_name);
        let start = node.start_position().row + 1;
        let end = node.end_position().row + 1;
        let temp_id = format!("struct_{qual}");

        let doc = self.take_doc();
        let mut attributes = BTreeMap::new();
        if let Some(d) = doc {
            attributes.insert("docstring".to_string(), serde_json::json!(d));
        }

        self.batch.entities.push(ExtractedEntity {
            temp_id: temp_id.clone(),
            kind: EntityKind::code_struct(),
            label: qual.clone(),
            locator: Some(ResourceLocator {
                uri: self.source_locator.to_string(),
                start_line: Some(start),
                end_line: Some(end),
                start_byte: Some(node.start_byte()),
                end_byte: Some(node.end_byte()),
            }),
            attributes,
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: self.content_hash,
        });

        self.batch.relations.push(ExtractedRelation {
            kind: RelationKind::defines(),
            participants: vec![
                (self.file_temp_id.to_string(), "source".to_string()),
                (temp_id.clone(), "target".to_string()),
            ],
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            attributes: BTreeMap::new(),
        });

        self.attach_rationale(&temp_id);
    }

    fn visit_enum(&mut self, node: &Node) {
        let enum_name = node
            .child_by_field_name("name")
            .map(|n| RustSourceAdapter::text_of(&n, self.source))
            .unwrap_or("AnonEnum");

        let qual = self.current_qual(enum_name);
        let start = node.start_position().row + 1;
        let end = node.end_position().row + 1;
        let temp_id = format!("enum_{qual}");

        let doc = self.take_doc();
        let mut attributes = BTreeMap::new();
        if let Some(d) = doc {
            attributes.insert("docstring".to_string(), serde_json::json!(d));
        }

        self.batch.entities.push(ExtractedEntity {
            temp_id: temp_id.clone(),
            kind: EntityKind::new("code:enum"),
            label: qual.clone(),
            locator: Some(ResourceLocator {
                uri: self.source_locator.to_string(),
                start_line: Some(start),
                end_line: Some(end),
                start_byte: Some(node.start_byte()),
                end_byte: Some(node.end_byte()),
            }),
            attributes,
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: self.content_hash,
        });

        self.batch.relations.push(ExtractedRelation {
            kind: RelationKind::defines(),
            participants: vec![
                (self.file_temp_id.to_string(), "source".to_string()),
                (temp_id.clone(), "target".to_string()),
            ],
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            attributes: BTreeMap::new(),
        });

        self.attach_rationale(&temp_id);
    }

    fn visit_trait(&mut self, node: &Node) {
        let trait_name = node
            .child_by_field_name("name")
            .map(|n| RustSourceAdapter::text_of(&n, self.source))
            .unwrap_or("AnonTrait");

        let qual = self.current_qual(trait_name);
        let start = node.start_position().row + 1;
        let end = node.end_position().row + 1;
        let temp_id = format!("trait_{qual}");

        let doc = self.take_doc();
        let mut attributes = BTreeMap::new();
        if let Some(d) = doc {
            attributes.insert("docstring".to_string(), serde_json::json!(d));
        }

        self.batch.entities.push(ExtractedEntity {
            temp_id: temp_id.clone(),
            kind: EntityKind::code_trait(),
            label: qual.clone(),
            locator: Some(ResourceLocator {
                uri: self.source_locator.to_string(),
                start_line: Some(start),
                end_line: Some(end),
                start_byte: Some(node.start_byte()),
                end_byte: Some(node.end_byte()),
            }),
            attributes,
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: self.content_hash,
        });

        self.batch.relations.push(ExtractedRelation {
            kind: RelationKind::defines(),
            participants: vec![
                (self.file_temp_id.to_string(), "source".to_string()),
                (temp_id.clone(), "target".to_string()),
            ],
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            attributes: BTreeMap::new(),
        });

        self.attach_rationale(&temp_id);

        self.scope.push(trait_name.to_string());
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }
        self.scope.pop();
    }

    fn visit_impl(&mut self, node: &Node) {
        let type_node = node.child_by_field_name("type");
        let trait_node = node.child_by_field_name("trait");

        let target_type = type_node
            .map(|n| {
                RustSourceAdapter::text_of(&n, self.source)
                    .trim()
                    .to_string()
            })
            .unwrap_or_else(|| "UnknownType".to_string());

        let trait_name = trait_node.map(|n| {
            RustSourceAdapter::text_of(&n, self.source)
                .trim()
                .to_string()
        });

        let prev_impl = self.current_impl.replace(target_type.clone());

        if let Some(trait_ref) = &trait_name {
            self.batch.unresolved.push(UnresolvedReference {
                source_temp_id: format!("struct_{target_type}"),
                target_ref: trait_ref.clone(),
                expected_kind: Some("code:trait".to_string()),
                relation_kind: RelationKind::implements(),
                role: "target".to_string(),
                locator: Some(ResourceLocator {
                    uri: self.source_locator.to_string(),
                    start_line: Some(node.start_position().row + 1),
                    end_line: Some(node.end_position().row + 1),
                    start_byte: None,
                    end_byte: None,
                }),
            });
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }

        self.current_impl = prev_impl;
    }

    fn visit_function(&mut self, node: &Node) {
        let fn_name = node
            .child_by_field_name("name")
            .map(|n| RustSourceAdapter::text_of(&n, self.source))
            .unwrap_or("anon_fn");

        let qual = self.current_qual(fn_name);
        let start = node.start_position().row + 1;
        let end = node.end_position().row + 1;
        let temp_id = format!("fn_{qual}");

        let doc = self.take_doc();
        let mut attributes = BTreeMap::new();
        if let Some(d) = doc {
            attributes.insert("docstring".to_string(), serde_json::json!(d));
        }

        let is_test = fn_name.starts_with("test_")
            || self.source[..node.start_byte()].ends_with("#[test]")
            || self.source[..node.start_byte()].ends_with("#[tokio::test]");

        if is_test {
            attributes.insert("is_test".to_string(), serde_json::json!(true));
        }

        self.batch.entities.push(ExtractedEntity {
            temp_id: temp_id.clone(),
            kind: if is_test {
                EntityKind::test_case()
            } else {
                EntityKind::code_function()
            },
            label: qual.clone(),
            locator: Some(ResourceLocator {
                uri: self.source_locator.to_string(),
                start_line: Some(start),
                end_line: Some(end),
                start_byte: Some(node.start_byte()),
                end_byte: Some(node.end_byte()),
            }),
            attributes,
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
            fingerprint: self.content_hash,
        });

        self.batch.relations.push(ExtractedRelation {
            kind: RelationKind::defines(),
            participants: vec![
                (self.file_temp_id.to_string(), "source".to_string()),
                (temp_id.clone(), "target".to_string()),
            ],
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            attributes: BTreeMap::new(),
        });

        self.attach_rationale(&temp_id);

        // Walk body for calls
        if let Some(body) = node.child_by_field_name("body") {
            let mut body_visitor = RustBodyVisitor {
                source: self.source,
                source_locator: self.source_locator,
                fn_temp_id: &temp_id,
                batch: self.batch,
            };
            body_visitor.walk(&body);
        }
    }

    fn visit_call(&mut self, _node: &Node) {}
}

struct RustBodyVisitor<'a, 'b> {
    source: &'a str,
    source_locator: &'a str,
    fn_temp_id: &'a str,
    batch: &'b mut ExtractionBatch,
}

impl<'a, 'b> RustBodyVisitor<'a, 'b> {
    fn walk(&mut self, node: &Node) {
        if node.kind() == "call_expression" {
            if let Some(fn_node) = node.child_by_field_name("function") {
                let call_text = RustSourceAdapter::text_of(&fn_node, self.source).trim();
                let start = node.start_position().row + 1;
                let end = node.end_position().row + 1;

                self.batch.unresolved.push(UnresolvedReference {
                    source_temp_id: self.fn_temp_id.to_string(),
                    target_ref: call_text.to_string(),
                    expected_kind: Some("code:function".to_string()),
                    relation_kind: RelationKind::calls(),
                    role: "callee".to_string(),
                    locator: Some(ResourceLocator {
                        uri: self.source_locator.to_string(),
                        start_line: Some(start),
                        end_line: Some(end),
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
