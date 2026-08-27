/* holosphere/src/contextgraph/adapters/code_ts.rs */
//!▫~•◦-------------------------------‣
//! # TypeScript AST Context Adapter (tree-sitter-typescript)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Maps TypeScript source code into universal ExtractedEntity and ExtractedRelation IR tokens.
//!
//! Superset of `code_js.rs`'s extraction surface: adds interfaces, type aliases, enums, and
//! `implements`/interface-`extends` edges. Does not handle JSX — see `code_tsx.rs` for that
//! dialect. See the Elevation Notes delivered alongside this file for the full assumption
//! surface (heritage-clause node-kind naming is a hypothesis, not tool-verified per file).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;

use tree_sitter::{Language, Node, Parser};

use super::super::adapter::{AdapterCapabilities, SourceAdapter, SourceInput};
use super::super::ir::{
    ExtractedEntity, ExtractedRelation, ExtractionBatch, SourceDescriptor, UnresolvedReference,
};
use super::super::schema::{EntityKind, Namespace, RelationKind, RelationOrigin, ResourceLocator};
use crate::transport::model_gateway::{EvidenceClass, VerificationState};
use crate::{HNSQRError, HNSQRResult};

const TEST_CALL_NAMES: &[&str] = &[
    "test",
    "it",
    "describe",
    "beforeEach",
    "afterEach",
    "beforeAll",
    "afterAll",
];

pub struct TsSourceAdapter;

impl Default for TsSourceAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl TsSourceAdapter {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn init_parser(language: Language) -> HNSQRResult<Parser> {
        let mut parser = Parser::new();
        parser.set_language(&language).map_err(|e| {
            HNSQRError::Internal(format!("tree-sitter-typescript init error: {e:?}"))
        })?;
        Ok(parser)
    }

    fn text_of<'a>(node: &Node, source: &'a str) -> &'a str {
        node.utf8_text(source.as_bytes()).unwrap_or("")
    }

    pub(crate) fn extract_with_language(
        source: &SourceInput,
        language: Language,
        source_type: &str,
    ) -> HNSQRResult<ExtractionBatch> {
        let text = match &source.text_content {
            Some(text) => text,
            None => {
                return Err(HNSQRError::InvalidRequest(
                    "source text_content required".to_string(),
                ));
            }
        };

        let mut parser = Self::init_parser(language)?;
        let tree = parser.parse(text, None).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Failed to parse {}", source.locator))
        })?;

        let root_node = tree.root_node();
        let content_hash = source.compute_fingerprint();

        let desc = SourceDescriptor {
            source_type: source_type.to_string(),
            locator: source.locator.clone(),
            content_hash,
            metadata: source.metadata.clone(),
        };

        let mut batch = ExtractionBatch::new(desc);

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

        let mut visitor = TsAstVisitor {
            source: text,
            source_locator: &source.locator,
            file_temp_id: &file_temp_id,
            content_hash,
            batch: &mut batch,
            scope: Vec::new(),
            current_class: None,
            pending_doc: Vec::new(),
            pending_rationale: Vec::new(),
        };

        visitor.walk(&root_node);

        Ok(batch)
    }
}

impl SourceAdapter for TsSourceAdapter {
    fn name(&self) -> &'static str {
        "typescript_treesitter_adapter"
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
        (source.locator.ends_with(".ts") && !source.locator.ends_with(".d.ts"))
            || source.source_type == "typescript"
    }

    fn extract(
        &self,
        source: &SourceInput,
        _namespace: &Namespace,
    ) -> HNSQRResult<ExtractionBatch> {
        Self::extract_with_language(
            source,
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            "typescript",
        )
    }
}

struct TsAstVisitor<'a, 'b> {
    source: &'a str,
    source_locator: &'a str,
    file_temp_id: &'a str,
    content_hash: [u8; 32],
    batch: &'b mut ExtractionBatch,
    scope: Vec<String>,
    current_class: Option<String>,
    pending_doc: Vec<String>,
    pending_rationale: Vec<(String, String, usize, usize)>,
}

impl<'a, 'b> TsAstVisitor<'a, 'b> {
    fn current_qual(&self, name: &str) -> String {
        if let Some(cls) = &self.current_class {
            if self.scope.is_empty() {
                format!("{cls}.{name}")
            } else {
                format!("{}.{cls}.{name}", self.scope.join("."))
            }
        } else if self.scope.is_empty() {
            name.to_string()
        } else {
            format!("{}.{name}", self.scope.join("."))
        }
    }

    fn walk(&mut self, node: &Node) {
        let kind = node.kind();

        if kind == "comment" {
            self.consume_comment(node);
            return;
        }

        match kind {
            "class_declaration" | "abstract_class_declaration" => self.visit_class(node),
            "interface_declaration" => self.visit_interface(node),
            "type_alias_declaration" => self.visit_type_alias(node),
            "enum_declaration" => self.visit_enum(node),
            "function_declaration" | "generator_function_declaration" => {
                self.visit_function(node, None)
            }
            "lexical_declaration" | "variable_declaration" => self.visit_variable_declaration(node),
            "call_expression" => self.visit_call_site(node),
            _ => self.walk_children(node),
        }
    }

    fn walk_children(&mut self, node: &Node) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }
    }

    fn consume_comment(&mut self, node: &Node) {
        let text = TsSourceAdapter::text_of(node, self.source).trim();
        let start = node.start_position().row + 1;
        let end = node.end_position().row + 1;

        if text.starts_with("/**") && text.len() > 4 {
            let cleaned = text
                .trim_start_matches("/**")
                .trim_end_matches("*/")
                .lines()
                .map(|l| l.trim().trim_start_matches('*').trim())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
                .join("\n");
            if !cleaned.is_empty() {
                self.pending_doc.push(cleaned);
            }
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

    fn define_and_push(
        &mut self,
        temp_id: String,
        kind: EntityKind,
        label: String,
        node: &Node,
        mut attributes: BTreeMap<String, serde_json::Value>,
    ) {
        let doc = self.take_doc();
        if let Some(d) = doc {
            attributes.insert("docstring".to_string(), serde_json::json!(d));
        }

        self.batch.entities.push(ExtractedEntity {
            temp_id: temp_id.clone(),
            kind,
            label,
            locator: Some(ResourceLocator {
                uri: self.source_locator.to_string(),
                start_line: Some(node.start_position().row + 1),
                end_line: Some(node.end_position().row + 1),
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

    fn visit_class(&mut self, node: &Node) {
        let class_name = node
            .child_by_field_name("name")
            .map(|n| TsSourceAdapter::text_of(&n, self.source))
            .unwrap_or("AnonClass");

        let qual = self.current_qual(class_name);
        let temp_id = format!("class_{qual}");
        let mut attributes = BTreeMap::new();
        if node.kind() == "abstract_class_declaration" {
            attributes.insert("is_abstract".to_string(), serde_json::json!(true));
        }

        self.define_and_push(
            temp_id.clone(),
            EntityKind::new("code:class"),
            qual,
            node,
            attributes,
        );

        self.extract_heritage(node, &temp_id);

        let prev_class = self.current_class.replace(class_name.to_string());
        self.walk_children(node);
        self.current_class = prev_class;
    }

    /// Scans for `extends`/`implements` heritage generically by node-kind substring rather
    /// than a fixed field name, because the exact nesting of `class_heritage` /
    /// `extends_clause` / `implements_clause` in tree-sitter-typescript's grammar is a
    /// [HYPOTHESIS] not confirmed against the pinned grammar version. `implements()` is
    /// reused as-is (proven on the reference adapter); `extends` uses the same
    /// `RelationKind::new("extends")` best guess as `code_js.rs` — see that file's note.
    fn extract_heritage(&mut self, node: &Node, owner_temp_id: &str) {
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            let ck = child.kind();
            let is_extends = ck == "class_heritage" || ck.contains("extends");
            let is_implements = ck.contains("implements");
            if !is_extends && !is_implements {
                continue;
            }

            let mut hc = child.walk();
            for hchild in child.children(&mut hc) {
                if matches!(
                    hchild.kind(),
                    "identifier"
                        | "member_expression"
                        | "type_identifier"
                        | "nested_type_identifier"
                        | "generic_type"
                ) {
                    let target_name = TsSourceAdapter::text_of(&hchild, self.source)
                        .trim()
                        .to_string();
                    let (relation_kind, expected_kind, role) = if is_implements {
                        (
                            RelationKind::implements(),
                            "code:interface",
                            "implements_target",
                        )
                    } else {
                        (RelationKind::new("extends"), "code:class", "extends_target")
                    };

                    self.batch.unresolved.push(UnresolvedReference {
                        source_temp_id: owner_temp_id.to_string(),
                        target_ref: target_name,
                        expected_kind: Some(expected_kind.to_string()),
                        relation_kind,
                        role: role.to_string(),
                        locator: Some(ResourceLocator {
                            uri: self.source_locator.to_string(),
                            start_line: Some(hchild.start_position().row + 1),
                            end_line: Some(hchild.end_position().row + 1),
                            start_byte: None,
                            end_byte: None,
                        }),
                    });
                }
            }
        }
    }

    fn visit_interface(&mut self, node: &Node) {
        let name = node
            .child_by_field_name("name")
            .map(|n| TsSourceAdapter::text_of(&n, self.source))
            .unwrap_or("AnonInterface");

        let qual = self.current_qual(name);
        let temp_id = format!("iface_{qual}");

        self.define_and_push(
            temp_id.clone(),
            EntityKind::new("code:interface"),
            qual,
            node,
            BTreeMap::new(),
        );

        // Interface-extends-interface (possibly multiple, comma-separated).
        self.extract_heritage(node, &temp_id);
    }

    fn visit_type_alias(&mut self, node: &Node) {
        let name = node
            .child_by_field_name("name")
            .map(|n| TsSourceAdapter::text_of(&n, self.source))
            .unwrap_or("AnonType");

        let qual = self.current_qual(name);
        let temp_id = format!("type_{qual}");

        self.define_and_push(
            temp_id,
            EntityKind::new("code:type_alias"),
            qual,
            node,
            BTreeMap::new(),
        );
    }

    fn visit_enum(&mut self, node: &Node) {
        let name = node
            .child_by_field_name("name")
            .map(|n| TsSourceAdapter::text_of(&n, self.source))
            .unwrap_or("AnonEnum");

        let qual = self.current_qual(name);
        let temp_id = format!("enum_{qual}");

        self.define_and_push(
            temp_id,
            EntityKind::new("code:enum"),
            qual,
            node,
            BTreeMap::new(),
        );
    }

    fn visit_function(&mut self, node: &Node, name_override: Option<&str>) {
        let fn_name = name_override.map(str::to_string).unwrap_or_else(|| {
            node.child_by_field_name("name")
                .map(|n| TsSourceAdapter::text_of(&n, self.source).to_string())
                .unwrap_or_else(|| "anon_fn".to_string())
        });

        let qual = self.current_qual(&fn_name);
        let temp_id = format!("fn_{qual}");

        self.define_and_push(
            temp_id.clone(),
            EntityKind::code_function(),
            qual,
            node,
            BTreeMap::new(),
        );

        if let Some(body) = node.child_by_field_name("body") {
            let mut body_visitor = TsBodyVisitor {
                source: self.source,
                source_locator: self.source_locator,
                fn_temp_id: &temp_id,
                batch: self.batch,
            };
            body_visitor.walk(&body);
        }
    }

    fn visit_variable_declaration(&mut self, node: &Node) {
        let mut cursor = node.walk();
        for declarator in node.children(&mut cursor) {
            if declarator.kind() != "variable_declarator" {
                continue;
            }
            let name_node = declarator.child_by_field_name("name");
            let value_node = declarator.child_by_field_name("value");
            if let (Some(name_n), Some(value_n)) = (name_node, value_node) {
                if matches!(
                    value_n.kind(),
                    "arrow_function" | "function" | "function_expression" | "generator_function"
                ) {
                    let name = TsSourceAdapter::text_of(&name_n, self.source).to_string();
                    self.visit_function(&value_n, Some(&name));
                }
            }
        }
    }

    fn visit_call_site(&mut self, node: &Node) {
        let Some(fn_node) = node.child_by_field_name("function") else {
            self.walk_children(node);
            return;
        };
        let callee_text = TsSourceAdapter::text_of(&fn_node, self.source);
        let base_ident = callee_text.split('.').next().unwrap_or(callee_text);

        if TEST_CALL_NAMES.contains(&base_ident) {
            if let Some(args) = node.child_by_field_name("arguments") {
                let mut cursor = args.walk();
                let children: Vec<Node> = args.children(&mut cursor).collect();
                let label_text = children.iter().find(|c| c.kind() == "string").map(|c| {
                    TsSourceAdapter::text_of(c, self.source)
                        .trim_matches(|ch| ch == '\'' || ch == '"' || ch == '`')
                        .to_string()
                });
                let callback = children.iter().find(|c| {
                    matches!(
                        c.kind(),
                        "arrow_function" | "function" | "function_expression"
                    )
                });

                if let (Some(label), Some(cb)) = (label_text, callback) {
                    let start = node.start_position().row + 1;
                    let qual = self.current_qual(&label);
                    let temp_id = format!("test_{start}_{qual}");

                    let mut attributes = BTreeMap::new();
                    attributes.insert("is_test".to_string(), serde_json::json!(true));
                    attributes.insert("test_kind".to_string(), serde_json::json!(base_ident));

                    self.define_and_push(
                        temp_id.clone(),
                        EntityKind::test_case(),
                        label,
                        node,
                        attributes,
                    );

                    if let Some(cb_body) = cb.child_by_field_name("body") {
                        let mut body_visitor = TsBodyVisitor {
                            source: self.source,
                            source_locator: self.source_locator,
                            fn_temp_id: &temp_id,
                            batch: self.batch,
                        };
                        body_visitor.walk(&cb_body);
                    }
                    return;
                }
            }
        }

        self.walk_children(node);
    }
}

struct TsBodyVisitor<'a, 'b> {
    source: &'a str,
    source_locator: &'a str,
    fn_temp_id: &'a str,
    batch: &'b mut ExtractionBatch,
}

impl<'a, 'b> TsBodyVisitor<'a, 'b> {
    fn walk(&mut self, node: &Node) {
        if node.kind() == "call_expression" {
            if let Some(fn_node) = node.child_by_field_name("function") {
                let call_text = TsSourceAdapter::text_of(&fn_node, self.source).trim();
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
