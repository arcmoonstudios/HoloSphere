/* holosphere/src/codegraph/languages/rust.rs */
//!▫~•◦-------------------------------‣
//! # Deterministic Tree-Sitter Rust AST Extractor & Rationale Parser
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Performs deterministic syntactic AST analysis on Rust source code, extracting
//! items, traits, implementations, call sites, type references, tests, and
//! architectural rationale comments without non-deterministic LLMs.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::path::Path;

use tree_sitter::{Node, Parser, Point};

use super::super::parser::{
    ExtractionContext, ExtractionResult, ImportItem, LanguageExtractor, UnresolvedCall,
    UnresolvedTypeRef,
};
use super::super::schema::{
    CodeEdge, CodeEdgeId, CodeNode, CodeNodeId, CodeNodeKind, CodeRelation, Language,
    RelationOrigin, SourceSpan,
};
use crate::transport::model_gateway::{EvidenceClass, VerificationState};
use crate::{HNSQRError, HNSQRResult};

pub struct RustExtractor;

impl RustExtractor {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    fn init_parser() -> HNSQRResult<Parser> {
        let mut parser = Parser::new();
        parser
            .set_language(&tree_sitter_rust::LANGUAGE.into())
            .map_err(|e| {
                HNSQRError::Internal(format!("Failed to initialize tree-sitter-rust: {e:?}"))
            })?;
        Ok(parser)
    }

    fn span_of(node: &Node) -> SourceSpan {
        let start = node.start_position();
        let end = node.end_position();
        SourceSpan::new(
            start.row + 1,
            start.column + 1,
            end.row + 1,
            end.column + 1,
            node.start_byte(),
            node.end_byte(),
        )
    }

    fn text_of<'a>(node: &Node, source: &'a str) -> &'a str {
        node.utf8_text(source.as_bytes()).unwrap_or("")
    }

    fn normalize_signature(sig: &str) -> String {
        sig.split_whitespace().collect::<Vec<_>>().join(" ")
    }
}

impl Default for RustExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl LanguageExtractor for RustExtractor {
    fn language(&self) -> Language {
        Language::Rust
    }

    fn extract(&self, ctx: &ExtractionContext) -> HNSQRResult<ExtractionResult> {
        let mut parser = Self::init_parser()?;
        let tree = parser.parse(ctx.source_code, None).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!("Failed to parse {}", ctx.relative_path.display()))
        })?;

        let root_node = tree.root_node();
        let mut result = ExtractionResult::default();

        let rel_path_str = ctx.relative_path.to_string_lossy().replace('\\', "/");
        let file_stem = ctx
            .relative_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        // 1. Create File Node
        let file_node_id = CodeNodeId::compute(
            ctx.workspace_id,
            &rel_path_str,
            CodeNodeKind::File,
            &rel_path_str,
            "",
        );
        let file_span = Self::span_of(&root_node);
        let file_node = CodeNode {
            id: file_node_id.clone(),
            kind: CodeNodeKind::File,
            name: file_stem.to_string(),
            qualified_name: rel_path_str.clone(),
            signature: None,
            language: Language::Rust,
            source_file: ctx.relative_path.clone(),
            source_span: file_span,
            symbol_hash: ctx.content_hash,
            file_hash: ctx.content_hash,
            docstring: None,
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        };
        result.nodes.push(file_node);

        // 2. Walk AST Items
        let mut visitor = RustVisitor {
            ctx,
            source: ctx.source_code,
            file_node_id: &file_node_id,
            current_scope: vec![rel_path_str.clone()],
            current_impl_target: None,
            current_impl_trait: None,
            result: &mut result,
            pending_doc: Vec::new(),
            pending_rationale: Vec::new(),
        };

        visitor.walk(&root_node);

        Ok(result)
    }
}

struct RustVisitor<'a, 'b> {
    ctx: &'a ExtractionContext<'a>,
    source: &'a str,
    file_node_id: &'a CodeNodeId,
    current_scope: Vec<String>,
    current_impl_target: Option<String>,
    current_impl_trait: Option<String>,
    result: &'b mut ExtractionResult,
    pending_doc: Vec<(String, SourceSpan)>,
    pending_rationale: Vec<(String, String, SourceSpan)>, // (kind, text, span)
}

impl<'a, 'b> RustVisitor<'a, 'b> {
    fn current_module_path(&self) -> String {
        self.current_scope.join("::")
    }

    fn walk(&mut self, node: &Node) {
        let kind = node.kind();

        // Check comments for rationale and docstrings
        if kind == "line_comment" || kind == "block_comment" {
            let text = RustExtractor::text_of(node, self.source).trim();
            let span = RustExtractor::span_of(node);

            if text.starts_with("///") || text.starts_with("//!") {
                let cleaned = text
                    .trim_start_matches("///")
                    .trim_start_matches("//!")
                    .trim()
                    .to_string();
                self.pending_doc.push((cleaned, span));
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
                    "AUDIT:",
                ] {
                    if upper.starts_with(marker) {
                        let tag = marker.trim_end_matches(':').to_lowercase();
                        self.pending_rationale
                            .push((tag, trimmed.to_string(), span));
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
            "type_item" => self.visit_type_alias(node),
            "function_item" => self.visit_function(node),
            "impl_item" => self.visit_impl(node),
            "macro_definition" => self.visit_macro(node),
            "const_item" | "static_item" => self.visit_const(node),
            "use_declaration" => self.visit_use(node),
            "call_expression" => self.visit_call(node),
            "macro_invocation" => self.visit_macro_invocation(node),
            _ => {
                let mut cursor = node.walk();
                for child in node.children(&mut cursor) {
                    self.walk(&child);
                }
            }
        }
    }

    fn take_docstring(&mut self) -> Option<String> {
        if self.pending_doc.is_empty() {
            None
        } else {
            let docs: Vec<String> = self.pending_doc.drain(..).map(|(d, _)| d).collect();
            Some(docs.join("\n"))
        }
    }

    fn attach_pending_rationale_to(&mut self, target_node_id: &CodeNodeId) {
        let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
        let mod_path = self.current_module_path();
        let items: Vec<_> = self.pending_rationale.drain(..).collect();
        for (tag, text, span) in items {
            let rationale_id = CodeNodeId::compute(
                self.ctx.workspace_id,
                &rel_path_str,
                CodeNodeKind::Rationale,
                &format!("rationale_{}_{}", span.start_line, span.start_col),
                &text,
            );

            let mut attributes = BTreeMap::new();
            attributes.insert("rationale_tag".to_string(), serde_json::json!(tag));
            attributes.insert("content".to_string(), serde_json::json!(text));

            let rationale_node = CodeNode {
                id: rationale_id.clone(),
                kind: CodeNodeKind::Rationale,
                name: format!("{tag}: {}", text.chars().take(40).collect::<String>()),
                qualified_name: format!("{mod_path}::{tag}_{}", span.start_line),
                signature: None,
                language: Language::Rust,
                source_file: self.ctx.relative_path.clone(),
                source_span: span,
                symbol_hash: [0u8; 32],
                file_hash: self.ctx.content_hash,
                docstring: None,
                attributes,
                evidence_class: EvidenceClass::Observation,
                verification_state: VerificationState::Verified,
            };

            let relation = if tag == "safety" {
                CodeRelation::Justifies
            } else {
                CodeRelation::Explains
            };

            let edge_id = CodeEdgeId::compute(
                &rationale_id,
                target_node_id,
                relation,
                RelationOrigin::Extracted,
                &span,
            );

            let edge = CodeEdge {
                id: edge_id,
                source: rationale_id,
                target: target_node_id.clone(),
                relation,
                origin: RelationOrigin::Extracted,
                confidence: 1.0,
                evidence: span,
                attributes: BTreeMap::new(),
            };

            self.result.nodes.push(rationale_node);
            self.result.edges.push(edge);
        }
    }

    fn visit_mod(&mut self, node: &Node) {
        let name_node = node.child_by_field_name("name");
        let mod_name = name_node
            .map(|n| RustExtractor::text_of(&n, self.source))
            .unwrap_or("anon_mod");

        let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
        let qual_name = format!("{}::{mod_name}", self.current_module_path());
        let span = RustExtractor::span_of(node);

        let mod_node_id = CodeNodeId::compute(
            self.ctx.workspace_id,
            &rel_path_str,
            CodeNodeKind::Module,
            &qual_name,
            "",
        );

        let doc = self.take_docstring();
        let mut attributes = BTreeMap::new();
        let is_test_mod = mod_name == "tests" || mod_name.ends_with("_tests");
        if is_test_mod {
            attributes.insert("is_test".to_string(), serde_json::json!(true));
        }

        let code_node = CodeNode {
            id: mod_node_id.clone(),
            kind: if is_test_mod {
                CodeNodeKind::Test
            } else {
                CodeNodeKind::Module
            },
            name: mod_name.to_string(),
            qualified_name: qual_name.clone(),
            signature: None,
            language: Language::Rust,
            source_file: self.ctx.relative_path.clone(),
            source_span: span,
            symbol_hash: [0u8; 32],
            file_hash: self.ctx.content_hash,
            docstring: doc,
            attributes,
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        };

        // File defines Module
        let def_edge_id = CodeEdgeId::compute(
            self.file_node_id,
            &mod_node_id,
            CodeRelation::Defines,
            RelationOrigin::Extracted,
            &span,
        );
        let def_edge = CodeEdge {
            id: def_edge_id,
            source: self.file_node_id.clone(),
            target: mod_node_id.clone(),
            relation: CodeRelation::Defines,
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            evidence: span,
            attributes: BTreeMap::new(),
        };

        self.attach_pending_rationale_to(&mod_node_id);
        self.result.nodes.push(code_node);
        self.result.edges.push(def_edge);

        self.current_scope.push(mod_name.to_string());
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }
        self.current_scope.pop();
    }

    fn visit_struct(&mut self, node: &Node) {
        let name_node = node.child_by_field_name("name");
        let struct_name = name_node
            .map(|n| RustExtractor::text_of(&n, self.source))
            .unwrap_or("AnonStruct");

        let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
        let qual_name = format!("{}::{struct_name}", self.current_module_path());
        let span = RustExtractor::span_of(node);

        let struct_node_id = CodeNodeId::compute(
            self.ctx.workspace_id,
            &rel_path_str,
            CodeNodeKind::Struct,
            &qual_name,
            "",
        );

        let doc = self.take_docstring();
        let code_node = CodeNode {
            id: struct_node_id.clone(),
            kind: CodeNodeKind::Struct,
            name: struct_name.to_string(),
            qualified_name: qual_name.clone(),
            signature: Some(format!("struct {struct_name}")),
            language: Language::Rust,
            source_file: self.ctx.relative_path.clone(),
            source_span: span,
            symbol_hash: [0u8; 32],
            file_hash: self.ctx.content_hash,
            docstring: doc,
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        };

        let def_edge_id = CodeEdgeId::compute(
            self.file_node_id,
            &struct_node_id,
            CodeRelation::Defines,
            RelationOrigin::Extracted,
            &span,
        );
        let def_edge = CodeEdge {
            id: def_edge_id,
            source: self.file_node_id.clone(),
            target: struct_node_id.clone(),
            relation: CodeRelation::Defines,
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            evidence: span,
            attributes: BTreeMap::new(),
        };

        self.attach_pending_rationale_to(&struct_node_id);
        self.result.nodes.push(code_node);
        self.result.edges.push(def_edge);

        // Walk field declarations
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "field_declaration" {
                if let Some(field_name_node) = child.child_by_field_name("name") {
                    let field_name = RustExtractor::text_of(&field_name_node, self.source);
                    let field_span = RustExtractor::span_of(&child);
                    let field_qual = format!("{qual_name}::{field_name}");
                    let field_id = CodeNodeId::compute(
                        self.ctx.workspace_id,
                        &rel_path_str,
                        CodeNodeKind::Field,
                        &field_qual,
                        "",
                    );

                    let field_node = CodeNode {
                        id: field_id.clone(),
                        kind: CodeNodeKind::Field,
                        name: field_name.to_string(),
                        qualified_name: field_qual,
                        signature: None,
                        language: Language::Rust,
                        source_file: self.ctx.relative_path.clone(),
                        source_span: field_span,
                        symbol_hash: [0u8; 32],
                        file_hash: self.ctx.content_hash,
                        docstring: None,
                        attributes: BTreeMap::new(),
                        evidence_class: EvidenceClass::Observation,
                        verification_state: VerificationState::Verified,
                    };

                    let contains_edge_id = CodeEdgeId::compute(
                        &struct_node_id,
                        &field_id,
                        CodeRelation::Contains,
                        RelationOrigin::Extracted,
                        &field_span,
                    );
                    let contains_edge = CodeEdge {
                        id: contains_edge_id,
                        source: struct_node_id.clone(),
                        target: field_id,
                        relation: CodeRelation::Contains,
                        origin: RelationOrigin::Extracted,
                        confidence: 1.0,
                        evidence: field_span,
                        attributes: BTreeMap::new(),
                    };

                    self.result.nodes.push(field_node);
                    self.result.edges.push(contains_edge);

                    // Track field type reference
                    if let Some(type_node) = child.child_by_field_name("type") {
                        let type_str = RustExtractor::text_of(&type_node, self.source).trim();
                        self.result.unresolved_types.push(UnresolvedTypeRef {
                            source_id: struct_node_id.clone(),
                            target_type: type_str.to_string(),
                            relation: CodeRelation::Uses,
                            span: RustExtractor::span_of(&type_node),
                        });
                    }
                }
            }
        }
    }

    fn visit_enum(&mut self, node: &Node) {
        let name_node = node.child_by_field_name("name");
        let enum_name = name_node
            .map(|n| RustExtractor::text_of(&n, self.source))
            .unwrap_or("AnonEnum");

        let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
        let qual_name = format!("{}::{enum_name}", self.current_module_path());
        let span = RustExtractor::span_of(node);

        let enum_node_id = CodeNodeId::compute(
            self.ctx.workspace_id,
            &rel_path_str,
            CodeNodeKind::Enum,
            &qual_name,
            "",
        );

        let doc = self.take_docstring();
        let code_node = CodeNode {
            id: enum_node_id.clone(),
            kind: CodeNodeKind::Enum,
            name: enum_name.to_string(),
            qualified_name: qual_name.clone(),
            signature: Some(format!("enum {enum_name}")),
            language: Language::Rust,
            source_file: self.ctx.relative_path.clone(),
            source_span: span,
            symbol_hash: [0u8; 32],
            file_hash: self.ctx.content_hash,
            docstring: doc,
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        };

        let def_edge_id = CodeEdgeId::compute(
            self.file_node_id,
            &enum_node_id,
            CodeRelation::Defines,
            RelationOrigin::Extracted,
            &span,
        );
        let def_edge = CodeEdge {
            id: def_edge_id,
            source: self.file_node_id.clone(),
            target: enum_node_id.clone(),
            relation: CodeRelation::Defines,
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            evidence: span,
            attributes: BTreeMap::new(),
        };

        self.attach_pending_rationale_to(&enum_node_id);
        self.result.nodes.push(code_node);
        self.result.edges.push(def_edge);

        // Walk variants
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "enum_variant" {
                if let Some(var_name_node) = child.child_by_field_name("name") {
                    let var_name = RustExtractor::text_of(&var_name_node, self.source);
                    let var_span = RustExtractor::span_of(&child);
                    let var_qual = format!("{qual_name}::{var_name}");
                    let var_id = CodeNodeId::compute(
                        self.ctx.workspace_id,
                        &rel_path_str,
                        CodeNodeKind::Variant,
                        &var_qual,
                        "",
                    );

                    let var_node = CodeNode {
                        id: var_id.clone(),
                        kind: CodeNodeKind::Variant,
                        name: var_name.to_string(),
                        qualified_name: var_qual,
                        signature: None,
                        language: Language::Rust,
                        source_file: self.ctx.relative_path.clone(),
                        source_span: var_span,
                        symbol_hash: [0u8; 32],
                        file_hash: self.ctx.content_hash,
                        docstring: None,
                        attributes: BTreeMap::new(),
                        evidence_class: EvidenceClass::Observation,
                        verification_state: VerificationState::Verified,
                    };

                    let contains_edge_id = CodeEdgeId::compute(
                        &enum_node_id,
                        &var_id,
                        CodeRelation::Contains,
                        RelationOrigin::Extracted,
                        &var_span,
                    );
                    let contains_edge = CodeEdge {
                        id: contains_edge_id,
                        source: enum_node_id.clone(),
                        target: var_id,
                        relation: CodeRelation::Contains,
                        origin: RelationOrigin::Extracted,
                        confidence: 1.0,
                        evidence: var_span,
                        attributes: BTreeMap::new(),
                    };

                    self.result.nodes.push(var_node);
                    self.result.edges.push(contains_edge);
                }
            }
        }
    }

    fn visit_trait(&mut self, node: &Node) {
        let name_node = node.child_by_field_name("name");
        let trait_name = name_node
            .map(|n| RustExtractor::text_of(&n, self.source))
            .unwrap_or("AnonTrait");

        let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
        let qual_name = format!("{}::{trait_name}", self.current_module_path());
        let span = RustExtractor::span_of(node);

        let trait_node_id = CodeNodeId::compute(
            self.ctx.workspace_id,
            &rel_path_str,
            CodeNodeKind::Trait,
            &qual_name,
            "",
        );

        let doc = self.take_docstring();
        let code_node = CodeNode {
            id: trait_node_id.clone(),
            kind: CodeNodeKind::Trait,
            name: trait_name.to_string(),
            qualified_name: qual_name.clone(),
            signature: Some(format!("trait {trait_name}")),
            language: Language::Rust,
            source_file: self.ctx.relative_path.clone(),
            source_span: span,
            symbol_hash: [0u8; 32],
            file_hash: self.ctx.content_hash,
            docstring: doc,
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        };

        let def_edge_id = CodeEdgeId::compute(
            self.file_node_id,
            &trait_node_id,
            CodeRelation::Defines,
            RelationOrigin::Extracted,
            &span,
        );
        let def_edge = CodeEdge {
            id: def_edge_id,
            source: self.file_node_id.clone(),
            target: trait_node_id.clone(),
            relation: CodeRelation::Defines,
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            evidence: span,
            attributes: BTreeMap::new(),
        };

        self.attach_pending_rationale_to(&trait_node_id);
        self.result.nodes.push(code_node);
        self.result.edges.push(def_edge);

        self.current_scope.push(trait_name.to_string());
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }
        self.current_scope.pop();
    }

    fn visit_type_alias(&mut self, node: &Node) {
        let name_node = node.child_by_field_name("name");
        let alias_name = name_node
            .map(|n| RustExtractor::text_of(&n, self.source))
            .unwrap_or("AnonType");

        let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
        let qual_name = format!("{}::{alias_name}", self.current_module_path());
        let span = RustExtractor::span_of(node);

        let type_node_id = CodeNodeId::compute(
            self.ctx.workspace_id,
            &rel_path_str,
            CodeNodeKind::TypeAlias,
            &qual_name,
            "",
        );

        let doc = self.take_docstring();
        let code_node = CodeNode {
            id: type_node_id.clone(),
            kind: CodeNodeKind::TypeAlias,
            name: alias_name.to_string(),
            qualified_name: qual_name,
            signature: Some(
                RustExtractor::text_of(node, self.source)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string(),
            ),
            language: Language::Rust,
            source_file: self.ctx.relative_path.clone(),
            source_span: span,
            symbol_hash: [0u8; 32],
            file_hash: self.ctx.content_hash,
            docstring: doc,
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        };

        let def_edge_id = CodeEdgeId::compute(
            self.file_node_id,
            &type_node_id,
            CodeRelation::Defines,
            RelationOrigin::Extracted,
            &span,
        );
        let def_edge = CodeEdge {
            id: def_edge_id,
            source: self.file_node_id.clone(),
            target: type_node_id.clone(),
            relation: CodeRelation::Defines,
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            evidence: span,
            attributes: BTreeMap::new(),
        };

        self.attach_pending_rationale_to(&type_node_id);
        self.result.nodes.push(code_node);
        self.result.edges.push(def_edge);
    }

    fn visit_impl(&mut self, node: &Node) {
        let type_node = node.child_by_field_name("type");
        let trait_node = node.child_by_field_name("trait");

        let target_type = type_node
            .map(|n| RustExtractor::text_of(&n, self.source).trim().to_string())
            .unwrap_or_else(|| "UnknownType".to_string());

        let trait_name =
            trait_node.map(|n| RustExtractor::text_of(&n, self.source).trim().to_string());

        let prev_impl_target = self.current_impl_target.replace(target_type.clone());
        let prev_impl_trait = self
            .current_impl_trait
            .replace(trait_name.clone().unwrap_or_default());

        if let Some(trait_ref) = &trait_name {
            // Record UnresolvedTypeRef for Implements relation
            let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
            let qual_name = format!("{}::{target_type}", self.current_module_path());
            let struct_node_id = CodeNodeId::compute(
                self.ctx.workspace_id,
                &rel_path_str,
                CodeNodeKind::Struct,
                &qual_name,
                "",
            );

            self.result.unresolved_types.push(UnresolvedTypeRef {
                source_id: struct_node_id,
                target_type: trait_ref.clone(),
                relation: CodeRelation::Implements,
                span: RustExtractor::span_of(node),
            });
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }

        self.current_impl_target = prev_impl_target;
        self.current_impl_trait = prev_impl_trait;
    }

    fn visit_function(&mut self, node: &Node) {
        let name_node = node.child_by_field_name("name");
        let fn_name = name_node
            .map(|n| RustExtractor::text_of(&n, self.source))
            .unwrap_or("anon_fn");

        let is_method = self.current_impl_target.is_some();
        let kind = if is_method {
            CodeNodeKind::Method
        } else {
            CodeNodeKind::Function
        };

        let qual_name = if let Some(impl_type) = &self.current_impl_target {
            format!("{}::{impl_type}::{fn_name}", self.current_module_path())
        } else {
            format!("{}::{fn_name}", self.current_module_path())
        };

        let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
        let span = RustExtractor::span_of(node);

        let raw_sig = RustExtractor::text_of(node, self.source)
            .lines()
            .take_while(|l| !l.contains('{'))
            .collect::<Vec<_>>()
            .join(" ");
        let normalized_sig = RustExtractor::normalize_signature(&raw_sig);

        let fn_node_id = CodeNodeId::compute(
            self.ctx.workspace_id,
            &rel_path_str,
            kind,
            &qual_name,
            &normalized_sig,
        );

        let doc = self.take_docstring();
        let mut attributes = BTreeMap::new();
        let is_test = fn_name.starts_with("test_")
            || self.source[..node.start_byte()].ends_with("#[test]")
            || self.source[..node.start_byte()].ends_with("#[tokio::test]")
            || self.current_scope.iter().any(|s| s == "tests");

        if is_test {
            attributes.insert("is_test".to_string(), serde_json::json!(true));
        }

        let code_node = CodeNode {
            id: fn_node_id.clone(),
            kind: if is_test { CodeNodeKind::Test } else { kind },
            name: fn_name.to_string(),
            qualified_name: qual_name,
            signature: Some(normalized_sig),
            language: Language::Rust,
            source_file: self.ctx.relative_path.clone(),
            source_span: span,
            symbol_hash: [0u8; 32],
            file_hash: self.ctx.content_hash,
            docstring: doc,
            attributes,
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        };

        let def_edge_id = CodeEdgeId::compute(
            self.file_node_id,
            &fn_node_id,
            CodeRelation::Defines,
            RelationOrigin::Extracted,
            &span,
        );
        let def_edge = CodeEdge {
            id: def_edge_id,
            source: self.file_node_id.clone(),
            target: fn_node_id.clone(),
            relation: CodeRelation::Defines,
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            evidence: span,
            attributes: BTreeMap::new(),
        };

        self.attach_pending_rationale_to(&fn_node_id);
        self.result.nodes.push(code_node);
        self.result.edges.push(def_edge);

        // Check parameter types & return types
        if let Some(params_node) = node.child_by_field_name("parameters") {
            let mut cursor = params_node.walk();
            for param in params_node.children(&mut cursor) {
                if let Some(type_node) = param.child_by_field_name("type") {
                    let type_str = RustExtractor::text_of(&type_node, self.source).trim();
                    self.result.unresolved_types.push(UnresolvedTypeRef {
                        source_id: fn_node_id.clone(),
                        target_type: type_str.to_string(),
                        relation: CodeRelation::Accepts,
                        span: RustExtractor::span_of(&type_node),
                    });
                }
            }
        }
        if let Some(ret_node) = node.child_by_field_name("return_type") {
            let ret_str = RustExtractor::text_of(&ret_node, self.source)
                .trim_start_matches("->")
                .trim();
            self.result.unresolved_types.push(UnresolvedTypeRef {
                source_id: fn_node_id.clone(),
                target_type: ret_str.to_string(),
                relation: CodeRelation::Returns,
                span: RustExtractor::span_of(&ret_node),
            });
        }

        // Walk body for calls
        if let Some(body) = node.child_by_field_name("body") {
            let mut body_visitor = FunctionBodyVisitor {
                source: self.source,
                fn_node_id: &fn_node_id,
                result: self.result,
            };
            body_visitor.walk(&body);
        }
    }

    fn visit_macro(&mut self, node: &Node) {
        let name_node = node.child_by_field_name("name");
        let macro_name = name_node
            .map(|n| RustExtractor::text_of(&n, self.source))
            .unwrap_or("anon_macro");

        let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
        let qual_name = format!("{}::{macro_name}", self.current_module_path());
        let span = RustExtractor::span_of(node);

        let macro_node_id = CodeNodeId::compute(
            self.ctx.workspace_id,
            &rel_path_str,
            CodeNodeKind::Macro,
            &qual_name,
            "",
        );

        let doc = self.take_docstring();
        let code_node = CodeNode {
            id: macro_node_id.clone(),
            kind: CodeNodeKind::Macro,
            name: macro_name.to_string(),
            qualified_name: qual_name,
            signature: Some(format!("macro_rules! {macro_name}")),
            language: Language::Rust,
            source_file: self.ctx.relative_path.clone(),
            source_span: span,
            symbol_hash: [0u8; 32],
            file_hash: self.ctx.content_hash,
            docstring: doc,
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        };

        let def_edge_id = CodeEdgeId::compute(
            self.file_node_id,
            &macro_node_id,
            CodeRelation::Defines,
            RelationOrigin::Extracted,
            &span,
        );
        let def_edge = CodeEdge {
            id: def_edge_id,
            source: self.file_node_id.clone(),
            target: macro_node_id.clone(),
            relation: CodeRelation::Defines,
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            evidence: span,
            attributes: BTreeMap::new(),
        };

        self.attach_pending_rationale_to(&macro_node_id);
        self.result.nodes.push(code_node);
        self.result.edges.push(def_edge);
    }

    fn visit_const(&mut self, node: &Node) {
        let name_node = node.child_by_field_name("name");
        let const_name = name_node
            .map(|n| RustExtractor::text_of(&n, self.source))
            .unwrap_or("ANON_CONST");

        let rel_path_str = self.ctx.relative_path.to_string_lossy().replace('\\', "/");
        let qual_name = format!("{}::{const_name}", self.current_module_path());
        let span = RustExtractor::span_of(node);

        let const_node_id = CodeNodeId::compute(
            self.ctx.workspace_id,
            &rel_path_str,
            CodeNodeKind::Constant,
            &qual_name,
            "",
        );

        let doc = self.take_docstring();
        let code_node = CodeNode {
            id: const_node_id.clone(),
            kind: CodeNodeKind::Constant,
            name: const_name.to_string(),
            qualified_name: qual_name,
            signature: Some(
                RustExtractor::text_of(node, self.source)
                    .lines()
                    .next()
                    .unwrap_or("")
                    .to_string(),
            ),
            language: Language::Rust,
            source_file: self.ctx.relative_path.clone(),
            source_span: span,
            symbol_hash: [0u8; 32],
            file_hash: self.ctx.content_hash,
            docstring: doc,
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        };

        let def_edge_id = CodeEdgeId::compute(
            self.file_node_id,
            &const_node_id,
            CodeRelation::Defines,
            RelationOrigin::Extracted,
            &span,
        );
        let def_edge = CodeEdge {
            id: def_edge_id,
            source: self.file_node_id.clone(),
            target: const_node_id.clone(),
            relation: CodeRelation::Defines,
            origin: RelationOrigin::Extracted,
            confidence: 1.0,
            evidence: span,
            attributes: BTreeMap::new(),
        };

        self.attach_pending_rationale_to(&const_node_id);
        self.result.nodes.push(code_node);
        self.result.edges.push(def_edge);
    }

    fn visit_use(&mut self, node: &Node) {
        let text = RustExtractor::text_of(node, self.source).trim();
        let span = RustExtractor::span_of(node);
        let cleaned = text
            .trim_start_matches("pub ")
            .trim_start_matches("use ")
            .trim_end_matches(';')
            .trim();

        // Extract individual symbols from use statements (e.g. use foo::bar::{A, B as C})
        self.extract_use_tree(self.file_node_id, "", cleaned, span);
    }

    fn extract_use_tree(
        &mut self,
        module_id: &CodeNodeId,
        prefix: &str,
        use_str: &str,
        span: SourceSpan,
    ) {
        if let Some((base, items)) = use_str.split_once("::{") {
            let items_str = items.trim_end_matches('}');
            let full_base = if prefix.is_empty() {
                base.to_string()
            } else {
                format!("{prefix}::{base}")
            };
            for item in items_str.split(',') {
                let trimmed = item.trim();
                if !trimmed.is_empty() {
                    self.extract_use_tree(module_id, &full_base, trimmed, span);
                }
            }
        } else {
            let full_path = if prefix.is_empty() {
                use_str.to_string()
            } else {
                format!("{prefix}::{use_str}")
            };
            let (import_path, imported_symbol, alias) =
                if let Some((path, alias_part)) = full_path.split_once(" as ") {
                    let sym = path.rsplit("::").next().unwrap_or(path);
                    (
                        path.to_string(),
                        sym.to_string(),
                        Some(alias_part.trim().to_string()),
                    )
                } else {
                    let sym = full_path.rsplit("::").next().unwrap_or(&full_path);
                    (full_path.clone(), sym.to_string(), None)
                };

            let is_glob = imported_symbol == "*";
            self.result.imports.push(ImportItem {
                module_node_id: module_id.clone(),
                import_path,
                imported_symbol,
                alias,
                is_glob,
                span,
            });
        }
    }

    fn visit_call(&mut self, _node: &Node) {
        // Handled in FunctionBodyVisitor
    }

    fn visit_macro_invocation(&mut self, _node: &Node) {
        // Handled in FunctionBodyVisitor
    }
}

struct FunctionBodyVisitor<'a, 'b> {
    source: &'a str,
    fn_node_id: &'a CodeNodeId,
    result: &'b mut ExtractionResult,
}

impl<'a, 'b> FunctionBodyVisitor<'a, 'b> {
    fn walk(&mut self, node: &Node) {
        let kind = node.kind();
        if kind == "call_expression" {
            if let Some(function_node) = node.child_by_field_name("function") {
                let span = RustExtractor::span_of(node);
                if function_node.kind() == "field_expression" {
                    // Method call: receiver.method(...)
                    if let (Some(argument), Some(field)) = (
                        function_node.child_by_field_name("argument"),
                        function_node.child_by_field_name("field"),
                    ) {
                        let receiver = RustExtractor::text_of(&argument, self.source)
                            .trim()
                            .to_string();
                        let target = RustExtractor::text_of(&field, self.source)
                            .trim()
                            .to_string();
                        self.result.unresolved_calls.push(UnresolvedCall {
                            caller_id: self.fn_node_id.clone(),
                            target_symbol: target,
                            receiver: Some(receiver),
                            span,
                        });
                    }
                } else {
                    // Direct call: func(...) or Type::func(...)
                    let call_target = RustExtractor::text_of(&function_node, self.source)
                        .trim()
                        .to_string();
                    let (receiver, target) =
                        if let Some((recv, tgt)) = call_target.rsplit_once("::") {
                            (Some(recv.to_string()), tgt.to_string())
                        } else {
                            (None, call_target)
                        };
                    self.result.unresolved_calls.push(UnresolvedCall {
                        caller_id: self.fn_node_id.clone(),
                        target_symbol: target,
                        receiver,
                        span,
                    });
                }
            }
        } else if kind == "macro_invocation" {
            if let Some(macro_node) = node.child_by_field_name("macro") {
                let macro_name = RustExtractor::text_of(&macro_node, self.source)
                    .trim()
                    .to_string();
                let span = RustExtractor::span_of(node);
                self.result.unresolved_calls.push(UnresolvedCall {
                    caller_id: self.fn_node_id.clone(),
                    target_symbol: macro_name,
                    receiver: None,
                    span,
                });
            }
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }
    }
}
