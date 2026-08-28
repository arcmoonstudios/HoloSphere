/* holosphere/src/codegraph/languages/generic.rs */
//!▫~•◦-------------------------------‣
//! # Shared Tree-Sitter and Structured-Document CodeGraph Extractors
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides deterministic front ends for the languages that share CodeGraph's common
//! file/symbol model. Language-specific semantic lowering can be layered on top without
//! making the registry silently reject a scanned source file.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;

use tree_sitter::{Language as TreeSitterLanguage, Node, Parser};

use super::super::parser::{
    ExtractionContext, ExtractionResult, LanguageExtractor, UnresolvedCall,
};
use super::super::schema::{
    CodeEdge, CodeEdgeId, CodeNode, CodeNodeId, CodeNodeKind, CodeRelation, Language,
    RelationOrigin, SourceSpan,
};
use crate::transport::model_gateway::{EvidenceClass, VerificationState};
use crate::{HNSQRError, HNSQRResult};

/// Tree-sitter front end for JavaScript-family and Python source files.
pub struct TreeSitterExtractor {
    language: Language,
    grammar: TreeSitterLanguage,
}

impl TreeSitterExtractor {
    #[must_use]
    pub fn typescript() -> Self {
        Self::new(
            Language::TypeScript,
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        )
    }

    #[must_use]
    pub fn tsx() -> Self {
        Self::new(Language::Tsx, tree_sitter_typescript::LANGUAGE_TSX.into())
    }

    #[must_use]
    pub fn javascript() -> Self {
        Self::new(
            Language::JavaScript,
            tree_sitter_javascript::LANGUAGE.into(),
        )
    }

    #[must_use]
    pub fn jsx() -> Self {
        Self::new(Language::Jsx, tree_sitter_javascript::LANGUAGE.into())
    }

    #[must_use]
    pub fn python() -> Self {
        Self::new(Language::Python, tree_sitter_python::LANGUAGE.into())
    }

    fn new(language: Language, grammar: TreeSitterLanguage) -> Self {
        Self { language, grammar }
    }
}

impl LanguageExtractor for TreeSitterExtractor {
    fn language(&self) -> Language {
        self.language
    }

    fn extract(&self, context: &ExtractionContext) -> HNSQRResult<ExtractionResult> {
        let mut parser = Parser::new();
        parser.set_language(&self.grammar).map_err(|error| {
            HNSQRError::Internal(format!(
                "Failed to initialize {} parser: {error:?}",
                self.language
            ))
        })?;
        let tree = parser.parse(context.source_code, None).ok_or_else(|| {
            HNSQRError::InvalidRequest(format!(
                "Failed to parse {}",
                context.relative_path.display()
            ))
        })?;
        // Fail-closed CodeGraph parsing contract: reject malformed source files to prevent partial/corrupt AST state from entering the typed CodeGraph.
        if tree.root_node().has_error() {
            return Err(HNSQRError::InvalidRequest(format!(
                "Syntax error while parsing {} as {}",
                context.relative_path.display(),
                self.language
            )));
        }

        let mut result = ExtractionResult::default();
        let file_id = push_file_node(context, self.language, &mut result, &tree.root_node());
        let mut visitor = SymbolVisitor {
            context,
            language: self.language,
            file_id: &file_id,
            result: &mut result,
            current_callable: None,
        };
        visitor.walk(&tree.root_node());
        Ok(result)
    }
}

struct SymbolVisitor<'a, 'b> {
    context: &'a ExtractionContext<'a>,
    language: Language,
    file_id: &'a CodeNodeId,
    result: &'b mut ExtractionResult,
    current_callable: Option<CodeNodeId>,
}

impl<'a, 'b> SymbolVisitor<'a, 'b> {
    fn walk(&mut self, node: &Node) {
        let mut callable_scope = None;
        if let Some(kind) = symbol_kind(node.kind()) {
            if let Some(name_node) = node.child_by_field_name("name") {
                let name = text_of(&name_node, self.context.source_code).trim();
                if !name.is_empty() {
                    let symbol_id = self.push_symbol(node, name, kind);
                    if matches!(kind, CodeNodeKind::Function | CodeNodeKind::Method) {
                        callable_scope = Some(std::mem::replace(
                            &mut self.current_callable,
                            Some(symbol_id),
                        ));
                    }
                }
            }
        }

        if node.kind() == "call_expression" {
            self.extract_call(node);
        }

        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            self.walk(&child);
        }

        if let Some(previous_callable) = callable_scope {
            self.current_callable = previous_callable;
        }
    }

    fn push_symbol(&mut self, node: &Node, name: &str, kind: CodeNodeKind) -> CodeNodeId {
        let path = normalized_path(self.context);
        let qualified_name = format!("{path}::{name}");
        let signature = text_of(node, self.context.source_code)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let id = CodeNodeId::compute(
            self.context.workspace_id,
            &path,
            kind,
            &qualified_name,
            &signature,
        );
        let span = span_of(node);
        self.result.nodes.push(CodeNode {
            id: id.clone(),
            kind,
            name: name.to_string(),
            qualified_name,
            signature: Some(signature),
            language: self.language,
            source_file: self.context.relative_path.clone(),
            source_span: span,
            symbol_hash: self.context.content_hash,
            file_hash: self.context.content_hash,
            docstring: None,
            attributes: BTreeMap::new(),
            evidence_class: EvidenceClass::Observation,
            verification_state: VerificationState::Verified,
        });
        push_contains_edge(self.result, self.file_id, &id, span);
        id
    }

    fn extract_call(&mut self, node: &Node) {
        let Some(caller_id) = &self.current_callable else {
            return;
        };
        let Some(function) = node.child_by_field_name("function") else {
            return;
        };
        let callee = text_of(&function, self.context.source_code).trim();
        if callee.is_empty() {
            return;
        }
        let (receiver, target_symbol) = callee
            .rsplit_once('.')
            .map_or((None, callee.to_string()), |(receiver, target)| {
                (Some(receiver.to_string()), target.to_string())
            });
        self.result.unresolved_calls.push(UnresolvedCall {
            caller_id: caller_id.clone(),
            target_symbol,
            receiver,
            span: span_of(node),
        });
    }
}

/// Structural extractor for Markdown, JSON, and TOML documents.
pub struct StructuredExtractor {
    language: Language,
}

impl StructuredExtractor {
    #[must_use]
    pub const fn markdown() -> Self {
        Self {
            language: Language::Markdown,
        }
    }

    #[must_use]
    pub const fn json() -> Self {
        Self {
            language: Language::Json,
        }
    }

    #[must_use]
    pub const fn toml() -> Self {
        Self {
            language: Language::Toml,
        }
    }
}

impl LanguageExtractor for StructuredExtractor {
    fn language(&self) -> Language {
        self.language
    }

    fn extract(&self, context: &ExtractionContext) -> HNSQRResult<ExtractionResult> {
        let mut result = ExtractionResult::default();
        let root_span = SourceSpan::new(
            1,
            1,
            context.source_code.lines().count().max(1),
            1,
            0,
            context.source_code.len(),
        );
        let file_id = push_file_node_with_span(context, self.language, &mut result, root_span);

        match self.language {
            Language::Markdown => {
                for (index, line) in context.source_code.lines().enumerate() {
                    let trimmed = line.trim();
                    let depth = trimmed
                        .chars()
                        .take_while(|character| *character == '#')
                        .count();
                    if depth == 0 || !trimmed.starts_with('#') {
                        continue;
                    }
                    let name = trimmed[depth..].trim();
                    if !name.is_empty() {
                        push_document_child(
                            context,
                            self.language,
                            &mut result,
                            &file_id,
                            name,
                            CodeNodeKind::Documentation,
                            SourceSpan::point(index + 1, depth + 1, 0),
                        );
                    }
                }
            }
            Language::Json => {
                let value: serde_json::Value =
                    serde_json::from_str(context.source_code).map_err(|error| {
                        HNSQRError::InvalidRequest(format!(
                            "Invalid JSON in {}: {error}",
                            context.relative_path.display()
                        ))
                    })?;
                if let serde_json::Value::Object(entries) = value {
                    for key in entries.keys() {
                        push_document_child(
                            context,
                            self.language,
                            &mut result,
                            &file_id,
                            key,
                            CodeNodeKind::Constant,
                            root_span,
                        );
                    }
                }
            }
            Language::Toml => {
                let value: toml::Value = toml::from_str(context.source_code).map_err(|error| {
                    HNSQRError::InvalidRequest(format!(
                        "Invalid TOML in {}: {error}",
                        context.relative_path.display()
                    ))
                })?;
                if let toml::Value::Table(entries) = value {
                    for key in entries.keys() {
                        push_document_child(
                            context,
                            self.language,
                            &mut result,
                            &file_id,
                            key,
                            CodeNodeKind::Constant,
                            root_span,
                        );
                    }
                }
            }
            _ => unreachable!("StructuredExtractor only supports document languages"),
        }
        Ok(result)
    }
}

fn symbol_kind(node_kind: &str) -> Option<CodeNodeKind> {
    match node_kind {
        "class_declaration" | "class_definition" | "abstract_class_declaration" => {
            Some(CodeNodeKind::Class)
        }
        "interface_declaration" => Some(CodeNodeKind::Interface),
        "enum_declaration" => Some(CodeNodeKind::Enum),
        "type_alias_declaration" => Some(CodeNodeKind::TypeAlias),
        "function_declaration" | "function_definition" | "generator_function_declaration" => {
            Some(CodeNodeKind::Function)
        }
        "method_definition" => Some(CodeNodeKind::Method),
        _ => None,
    }
}

fn push_file_node(
    context: &ExtractionContext,
    language: Language,
    result: &mut ExtractionResult,
    root: &Node,
) -> CodeNodeId {
    push_file_node_with_span(context, language, result, span_of(root))
}

fn push_file_node_with_span(
    context: &ExtractionContext,
    language: Language,
    result: &mut ExtractionResult,
    span: SourceSpan,
) -> CodeNodeId {
    let path = normalized_path(context);
    let id = CodeNodeId::compute(context.workspace_id, &path, CodeNodeKind::File, &path, "");
    let name = context
        .relative_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("unknown")
        .to_string();
    result.nodes.push(CodeNode {
        id: id.clone(),
        kind: CodeNodeKind::File,
        name,
        qualified_name: path,
        signature: None,
        language,
        source_file: context.relative_path.clone(),
        source_span: span,
        symbol_hash: context.content_hash,
        file_hash: context.content_hash,
        docstring: None,
        attributes: BTreeMap::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
    });
    id
}

fn push_document_child(
    context: &ExtractionContext,
    language: Language,
    result: &mut ExtractionResult,
    file_id: &CodeNodeId,
    name: &str,
    kind: CodeNodeKind,
    span: SourceSpan,
) {
    let path = normalized_path(context);
    let qualified_name = format!("{path}::{name}");
    let id = CodeNodeId::compute(context.workspace_id, &path, kind, &qualified_name, name);
    result.nodes.push(CodeNode {
        id: id.clone(),
        kind,
        name: name.to_string(),
        qualified_name,
        signature: None,
        language,
        source_file: context.relative_path.clone(),
        source_span: span,
        symbol_hash: context.content_hash,
        file_hash: context.content_hash,
        docstring: None,
        attributes: BTreeMap::new(),
        evidence_class: EvidenceClass::Observation,
        verification_state: VerificationState::Verified,
    });
    push_contains_edge(result, file_id, &id, span);
}

fn push_contains_edge(
    result: &mut ExtractionResult,
    source: &CodeNodeId,
    target: &CodeNodeId,
    evidence: SourceSpan,
) {
    result.edges.push(CodeEdge {
        id: CodeEdgeId::compute(
            source,
            target,
            CodeRelation::Contains,
            RelationOrigin::Extracted,
            &evidence,
        ),
        source: source.clone(),
        target: target.clone(),
        relation: CodeRelation::Contains,
        origin: RelationOrigin::Extracted,
        confidence: RelationOrigin::Extracted.default_confidence(),
        evidence,
        attributes: BTreeMap::new(),
    });
}

fn normalized_path(context: &ExtractionContext) -> String {
    context.relative_path.to_string_lossy().replace('\\', "/")
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn tsx_extractor_parses_jsx_and_emits_component_symbol() {
        let path = PathBuf::from("Banner.tsx");
        let context = ExtractionContext {
            workspace_id: "test",
            relative_path: &path,
            source_code: "export function Banner() { renderMain(); return <main />; }",
            content_hash: [7; 32],
        };
        let extracted = TreeSitterExtractor::tsx().extract(&context).unwrap();
        assert!(
            extracted
                .nodes
                .iter()
                .any(|node| node.name == "Banner" && node.kind == CodeNodeKind::Function)
        );
        assert!(
            extracted
                .unresolved_calls
                .iter()
                .any(|call| call.target_symbol == "renderMain")
        );
    }

    #[test]
    fn document_extractors_emit_top_level_structure() {
        let json_path = PathBuf::from("package.json");
        let markdown_path = PathBuf::from("README.md");
        let json_context = ExtractionContext {
            workspace_id: "test",
            relative_path: &json_path,
            source_code: r#"{"name":"holo"}"#,
            content_hash: [7; 32],
        };
        let markdown_context = ExtractionContext {
            workspace_id: "test",
            relative_path: &markdown_path,
            source_code: "# Overview",
            content_hash: [7; 32],
        };
        let json = StructuredExtractor::json().extract(&json_context).unwrap();
        let markdown = StructuredExtractor::markdown()
            .extract(&markdown_context)
            .unwrap();
        assert!(json.nodes.iter().any(|node| node.name == "name"));
        assert!(markdown.nodes.iter().any(|node| node.name == "Overview"));
    }

    #[test]
    fn every_tree_sitter_dialect_emits_its_declared_language() {
        let cases = [
            (
                TreeSitterExtractor::typescript(),
                "export function typed() {}",
                "a.ts",
            ),
            (
                TreeSitterExtractor::tsx(),
                "export function view() { return <main />; }",
                "a.tsx",
            ),
            (
                TreeSitterExtractor::javascript(),
                "function script() {}",
                "a.js",
            ),
            (
                TreeSitterExtractor::jsx(),
                "function view() { return <main />; }",
                "a.jsx",
            ),
            (
                TreeSitterExtractor::python(),
                "def pythonic():\n    return 1\n",
                "a.py",
            ),
        ];
        for (extractor, source_code, path) in cases {
            let path = PathBuf::from(path);
            let expected_language = extractor.language();
            let context = ExtractionContext {
                workspace_id: "test",
                relative_path: &path,
                source_code,
                content_hash: [9; 32],
            };
            let extracted = extractor.extract(&context).unwrap();
            assert!(
                extracted
                    .nodes
                    .iter()
                    .any(|node| node.language == expected_language)
            );
            assert!(
                extracted
                    .nodes
                    .iter()
                    .any(|node| node.kind == CodeNodeKind::Function)
            );
        }
    }

    #[test]
    fn toml_extractor_emits_top_level_keys() {
        let path = PathBuf::from("Cargo.toml");
        let context = ExtractionContext {
            workspace_id: "test",
            relative_path: &path,
            source_code: "name = 'holosphere'",
            content_hash: [8; 32],
        };
        let extracted = StructuredExtractor::toml().extract(&context).unwrap();
        assert!(extracted.nodes.iter().any(|node| node.name == "name"));
    }

    #[test]
    fn syntax_errors_are_rejected_before_graph_ingestion() {
        let path = PathBuf::from("broken.ts");
        let context = ExtractionContext {
            workspace_id: "test",
            relative_path: &path,
            source_code: "function incomplete(",
            content_hash: [0; 32],
        };
        assert!(TreeSitterExtractor::typescript().extract(&context).is_err());
    }
}
