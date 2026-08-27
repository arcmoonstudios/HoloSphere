/* holosphere/src/codegraph/parser.rs */
//!▫~•◦-------------------------------‣
//! # Language-Agnostic CodeGraph AST Extractor Trait & Intermediate Representation
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Defines the uniform abstraction for AST extractors across programming languages,
//! collecting structural nodes, direct edges, rationale notes, and unresolved references.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::schema::{CodeEdge, CodeNode, CodeNodeId, CodeRelation, Language, SourceSpan};
use crate::HNSQRResult;

/// Context passed into a language-specific AST extractor.
#[derive(Clone, Debug)]
pub struct ExtractionContext<'a> {
    pub workspace_id: &'a str,
    pub relative_path: &'a PathBuf,
    pub source_code: &'a str,
    pub content_hash: [u8; 32],
}

/// Unresolved function/method invocation identified during AST parsing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnresolvedCall {
    pub caller_id: CodeNodeId,
    pub target_symbol: String,
    pub receiver: Option<String>,
    pub span: SourceSpan,
}

/// Unresolved type reference (struct, trait, return type, argument type) identified during parsing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnresolvedTypeRef {
    pub source_id: CodeNodeId,
    pub target_type: String,
    pub relation: CodeRelation,
    pub span: SourceSpan,
}

/// Discovered import/use declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImportItem {
    pub module_node_id: CodeNodeId,
    pub import_path: String,
    pub imported_symbol: String,
    pub alias: Option<String>,
    pub is_glob: bool,
    pub span: SourceSpan,
}

/// Complete output emitted from a single source file's AST extraction pass.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub nodes: Vec<CodeNode>,
    pub edges: Vec<CodeEdge>,
    pub unresolved_calls: Vec<UnresolvedCall>,
    pub unresolved_types: Vec<UnresolvedTypeRef>,
    pub imports: Vec<ImportItem>,
}

/// Uniform trait implemented by each programming language front end.
pub trait LanguageExtractor: Send + Sync {
    fn language(&self) -> Language;
    fn extract(&self, context: &ExtractionContext) -> HNSQRResult<ExtractionResult>;
}
