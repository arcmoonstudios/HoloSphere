/* holosphere/src/codegraph/schema.rs */
//!▫~•◦-------------------------------‣
//! # HoloSphere CodeGraph — Authoritative Code AST & Topology Schema
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides deterministic intermediate representations for repository-level AST symbols,
//! relational topology, source spans, stable identity hashing, and epistemic origin attribution.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::transport::model_gateway::{EvidenceClass, VerificationState};

/// Supported language taxonomy for syntax parsing and extraction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Language {
    Rust,
    TypeScript,
    Tsx,
    JavaScript,
    Jsx,
    Python,
    Markdown,
    Json,
    Toml,
    Go,
    Java,
    C,
    Cpp,
    Unknown,
}

impl Language {
    #[must_use]
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Self::Rust,
            "ts" => Self::TypeScript,
            "tsx" => Self::Tsx,
            "js" | "mjs" | "cjs" => Self::JavaScript,
            "jsx" => Self::Jsx,
            "py" | "pyi" => Self::Python,
            "md" | "markdown" => Self::Markdown,
            "json" => Self::Json,
            "toml" => Self::Toml,
            "go" => Self::Go,
            "java" => Self::Java,
            "c" | "h" => Self::C,
            "cpp" | "cxx" | "cc" | "hpp" => Self::Cpp,
            _ => Self::Unknown,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Rust => "rust",
            Self::TypeScript => "typescript",
            Self::Tsx => "tsx",
            Self::JavaScript => "javascript",
            Self::Jsx => "jsx",
            Self::Python => "python",
            Self::Markdown => "markdown",
            Self::Json => "json",
            Self::Toml => "toml",
            Self::Go => "go",
            Self::Java => "java",
            Self::C => "c",
            Self::Cpp => "cpp",
            Self::Unknown => "unknown",
        }
    }

    /// Whether the built-in CodeGraph registry has a concrete extractor for this language.
    #[must_use]
    pub const fn has_builtin_extractor(self) -> bool {
        matches!(
            self,
            Self::Rust
                | Self::TypeScript
                | Self::Tsx
                | Self::JavaScript
                | Self::Jsx
                | Self::Python
                | Self::Markdown
                | Self::Json
                | Self::Toml
        )
    }
}

impl fmt::Display for Language {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
mod language_tests {
    use super::Language;

    #[test]
    fn contextgraph_language_extensions_have_builtin_extractors() {
        for extension in ["rs", "ts", "tsx", "js", "jsx", "py", "md", "json", "toml"] {
            assert!(Language::from_extension(extension).has_builtin_extractor());
        }
        assert!(!Language::from_extension("go").has_builtin_extractor());
    }
}

/// Exact byte and line-column bounding span in a source code file.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct SourceSpan {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub byte_start: usize,
    pub byte_end: usize,
}

impl SourceSpan {
    #[must_use]
    pub const fn new(
        start_line: usize,
        start_col: usize,
        end_line: usize,
        end_col: usize,
        byte_start: usize,
        byte_end: usize,
    ) -> Self {
        Self {
            start_line,
            start_col,
            end_line,
            end_col,
            byte_start,
            byte_end,
        }
    }

    #[must_use]
    pub const fn point(line: usize, col: usize, byte_offset: usize) -> Self {
        Self {
            start_line: line,
            start_col: col,
            end_line: line,
            end_col: col,
            byte_start: byte_offset,
            byte_end: byte_offset,
        }
    }
}

impl fmt::Display for SourceSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "L{}:{}-L{}:{} (bytes {}..{})",
            self.start_line,
            self.start_col,
            self.end_line,
            self.end_col,
            self.byte_start,
            self.byte_end
        )
    }
}

/// Structural categorization of CodeGraph symbols and hierarchy elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeNodeKind {
    // Hierarchy & Containment
    Repository,
    Directory,
    File,
    Module,
    Namespace,

    // Declarations & Types
    Class,
    Struct,
    Trait,
    Interface,
    Enum,
    Variant,
    TypeAlias,

    // Callables
    Function,
    Method,
    Constructor,
    Macro,

    // Members & Variables
    Field,
    Constant,

    // Meta & Architectural Entities
    Import,
    Test,
    Documentation,
    Rationale,
}

impl CodeNodeKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Repository => "repository",
            Self::Directory => "directory",
            Self::File => "file",
            Self::Module => "module",
            Self::Namespace => "namespace",
            Self::Class => "class",
            Self::Struct => "struct",
            Self::Trait => "trait",
            Self::Interface => "interface",
            Self::Enum => "enum",
            Self::Variant => "variant",
            Self::TypeAlias => "type_alias",
            Self::Function => "function",
            Self::Method => "method",
            Self::Constructor => "constructor",
            Self::Macro => "macro",
            Self::Field => "field",
            Self::Constant => "constant",
            Self::Import => "import",
            Self::Test => "test",
            Self::Documentation => "documentation",
            Self::Rationale => "rationale",
        }
    }
}

impl fmt::Display for CodeNodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Directed relational semantic between two CodeGraph nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodeRelation {
    Defines,
    Contains,

    Imports,
    Exports,

    Calls,
    Uses,
    References,

    Implements,
    Inherits,

    Reads,
    Writes,

    Constructs,

    Tests,
    Documents,
    Explains,
    Justifies,

    Returns,
    Accepts,

    DependsOn,
}

impl CodeRelation {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Defines => "defines",
            Self::Contains => "contains",
            Self::Imports => "imports",
            Self::Exports => "exports",
            Self::Calls => "calls",
            Self::Uses => "uses",
            Self::References => "references",
            Self::Implements => "implements",
            Self::Inherits => "inherits",
            Self::Reads => "reads",
            Self::Writes => "writes",
            Self::Constructs => "constructs",
            Self::Tests => "tests",
            Self::Documents => "documents",
            Self::Explains => "explains",
            Self::Justifies => "justifies",
            Self::Returns => "returns",
            Self::Accepts => "accepts",
            Self::DependsOn => "depends_on",
        }
    }
}

impl fmt::Display for CodeRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Epistemic origin category for code graph relationships.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationOrigin {
    /// Proved directly by syntactic AST construction (e.g. `use foo::Bar;` or explicit function definition).
    Extracted,
    /// Resolved statically across file / symbol tables with high confidence.
    Resolved,
    /// Multiple legal candidate targets exist; preserved without artificial guessing.
    Ambiguous,
}

impl RelationOrigin {
    #[must_use]
    pub const fn default_confidence(self) -> f32 {
        match self {
            Self::Extracted => 1.0,
            Self::Resolved => 0.95,
            Self::Ambiguous => 0.5,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Extracted => "extracted",
            Self::Resolved => "resolved",
            Self::Ambiguous => "ambiguous",
        }
    }
}

impl fmt::Display for RelationOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Deterministic, position-independent identifier for a code node.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CodeNodeId(pub String);

impl CodeNodeId {
    /// Computes a stable hash ID invariant to line additions/deletions.
    #[must_use]
    pub fn compute(
        workspace_id: &str,
        relative_path: &str,
        kind: CodeNodeKind,
        qualified_name: &str,
        normalized_signature: &str,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_CODE_NODE_V1:");
        hasher.update(workspace_id.as_bytes());
        hasher.update(b"|");
        hasher.update(relative_path.replace('\\', "/").as_bytes());
        hasher.update(b"|");
        hasher.update(kind.as_str().as_bytes());
        hasher.update(b"|");
        hasher.update(qualified_name.as_bytes());
        hasher.update(b"|");
        hasher.update(normalized_signature.as_bytes());

        let hex = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        Self(format!("sym_{hex}"))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for CodeNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CodeNodeId({})", self.0)
    }
}

impl fmt::Display for CodeNodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Deterministic identifier for a code graph relationship edge.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct CodeEdgeId(pub String);

impl CodeEdgeId {
    #[must_use]
    pub fn compute(
        source: &CodeNodeId,
        target: &CodeNodeId,
        relation: CodeRelation,
        origin: RelationOrigin,
        evidence: &SourceSpan,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_CODE_EDGE_V1:");
        hasher.update(source.as_str().as_bytes());
        hasher.update(b"->");
        hasher.update(target.as_str().as_bytes());
        hasher.update(b":");
        hasher.update(relation.as_str().as_bytes());
        hasher.update(b":");
        hasher.update(origin.as_str().as_bytes());
        hasher.update(b":");
        hasher.update(format!("{evidence}").as_bytes());

        let hex = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        Self(format!("edge_{hex}"))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for CodeEdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CodeEdgeId({})", self.0)
    }
}

impl fmt::Display for CodeEdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Full authoritative record of one code symbol, item, or rationale element.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeNode {
    pub id: CodeNodeId,
    pub kind: CodeNodeKind,

    pub name: String,
    pub qualified_name: String,
    pub signature: Option<String>,

    pub language: Language,

    pub source_file: PathBuf,
    pub source_span: SourceSpan,

    pub symbol_hash: [u8; 32],
    pub file_hash: [u8; 32],

    pub docstring: Option<String>,
    pub attributes: BTreeMap<String, serde_json::Value>,

    pub evidence_class: EvidenceClass,
    pub verification_state: VerificationState,
}

/// Authoritative directed edge between two code nodes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodeEdge {
    pub id: CodeEdgeId,
    pub source: CodeNodeId,
    pub target: CodeNodeId,

    pub relation: CodeRelation,
    pub origin: RelationOrigin,
    pub confidence: f32,

    pub evidence: SourceSpan,
    pub attributes: BTreeMap<String, serde_json::Value>,
}

/// Atomic mutation payload representing repository changes to be committed to HoloSphere.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct CodeGraphDelta {
    pub workspace_id: String,
    pub insert_nodes: Vec<CodeNode>,
    pub delete_nodes: Vec<CodeNodeId>,

    pub insert_edges: Vec<CodeEdge>,
    pub delete_edges: Vec<CodeEdgeId>,

    pub touched_files: Vec<PathBuf>,
}
