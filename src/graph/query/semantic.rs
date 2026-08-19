/* hnsqr/src/graph/query/semantic.rs */
//!▫~•◦-------------------------------‣
//! # Semantic Analyzer — Name Resolution and Symbol Binding
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Converts string aliases in the `QueryAst` into `SymbolId`s and validates
//! that every referenced alias is declared before use.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣
 
use crate::graph::query::ast::{GraphPattern, QueryAst};
use crate::graph::query::symbols::SymbolTable;

/// Errors produced during semantic analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SemanticError {
    UndeclaredAlias(String),
    DuplicateAlias(String),
    InvalidHopRange { min: u8, max: u8 },
    UnsupportedFeature(String),
}

impl std::fmt::Display for SemanticError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UndeclaredAlias(a) => write!(f, "Undeclared alias: `{a}`"),
            Self::DuplicateAlias(a) => write!(f, "Duplicate alias: `{a}`"),
            Self::InvalidHopRange { min, max } => {
                write!(f, "Invalid hop range: min={min} > max={max}")
            }
            Self::UnsupportedFeature(s) => write!(f, "Unsupported in v1: {s}"),
        }
    }
}

impl std::error::Error for SemanticError {}

/// Performs semantic analysis on a parsed `QueryAst`.
pub struct SemanticAnalyzer;

impl SemanticAnalyzer {
    /// Analyses the AST and returns a populated `SymbolTable`, or a list of errors.
    pub fn analyse(ast: &QueryAst) -> Result<SymbolTable, Vec<SemanticError>> {
        let mut symbols = SymbolTable::default();
        let mut errors = Vec::new();

        // Register aliases from VECTOR MATCH.
        if let Some(vm) = &ast.vector_match {
            symbols.intern(&vm.binding);
        }

        // Process patterns in order: NodePattern declarations before Expand uses.
        for pattern in &ast.patterns {
            match pattern {
                GraphPattern::NodePattern { alias, .. } => {
                    if symbols.get(alias).is_some() {
                        // Re-binding the same alias is allowed in MATCH (it constrains).
                    } else {
                        symbols.intern(alias);
                    }
                }
                GraphPattern::Expand {
                    src_alias,
                    rel_alias,
                    dst_alias,
                    min_hops,
                    max_hops,
                    ..
                } => {
                    if *min_hops > *max_hops {
                        errors.push(SemanticError::InvalidHopRange {
                            min: *min_hops,
                            max: *max_hops,
                        });
                    }
                    // src must already be declared (either from NodePattern or VECTOR MATCH).
                    if symbols.get(src_alias).is_none() {
                        errors.push(SemanticError::UndeclaredAlias(src_alias.clone()));
                    }
                    // dst is declared here.
                    symbols.intern(dst_alias);
                    // rel alias is optional.
                    if let Some(ra) = rel_alias {
                        symbols.intern(ra);
                    }
                }
                GraphPattern::OptionalExpand {
                    src_alias,
                    rel_alias,
                    dst_alias,
                    min_hops,
                    max_hops,
                    ..
                } => {
                    if *min_hops > *max_hops {
                        errors.push(SemanticError::InvalidHopRange {
                            min: *min_hops,
                            max: *max_hops,
                        });
                    }
                    if symbols.get(src_alias).is_none() {
                        errors.push(SemanticError::UndeclaredAlias(src_alias.clone()));
                    }
                    symbols.intern(dst_alias);
                    if let Some(ra) = rel_alias {
                        symbols.intern(ra);
                    }
                }
            }
        }

        // Validate RETURN aliases.
        for item in &ast.return_clause.items {
            let alias = match item {
                crate::graph::query::ast::ReturnItem::Alias(a) => a.as_str(),
                crate::graph::query::ast::ReturnItem::PropertyRef { alias, .. } => alias.as_str(),
            };
            if symbols.get(alias).is_none() {
                errors.push(SemanticError::UndeclaredAlias(alias.to_string()));
            }
        }

        if errors.is_empty() {
            Ok(symbols)
        } else {
            Err(errors)
        }
    }
}
