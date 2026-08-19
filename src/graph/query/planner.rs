/* hnsqr/src/graph/query/planner.rs */
//!▫~•◦-------------------------------‣
//! # Query Planner — Text → Executable Physical Plan
//!▫~•◦-------------------------------------------------------------------‣
//!
//! `QueryPlanner::compile` is the single entry-point for taking a GraphQuery
//! query string all the way through to an executable `PhysicalPlan`:
//!
//! ```text
//! &str  ──parse_query──►  QueryAst
//!         ──SemanticAnalyzer::analyse──►  SymbolTable
//!         ──LogicalPlan::from_ast──►  LogicalPlan
//!         ──Optimizer::optimise──►  LogicalPlan
//!         ──PhysicalPlan::lower──►  PhysicalPlan
//! ```
//!
//! The output `CompiledQuery` bundles the `PhysicalPlan`, `SymbolTable`,
//! and the original `QueryAst` so callers can:
//! - Execute via `ExecutionContext::execute(&plan)` or
//!   `ExecutionContext::execute_with_vector_engine(&plan, index, vectors)`.
//! - Inspect via `ExplainOutput::render(&plan)`.
//! - Propagate mutation clauses via `ExecutionContext::compile_mutations`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::catalog::labels::LabelCatalog;
use crate::graph::catalog::relationships::RelTypeCatalog;
use crate::graph::query::ast::{
    GraphPattern, QueryAst, ReturnItem,
};
use crate::graph::query::logical::LogicalPlan;
use crate::graph::query::optimizer::Optimizer;
use crate::graph::query::parser::{parse_query, ParseError};
use crate::graph::query::physical::PhysicalPlan;
use crate::graph::query::semantic::{SemanticAnalyzer, SemanticError};
use crate::graph::query::symbols::{SymbolId, SymbolTable};

// ── CompileError ──────────────────────────────────────────────────────────────

/// Errors produced by `QueryPlanner::compile`.
#[derive(Debug)]
pub enum CompileError {
    Parse(ParseError),
    Semantic(Vec<SemanticError>),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(e) => write!(f, "Parse: {e}"),
            Self::Semantic(errs) => {
                for e in errs {
                    writeln!(f, "Semantic: {e}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CompileError {}

// ── CompiledQuery ─────────────────────────────────────────────────────────────

/// The fully compiled and optimised query, ready for execution.
pub struct CompiledQuery {
    pub plan: PhysicalPlan,
    pub symbols: SymbolTable,
    pub ast: QueryAst,
    /// Maps column index → alias string for result labelling.
    pub column_names: Vec<String>,
}

// ── QueryPlanner ─────────────────────────────────────────────────────────────

/// Compiles GraphQuery text all the way to a `PhysicalPlan`.
pub struct QueryPlanner;

impl QueryPlanner {
    /// Parses, semantically analyses, optimises and lowers a GraphQuery query.
    ///
    /// `label_catalog` and `rel_catalog` are borrowed at parse time so label
    /// and relationship-type names are resolved to compact IDs without a
    /// second catalog-lookup pass.
    pub fn compile(
        src: &str,
        label_catalog: &LabelCatalog,
        rel_catalog: &RelTypeCatalog,
        stats: Option<&crate::graph::stats::cardinality::GraphCardinalityStats>,
    ) -> Result<CompiledQuery, CompileError> {
        // 1. Parse → QueryAst (zero-copy lexer)
        let ast = parse_query(src, label_catalog, rel_catalog).map_err(CompileError::Parse)?;

        // 2. Semantic analysis → SymbolTable
        let symbols =
            SemanticAnalyzer::analyse(&ast).map_err(CompileError::Semantic)?;

        // 3. Lower AST → LogicalPlan
        let logical = Self::build_logical_plan(&ast, &symbols);

        // 4. Heuristic optimisation
        let logical = Optimizer::optimise(logical, stats);

        // 5. Physical lowering
        let plan = PhysicalPlan::lower(logical);

        // 6. Build ordered column name list from RETURN clause
        let column_names = ast
            .return_clause
            .items
            .iter()
            .map(|item| match item {
                ReturnItem::Alias(a) => a.clone(),
                ReturnItem::PropertyRef { alias, key } => format!("{alias}.{key}"),
            })
            .collect();

        Ok(CompiledQuery { plan, symbols, ast, column_names })
    }

    // ── AST → LogicalPlan ─────────────────────────────────────────────────

    fn build_logical_plan(ast: &QueryAst, symbols: &SymbolTable) -> LogicalPlan {
        // Determine the root scan.
        let mut plan: LogicalPlan = if let Some(vm) = &ast.vector_match {
            let binding = symbols.get(&vm.binding).unwrap_or(SymbolId(0));
            LogicalPlan::VectorSeed {
                binding,
                query_param: vm.query_param.clone(),
                k: vm.k,
                contract: vm.contract,
            }
        } else {
            // Find the first NodePattern to seed the scan.
            let first_node = ast.patterns.iter().find_map(|p| {
                if let GraphPattern::NodePattern { alias, label, .. } = p {
                    Some((alias.clone(), *label))
                } else {
                    None
                }
            });

            match first_node {
                Some((alias, label)) => {
                    let binding = symbols.get(&alias).unwrap_or(SymbolId(0));
                    // Collect predicates associated with this node.
                    let preds = ast.patterns.iter().find_map(|p| {
                        if let GraphPattern::NodePattern { alias: a, predicates, .. } = p {
                            if a == &alias { Some(predicates.clone()) } else { None }
                        } else {
                            None
                        }
                    }).unwrap_or_default();
                    LogicalPlan::NodeScan {
                        binding,
                        label_filter: label,
                        predicates: preds,
                    }
                }
                None => {
                    // Fallback: scan all nodes.
                    LogicalPlan::NodeScan {
                        binding: SymbolId(0),
                        label_filter: None,
                        predicates: Vec::new(),
                    }
                }
            }
        };

        // Chain expand / optional-expand patterns.
        for pattern in &ast.patterns {
            match pattern {
                GraphPattern::NodePattern { .. } => {
                    // Already used as root scan or emitted as destination in Expand.
                }
                GraphPattern::Expand {
                    src_alias,
                    rel_alias,
                    rel_type,
                    dst_alias,
                    direction,
                    min_hops,
                    max_hops,
                } => {
                    let src_binding = symbols.get(src_alias).unwrap_or(SymbolId(0));
                    let rel_binding = rel_alias.as_deref().and_then(|r| symbols.get(r));
                    let dst_binding = symbols.get(dst_alias).unwrap_or(SymbolId(0));
                    plan = LogicalPlan::Expand {
                        input: Box::new(plan),
                        src_binding,
                        rel_binding,
                        dst_binding,
                        rel_type_filter: *rel_type,
                        direction: *direction,
                        min_hops: *min_hops,
                        max_hops: *max_hops,
                    };
                }
                GraphPattern::OptionalExpand {
                    src_alias,
                    rel_alias,
                    rel_type,
                    dst_alias,
                    direction,
                    min_hops,
                    max_hops,
                } => {
                    let src_binding = symbols.get(src_alias).unwrap_or(SymbolId(0));
                    let rel_binding = rel_alias.as_deref().and_then(|r| symbols.get(r));
                    let dst_binding = symbols.get(dst_alias).unwrap_or(SymbolId(0));
                    plan = LogicalPlan::OptionalExpand {
                        input: Box::new(plan),
                        src_binding,
                        rel_binding,
                        dst_binding,
                        rel_type_filter: *rel_type,
                        direction: *direction,
                        min_hops: *min_hops,
                        max_hops: *max_hops,
                    };
                }
            }
        }

        // WHERE predicates → Filter node.
        if !ast.where_clause.predicates.is_empty() {
            plan = LogicalPlan::Filter {
                input: Box::new(plan),
                predicates: ast.where_clause.predicates.clone(),
            };
        }

        // RETURN projection.
        let output_symbols: Vec<SymbolId> = ast
            .return_clause
            .items
            .iter()
            .filter_map(|item| {
                let alias = match item {
                    ReturnItem::Alias(a) => a.as_str(),
                    ReturnItem::PropertyRef { alias, .. } => alias.as_str(),
                };
                symbols.get(alias)
            })
            .collect();

        if !output_symbols.is_empty() {
            plan = LogicalPlan::Project {
                input: Box::new(plan),
                output_bindings: output_symbols,
            };
        }

        // LIMIT.
        if let Some(limit) = ast.return_clause.limit {
            plan = LogicalPlan::Limit { input: Box::new(plan), count: limit };
        }

        plan
    }
}
