/* hnsqr/src/graph/query/mod.rs */
//!▫~•◦-------------------------------‣
//! # Graph Query Engine — Six-Stage GraphQuery-Compatible Pipeline
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Stages:
//!   1. `lexer`    — zero-copy tokenisation of `&str` input
//!   2. `parser`   — token stream → `QueryAst` (one alloc per identifier)
//!   3. `semantic` — name resolution, symbol binding
//!   4. `logical`  — `QueryAst` → logical plan (relational-algebra IR)
//!   5. `physical` — logical plan → morsel-driven physical operators
//!   6. `executor` — physical plan → result rows
//!
//! Use `QueryPlanner::compile(src, label_catalog, rel_catalog)` to go
//! from a GraphQuery `&str` directly to an executable `PhysicalPlan` in one
//! call, or use the individual stages for testing and introspection.
//!
//! ## Supported subset — HNSQR Graph Query Profile v1
//! MATCH, OPTIONAL MATCH, WHERE (scalar predicates), RETURN, LIMIT,
//! variable-length paths (`-[*min..max]->`), VECTOR MATCH extension
//! (Certified / HighRecall / Bounded), CREATE node, DELETE alias.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod ast;
pub mod executor;
pub mod explain;
pub mod lexer;
pub mod logical;
pub mod morsel;
pub mod optimizer;
pub mod parser;
pub mod physical;
pub mod planner;
pub mod semantic;
pub mod symbols;

pub use ast::{Direction, GraphPattern, QueryAst, ReturnClause, WhereClause};
pub use executor::{ExecutionContext, QueryResult};
pub use explain::ExplainOutput;
pub use lexer::{Lexer, Token};
pub use logical::LogicalPlan;
pub use morsel::{BindingColumn, Morsel};
pub use parser::{ParseError, Parser, parse_query};
pub use physical::PhysicalPlan;
pub use planner::{CompileError, CompiledQuery, QueryPlanner};
pub use semantic::{SemanticAnalyzer, SemanticError};
pub use symbols::{SymbolId, SymbolTable};
