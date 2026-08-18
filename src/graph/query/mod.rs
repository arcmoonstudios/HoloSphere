/* hnsqr/src/graph/query/mod.rs */
//!▫~•◦-------------------------------‣
//! # Graph Query Engine — Six-Stage Cypher-Compatible Pipeline
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Stages:
//!   1. `parser`   — text → concrete syntax tree
//!   2. `ast`      — CST → typed AST
//!   3. `semantic` — name resolution, symbol binding
//!   4. `logical`  — AST → logical plan (RelationalAlgebra-style)
//!   5. `physical` — logical plan → morsel-driven physical operators
//!   6. `executor` — physical plan → result rows
//!
//! The supported subset is "HNSQR Graph Query Profile v1":
//!   MATCH, WHERE (simple predicates), RETURN, LIMIT, WITH (pass-through),
//!   VECTOR MATCH extension (Certified / HighRecall / Bounded contract).
//!
//! Cypher features deferred to v2+: OPTIONAL MATCH, variable-length paths,
//! aggregation, ORDER BY, CREATE/DELETE/MERGE/SET within queries, subqueries.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod ast;
pub mod executor;
pub mod explain;
pub mod logical;
pub mod morsel;
pub mod optimizer;
pub mod physical;
pub mod semantic;
pub mod symbols;

pub use ast::{Direction, GraphPattern, QueryAst, ReturnClause, WhereClause};
pub use executor::{ExecutionContext, QueryResult};
pub use explain::ExplainOutput;
pub use logical::LogicalPlan;
pub use morsel::{BindingColumn, Morsel};
pub use physical::PhysicalPlan;
pub use semantic::{SemanticAnalyzer, SemanticError};
pub use symbols::{SymbolId, SymbolTable};
