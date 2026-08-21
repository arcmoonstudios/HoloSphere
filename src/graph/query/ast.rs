/* hnsqr/src/graph/query/ast.rs */
//!▫~•◦-------------------------------‣
//! # Graph Query AST
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Represents the post-parse, pre-semantic structure of a graph query.
//! String aliases are still present at this stage; they are replaced by
//! `SymbolId`s during semantic analysis.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::catalog::labels::LabelId;
use crate::graph::catalog::relationships::RelTypeId;

/// Traversal direction in a pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Outgoing,
    Incoming,
    Undirected,
}

/// A single predicate operand.
#[derive(Clone, Debug, PartialEq)]
pub enum PredicateValue {
    Literal(serde_json::Value),
    Parameter(String),
    PropertyRef { alias: String, key: String },
}

/// Simple scalar predicate supported in v1 WHERE clauses.
#[derive(Clone, Debug, PartialEq)]
pub enum ScalarPredicate {
    Eq(PredicateValue, PredicateValue),
    Ne(PredicateValue, PredicateValue),
    Lt(PredicateValue, PredicateValue),
    Le(PredicateValue, PredicateValue),
    Gt(PredicateValue, PredicateValue),
    Ge(PredicateValue, PredicateValue),
    IsNull(PredicateValue),
    IsNotNull(PredicateValue),
    And(Box<ScalarPredicate>, Box<ScalarPredicate>),
    Or(Box<ScalarPredicate>, Box<ScalarPredicate>),
    Not(Box<ScalarPredicate>),
}

/// A pattern element — either a node or an expand step.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphPattern {
    /// `(alias:Label)` — node with optional label filter.
    NodePattern {
        alias: String,
        label: Option<LabelId>,
        predicates: Vec<ScalarPredicate>,
    },
    /// `(src)-[r:TYPE]->(dst)` — directed or undirected expand.
    Expand {
        src_alias: String,
        rel_alias: Option<String>,
        rel_type: Option<RelTypeId>,
        dst_alias: String,
        direction: Direction,
        /// Bounded variable-length path range; 1..1 for single-hop.
        min_hops: u8,
        max_hops: u8,
    },
    /// `OPTIONAL MATCH (src)-[r:TYPE]->(dst)` — left outer join expansion.
    OptionalExpand {
        src_alias: String,
        rel_alias: Option<String>,
        rel_type: Option<RelTypeId>,
        dst_alias: String,
        direction: Direction,
        min_hops: u8,
        max_hops: u8,
    },
}

/// Inline vector search clause (HNSQR extension).
#[derive(Clone, Debug, PartialEq)]
pub struct VectorMatchClause {
    /// Alias that receives vector-matched results.
    pub binding: String,
    /// Query text or parameter name.
    pub query_param: String,
    pub k: usize,
    pub contract: VectorContract,
}

/// Retrieval contract for a VECTOR MATCH clause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorContract {
    Certified,
    HighRecall,
    Bounded,
}

/// WHERE clause (v1: flat conjunction of scalar predicates).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WhereClause {
    pub predicates: Vec<ScalarPredicate>,
}

/// RETURN clause.
#[derive(Clone, Debug, PartialEq)]
pub struct ReturnClause {
    pub items: Vec<ReturnItem>,
    pub limit: Option<usize>,
}

/// A single item in a RETURN list.
#[derive(Clone, Debug, PartialEq)]
pub enum ReturnItem {
    Alias(String),
    PropertyRef { alias: String, key: String },
}

/// Top-level query AST.
#[derive(Clone, Debug, PartialEq)]
pub struct QueryAst {
    /// Optional leading VECTOR MATCH clause.
    pub vector_match: Option<VectorMatchClause>,
    /// Ordered list of MATCH pattern steps.
    pub patterns: Vec<GraphPattern>,
    pub where_clause: WhereClause,
    pub return_clause: ReturnClause,
    /// Optional mutation clauses (CREATE, DELETE, SET, MERGE) to replicate through Raft.
    pub mutations: Vec<GraphMutationClause>,
    /// Optional UNWIND clause for list expansions.
    pub unwind: Option<UnwindClause>,
    /// Optional CALL { ... } subquery clauses.
    pub subqueries: Vec<CallSubqueryClause>,
}

/// UNWIND clause expanding a list expression into individual alias rows.
#[derive(Clone, Debug, PartialEq)]
pub struct UnwindClause {
    pub expression: String,
    pub alias: String,
}

/// CALL { ... } subquery clause for nested transaction isolation.
#[derive(Clone, Debug, PartialEq)]
pub struct CallSubqueryClause {
    pub subquery: String,
}

/// In-query mutation clause descriptor.
#[derive(Clone, Debug, PartialEq)]
pub enum GraphMutationClause {
    CreateNode {
        alias: String,
        labels: Vec<crate::graph::catalog::labels::LabelId>,
        properties: std::collections::HashMap<String, serde_json::Value>,
    },
    CreateRelationship {
        src_alias: String,
        dst_alias: String,
        rel_type: crate::graph::catalog::relationships::RelTypeId,
        properties: std::collections::HashMap<String, serde_json::Value>,
        weight: f32,
    },
    MergePattern {
        pattern: GraphPattern,
    },
    DeleteAlias(String),
}
