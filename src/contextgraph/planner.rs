/* holosphere/src/contextgraph/planner.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Query Planner & Budget Governor
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Intelligently selects the minimal, optimal query execution plan (exact lookup, lexical,
//! semantic vector, hypergraph traversal, pathfinding, or temporal diff) to prevent unneeded compute.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// Query Budget boundaries to guarantee bounded model context consumption.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextBudget {
    pub max_results: usize,
    pub max_chars: usize,
    pub max_depth: usize,
}

impl Default for ContextBudget {
    fn default() -> Self {
        Self {
            max_results: 20,
            max_chars: 12_000,
            max_depth: 3,
        }
    }
}

/// Execution strategy selected by the planner.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryPlan {
    ExactEntityLookup,
    LexicalSearch,
    SemanticSearch,
    GraphTraversal,
    HybridSeedTraversal,
    PathSearch,
    TemporalDiff,
    ImpactTraversal,
}

/// User query request specification.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContextQueryRequest {
    pub query: Option<String>,
    pub entity_id: Option<String>,
    pub from_entity: Option<String>,
    pub to_entity: Option<String>,
    pub strategy: Option<String>,
    pub snapshot_lsn: Option<u64>,
    pub compare_lsn: Option<u64>,
    pub kinds: Option<Vec<String>>,
    pub budget: Option<ContextBudget>,
}

pub struct QueryPlanner;

impl QueryPlanner {
    /// Selects optimal plan according to query intent and arguments.
    #[must_use]
    pub fn plan(req: &ContextQueryRequest) -> QueryPlan {
        if req.compare_lsn.is_some() {
            return QueryPlan::TemporalDiff;
        }

        if let (Some(_), Some(_)) = (&req.from_entity, &req.to_entity) {
            return QueryPlan::PathSearch;
        }

        if let Some(strat) = &req.strategy {
            match strat.to_lowercase().as_str() {
                "impact" | "blast_radius" => return QueryPlan::ImpactTraversal,
                "traverse" | "graph" => return QueryPlan::GraphTraversal,
                "exact" => return QueryPlan::ExactEntityLookup,
                "path" => return QueryPlan::PathSearch,
                "diff" => return QueryPlan::TemporalDiff,
                "semantic" => return QueryPlan::SemanticSearch,
                "lexical" => return QueryPlan::LexicalSearch,
                _ => {}
            }
        }

        if let Some(_ent_id) = &req.entity_id {
            if req.query.is_none() {
                return QueryPlan::ExactEntityLookup;
            }
        }

        if let Some(q) = &req.query {
            if q.contains("::") || q.starts_with("sym_") || q.starts_with("ent_") {
                return QueryPlan::ExactEntityLookup;
            }
            if q.to_lowercase().starts_with("what breaks") || q.to_lowercase().contains("impact") {
                return QueryPlan::ImpactTraversal;
            }
            if q.to_lowercase().contains("path from") || q.to_lowercase().contains("how does") {
                return QueryPlan::PathSearch;
            }
        }

        QueryPlan::HybridSeedTraversal
    }
}
