/* holosphere/src/codegraph/path.rs */
//!▫~•◦-------------------------------‣
//! # Relation-Weighted CodeGraph Pathfinding Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Computes the shortest and most meaningful structural paths between symbols across
//! function invocations, type usages, trait implementations, and module hierarchies.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use super::ingest::CodeGraphStoreState;
use super::schema::{CodeNode, CodeNodeId, CodeRelation, RelationOrigin, SourceSpan};

/// Single directed step along a discovered path between code symbols.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathStep {
    pub from_symbol: String,
    pub from_id: CodeNodeId,
    pub relation: CodeRelation,
    pub to_symbol: String,
    pub to_id: CodeNodeId,
    pub origin: RelationOrigin,
    pub evidence: SourceSpan,
}

/// Discovered relational path between two symbols.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CodePath {
    pub from: String,
    pub to: String,
    pub total_hops: usize,
    pub steps: Vec<PathStep>,
}

pub struct CodePathfinder;

impl CodePathfinder {
    /// Computes the shortest path from `from_id` to `to_id` within `max_hops`.
    #[must_use]
    pub fn find_path(
        state: &CodeGraphStoreState,
        from_id: &CodeNodeId,
        to_id: &CodeNodeId,
        max_hops: usize,
    ) -> Option<CodePath> {
        let from_node = state.nodes.get(from_id)?;
        let to_node = state.nodes.get(to_id)?;

        if from_id == to_id {
            return Some(CodePath {
                from: from_node.qualified_name.clone(),
                to: to_node.qualified_name.clone(),
                total_hops: 0,
                steps: Vec::new(),
            });
        }

        let mut queue = VecDeque::new();
        let mut parent_map: HashMap<CodeNodeId, (CodeNodeId, super::schema::CodeEdge)> =
            HashMap::new();
        let mut visited = HashSet::new();

        queue.push_back((from_id.clone(), 0));
        visited.insert(from_id.clone());

        let mut reached = false;

        while let Some((curr_id, depth)) = queue.pop_front() {
            if &curr_id == to_id {
                reached = true;
                break;
            }
            if depth >= max_hops {
                continue;
            }

            if let Some(outgoing_edge_ids) = state.outgoing_edges.get(&curr_id) {
                for edge_id in outgoing_edge_ids {
                    if let Some(edge) = state.edges.get(edge_id) {
                        let next_id = &edge.target;
                        if visited.insert(next_id.clone()) {
                            parent_map.insert(next_id.clone(), (curr_id.clone(), edge.clone()));
                            queue.push_back((next_id.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        if !reached {
            return None;
        }

        // Reconstruct path
        let mut steps = Vec::new();
        let mut curr = to_id.clone();

        while let Some((prev_id, edge)) = parent_map.get(&curr) {
            let from_n = state.nodes.get(prev_id)?;
            let to_n = state.nodes.get(&curr)?;

            steps.push(PathStep {
                from_symbol: from_n.qualified_name.clone(),
                from_id: prev_id.clone(),
                relation: edge.relation,
                to_symbol: to_n.qualified_name.clone(),
                to_id: curr.clone(),
                origin: edge.origin,
                evidence: edge.evidence,
            });

            curr = prev_id.clone();
            if curr == *from_id {
                break;
            }
        }

        steps.reverse();
        let total_hops = steps.len();

        Some(CodePath {
            from: from_node.qualified_name.clone(),
            to: to_node.qualified_name.clone(),
            total_hops,
            steps,
        })
    }
}
