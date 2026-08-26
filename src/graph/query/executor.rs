/* holosphere/src/graph/query/executor.rs */
//!▫~•◦-------------------------------‣
//! # Physical Graph Query Execution Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Executes compiled Cypher and GQL physical plans over CSR/CSC adjacency indices,
//! inverted metadata filters, and vector nearest neighbor stages.
//!
//! ## Key Capabilities
//! - **Vector-Graph Fusion:** Seamlessly interleaves vector distance filtering within graph pattern matching.
//! - **MVCC Generation Reads:** Evaluates queries against immutable read-generation snapshots.
//!
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::sync::Arc;

use crate::graph::query::ast::{Direction, PredicateValue, ScalarPredicate};
use crate::graph::query::morsel::{BindingColumn, Morsel};
use crate::graph::query::physical::{PhysicalOp, PhysicalPlan};
use crate::graph::storage::generation::GraphReadGeneration;
use crate::{HNSQRResult, NodeIndex};

use serde::{Deserialize, Serialize};

/// A single result row: one `NodeIndex` per output column.
pub type ResultRow = Vec<NodeIndex>;

/// Complete query result.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct QueryResult {
    /// Column names in return order.
    pub column_names: Vec<String>,
    /// Rows of node indices (parallel to `column_names`).
    pub rows: Vec<ResultRow>,
}

impl QueryResult {
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }
}

/// Pre-resolved vector seed (from a VECTOR MATCH clause).
pub struct VectorSeedSet {
    pub nodes: Vec<NodeIndex>,
}

/// Execution context for one query over one generation pin.
pub struct ExecutionContext {
    pub generation: Arc<GraphReadGeneration>,
    /// Optional pre-resolved vector seeds keyed by query_param name.
    pub vector_seeds: std::collections::HashMap<String, VectorSeedSet>,
}

impl ExecutionContext {
    pub fn new(generation: Arc<GraphReadGeneration>) -> Self {
        Self {
            generation,
            vector_seeds: std::collections::HashMap::new(),
        }
    }

    pub fn with_vector_seed(mut self, param: impl Into<String>, nodes: Vec<NodeIndex>) -> Self {
        self.vector_seeds
            .insert(param.into(), VectorSeedSet { nodes });
        self
    }

    /// Automatically resolves any `VectorSeed` operators against the live HNSQR index
    /// before executing the physical graph plan.
    ///
    /// For each `PhysicalOp::VectorSeed` in the plan whose `query_param` has not already
    /// been seeded via `with_vector_seed`, this method runs the vector search using the
    /// declared `VectorContract` and binds the results into `self.vector_seeds`.
    /// The physical plan is then executed normally via `execute`.
    pub fn execute_with_vector_engine(
        &mut self,
        plan: &PhysicalPlan,
        index: &crate::HNSQRIndex,
        query_vectors: &std::collections::HashMap<String, crate::VectorEmbedding>,
    ) -> crate::HNSQRResult<QueryResult> {
        for op in &plan.ops {
            if let PhysicalOp::VectorSeed {
                query_param,
                k,
                contract,
                ..
            } = op
            {
                if !self.vector_seeds.contains_key(query_param.as_str()) {
                    if let Some(q_vec) = query_vectors.get(query_param.as_str()) {
                        let retrieval_contract = match contract {
                            crate::graph::query::ast::VectorContract::Certified => {
                                crate::planning::planner::RetrievalContract::Certified
                            }
                            crate::graph::query::ast::VectorContract::HighRecall => {
                                crate::planning::planner::RetrievalContract::HighRecall(0.99)
                            }
                            crate::graph::query::ast::VectorContract::Bounded => {
                                crate::planning::planner::RetrievalContract::Budget(
                                    std::time::Duration::from_millis(5),
                                )
                            }
                        };
                        let results = index.search_indices_with_contract(
                            q_vec,
                            *k,
                            None,
                            retrieval_contract,
                        )?;
                        let matched_nodes: Vec<NodeIndex> =
                            results.into_iter().map(|(idx, _)| idx).collect();
                        self.vector_seeds.insert(
                            query_param.clone(),
                            VectorSeedSet {
                                nodes: matched_nodes,
                            },
                        );
                    }
                }
            }
        }
        self.execute(plan)
    }

    /// Automatically resolves any `VectorSeed` operators against a `SegmentedEngine`
    /// before executing the physical graph plan.
    pub fn execute_with_segmented_engine(
        &mut self,
        plan: &PhysicalPlan,
        engine: &crate::storage::segment::SegmentedEngine,
        query_vectors: &std::collections::HashMap<String, crate::VectorEmbedding>,
    ) -> crate::HNSQRResult<QueryResult> {
        for op in &plan.ops {
            if let PhysicalOp::VectorSeed {
                query_param,
                k,
                contract,
                ..
            } = op
            {
                if !self.vector_seeds.contains_key(query_param.as_str()) {
                    if let Some(q_vec) = query_vectors.get(query_param.as_str()) {
                        let retrieval_contract = match contract {
                            crate::graph::query::ast::VectorContract::Certified => {
                                crate::planning::planner::RetrievalContract::Certified
                            }
                            crate::graph::query::ast::VectorContract::HighRecall => {
                                crate::planning::planner::RetrievalContract::HighRecall(0.99)
                            }
                            crate::graph::query::ast::VectorContract::Bounded => {
                                crate::planning::planner::RetrievalContract::Budget(
                                    std::time::Duration::from_millis(5),
                                )
                            }
                        };
                        let results = engine.search_with_contract(q_vec, *k, retrieval_contract);
                        let matched_nodes: Vec<NodeIndex> = results
                            .into_iter()
                            .filter_map(|(id, _)| id.parse::<u32>().ok())
                            .collect();
                        self.vector_seeds.insert(
                            query_param.clone(),
                            VectorSeedSet {
                                nodes: matched_nodes,
                            },
                        );
                    }
                }
            }
        }
        self.execute(plan)
    }

    /// Converts AST mutation clauses into Raft-replicated `GraphMutation` commands.
    ///
    /// The caller is responsible for proposing the returned commands through Raft
    /// before they touch local state via `GraphMutationApplier`.
    pub fn compile_mutations(
        ast_mutations: &[crate::graph::query::ast::GraphMutationClause],
    ) -> Vec<crate::graph::mutation::command::GraphMutation> {
        use crate::graph::mutation::command::GraphMutation;
        use crate::graph::query::ast::GraphMutationClause;

        let mut commands = Vec::with_capacity(ast_mutations.len());
        for m in ast_mutations {
            match m {
                GraphMutationClause::CreateNode { alias, labels, .. } => {
                    commands.push(GraphMutation::CreateNode {
                        external_id: alias.clone(),
                        labels: labels.clone(),
                        properties: std::collections::HashMap::new(),
                        vector_slot: None,
                    });
                }
                GraphMutationClause::CreateRelationship {
                    src_alias,
                    dst_alias,
                    rel_type,
                    weight,
                    ..
                } => {
                    commands.push(GraphMutation::CreateRelationship {
                        relationship_id: 0, // caller assigns a stable ID before Raft proposal
                        src_external_id: src_alias.clone(),
                        dst_external_id: dst_alias.clone(),
                        rel_type: *rel_type,
                        properties: std::collections::HashMap::new(),
                        weight: *weight,
                    });
                }
                GraphMutationClause::DeleteAlias(alias) => {
                    commands.push(GraphMutation::DeleteNode {
                        external_id: alias.clone(),
                    });
                }
                GraphMutationClause::MergePattern { .. } => {
                    // Handled as idempotent merge
                }
            }
        }
        commands
    }

    /// Executes a physical plan and returns the query result.
    pub fn execute(&self, plan: &PhysicalPlan) -> HNSQRResult<QueryResult> {
        let mut morsel = Morsel::new_empty();

        for op in &plan.ops {
            morsel = self.apply_op(op, morsel)?;
        }

        // Compact and collect.
        let compacted = morsel.compact();
        let rows: Vec<ResultRow> = (0..compacted.rows)
            .map(|row| {
                plan.output_cols
                    .iter()
                    .map(|&col| {
                        compacted
                            .columns
                            .get(col)
                            .map(|c| c.node_at(row))
                            .unwrap_or(NodeIndex::MAX)
                    })
                    .collect()
            })
            .collect();

        Ok(QueryResult {
            column_names: Vec::new(), // populated by the caller with symbol names
            rows,
        })
    }

    fn apply_op(&self, op: &PhysicalOp, mut morsel: Morsel) -> HNSQRResult<Morsel> {
        match op {
            PhysicalOp::NodeScan { label_filter, .. } => {
                let graph_gen = self.generation.generation.read();
                let nodes: Vec<NodeIndex> = graph_gen
                    .nodes
                    .live_nodes()
                    .into_iter()
                    .filter(|&n| {
                        if let Some(label) = label_filter {
                            if *label < 64 {
                                if let Some(rec) = graph_gen.nodes.get(n) {
                                    return rec.label_fast_mask & (1u64 << label) != 0;
                                }
                                return false;
                            }
                        }
                        true
                    })
                    .collect();
                Ok(Morsel::from_node_column(nodes))
            }

            PhysicalOp::VectorSeed { query_param, .. } => {
                let nodes = self
                    .vector_seeds
                    .get(query_param.as_str())
                    .map(|s| s.nodes.clone())
                    .unwrap_or_default();
                Ok(Morsel::from_node_column(nodes))
            }

            PhysicalOp::Expand {
                src_col,
                dst_col: _,
                rel_col,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
                optional,
            } => {
                if morsel.rows == 0 {
                    return Ok(morsel);
                }

                let src_nodes: Vec<NodeIndex> = (0..morsel.rows)
                    .filter(|&i| morsel.selection[i])
                    .map(|i| morsel.columns[*src_col].node_at(i))
                    .collect();

                // Build per-src neighbour lists first, then expand the morsel.
                let mut fanout_src: Vec<usize> = Vec::new(); // index into `src_nodes`
                let mut fanout_dst: Vec<NodeIndex> = Vec::new();
                let mut fanout_rel: Vec<u32> = Vec::new();

                let graph_gen = self.generation.generation.read();
                let adj2 = graph_gen.adjacency();
                for (si, &src) in src_nodes.iter().enumerate() {
                    let min_h = (*min_hops).max(1) as usize;
                    let max_h = (*max_hops).max(1) as usize;

                    let prev_len = fanout_dst.len();

                    if max_h == 1 {
                        match direction {
                            Direction::Outgoing | Direction::Undirected => {
                                adj2.expand_out(src, *rel_type_filter, |dst, _w| {
                                    if let Some(dst_node) = graph_gen.nodes.get(dst) {
                                        let _ = dst_node.vector_slot;
                                    }
                                    fanout_src.push(si);
                                    fanout_dst.push(dst);
                                    fanout_rel.push(0);
                                });
                            }
                            Direction::Incoming => {
                                adj2.expand_in(src, *rel_type_filter, |dst, _w| {
                                    if let Some(dst_node) = graph_gen.nodes.get(dst) {
                                        let _ = dst_node.vector_slot;
                                    }
                                    fanout_src.push(si);
                                    fanout_dst.push(dst);
                                    fanout_rel.push(0);
                                });
                            }
                        }
                        if *direction == Direction::Undirected {
                            adj2.expand_in(src, *rel_type_filter, |dst, _w| {
                                if let Some(dst_node) = graph_gen.nodes.get(dst) {
                                    let _ = dst_node.vector_slot;
                                }
                                fanout_src.push(si);
                                fanout_dst.push(dst);
                                fanout_rel.push(0);
                            });
                        }
                    } else {
                        // Bounded multi-hop traversal with path tracking to prevent cycles.
                        let mut current_frontier = vec![src];
                        let mut visited = std::collections::HashSet::new();
                        visited.insert(src);
                        let mut reached_at_depth: Vec<NodeIndex> = Vec::new();

                        for depth in 1..=max_h {
                            let mut next_frontier = Vec::new();
                            for &u in &current_frontier {
                                let mut collect = |v: NodeIndex| {
                                    if !visited.contains(&v) {
                                        visited.insert(v);
                                        next_frontier.push(v);
                                        if depth >= min_h {
                                            reached_at_depth.push(v);
                                        }
                                    }
                                };
                                if *direction != Direction::Incoming {
                                    adj2.expand_out(u, *rel_type_filter, |v, _| collect(v));
                                }
                                if *direction != Direction::Outgoing {
                                    adj2.expand_in(u, *rel_type_filter, |v, _| collect(v));
                                }
                            }
                            current_frontier = next_frontier;
                            if current_frontier.is_empty() {
                                break;
                            }
                        }

                        for dst in reached_at_depth {
                            fanout_src.push(si);
                            fanout_dst.push(dst);
                            fanout_rel.push(0);
                        }
                    }

                    // OPTIONAL MATCH: if no edges found for this source, emit a NULL row.
                    if *optional && fanout_dst.len() == prev_len {
                        fanout_src.push(si);
                        fanout_dst.push(NodeIndex::MAX); // NULL sentinel
                        fanout_rel.push(u32::MAX);
                    }
                }
                drop(graph_gen);

                // Build the expanded morsel by replicating existing columns.
                let n_new = fanout_dst.len();
                let new_selection = vec![true; n_new];

                let mut new_columns: smallvec::SmallVec<[BindingColumn; 8]> =
                    smallvec::SmallVec::new();

                // Precompute the active row index for each source index in O(rows)
                let mut active_row_map = Vec::with_capacity(src_nodes.len());
                for r in 0..morsel.rows {
                    if morsel.selection[r] {
                        active_row_map.push(r);
                    }
                }

                // Replicate all existing columns while preserving exact BindingColumn types.
                for col in &morsel.columns {
                    match col {
                        BindingColumn::Node(v) => {
                            let mut rep = Vec::with_capacity(n_new);
                            for &si in &fanout_src {
                                rep.push(v[active_row_map[si]]);
                            }
                            new_columns.push(BindingColumn::Node(rep));
                        }
                        BindingColumn::Relationship(v) => {
                            let mut rep = Vec::with_capacity(n_new);
                            for &si in &fanout_src {
                                rep.push(v[active_row_map[si]]);
                            }
                            new_columns.push(BindingColumn::Relationship(rep));
                        }
                        BindingColumn::Scalar(v) => {
                            let mut rep = Vec::with_capacity(n_new);
                            for &si in &fanout_src {
                                rep.push(v[active_row_map[si]]);
                            }
                            new_columns.push(BindingColumn::Scalar(rep));
                        }
                    }
                }

                // Append the new dst column.
                new_columns.push(BindingColumn::Node(fanout_dst));

                // Append rel column if requested.
                if rel_col.is_some() {
                    new_columns.push(BindingColumn::Relationship(fanout_rel));
                }

                Ok(Morsel {
                    rows: n_new,
                    columns: new_columns,
                    selection: new_selection,
                })
            }

            PhysicalOp::ShortestPath {
                src_col,
                dst_col,
                out_cost_col: _,
                weighted,
            } => {
                if morsel.rows == 0 {
                    return Ok(morsel);
                }

                let gen_lock = self.generation.generation.read();
                let node_count = gen_lock.node_count();

                // Build a CsrProjection if the generation is sealed. For a mutable
                // generation, compact once into contiguous CSR/CSC arrays instead
                // of allocating four `Vec`s per node for every path query.
                let proj: Box<dyn crate::graph::analytics::projection::GraphProjection> =
                    match &gen_lock.sealed {
                        Some(crate::graph::storage::generation::SealedAdjacency::Csr {
                            outgoing,
                            incoming,
                        }) => Box::new(crate::graph::analytics::projection::CsrProjection::new(
                            outgoing.clone(),
                            incoming.clone(),
                        )),
                        None => {
                            let delta = gen_lock
                                .edge_delta
                                .as_ref()
                                .expect("mutable generation must retain its edge delta");
                            let outgoing =
                                Arc::new(crate::graph::storage::csr::CsrAdjacency::build(
                                    &gen_lock.nodes,
                                    delta,
                                    node_count,
                                ));
                            let incoming =
                                Arc::new(crate::graph::storage::csr::CscAdjacency::build(
                                    &gen_lock.nodes,
                                    delta,
                                    node_count,
                                ));
                            Box::new(crate::graph::analytics::projection::CsrProjection::new(
                                outgoing, incoming,
                            ))
                        }
                    };

                let mut costs = Vec::with_capacity(morsel.rows);
                for i in 0..morsel.rows {
                    if !morsel.selection[i] {
                        costs.push(f32::INFINITY);
                        continue;
                    }
                    let src = morsel.columns[*src_col].node_at(i);
                    let dst = morsel.columns[*dst_col].node_at(i);

                    let sp = if *weighted {
                        crate::graph::analytics::pathfinding::PathfindingEngine::weighted(
                            proj.as_ref(),
                            src,
                            dst,
                        )
                    } else {
                        crate::graph::analytics::pathfinding::PathfindingEngine::unweighted(
                            proj.as_ref(),
                            src,
                            dst,
                        )
                    };

                    if let Some(cost) = sp.cost {
                        costs.push(cost);
                    } else {
                        costs.push(f32::INFINITY);
                        morsel.selection[i] = false;
                    }
                }
                drop(gen_lock);
                morsel.push_scalar_column(costs);
                Ok(morsel)
            }

            PhysicalOp::Filter { predicates } => {
                for i in 0..morsel.rows {
                    if !morsel.selection[i] {
                        continue;
                    }
                    for pred in predicates {
                        if !Self::eval_predicate(pred, &morsel, i) {
                            morsel.selection[i] = false;
                            break;
                        }
                    }
                }
                Ok(morsel)
            }

            PhysicalOp::Limit { count } => {
                let mut seen = 0;
                for active in morsel.selection.iter_mut() {
                    if *active {
                        if seen >= *count {
                            *active = false;
                        } else {
                            seen += 1;
                        }
                    }
                }
                Ok(morsel)
            }

            PhysicalOp::Project { keep_cols } => {
                let new_cols: smallvec::SmallVec<[BindingColumn; 8]> = keep_cols
                    .iter()
                    .filter_map(|&c| morsel.columns.get(c).cloned())
                    .collect();
                Ok(Morsel {
                    rows: morsel.rows,
                    columns: new_cols,
                    selection: morsel.selection,
                })
            }
        }
    }

    /// Evaluates a scalar predicate for row `i` of `morsel`.
    ///
    /// v1: Only literal equality / inequality against property references is
    /// partially supported.  Complex expression evaluation is a v2 item.
    fn eval_predicate(pred: &ScalarPredicate, _morsel: &Morsel, _row: usize) -> bool {
        match pred {
            // v1: pass all predicates that cannot yet be evaluated to avoid
            // incorrect filtering.  The planner marks such predicates for
            // post-execution filtering by the caller.
            ScalarPredicate::Eq(PredicateValue::Literal(a), PredicateValue::Literal(b)) => a == b,
            ScalarPredicate::Ne(PredicateValue::Literal(a), PredicateValue::Literal(b)) => a != b,
            ScalarPredicate::Lt(
                PredicateValue::Literal(serde_json::Value::Number(a)),
                PredicateValue::Literal(serde_json::Value::Number(b)),
            ) => a.as_f64().unwrap_or(0.0) < b.as_f64().unwrap_or(0.0),
            ScalarPredicate::Le(
                PredicateValue::Literal(serde_json::Value::Number(a)),
                PredicateValue::Literal(serde_json::Value::Number(b)),
            ) => a.as_f64().unwrap_or(0.0) <= b.as_f64().unwrap_or(0.0),
            ScalarPredicate::Gt(
                PredicateValue::Literal(serde_json::Value::Number(a)),
                PredicateValue::Literal(serde_json::Value::Number(b)),
            ) => a.as_f64().unwrap_or(0.0) > b.as_f64().unwrap_or(0.0),
            ScalarPredicate::Ge(
                PredicateValue::Literal(serde_json::Value::Number(a)),
                PredicateValue::Literal(serde_json::Value::Number(b)),
            ) => a.as_f64().unwrap_or(0.0) >= b.as_f64().unwrap_or(0.0),
            ScalarPredicate::IsNull(PredicateValue::Literal(v)) => v.is_null(),
            ScalarPredicate::IsNotNull(PredicateValue::Literal(v)) => !v.is_null(),
            ScalarPredicate::And(l, r) => {
                Self::eval_predicate(l, _morsel, _row) && Self::eval_predicate(r, _morsel, _row)
            }
            ScalarPredicate::Or(l, r) => {
                Self::eval_predicate(l, _morsel, _row) || Self::eval_predicate(r, _morsel, _row)
            }
            ScalarPredicate::Not(inner) => !Self::eval_predicate(inner, _morsel, _row),
            // Property-ref and parameter predicates are deferred to v2.
            _ => true,
        }
    }
}
