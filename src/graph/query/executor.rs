/* hnsqr/src/graph/query/executor.rs */
//!▫~•◦-------------------------------‣
//! # Query Executor — Morsel-Driven Physical Plan Evaluation
//!▫~•◦-------------------------------------------------------------------‣
//!
//! The executor drives a `PhysicalPlan` over a pinned `GraphReadGeneration`
//! and an optional pre-resolved vector-seed set.
//!
//! All adjacency expansion uses the generation's `AdjacencyBlock` — either
//! the mutable delta or the sealed CSR/CSC — without the caller caring which.

use std::sync::Arc;

use crate::graph::query::ast::{Direction, ScalarPredicate, PredicateValue};
use crate::graph::query::morsel::{BindingColumn, Morsel};
use crate::graph::query::physical::{PhysicalOp, PhysicalPlan};
use crate::graph::storage::generation::GraphReadGeneration;
use crate::{HNSQRResult, NodeIndex};

/// A single result row: one `NodeIndex` per output column.
pub type ResultRow = Vec<NodeIndex>;

/// Complete query result.
#[derive(Debug, Default)]
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
        self.vector_seeds.insert(param.into(), VectorSeedSet { nodes });
        self
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

            PhysicalOp::Expand { src_col, dst_col: _, rel_col, rel_type_filter, direction } => {
                if morsel.rows == 0 {
                    return Ok(morsel);
                }

                let src_nodes: Vec<NodeIndex> = (0..morsel.rows)
                    .filter(|&i| morsel.selection[i])
                    .map(|i| morsel.columns[*src_col].node_at(i))
                    .collect();

                // We need to fan-out: one src may produce multiple dst rows.
                // Build per-src neighbour lists first, then expand the morsel.
                let mut fanout_src: Vec<usize> = Vec::new(); // index into `src_nodes`
                let mut fanout_dst: Vec<NodeIndex> = Vec::new();
                let mut fanout_rel: Vec<u32> = Vec::new();

                let graph_gen = self.generation.generation.read();
                let adj2 = graph_gen.adjacency();
                for (si, &src) in src_nodes.iter().enumerate() {
                    match direction {
                        Direction::Outgoing | Direction::Undirected => {
                            adj2.expand_out(src, *rel_type_filter, |dst, _w| {
                                fanout_src.push(si);
                                fanout_dst.push(dst);
                                fanout_rel.push(0); // rel_id placeholder
                            });
                        }
                        Direction::Incoming => {
                            adj2.expand_in(src, *rel_type_filter, |dst, _w| {
                                fanout_src.push(si);
                                fanout_dst.push(dst);
                                fanout_rel.push(0);
                            });
                        }
                    }
                    if *direction == Direction::Undirected {
                        adj2.expand_in(src, *rel_type_filter, |dst, _w| {
                            fanout_src.push(si);
                            fanout_dst.push(dst);
                            fanout_rel.push(0);
                        });
                    }
                }
                drop(graph_gen);

                // Build the expanded morsel by replicating existing columns.
                let n_new = fanout_dst.len();
                let new_selection = vec![true; n_new];

                let mut new_columns: smallvec::SmallVec<[BindingColumn; 8]> =
                    smallvec::SmallVec::new();

                // Replicate all existing columns.
                for col in &morsel.columns {
                    let replicated: Vec<NodeIndex> = fanout_src
                        .iter()
                        .map(|&si| {
                            let orig_active_idx = si;
                            let mut ai = 0;
                            let mut orig_row = 0;
                            for r in 0..morsel.rows {
                                if morsel.selection[r] {
                                    if ai == orig_active_idx {
                                        orig_row = r;
                                        break;
                                    }
                                    ai += 1;
                                }
                            }
                            match col {
                                BindingColumn::Node(v) => v[orig_row],
                                BindingColumn::Relationship(v) => v[orig_row],
                                BindingColumn::Scalar(v) => v[orig_row] as NodeIndex,
                            }
                        })
                        .collect();
                    new_columns.push(BindingColumn::Node(replicated));
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
            ScalarPredicate::And(l, r) => {
                Self::eval_predicate(l, _morsel, _row)
                    && Self::eval_predicate(r, _morsel, _row)
            }
            ScalarPredicate::Or(l, r) => {
                Self::eval_predicate(l, _morsel, _row)
                    || Self::eval_predicate(r, _morsel, _row)
            }
            ScalarPredicate::Not(inner) => !Self::eval_predicate(inner, _morsel, _row),
            // Property-ref and parameter predicates are deferred to v2.
            _ => true,
        }
    }
}
