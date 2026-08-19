/* hnsqr/src/graph/query/physical.rs */
//!▫~•◦-------------------------------‣
//! # Physical Plan — Morsel-Driven Operator Tree
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Translates a `LogicalPlan` into a sequence of physical operators that
//! work over `Morsel` batches.  Each operator is a pure function:
//! `fn(Morsel) -> Morsel` — no hidden state, no allocation after planning.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣
 

use crate::graph::catalog::labels::LabelId;
use crate::graph::catalog::relationships::RelTypeId;
use crate::graph::query::ast::{Direction, ScalarPredicate, VectorContract};
use crate::graph::query::logical::LogicalPlan;
use crate::graph::query::symbols::SymbolId;

/// A single physical operator step.
#[derive(Clone, Debug)]
pub enum PhysicalOp {
    /// Scan all live nodes in the generation, filtering by label bitmask.
    NodeScan {
        binding_col: usize,
        label_filter: Option<LabelId>,
    },
    /// Seed the morsel from a pre-resolved vector-search result.
    VectorSeed {
        binding_col: usize,
        query_param: String,
        k: usize,
        contract: VectorContract,
    },
    /// Expand one hop from `src_col` to a new `dst_col`.
    Expand {
        src_col: usize,
        dst_col: usize,
        rel_col: Option<usize>,
        rel_type_filter: Option<RelTypeId>,
        direction: Direction,
        min_hops: u8,
        max_hops: u8,
        /// When `true`, rows with zero matching edges are preserved with a NULL sentinel.
        optional: bool,
    },
    /// Shortest-path evaluation between `src_col` and `dst_col` bindings.
    ShortestPath {
        src_col: usize,
        dst_col: usize,
        out_cost_col: usize,
        weighted: bool,
    },
    /// Apply scalar predicates; deactivates rows that fail.
    Filter {
        predicates: Vec<ScalarPredicate>,
    },
    /// Remove inactive rows and truncate to `count`.
    Limit {
        count: usize,
    },
    /// Keep only the listed column indices.
    Project {
        keep_cols: Vec<usize>,
    },
}

/// An ordered sequence of physical operators plus a binding-column registry.
#[derive(Clone, Debug)]
pub struct PhysicalPlan {
    pub ops: Vec<PhysicalOp>,
    /// Maps `SymbolId` → column index within the working morsel.
    pub col_of: Vec<(SymbolId, usize)>,
    pub output_cols: Vec<usize>,
}

impl PhysicalPlan {
    /// Lowers a `LogicalPlan` into a flat `PhysicalPlan`.
    ///
    /// `col_counter` is incremented for each new binding column allocated.
    pub fn lower(plan: LogicalPlan) -> Self {
        let mut ops = Vec::new();
        let mut col_of: Vec<(SymbolId, usize)> = Vec::new();
        let mut col_counter = 0usize;

        Self::lower_node(&plan, &mut ops, &mut col_of, &mut col_counter);

        let output_cols = col_of.iter().map(|(_, c)| *c).collect();
        PhysicalPlan { ops, col_of, output_cols }
    }

    fn lower_node(
        plan: &LogicalPlan,
        ops: &mut Vec<PhysicalOp>,
        col_of: &mut Vec<(SymbolId, usize)>,
        col_counter: &mut usize,
    ) {
        match plan {
            LogicalPlan::NodeScan { binding, label_filter, predicates } => {
                let col = *col_counter;
                *col_counter += 1;
                col_of.push((*binding, col));
                ops.push(PhysicalOp::NodeScan {
                    binding_col: col,
                    label_filter: *label_filter,
                });
                if !predicates.is_empty() {
                    ops.push(PhysicalOp::Filter { predicates: predicates.clone() });
                }
            }
            LogicalPlan::VectorSeed { binding, query_param, k, contract } => {
                let col = *col_counter;
                *col_counter += 1;
                col_of.push((*binding, col));
                ops.push(PhysicalOp::VectorSeed {
                    binding_col: col,
                    query_param: query_param.clone(),
                    k: *k,
                    contract: *contract,
                });
            }
            LogicalPlan::Expand {
                input,
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            } => {
                Self::lower_node(input, ops, col_of, col_counter);

                let src_col = col_of
                    .iter()
                    .find(|(s, _)| s == src_binding)
                    .map(|(_, c)| *c)
                    .unwrap_or(0);

                let rel_col = rel_binding.map(|rb| {
                    let c = *col_counter;
                    *col_counter += 1;
                    col_of.push((rb, c));
                    c
                });

                let dst_col = *col_counter;
                *col_counter += 1;
                col_of.push((*dst_binding, dst_col));

                ops.push(PhysicalOp::Expand {
                    src_col,
                    dst_col,
                    rel_col,
                    rel_type_filter: *rel_type_filter,
                    direction: *direction,
                    min_hops: *min_hops,
                    max_hops: *max_hops,
                    optional: false,
                });
            }
            LogicalPlan::OptionalExpand {
                input,
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            } => {
                Self::lower_node(input, ops, col_of, col_counter);

                let src_col = col_of
                    .iter()
                    .find(|(s, _)| s == src_binding)
                    .map(|(_, c)| *c)
                    .unwrap_or(0);

                let rel_col = rel_binding.map(|rb| {
                    let c = *col_counter;
                    *col_counter += 1;
                    col_of.push((rb, c));
                    c
                });

                let dst_col = *col_counter;
                *col_counter += 1;
                col_of.push((*dst_binding, dst_col));

                ops.push(PhysicalOp::Expand {
                    src_col,
                    dst_col,
                    rel_col,
                    rel_type_filter: *rel_type_filter,
                    direction: *direction,
                    min_hops: *min_hops,
                    max_hops: *max_hops,
                    optional: true,
                });
            }
            LogicalPlan::Filter { input, predicates } => {
                Self::lower_node(input, ops, col_of, col_counter);
                if !predicates.is_empty() {
                    ops.push(PhysicalOp::Filter { predicates: predicates.clone() });
                }
            }
            LogicalPlan::Limit { input, count } => {
                Self::lower_node(input, ops, col_of, col_counter);
                ops.push(PhysicalOp::Limit { count: *count });
            }
            LogicalPlan::Project { input, output_bindings } => {
                Self::lower_node(input, ops, col_of, col_counter);
                let keep: Vec<usize> = output_bindings
                    .iter()
                    .filter_map(|sb| col_of.iter().find(|(s, _)| s == sb).map(|(_, c)| *c))
                    .collect();
                ops.push(PhysicalOp::Project { keep_cols: keep });
            }
        }
    }
}
