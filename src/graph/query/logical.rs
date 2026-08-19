/* hnsqr/src/graph/query/logical.rs */
//!▫~•◦-------------------------------‣
//! # Logical Plan — Relational-Algebra IR
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::catalog::labels::LabelId;
use crate::graph::catalog::relationships::RelTypeId;
use crate::graph::query::ast::{Direction, ScalarPredicate, VectorContract};
use crate::graph::query::symbols::SymbolId;

/// A node in the logical plan tree.
#[derive(Clone, Debug)]
pub enum LogicalPlan {
    /// Scan all live nodes, optionally filtered by label.
    NodeScan {
        binding: SymbolId,
        label_filter: Option<LabelId>,
        predicates: Vec<ScalarPredicate>,
    },
    /// Seed from a VECTOR MATCH result set (already resolved to NodeIndex list).
    VectorSeed {
        binding: SymbolId,
        query_param: String,
        k: usize,
        contract: VectorContract,
    },
    /// Expand one hop from `src_binding` to `dst_binding`.
    Expand {
        input: Box<LogicalPlan>,
        src_binding: SymbolId,
        rel_binding: Option<SymbolId>,
        dst_binding: SymbolId,
        rel_type_filter: Option<RelTypeId>,
        direction: Direction,
        min_hops: u8,
        max_hops: u8,
    },
    /// Optional expand preserving rows with NULL when no edges match.
    OptionalExpand {
        input: Box<LogicalPlan>,
        src_binding: SymbolId,
        rel_binding: Option<SymbolId>,
        dst_binding: SymbolId,
        rel_type_filter: Option<RelTypeId>,
        direction: Direction,
        min_hops: u8,
        max_hops: u8,
    },
    /// Apply scalar predicates to rows from `input`.
    Filter {
        input: Box<LogicalPlan>,
        predicates: Vec<ScalarPredicate>,
    },
    /// Project a subset of bindings.
    Project {
        input: Box<LogicalPlan>,
        output_bindings: Vec<SymbolId>,
    },
    /// Limit row count.
    Limit {
        input: Box<LogicalPlan>,
        count: usize,
    },
}

impl LogicalPlan {
    /// Returns the set of bindings produced by this plan node.
    pub fn output_bindings(&self) -> Vec<SymbolId> {
        match self {
            Self::NodeScan { binding, .. } | Self::VectorSeed { binding, .. } => {
                vec![*binding]
            }
            Self::Expand { input, dst_binding, rel_binding, .. }
            | Self::OptionalExpand { input, dst_binding, rel_binding, .. } => {
                let mut b = input.output_bindings();
                if let Some(rb) = rel_binding {
                    b.push(*rb);
                }
                b.push(*dst_binding);
                b
            }
            Self::Filter { input, .. } | Self::Limit { input, .. } => {
                input.output_bindings()
            }
            Self::Project { output_bindings, .. } => output_bindings.clone(),
        }
    }
}
