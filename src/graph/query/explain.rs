/* holosphere/src/graph/query/explain.rs */
//!▫~•◦-------------------------------‣
//! # EXPLAIN — Human-Readable Physical Plan Rendering
//!▫~•◦-------------------------------------------------------------------‣

use crate::graph::query::physical::{PhysicalOp, PhysicalPlan};

/// Human-readable explanation of a physical plan.
#[derive(Debug, Default)]
pub struct ExplainOutput {
    pub lines: Vec<String>,
}

impl ExplainOutput {
    pub fn render(plan: &PhysicalPlan) -> Self {
        let mut lines = Vec::new();
        lines.push("=== HNSQR GRAPH QUERY PLAN ===".to_string());
        for (i, op) in plan.ops.iter().enumerate() {
            lines.push(format!("  [{i}] {}", Self::fmt_op(op)));
        }
        lines.push(format!("  Output columns: {:?}", plan.output_cols));
        Self { lines }
    }

    fn fmt_op(op: &PhysicalOp) -> String {
        match op {
            PhysicalOp::NodeScan {
                binding_col,
                label_filter,
            } => format!("NodeScan(col={binding_col}, label={label_filter:?})"),
            PhysicalOp::VectorSeed {
                binding_col,
                query_param,
                k,
                contract,
            } => format!(
                "VectorSeed(col={binding_col}, param={query_param}, k={k}, contract={contract:?})"
            ),
            PhysicalOp::Expand {
                src_col,
                dst_col,
                rel_type_filter,
                direction,
                ..
            } => format!(
                "Expand(src_col={src_col}, dst_col={dst_col}, type={rel_type_filter:?}, dir={direction:?})"
            ),
            PhysicalOp::Filter { predicates } => format!("Filter({} predicates)", predicates.len()),
            PhysicalOp::Limit { count } => format!("Limit({count})"),
            PhysicalOp::Project { keep_cols } => format!("Project({keep_cols:?})"),
            PhysicalOp::ShortestPath {
                src_col,
                dst_col,
                weighted,
                ..
            } => format!("ShortestPath(src_col={src_col}, dst_col={dst_col}, weighted={weighted})"),
        }
    }

    /// Formats all lines as a single string for logging.
    pub fn to_string(&self) -> String {
        self.lines.join("\n")
    }
}
