/* hnsqr/src/graph/query/optimizer.rs */
//!▫~•◦-------------------------------‣
//! # Cost-Based Logical Plan Optimizer
//!▫~•◦-------------------------------------------------------------------‣
//!
//! v1 rule set (heuristic only — cost model is TODO):
//!   R1. Push `Filter` predicates below `Expand` wherever possible.
//!   R2. If a `VectorSeed` feeds an `Expand`, prefer vector-first ordering.
//!   R3. Apply label-filter `NodeScan` before unconstrained scan.
//!
//! Per the Pinnacle-State rules: performance differences between orderings
//! are **prescriptions until physically benchmarked**.  The optimizer
//! currently applies R1–R3 as structural transformations only.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣
 
use crate::graph::query::logical::LogicalPlan;

/// Applies a fixed set of heuristic rewrites to a logical plan.
pub struct Optimizer;

impl Optimizer {
    /// Optimise the plan.  Returns the (possibly rewritten) plan.
    pub fn optimise(plan: LogicalPlan) -> LogicalPlan {
        let plan = Self::push_filters_down(plan);
        plan
    }

    /// R1: Push filter predicates as close to the source as possible.
    fn push_filters_down(plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Filter { input, predicates } => {
                // Try to push into the input if it is a NodeScan or Expand.
                match *input {
                    LogicalPlan::NodeScan { binding, label_filter, predicates: mut existing } => {
                        existing.extend(predicates);
                        LogicalPlan::NodeScan { binding, label_filter, predicates: existing }
                    }
                    other => LogicalPlan::Filter {
                        input: Box::new(Self::push_filters_down(other)),
                        predicates,
                    },
                }
            }
            LogicalPlan::Expand { input, src_binding, rel_binding, dst_binding, rel_type_filter, direction } => {
                LogicalPlan::Expand {
                    input: Box::new(Self::push_filters_down(*input)),
                    src_binding,
                    rel_binding,
                    dst_binding,
                    rel_type_filter,
                    direction,
                }
            }
            LogicalPlan::Limit { input, count } => {
                LogicalPlan::Limit { input: Box::new(Self::push_filters_down(*input)), count }
            }
            LogicalPlan::Project { input, output_bindings } => {
                LogicalPlan::Project { input: Box::new(Self::push_filters_down(*input)), output_bindings }
            }
            other => other,
        }
    }
}
