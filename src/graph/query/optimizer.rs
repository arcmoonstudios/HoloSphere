/* hnsqr/src/graph/query/optimizer.rs */
//!▫~•◦-------------------------------‣
//! # Cost-Based Logical Plan Optimizer
//!▫~•◦-------------------------------------------------------------------‣
//!
//! v1 rule set:
//!   R1. Push `Filter` predicates into `NodeScan` wherever semantics allow.
//!   R2. Reorder adjacent independent mandatory `Expand` operators by estimated fan-out.
//!   R3. Preserve user order when statistics are absent or dependency constraints prevent swapping.
//!
//! The cost model is deliberately compact and deterministic. It uses sealed-generation
//! cardinality statistics to estimate per-expand fan-out from average degree and optional
//! relationship-type selectivity. It never reorders `OptionalExpand`, because outer-join
//! semantics make those transformations non-commutative.
//!
//! Performance differences between valid orderings remain [BENCH REQUIRED].
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::graph::query::ast::Direction;
use crate::graph::query::logical::LogicalPlan;
use crate::graph::query::symbols::SymbolId;
use crate::graph::stats::cardinality::GraphCardinalityStats;

/// Applies deterministic semantic-preserving rewrites to a logical plan.
pub struct Optimizer;

impl Optimizer {
    /// Optimises the plan using available cardinality statistics.
    pub fn optimise(plan: LogicalPlan, stats: Option<&GraphCardinalityStats>) -> LogicalPlan {
        let plan = Self::push_filters_down(plan);
        Self::reorder_expands_by_cost(plan, stats)
    }

    /// R1: Push filter predicates as close to the source as possible.
    fn push_filters_down(plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Filter { input, predicates } => match *input {
                LogicalPlan::NodeScan {
                    binding,
                    label_filter,
                    predicates: mut existing,
                } => {
                    existing.extend(predicates);
                    LogicalPlan::NodeScan {
                        binding,
                        label_filter,
                        predicates: existing,
                    }
                }
                other => LogicalPlan::Filter {
                    input: Box::new(Self::push_filters_down(other)),
                    predicates,
                },
            },
            LogicalPlan::Expand {
                input,
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            } => LogicalPlan::Expand {
                input: Box::new(Self::push_filters_down(*input)),
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            },
            LogicalPlan::OptionalExpand {
                input,
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            } => LogicalPlan::OptionalExpand {
                input: Box::new(Self::push_filters_down(*input)),
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            },
            LogicalPlan::Limit { input, count } => LogicalPlan::Limit {
                input: Box::new(Self::push_filters_down(*input)),
                count,
            },
            LogicalPlan::Project {
                input,
                output_bindings,
            } => LogicalPlan::Project {
                input: Box::new(Self::push_filters_down(*input)),
                output_bindings,
            },
            other => other,
        }
    }

    /// R2: Reorder adjacent, dependency-independent mandatory expands so the lower
    /// estimated fan-out runs first. Statistics are advisory only; if they are absent,
    /// the original logical order is preserved exactly.
    fn reorder_expands_by_cost(
        plan: LogicalPlan,
        stats: Option<&GraphCardinalityStats>,
    ) -> LogicalPlan {
        let Some(stats) = stats else {
            return Self::recurse_without_reorder(plan);
        };

        match plan {
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
                let optimised_input = Self::reorder_expands_by_cost(*input, Some(stats));

                match optimised_input {
                    LogicalPlan::Expand {
                        input: lower_input,
                        src_binding: lower_src,
                        rel_binding: lower_rel,
                        dst_binding: lower_dst,
                        rel_type_filter: lower_rel_type,
                        direction: lower_direction,
                        min_hops: lower_min_hops,
                        max_hops: lower_max_hops,
                    } => {
                        let base_bindings = lower_input.output_bindings();
                        let independent = Self::contains_binding(&base_bindings, lower_src)
                            && Self::contains_binding(&base_bindings, src_binding)
                            && lower_dst != src_binding
                            && dst_binding != lower_src
                            && lower_dst != dst_binding;

                        if independent {
                            let upper_cost = Self::estimate_expand_fanout(
                                stats,
                                rel_type_filter,
                                direction,
                                min_hops,
                                max_hops,
                            );
                            let lower_cost = Self::estimate_expand_fanout(
                                stats,
                                lower_rel_type,
                                lower_direction,
                                lower_min_hops,
                                lower_max_hops,
                            );

                            if upper_cost < lower_cost {
                                let upper_first = LogicalPlan::Expand {
                                    input: lower_input,
                                    src_binding,
                                    rel_binding,
                                    dst_binding,
                                    rel_type_filter,
                                    direction,
                                    min_hops,
                                    max_hops,
                                };
                                return LogicalPlan::Expand {
                                    input: Box::new(upper_first),
                                    src_binding: lower_src,
                                    rel_binding: lower_rel,
                                    dst_binding: lower_dst,
                                    rel_type_filter: lower_rel_type,
                                    direction: lower_direction,
                                    min_hops: lower_min_hops,
                                    max_hops: lower_max_hops,
                                };
                            }
                        }

                        LogicalPlan::Expand {
                            input: Box::new(LogicalPlan::Expand {
                                input: lower_input,
                                src_binding: lower_src,
                                rel_binding: lower_rel,
                                dst_binding: lower_dst,
                                rel_type_filter: lower_rel_type,
                                direction: lower_direction,
                                min_hops: lower_min_hops,
                                max_hops: lower_max_hops,
                            }),
                            src_binding,
                            rel_binding,
                            dst_binding,
                            rel_type_filter,
                            direction,
                            min_hops,
                            max_hops,
                        }
                    }
                    other => LogicalPlan::Expand {
                        input: Box::new(other),
                        src_binding,
                        rel_binding,
                        dst_binding,
                        rel_type_filter,
                        direction,
                        min_hops,
                        max_hops,
                    },
                }
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
            } => LogicalPlan::OptionalExpand {
                input: Box::new(Self::reorder_expands_by_cost(*input, Some(stats))),
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            },
            LogicalPlan::Filter { input, predicates } => LogicalPlan::Filter {
                input: Box::new(Self::reorder_expands_by_cost(*input, Some(stats))),
                predicates,
            },
            LogicalPlan::Limit { input, count } => LogicalPlan::Limit {
                input: Box::new(Self::reorder_expands_by_cost(*input, Some(stats))),
                count,
            },
            LogicalPlan::Project {
                input,
                output_bindings,
            } => LogicalPlan::Project {
                input: Box::new(Self::reorder_expands_by_cost(*input, Some(stats))),
                output_bindings,
            },
            other => other,
        }
    }

    fn recurse_without_reorder(plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Expand {
                input,
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            } => LogicalPlan::Expand {
                input: Box::new(Self::recurse_without_reorder(*input)),
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            },
            LogicalPlan::OptionalExpand {
                input,
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            } => LogicalPlan::OptionalExpand {
                input: Box::new(Self::recurse_without_reorder(*input)),
                src_binding,
                rel_binding,
                dst_binding,
                rel_type_filter,
                direction,
                min_hops,
                max_hops,
            },
            LogicalPlan::Filter { input, predicates } => LogicalPlan::Filter {
                input: Box::new(Self::recurse_without_reorder(*input)),
                predicates,
            },
            LogicalPlan::Limit { input, count } => LogicalPlan::Limit {
                input: Box::new(Self::recurse_without_reorder(*input)),
                count,
            },
            LogicalPlan::Project {
                input,
                output_bindings,
            } => LogicalPlan::Project {
                input: Box::new(Self::recurse_without_reorder(*input)),
                output_bindings,
            },
            other => other,
        }
    }

    #[inline]
    fn contains_binding(bindings: &[SymbolId], needle: SymbolId) -> bool {
        bindings.contains(&needle)
    }

    /// Estimates result fan-out for one expansion. Relationship cardinality is converted
    /// to an average per-node degree; variable-length traversals use a bounded geometric
    /// sum over the requested hop range.
    fn estimate_expand_fanout(
        stats: &GraphCardinalityStats,
        rel_type_filter: Option<u16>,
        direction: Direction,
        min_hops: u8,
        max_hops: u8,
    ) -> f64 {
        let mut degree = match rel_type_filter {
            Some(rel_type) if stats.nodes > 0 => stats
                .relationship_cardinality
                .get(&rel_type)
                .copied()
                .unwrap_or(0) as f64
                / stats.nodes as f64,
            _ => stats.average_out_degree,
        };

        if matches!(direction, Direction::Undirected) {
            degree *= 2.0;
        }

        let min_hops = min_hops.max(1);
        let max_hops = max_hops.max(min_hops);
        let mut fanout = 0.0;
        for hop in min_hops..=max_hops {
            fanout += degree.powi(i32::from(hop));
        }
        fanout
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn stats() -> GraphCardinalityStats {
        let mut relationship_cardinality = HashMap::new();
        relationship_cardinality.insert(1, 8_000);
        relationship_cardinality.insert(2, 100);
        GraphCardinalityStats {
            nodes: 1_000,
            edges: 8_100,
            relationship_cardinality,
            average_out_degree: 8.1,
            ..GraphCardinalityStats::default()
        }
    }

    fn independent_two_expand_plan() -> LogicalPlan {
        LogicalPlan::Expand {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(LogicalPlan::NodeScan {
                    binding: SymbolId(0),
                    label_filter: None,
                    predicates: Vec::new(),
                }),
                src_binding: SymbolId(0),
                rel_binding: None,
                dst_binding: SymbolId(1),
                rel_type_filter: Some(1),
                direction: Direction::Outgoing,
                min_hops: 1,
                max_hops: 1,
            }),
            src_binding: SymbolId(0),
            rel_binding: None,
            dst_binding: SymbolId(2),
            rel_type_filter: Some(2),
            direction: Direction::Outgoing,
            min_hops: 1,
            max_hops: 1,
        }
    }

    #[test]
    fn cheaper_independent_expand_is_moved_closer_to_source() {
        let optimised = Optimizer::optimise(independent_two_expand_plan(), Some(&stats()));
        let LogicalPlan::Expand {
            input,
            rel_type_filter: outer_type,
            ..
        } = optimised
        else {
            panic!("expected outer expand");
        };
        let LogicalPlan::Expand {
            rel_type_filter: inner_type,
            ..
        } = *input
        else {
            panic!("expected inner expand");
        };

        assert_eq!(inner_type, Some(2));
        assert_eq!(outer_type, Some(1));
    }

    #[test]
    fn dependent_expand_order_is_preserved() {
        let plan = LogicalPlan::Expand {
            input: Box::new(LogicalPlan::Expand {
                input: Box::new(LogicalPlan::NodeScan {
                    binding: SymbolId(0),
                    label_filter: None,
                    predicates: Vec::new(),
                }),
                src_binding: SymbolId(0),
                rel_binding: None,
                dst_binding: SymbolId(1),
                rel_type_filter: Some(1),
                direction: Direction::Outgoing,
                min_hops: 1,
                max_hops: 1,
            }),
            src_binding: SymbolId(1),
            rel_binding: None,
            dst_binding: SymbolId(2),
            rel_type_filter: Some(2),
            direction: Direction::Outgoing,
            min_hops: 1,
            max_hops: 1,
        };

        let optimised = Optimizer::optimise(plan, Some(&stats()));
        let LogicalPlan::Expand {
            input,
            rel_type_filter: outer_type,
            ..
        } = optimised
        else {
            panic!("expected outer expand");
        };
        let LogicalPlan::Expand {
            rel_type_filter: inner_type,
            ..
        } = *input
        else {
            panic!("expected inner expand");
        };

        assert_eq!(inner_type, Some(1));
        assert_eq!(outer_type, Some(2));
    }

    #[test]
    fn no_stats_preserves_original_expand_order() {
        let optimised = Optimizer::optimise(independent_two_expand_plan(), None);
        let LogicalPlan::Expand {
            input,
            rel_type_filter: outer_type,
            ..
        } = optimised
        else {
            panic!("expected outer expand");
        };
        let LogicalPlan::Expand {
            rel_type_filter: inner_type,
            ..
        } = *input
        else {
            panic!("expected inner expand");
        };

        assert_eq!(inner_type, Some(1));
        assert_eq!(outer_type, Some(2));
    }
}
