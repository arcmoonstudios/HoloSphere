/* holosphere/src/entity/exact/mod.rs */
//!▫~•◦-------------------------------‣
//! # E-Constrained Exact Retrieval Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides bit-exact Top-K vector scoring strictly constrained by
//! graph/epistemic/temporal eligibility sets E = {e | P(e, S_k) = true}.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod dense;
pub mod metric;
pub mod planner;
pub mod sparse;

pub use dense::masked_dense_scan;
pub use metric::{
    CosineMetric, DistanceFunction, EuclideanMetric, ExactVectorMetric, InnerProductMetric,
    ProjectiveOverlapMetric, ScoredEntity, resolve_metric,
};
pub use planner::{
    ExactEligibilityCostModel, ExactEligibilityProof, ExactRetrievalContext, ExactScanOperator,
    ExactScanPlan, exact_top_k, exact_top_k_scalar,
};
pub use sparse::sparse_gather_scan;
