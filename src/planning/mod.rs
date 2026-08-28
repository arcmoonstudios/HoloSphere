/* holosphere/src/planning/mod.rs */
//!▫~•◦-------------------------------‣
//! # Query Planning & Automated Index Calibration Subsystem
//!▫~•◦-------------------------------------------------------------------‣
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod affect;
pub mod autoforge;
pub mod planner;

pub use affect::{AffectiveRegime, AffectiveStateTensor8D};
pub use autoforge::{
    AutoForge, DerivedPhysicalConfig, OperatorIntent, OperatorIntentConfig, PlannerProfile,
};
pub use planner::{
    CalibratedRouteDecider, ExactScanCrossoverModel, ExecutionPlan, ExecutionProof, QueryModality,
    RetrievalContract, UniversalPlanner,
};
