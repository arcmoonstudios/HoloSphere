/* hnsqr/src/capacity/mod.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Capacity Planning Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides mathematical capacity and resource sizing for production deployments.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod planner;

pub use planner::{
    CapacityPlanner, CapacityRequirements, ClusterCapacityPlan, MachineTelemetryProfile,
};
