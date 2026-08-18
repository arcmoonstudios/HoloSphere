/* hnsqr/src/kubernetes/mod.rs */
//!▫~•◦-------------------------------‣
//! # HNSQR Cloud Native Kubernetes Operator Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides Custom Resource Definitions (CRDs) and reconciliation controllers.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod autoscaler;
pub mod operator;

pub use autoscaler::{AutoscalerMetrics, AutoscalerRecommendation, NativeAutoscaler};
pub use operator::{
    HNSQRClusterSpec, HNSQRClusterStatus, KubernetesOperator, OperatorLifecyclePhase,
};

