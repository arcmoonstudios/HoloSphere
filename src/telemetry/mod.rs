/* holosphere/src/telemetry/mod.rs */
//!▫~•◦-------------------------------‣
//! # Production Observability & Prometheus Exporter
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides lock-free metrics collection, OpenMetrics/Prometheus formatted exports,
//! and enterprise readiness and liveness endpoints.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod metrics;
pub mod slo;
pub mod tracing;

pub use metrics::{EngineMetrics, PrometheusExporter};
pub use slo::{SloAlertSeverity, SloManager, SloReport, SloTargetConfig};
pub use tracing::{ExecutionSpan, SpanRecord, TraceContext};
