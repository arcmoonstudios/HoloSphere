/* hnsqr/src/telemetry/metrics.rs */
//!▫~•◦-------------------------------‣
//! # Lock-Free Prometheus Metrics Exporter
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Exposes production OpenMetrics/Prometheus telemetry for retrieval stages,
//! WAL durability, metadata pressure, and cluster state.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Comprehensive lock-free engine metrics container.
#[derive(Clone, Debug, Default)]
pub struct EngineMetrics {
    pub queries_total: Arc<AtomicU64>,
    pub query_latency_micros_total: Arc<AtomicU64>,
    pub exact_simd_evaluations: Arc<AtomicU64>,
    pub proof_regions_pruned: Arc<AtomicU64>,
    pub lutz_l0_pruned: Arc<AtomicU64>,
    pub lutz_l1_pruned: Arc<AtomicU64>,
    pub wal_appends_total: Arc<AtomicU64>,
    pub wal_bytes_written: Arc<AtomicU64>,
    pub wal_fsync_micros_total: Arc<AtomicU64>,
    pub metadata_memory_bytes: Arc<AtomicUsize>,
    pub cluster_epoch: Arc<AtomicU64>,
    pub tenant_queries: Arc<RwLock<HashMap<String, u64>>>,
    pub tenant_storage_gb: Arc<RwLock<HashMap<String, f64>>>,
}

impl EngineMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Replaces the exported billing snapshot for one tenant.
    pub fn set_tenant_usage(&self, tenant: impl Into<String>, queries: u64, storage_gb: f64) {
        let tenant = tenant.into();
        self.tenant_queries.write().insert(tenant.clone(), queries);
        self.tenant_storage_gb.write().insert(tenant, storage_gb);
    }
}

/// Formatter generating Prometheus / OpenMetrics compliant text output.
pub struct PrometheusExporter;

impl PrometheusExporter {
    pub fn format(metrics: &EngineMetrics) -> String {
        let mut out = String::with_capacity(2048);

        out.push_str("# HELP hnsqr_queries_total Total search queries processed.\n");
        out.push_str("# TYPE hnsqr_queries_total counter\n");
        out.push_str(&format!(
            "hnsqr_queries_total {}\n",
            metrics.queries_total.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP hnsqr_query_latency_micros_total Accumulated query latency.\n");
        out.push_str("# TYPE hnsqr_query_latency_micros_total counter\n");
        out.push_str(&format!(
            "hnsqr_query_latency_micros_total {}\n",
            metrics.query_latency_micros_total.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP hnsqr_exact_simd_evaluations Total vectors evaluated via exact SIMD.\n",
        );
        out.push_str("# TYPE hnsqr_exact_simd_evaluations counter\n");
        out.push_str(&format!(
            "hnsqr_exact_simd_evaluations {}\n",
            metrics.exact_simd_evaluations.load(Ordering::Relaxed)
        ));

        out.push_str(
            "# HELP hnsqr_proof_regions_pruned Subtree envelopes pruned by proof bounds.\n",
        );
        out.push_str("# TYPE hnsqr_proof_regions_pruned counter\n");
        out.push_str(&format!(
            "hnsqr_proof_regions_pruned {}\n",
            metrics.proof_regions_pruned.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP hnsqr_lutz_l0_pruned Candidates pruned by LUTz L0 bound.\n");
        out.push_str("# TYPE hnsqr_lutz_l0_pruned counter\n");
        out.push_str(&format!(
            "hnsqr_lutz_l0_pruned {}\n",
            metrics.lutz_l0_pruned.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP hnsqr_wal_appends_total Total mutations appended to WAL.\n");
        out.push_str("# TYPE hnsqr_wal_appends_total counter\n");
        out.push_str(&format!(
            "hnsqr_wal_appends_total {}\n",
            metrics.wal_appends_total.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP hnsqr_wal_bytes_written Total bytes written to WAL.\n");
        out.push_str("# TYPE hnsqr_wal_bytes_written counter\n");
        out.push_str(&format!(
            "hnsqr_wal_bytes_written {}\n",
            metrics.wal_bytes_written.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP hnsqr_metadata_memory_bytes Total tracked metadata heap usage.\n");
        out.push_str("# TYPE hnsqr_metadata_memory_bytes gauge\n");
        out.push_str(&format!(
            "hnsqr_metadata_memory_bytes {}\n",
            metrics.metadata_memory_bytes.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP hnsqr_cluster_epoch Current active cluster topology epoch.\n");
        out.push_str("# TYPE hnsqr_cluster_epoch gauge\n");
        out.push_str(&format!(
            "hnsqr_cluster_epoch {}\n",
            metrics.cluster_epoch.load(Ordering::Relaxed)
        ));

        if let Some(queries) = metrics.tenant_queries.try_read() {
            out.push_str("# HELP hnsqr_tenant_queries_total Usage queries per tenant namespace.\n");
            out.push_str("# TYPE hnsqr_tenant_queries_total counter\n");
            for (tenant, count) in queries.iter() {
                out.push_str(&format!(
                    "hnsqr_tenant_queries_total{{tenant=\"{}\"}} {count}\n",
                    prometheus_label_value(tenant)
                ));
            }
        }
        if let Some(storage) = metrics.tenant_storage_gb.try_read() {
            out.push_str(
                "# HELP hnsqr_tenant_storage_gb Storage volume per tenant in gigabytes.\n",
            );
            out.push_str("# TYPE hnsqr_tenant_storage_gb gauge\n");
            for (tenant, gb) in storage.iter() {
                out.push_str(&format!(
                    "hnsqr_tenant_storage_gb{{tenant=\"{}\"}} {gb:.4}\n",
                    prometheus_label_value(tenant)
                ));
            }
        }

        out
    }
}

fn prometheus_label_value(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exports_escaped_per_tenant_usage() {
        let metrics = EngineMetrics::new();
        metrics.set_tenant_usage("tenant\"alpha", 42, 1.25);

        let output = PrometheusExporter::format(&metrics);
        assert!(output.contains("hnsqr_tenant_queries_total{tenant=\"tenant\\\"alpha\"} 42"));
        assert!(output.contains("hnsqr_tenant_storage_gb{tenant=\"tenant\\\"alpha\"} 1.2500"));
    }
}
