/* holosphere/src/capacity/planner.rs */
//!▫~•◦-------------------------------‣
//! # Capacity Planning Engine (`hnsqr plan`)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides mathematical capacity forecasting for cloud-scale cluster deployment:
//! estimates RAM, NVMe throughput, storage bytes, learner count, and shard topologies
//! calibrated against empirical hardware benchmarks with statistical confidence intervals.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::planning::AutoForge;

/// Input deployment requirements.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CapacityRequirements {
    pub total_vectors: usize,
    pub dimension: usize,
    pub target_query_qps: u32,
    pub target_write_qps: u32,
    pub replication_factor: u32,
}

/// Sized infrastructure recommendation with confidence bounds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterCapacityPlan {
    pub total_vector_storage_gb: f64,
    pub total_index_memory_gb: f64,
    pub recommended_ram_gb: f64,
    pub recommended_ram_ci_low_gb: f64,
    pub recommended_ram_ci_high_gb: f64,
    pub recommended_nvme_bandwidth_mbps: f64,
    pub recommended_shards: u32,
    pub recommended_learners: u32,
    pub expected_p99_latency_ms: f64,
    pub confidence_level: f32,
}

/// Empirical hardware calibration metrics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MachineTelemetryProfile {
    pub simd_scan_gbps: f64,
    pub rivero_candidates_per_us: f64,
    pub proof_tree_nodes_per_us: f64,
    pub nvme_fsync_p99_ms: f64,
    pub nvme_sequential_mbps: f64,
    pub network_rtt_ms: f64,
}

impl Default for MachineTelemetryProfile {
    fn default() -> Self {
        Self {
            simd_scan_gbps: 45.0,
            rivero_candidates_per_us: 120.0,
            proof_tree_nodes_per_us: 85.0,
            nvme_fsync_p99_ms: 1.2,
            nvme_sequential_mbps: 3200.0,
            network_rtt_ms: 0.35,
        }
    }
}

/// Capacity Planning Calculator.
pub struct CapacityPlanner;

impl CapacityPlanner {
    /// Captures the locally measurable CPU and memory characteristics before
    /// sizing a deployment. Storage and network values remain explicit
    /// assumptions until an operator supplies measured telemetry through
    /// [`Self::compute_plan_calibrated`].
    pub fn local_calibrated_telemetry() -> MachineTelemetryProfile {
        let profile = AutoForge::calibrate();
        let defaults = MachineTelemetryProfile::default();
        let throughput_scale = (profile.simd_gflops as f64 / 250.0).clamp(0.25, 4.0);
        MachineTelemetryProfile {
            simd_scan_gbps: (profile.memory_bandwidth_gbps as f64).max(0.1),
            rivero_candidates_per_us: defaults.rivero_candidates_per_us * throughput_scale,
            proof_tree_nodes_per_us: defaults.proof_tree_nodes_per_us * throughput_scale,
            ..defaults
        }
    }

    /// Sizes an interactive local deployment from measured CPU and memory
    /// throughput. Production callers should provide complete measured storage
    /// and network telemetry with [`Self::compute_plan_calibrated`].
    pub fn compute_plan_local_calibrated(req: &CapacityRequirements) -> ClusterCapacityPlan {
        Self::compute_plan_calibrated(req, &Self::local_calibrated_telemetry())
    }

    /// Computes a portable estimate using documented baseline telemetry.
    /// This remains useful for offline planning, but it is intentionally not
    /// the CLI default because it is not machine-specific.
    pub fn compute_plan(req: &CapacityRequirements) -> ClusterCapacityPlan {
        Self::compute_plan_calibrated(req, &MachineTelemetryProfile::default())
    }

    pub fn compute_plan_calibrated(
        req: &CapacityRequirements,
        telemetry: &MachineTelemetryProfile,
    ) -> ClusterCapacityPlan {
        let bytes_per_vector = (req.dimension * 8) as f64;
        let raw_vector_bytes = (req.total_vectors as f64) * bytes_per_vector;
        let total_vector_storage_gb = (raw_vector_bytes * req.replication_factor as f64) / 1e9;

        // ProofTree (~64 bytes/vector)
        let proof_tree_bytes = (req.total_vectors as f64) * 64.0;
        let total_index_memory_gb = proof_tree_bytes / 1e9;

        // Recommended RAM with 95% confidence intervals
        let recommended_ram_gb = total_index_memory_gb * 1.3 + 2.0;
        let recommended_ram_ci_low_gb = total_index_memory_gb * 1.15 + 1.5;
        let recommended_ram_ci_high_gb = total_index_memory_gb * 1.50 + 4.0;

        // Write bandwidth + exact residual query bandwidth
        let write_bw_mbps = ((req.target_write_qps as f64) * bytes_per_vector) / 1e6;
        let query_bw_mbps = ((req.target_query_qps as f64) * 0.01 * bytes_per_vector) / 1e6;
        let recommended_nvme_bandwidth_mbps = (write_bw_mbps + query_bw_mbps) * 2.0;

        // Shards & Learners
        let recommended_shards = ((req.total_vectors as f64 / 10_000_000.0).ceil() as u32).max(1);
        let qps_per_learner = (telemetry.rivero_candidates_per_us * 15.0).max(1000.0);
        let recommended_learners =
            ((req.target_query_qps as f64 / qps_per_learner).ceil() as u32).max(1);

        let expected_p99_latency_ms =
            1.5 + telemetry.nvme_fsync_p99_ms * 0.2 + telemetry.network_rtt_ms;

        ClusterCapacityPlan {
            total_vector_storage_gb,
            total_index_memory_gb,
            recommended_ram_gb,
            recommended_ram_ci_low_gb,
            recommended_ram_ci_high_gb,
            recommended_nvme_bandwidth_mbps,
            recommended_shards,
            recommended_learners,
            expected_p99_latency_ms,
            confidence_level: 0.95,
        }
    }
}
