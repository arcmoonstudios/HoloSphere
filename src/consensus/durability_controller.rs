/* holosphere/src/consensus/durability_controller.rs */
//!▫~•◦-------------------------------‣
//! # Adaptive Durability Batching & Latency Controller
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Measures NVMe fsync latency, outstanding WAL bytes, mutation arrival rate,
//! and replication RTT to dynamically tune group-commit batch windows and flush
//! cadences without requiring operators to manually adjust low-level physics.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::planning::autoforge::OperatorIntent;

/// Measured runtime metrics used by the Durability Controller.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct StorageTelemetry {
    pub p50_fsync_micros: u64,
    pub p99_fsync_micros: u64,
    pub mutation_arrival_rate_per_sec: u64,
    pub outstanding_wal_bytes: u64,
    pub replication_rtt_micros: u64,
}

/// Dynamic batching parameters derived by the Durability Controller.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct DurabilityBatchPlan {
    pub max_batch_size: usize,
    pub max_batch_delay_micros: u64,
    pub flush_cadence_micros: u64,
    pub pipeline_window_depth: usize,
    pub is_direct_io: bool,
}

/// Adaptive Durability Controller.
pub struct DurabilityController {
    intent: RwLock<OperatorIntent>,
    telemetry: RwLock<StorageTelemetry>,
    max_commit_sla_micros: AtomicU64,
    current_plan: RwLock<DurabilityBatchPlan>,
}

impl Default for DurabilityController {
    fn default() -> Self {
        Self::new(OperatorIntent::CertifiedExact, 20_000) // 20ms default commit SLA
    }
}

impl DurabilityController {
    pub fn new(intent: OperatorIntent, max_commit_sla_micros: u64) -> Self {
        let controller = Self {
            intent: RwLock::new(intent),
            telemetry: RwLock::new(StorageTelemetry::default()),
            max_commit_sla_micros: AtomicU64::new(max_commit_sla_micros),
            current_plan: RwLock::new(DurabilityBatchPlan {
                max_batch_size: 64,
                max_batch_delay_micros: 2000,
                flush_cadence_micros: 1000,
                pipeline_window_depth: 4,
                is_direct_io: false,
            }),
        };
        controller.recalculate_plan();
        controller
    }

    /// Updates measured telemetry from storage and network lanes.
    pub fn record_telemetry(&self, telemetry: StorageTelemetry) {
        *self.telemetry.write() = telemetry;
        self.recalculate_plan();
    }

    /// Updates operator intent and triggers batching plan adaptation.
    pub fn set_operator_intent(&self, intent: OperatorIntent) {
        *self.intent.write() = intent;
        self.recalculate_plan();
    }

    /// Recalculates batching windows and flush cadences to satisfy the SLA.
    pub fn recalculate_plan(&self) {
        let intent = self.intent.read().clone();
        let telemetry = *self.telemetry.read();
        let sla_micros = self.max_commit_sla_micros.load(Ordering::Relaxed);

        let (max_batch, max_delay, cadence, depth) = match intent {
            OperatorIntent::LatencyBudget { max_micros } => {
                let target_micros = max_micros.min(sla_micros);
                let delay = (target_micros / 4).max(200);
                (32, delay, delay / 2, 2)
            }
            OperatorIntent::ResourceBudget { .. } => {
                // High throughput, large batching
                (512, 10_000.min(sla_micros), 5_000, 8)
            }
            _ => {
                // Adaptive default based on fsync latency
                if telemetry.p99_fsync_micros > 10_000 {
                    // Slow disk: accumulate larger batches to amortize fsync
                    (256, (sla_micros / 2).max(1000), 2000, 6)
                } else if telemetry.mutation_arrival_rate_per_sec > 10_000 {
                    // High ingest rate: fast pipeline batching
                    (128, 1000, 500, 4)
                } else {
                    // Balanced
                    (64, 2000, 1000, 4)
                }
            }
        };

        *self.current_plan.write() = DurabilityBatchPlan {
            max_batch_size: max_batch,
            max_batch_delay_micros: max_delay,
            flush_cadence_micros: cadence,
            pipeline_window_depth: depth,
            is_direct_io: telemetry.outstanding_wal_bytes > 64 * 1024 * 1024,
        };
    }

    /// Returns the active batch plan.
    pub fn current_plan(&self) -> DurabilityBatchPlan {
        *self.current_plan.read()
    }

    /// Generates human-readable EXPLAIN diagnostic for the active batching strategy.
    pub fn explain_plan(&self) -> String {
        let plan = *self.current_plan.read();
        let telemetry = *self.telemetry.read();
        format!(
            "DurabilityController Plan:\n  - Max Batch Size: {}\n  - Max Batch Delay: {} µs\n  - Flush Cadence: {} µs\n  - In-Flight Replication Windows: {}\n  - Measured P99 Fsync: {} µs\n  - Ingest Rate: {} ops/sec",
            plan.max_batch_size,
            plan.max_batch_delay_micros,
            plan.flush_cadence_micros,
            plan.pipeline_window_depth,
            telemetry.p99_fsync_micros,
            telemetry.mutation_arrival_rate_per_sec
        )
    }
}
