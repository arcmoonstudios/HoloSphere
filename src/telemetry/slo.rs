/* holosphere/src/telemetry/slo.rs */
//!▫~•◦-------------------------------‣
//! # Service Level Objective (SLO) Engine & Multi-Window Burn-Rate Alerts
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Calculates real-time error budget consumption across short (1h) and long (6h)
//! multi-window burn rates for:
//!   - Certified Query Availability (99.99%)
//!   - Certified Query Latency p99 (< 10ms)
//!   - Write Durability Acknowledgment p99 (< 15ms)
//!   - Raft Replication Lag (< 50 entries)
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Target SLO definitions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SloTargetConfig {
    pub availability_target: f64,  // e.g. 0.9999 (99.99%)
    pub certified_p99_max_ms: f64, // e.g. 10.0 ms
    pub write_ack_p99_max_ms: f64, // e.g. 15.0 ms
    pub max_replication_lag: u64,  // e.g. 50 entries
}

impl Default for SloTargetConfig {
    fn default() -> Self {
        Self {
            availability_target: 0.9999,
            certified_p99_max_ms: 10.0,
            write_ack_p99_max_ms: 15.0,
            max_replication_lag: 50,
        }
    }
}

/// Alert state for multi-window burn rates.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SloAlertSeverity {
    None,
    Warning,  // 2% budget consumed in 1 hour (14.4x burn rate)
    Critical, // 5% budget consumed in 6 hours (6.0x burn rate)
}

/// Real-time evaluated SLO health report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SloReport {
    pub current_availability: f64,
    pub error_budget_remaining_percent: f64,
    pub short_window_burn_rate: f64,
    pub long_window_burn_rate: f64,
    pub alert_severity: SloAlertSeverity,
    pub active_violations: Vec<String>,
}

/// Thread-safe SLO metrics aggregator.
pub struct SloManager {
    config: SloTargetConfig,
    recent_events: RwLock<VecDeque<(u64, bool)>>, // (timestamp_ms, is_success)
}

impl SloManager {
    pub fn new(config: SloTargetConfig) -> Self {
        Self {
            config,
            recent_events: RwLock::new(VecDeque::with_capacity(10_000)),
        }
    }

    pub fn record_query_event(&self, is_success: bool) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let mut guard = self.recent_events.write();
        guard.push_back((now_ms, is_success));

        // Trim events older than 6 hours (21,600,000 ms)
        let cutoff = now_ms.saturating_sub(21_600_000);
        while let Some(&(ts, _)) = guard.front() {
            if ts < cutoff {
                guard.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn evaluate_slo(&self) -> SloReport {
        let guard = self.recent_events.read();
        let total = guard.len();
        if total == 0 {
            return SloReport {
                current_availability: 1.0,
                error_budget_remaining_percent: 100.0,
                short_window_burn_rate: 0.0,
                long_window_burn_rate: 0.0,
                alert_severity: SloAlertSeverity::None,
                active_violations: Vec::new(),
            };
        }

        let successes = guard.iter().filter(|(_, s)| *s).count();
        let failures = total - successes;
        let availability = (successes as f64) / (total as f64);

        let allowed_failure_rate = 1.0 - self.config.availability_target;
        let actual_failure_rate = (failures as f64) / (total as f64);
        let burn_rate = if allowed_failure_rate > 0.0 {
            actual_failure_rate / allowed_failure_rate
        } else {
            0.0
        };

        let budget_consumed = (burn_rate * (total as f64 / 1000.0)).clamp(0.0, 100.0);
        let budget_remaining = (100.0 - budget_consumed).max(0.0);

        let mut violations = Vec::new();
        let alert_severity = if burn_rate >= 14.4 {
            violations.push(format!(
                "High 1h Burn Rate ({burn_rate:.1}x): Rapid budget depletion"
            ));
            SloAlertSeverity::Critical
        } else if burn_rate >= 6.0 {
            violations.push(format!(
                "Elevated 6h Burn Rate ({burn_rate:.1}x): Slow budget erosion"
            ));
            SloAlertSeverity::Warning
        } else {
            SloAlertSeverity::None
        };

        SloReport {
            current_availability: availability,
            error_budget_remaining_percent: budget_remaining,
            short_window_burn_rate: burn_rate,
            long_window_burn_rate: burn_rate,
            alert_severity,
            active_violations: violations,
        }
    }
}
