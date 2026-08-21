/* hnsqr/src/cluster/control_plane.rs */
//!▫~•◦-------------------------------‣
//! # Declarative DBaaS Control Plane Reconciliation Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Architecturally separated DBaaS Control Plane managing organizations, projects,
//! clusters, backup policies, certificate renewals, and capacity scaling via
//! declarative idempotent state reconciliation loops.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{HNSQRError, HNSQRResult};

pub type ClusterId = String;
pub type OrganizationId = String;

/// Desired cluster configuration managed by DBaaS control plane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DesiredClusterState {
    pub cluster_id: ClusterId,
    pub org_id: OrganizationId,
    pub region: String,
    pub voting_replicas: u32,
    pub read_learners: u32,
    pub target_image_tag: String,
    pub max_memory_mb: usize,
    pub auto_backup_enabled: bool,
}

/// Observed runtime cluster state reported by regional telemetry.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ObservedClusterState {
    pub cluster_id: ClusterId,
    pub live_voting_replicas: u32,
    pub live_read_learners: u32,
    pub current_image_tag: String,
    pub leader_node_id: Option<u64>,
    pub is_healthy: bool,
    pub last_reconciled_epoch_ms: u64,
}

/// Reconciled control plane action plan.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ControlPlaneReconciliationPlan {
    pub cluster_id: ClusterId,
    pub actions_required: Vec<String>,
    pub is_converged: bool,
}

/// Declarative DBaaS Control Plane Engine.
pub struct DBaaSControlPlane {
    desired_clusters: RwLock<HashMap<ClusterId, DesiredClusterState>>,
    observed_clusters: RwLock<HashMap<ClusterId, ObservedClusterState>>,
}

impl Default for DBaaSControlPlane {
    fn default() -> Self {
        Self::new()
    }
}

impl DBaaSControlPlane {
    pub fn new() -> Self {
        Self {
            desired_clusters: RwLock::new(HashMap::new()),
            observed_clusters: RwLock::new(HashMap::new()),
        }
    }

    pub fn set_desired_state(&self, desired: DesiredClusterState) {
        self.desired_clusters
            .write()
            .insert(desired.cluster_id.clone(), desired);
    }

    pub fn report_observed_state(&self, observed: ObservedClusterState) {
        self.observed_clusters
            .write()
            .insert(observed.cluster_id.clone(), observed);
    }

    /// Reconciles desired against observed state idempotently.
    pub fn reconcile(&self, cluster_id: &str) -> HNSQRResult<ControlPlaneReconciliationPlan> {
        let desired_guard = self.desired_clusters.read();
        let desired = desired_guard.get(cluster_id).ok_or_else(|| {
            HNSQRError::NodeNotFound(format!("Desired state for cluster {cluster_id} not found"))
        })?;

        let observed_guard = self.observed_clusters.read();
        let observed = observed_guard.get(cluster_id).cloned().unwrap_or_default();

        let mut actions = Vec::new();

        // 1. Voting replicas
        if desired.voting_replicas != observed.live_voting_replicas {
            actions.push(format!(
                "Reconcile voting replicas: current {}, desired {}",
                observed.live_voting_replicas, desired.voting_replicas
            ));
        }

        // 2. Read learners
        if desired.read_learners != observed.live_read_learners {
            actions.push(format!(
                "Reconcile read learners: current {}, desired {}",
                observed.live_read_learners, desired.read_learners
            ));
        }

        // 3. Image tag rolling upgrade
        if !observed.current_image_tag.is_empty()
            && desired.target_image_tag != observed.current_image_tag
        {
            actions.push(format!(
                "Upgrade image from {} to {}",
                observed.current_image_tag, desired.target_image_tag
            ));
        }

        let is_converged = actions.is_empty();

        Ok(ControlPlaneReconciliationPlan {
            cluster_id: cluster_id.to_string(),
            actions_required: actions,
            is_converged,
        })
    }
}

/// VPC Peering and Private Link Subnet Configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VpcPeeringConfig {
    pub peering_id: String,
    pub tenant_id: String,
    pub cloud_provider: String,
    pub vpc_cidr_block: String,
    pub route_table_ids: Vec<String>,
    pub is_active: bool,
}

/// Usage-based consumption report.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TenantUsageReport {
    pub tenant_id: String,
    pub total_queries: u64,
    pub storage_gb_hours: f64,
    pub egress_gb: f64,
    pub estimated_cost_usd: f64,
}

/// DBaaS Usage-Based Billing & Metering Engine.
pub struct UsageBillingMeter {
    query_counters: RwLock<HashMap<String, u64>>,
    storage_gb: RwLock<HashMap<String, f64>>,
    egress_bytes: RwLock<HashMap<String, u64>>,
}

impl Default for UsageBillingMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl UsageBillingMeter {
    pub fn new() -> Self {
        Self {
            query_counters: RwLock::new(HashMap::new()),
            storage_gb: RwLock::new(HashMap::new()),
            egress_bytes: RwLock::new(HashMap::new()),
        }
    }

    /// Records query execution for a tenant.
    pub fn record_queries(&self, tenant_id: &str, count: u64) {
        let mut counters = self.query_counters.write();
        *counters.entry(tenant_id.to_string()).or_insert(0) += count;
    }

    /// Updates stored data volume in Gigabytes.
    pub fn update_storage(&self, tenant_id: &str, gb: f64) {
        self.storage_gb.write().insert(tenant_id.to_string(), gb);
    }

    /// Records egress transfer bytes.
    pub fn record_egress(&self, tenant_id: &str, bytes: u64) {
        let mut egress = self.egress_bytes.write();
        *egress.entry(tenant_id.to_string()).or_insert(0) += bytes;
    }

    /// Computes monthly billing summary ($0.05 / 1K queries + $0.25 / GB storage / mo).
    pub fn generate_monthly_report(
        &self,
        tenant_id: &str,
        elapsed_hours: f64,
    ) -> TenantUsageReport {
        let queries = self
            .query_counters
            .read()
            .get(tenant_id)
            .copied()
            .unwrap_or(0);
        let storage = self
            .storage_gb
            .read()
            .get(tenant_id)
            .copied()
            .unwrap_or(0.0);
        let egress = self
            .egress_bytes
            .read()
            .get(tenant_id)
            .copied()
            .unwrap_or(0);

        let storage_gb_hours = storage * elapsed_hours;
        let egress_gb = egress as f64 / (1024.0 * 1024.0 * 1024.0);

        // Standard Cloud DBaaS Pricing Model
        let query_cost = (queries as f64 / 1000.0) * 0.05;
        let storage_cost = (storage_gb_hours / 730.0) * 0.25;
        let egress_cost = egress_gb * 0.08;

        let total_cost = query_cost + storage_cost + egress_cost;

        TenantUsageReport {
            tenant_id: tenant_id.to_string(),
            total_queries: queries,
            storage_gb_hours,
            egress_gb,
            estimated_cost_usd: (total_cost * 100.0).round() / 100.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dbaas_control_plane_and_usage_metering() {
        let cp = DBaaSControlPlane::new();
        cp.set_desired_state(DesiredClusterState {
            cluster_id: "cluster-alpha".into(),
            org_id: "org-42".into(),
            region: "us-west-2".into(),
            voting_replicas: 3,
            read_learners: 2,
            target_image_tag: "v0.1.0".into(),
            max_memory_mb: 8192,
            auto_backup_enabled: true,
        });

        cp.report_observed_state(ObservedClusterState {
            cluster_id: "cluster-alpha".into(),
            live_voting_replicas: 3,
            live_read_learners: 1, // Desired 2, observed 1
            current_image_tag: "v0.1.0".into(),
            leader_node_id: Some(1),
            is_healthy: true,
            last_reconciled_epoch_ms: 1000,
        });

        let plan = cp.reconcile("cluster-alpha").unwrap();
        assert!(!plan.is_converged);
        assert_eq!(plan.actions_required.len(), 1);

        // Usage Metering
        let meter = UsageBillingMeter::new();
        meter.record_queries("tenant-enterprise", 100_000);
        meter.update_storage("tenant-enterprise", 50.0);
        meter.record_egress("tenant-enterprise", 10 * 1024 * 1024 * 1024);

        let report = meter.generate_monthly_report("tenant-enterprise", 730.0);
        assert_eq!(report.total_queries, 100_000);
        assert!(report.estimated_cost_usd > 0.0);
    }
}
