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

use std::collections::HashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

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
        self.desired_clusters.write().insert(desired.cluster_id.clone(), desired);
    }

    pub fn report_observed_state(&self, observed: ObservedClusterState) {
        self.observed_clusters.write().insert(observed.cluster_id.clone(), observed);
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
        if !observed.current_image_tag.is_empty() && desired.target_image_tag != observed.current_image_tag {
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
