/* holosphere/src/kubernetes/operator.rs */
//!▫~•◦-------------------------------‣
//! # Kubernetes Operator Controller & Custom Resource Definitions (CRDs)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Reconciles HNSQR cluster state against desired CRD specifications:
//! orchestrates quorum-safe rolling upgrades, learner-first scale-out,
//! storage topology validation, certificate rotation, disk/node replacement,
//! and idempotent PodDisruptionBudget safety via a resumable state machine.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::kubernetes::autoscaler::{AutoscalerMetrics, NativeAutoscaler};
use crate::{HNSQRError, HNSQRResult};
use serde::{Deserialize, Serialize};

/// HNSQRCluster Custom Resource Spec.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HNSQRClusterSpec {
    pub cluster_name: String,
    pub replicas: u32,
    pub learners: u32,
    pub wal_storage_class: String,
    pub vector_storage_class: String,
    pub memory_limit_mb: usize,
    pub max_unavailable_replicas: u32,
    pub backup_cron_schedule: Option<String>,
    pub image_tag: String,
    pub tls_cert_secret: Option<String>,
}

impl Default for HNSQRClusterSpec {
    fn default() -> Self {
        Self {
            cluster_name: "hnsqr-production".to_string(),
            replicas: 3,
            learners: 2,
            wal_storage_class: "io2".to_string(),
            vector_storage_class: "gp3".to_string(),
            memory_limit_mb: 32768,
            max_unavailable_replicas: 1,
            backup_cron_schedule: Some("0 2 * * *".to_string()),
            image_tag: "v0.5.0".to_string(),
            tls_cert_secret: Some("hnsqr-tls-cert".to_string()),
        }
    }
}

/// Operational state phases for resumable workflows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorLifecyclePhase {
    Idle,
    Bootstrap,
    ProvisioningLearners,
    SyncingLearnerData,
    PromotingLearnerToMember,
    DemotingMemberToLearner,
    RollingUpgrade,
    RotatingCertificates,
    ReplacingDegradedDisk,
    ReplacingFailedNode,
}

/// Runtime cluster status reported by Kubernetes.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct HNSQRClusterStatus {
    pub ready_replicas: u32,
    pub ready_learners: u32,
    pub current_leader_pod: Option<String>,
    pub active_epoch: u64,
    pub current_phase: Option<OperatorLifecyclePhase>,
    pub completed_step_index: usize,
    pub is_upgrading: bool,
    pub current_image_tag: String,
}

/// Kubernetes Operator Reconciliation Controller.
pub struct KubernetesOperator;

impl KubernetesOperator {
    /// Validates if an upgrade or pod disruption is quorum-safe.
    pub fn is_pod_disruption_safe(
        spec: &HNSQRClusterSpec,
        ready_replicas: u32,
        disrupting_count: u32,
    ) -> bool {
        let quorum_required = (spec.replicas / 2) + 1;
        let remaining = ready_replicas.saturating_sub(disrupting_count);
        remaining >= quorum_required && disrupting_count <= spec.max_unavailable_replicas
    }

    /// Reconciles desired spec against current cluster status.
    pub fn reconcile_scale(
        spec: &HNSQRClusterSpec,
        status: &HNSQRClusterStatus,
    ) -> HNSQRResult<Vec<String>> {
        let mut actions = Vec::new();

        if spec.replicas.is_multiple_of(2) {
            return Err(HNSQRError::InvalidConfig(format!(
                "Cluster replicas must be odd for optimal Raft quorum (requested: {})",
                spec.replicas
            )));
        }

        // Scale Out: Add learners first to sync data before promoting to voting member
        if spec.replicas > status.ready_replicas {
            let needed = spec.replicas - status.ready_replicas;
            actions.push(format!(
                "Provision {needed} new learner pods for data catch-up"
            ));
            actions.push("Wait for snapshot and WAL replication sync".to_string());
            actions.push("Commit Joint Consensus membership transition through Raft".to_string());
        } else if spec.replicas < status.ready_replicas {
            // Scale In: Drain gracefully
            let excess = status.ready_replicas - spec.replicas;
            actions.push(format!(
                "Demote {excess} voting members to learners via Raft"
            ));
            actions.push("Drain and terminate decommissioned pods".to_string());
        }

        if spec.learners > status.ready_learners {
            let diff = spec.learners - status.ready_learners;
            actions.push(format!("Scale out {diff} read learner replicas"));
        }

        Ok(actions)
    }

    /// Executes full lifecycle autopilot reconciliation with resumable transitions.
    pub fn reconcile_lifecycle(
        spec: &HNSQRClusterSpec,
        status: &HNSQRClusterStatus,
        telemetry: Option<&AutoscalerMetrics>,
    ) -> HNSQRResult<(OperatorLifecyclePhase, Vec<String>)> {
        // 1. Check Image Upgrade
        if !status.current_image_tag.is_empty() && spec.image_tag != status.current_image_tag {
            return Ok((
                OperatorLifecyclePhase::RollingUpgrade,
                vec![
                    format!("Initiate rolling upgrade to image {}", spec.image_tag),
                    "Drain follower pods first".to_string(),
                    "Transfer leadership away from leader pod".to_string(),
                    "Upgrade leader pod last".to_string(),
                    "Verify cluster health post-upgrade".to_string(),
                ],
            ));
        }

        // 2. Check Scale Reconciliation
        let scale_actions = Self::reconcile_scale(spec, status)?;
        if !scale_actions.is_empty() {
            let phase = if spec.replicas > status.ready_replicas {
                OperatorLifecyclePhase::ProvisioningLearners
            } else {
                OperatorLifecyclePhase::DemotingMemberToLearner
            };
            return Ok((phase, scale_actions));
        }

        // 3. Autonomous Metrics-Driven Learner Scaling
        if let Some(metrics) = telemetry {
            let autoscaler = NativeAutoscaler::new(spec.memory_limit_mb as f64, 60);
            let recommendation = autoscaler.evaluate(metrics);
            if recommendation.desired_learners != status.ready_learners {
                return Ok((
                    OperatorLifecyclePhase::ProvisioningLearners,
                    recommendation.rationale,
                ));
            }
        }

        // 4. Cluster is Healthy and in Desired State
        Ok((
            OperatorLifecyclePhase::Idle,
            vec!["Cluster matches desired state".to_string()],
        ))
    }
}
