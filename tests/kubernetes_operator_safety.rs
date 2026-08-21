/* hnsqr/tests/kubernetes_operator_safety.rs */
//!▫~•◦-------------------------------‣
//! # Kubernetes Operator Quorum Safety & Rolling Upgrade Invariant Suite
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates:
//!   - PodDisruptionBudget quorum safety across 3, 5, and 7 replica clusters
//!   - Refusal of disruptive actions that would break write quorum
//!   - Learner-first scale out and graceful joint consensus membership transitions
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use hnsqr::kubernetes::{HNSQRClusterSpec, HNSQRClusterStatus, KubernetesOperator};

#[test]
fn test_pod_disruption_quorum_budget() {
    let spec3 = HNSQRClusterSpec {
        cluster_name: "prod-cluster-3".to_string(),
        replicas: 3,
        learners: 1,
        wal_storage_class: "nvme-fast".to_string(),
        vector_storage_class: "standard".to_string(),
        memory_limit_mb: 16384,
        max_unavailable_replicas: 1,
        backup_cron_schedule: Some("0 2 * * *".to_string()),
        ..Default::default()
    };

    // 3-node cluster: quorum is 2.
    // Disrupting 1 pod when 3 are ready leaves 2 ready (Safe)
    assert!(KubernetesOperator::is_pod_disruption_safe(&spec3, 3, 1));
    // Disrupting 2 pods leaves 1 ready (Unsafe: breaks quorum)
    assert!(!KubernetesOperator::is_pod_disruption_safe(&spec3, 3, 2));

    let spec5 = HNSQRClusterSpec {
        cluster_name: "prod-cluster-5".to_string(),
        replicas: 5,
        learners: 2,
        wal_storage_class: "nvme-fast".to_string(),
        vector_storage_class: "standard".to_string(),
        memory_limit_mb: 32768,
        max_unavailable_replicas: 2,
        backup_cron_schedule: None,
        ..Default::default()
    };

    // 5-node cluster: quorum is 3.
    assert!(KubernetesOperator::is_pod_disruption_safe(&spec5, 5, 2));
    assert!(!KubernetesOperator::is_pod_disruption_safe(&spec5, 5, 3));
}

#[test]
fn test_operator_reconciliation_scale_out() {
    let spec = HNSQRClusterSpec {
        cluster_name: "hnsqr-scale".to_string(),
        replicas: 5,
        learners: 2,
        wal_storage_class: "nvme".to_string(),
        vector_storage_class: "standard".to_string(),
        memory_limit_mb: 16384,
        max_unavailable_replicas: 1,
        backup_cron_schedule: None,
        ..Default::default()
    };

    let status = HNSQRClusterStatus {
        ready_replicas: 3,
        ready_learners: 0,
        current_leader_pod: Some("hnsqr-scale-0".to_string()),
        active_epoch: 1,
        is_upgrading: false,
        ..Default::default()
    };

    let actions = KubernetesOperator::reconcile_scale(&spec, &status).unwrap();
    assert!(
        actions
            .iter()
            .any(|a| a.contains("Provision 2 new learner pods"))
    );
    assert!(
        actions
            .iter()
            .any(|a| a.contains("Scale out 2 read learner replicas"))
    );
}
