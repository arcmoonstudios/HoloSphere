/* hnsqr/src/security/tenant.rs */
//!▫~•◦-------------------------------‣
//! # Multi-Tenant Namespace Isolation & Resource Quota Governance
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides strict logical and physical namespace segregation across
//! tenants, collections, vector partitions, and metadata domains.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::{HNSQRError, HNSQRResult};

/// Resource quota limits for a tenant.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TenantQuota {
    pub max_vectors: usize,
    pub max_memory_bytes: usize,
    pub max_qps: u32,
}

impl Default for TenantQuota {
    fn default() -> Self {
        Self {
            max_vectors: 1_000_000,
            max_memory_bytes: 4 * 1024 * 1024 * 1024, // 4 GB
            max_qps: 1000,
        }
    }
}

/// Dynamic usage tracking for a tenant.
#[derive(Debug, Default)]
pub struct TenantUsage {
    pub current_vectors: AtomicUsize,
    pub current_memory_bytes: AtomicUsize,
    pub queries_processed: AtomicU64,
}

/// Tenant isolation namespace context.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantNamespace {
    pub tenant_id: String,
    pub collection_id: String,
}

impl TenantNamespace {
    pub fn new(tenant_id: impl Into<String>, collection_id: impl Into<String>) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            collection_id: collection_id.into(),
        }
    }

    /// Formats a canonical namespaced external ID.
    pub fn qualify_id(&self, local_id: &str) -> String {
        format!("{}:{}:{}", self.tenant_id, self.collection_id, local_id)
    }

    /// Extracts the local ID if it matches this namespace, or rejects cross-tenant IDs.
    pub fn parse_qualified_id<'a>(&self, qualified_id: &'a str) -> Option<&'a str> {
        let prefix = format!("{}:{}:", self.tenant_id, self.collection_id);
        qualified_id.strip_prefix(&prefix)
    }
}

/// Caller query context carrying namespace identity.
#[derive(Clone, Debug)]
pub struct TenantContext {
    pub namespace: TenantNamespace,
    pub auth_key_id: Option<String>,
}

/// Multi-tenant resource coordinator.
pub struct TenantManager {
    tenants: RwLock<HashMap<String, (TenantQuota, Arc<TenantUsage>)>>,
}

impl Default for TenantManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TenantManager {
    pub fn new() -> Self {
        Self {
            tenants: RwLock::new(HashMap::new()),
        }
    }

    /// Registers a tenant with custom quota.
    pub fn register_tenant(&self, tenant_id: &str, quota: TenantQuota) {
        self.tenants.write().insert(
            tenant_id.to_string(),
            (quota, Arc::new(TenantUsage::default())),
        );
    }

    /// Checks if a tenant has capacity to insert `num_vectors` and `bytes`.
    pub fn check_admission(
        &self,
        tenant_id: &str,
        num_vectors: usize,
        bytes: usize,
    ) -> HNSQRResult<()> {
        let guard = self.tenants.read();
        if let Some((quota, usage)) = guard.get(tenant_id) {
            let cur_v = usage.current_vectors.load(Ordering::Relaxed);
            let cur_m = usage.current_memory_bytes.load(Ordering::Relaxed);

            if cur_v + num_vectors > quota.max_vectors {
                return Err(HNSQRError::Internal(format!(
                    "Tenant '{tenant_id}' vector quota exceeded: limit {}",
                    quota.max_vectors
                )));
            }

            if cur_m + bytes > quota.max_memory_bytes {
                return Err(HNSQRError::Internal(format!(
                    "Tenant '{tenant_id}' memory quota exceeded: limit {} MB",
                    quota.max_memory_bytes / (1024 * 1024)
                )));
            }

            usage
                .current_vectors
                .fetch_add(num_vectors, Ordering::Relaxed);
            usage
                .current_memory_bytes
                .fetch_add(bytes, Ordering::Relaxed);
        }
        Ok(())
    }

    /// Records query execution for rate and telemetry tracking.
    pub fn record_query(&self, tenant_id: &str) {
        let guard = self.tenants.read();
        if let Some((_, usage)) = guard.get(tenant_id) {
            usage.queries_processed.fetch_add(1, Ordering::Relaxed);
        }
    }
}
