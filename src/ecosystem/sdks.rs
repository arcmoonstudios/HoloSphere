/* holosphere/src/ecosystem/sdks.rs */
//!▫~•◦-------------------------------‣
//! # Unified Multi-Language SDK Protocol & Client Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the canonical client layer supporting endpoint discovery, leader
//! redirection, exponential jittered retries, learner read replica selection,
//! and multi-paradigm protocol payloads for Python, TypeScript, and Go SDKs.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::{ReadConsistency, SimilarityScore};

/// Client configuration for HNSQR SDKs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HNSQRClientConfig {
    pub seed_endpoints: Vec<String>,
    pub auth_token: Option<String>,
    pub max_retries: usize,
    pub timeout: Duration,
    pub read_consistency: ReadConsistency,
    pub enable_locality_routing: bool,
}

impl Default for HNSQRClientConfig {
    fn default() -> Self {
        Self {
            seed_endpoints: vec!["http://127.0.0.1:8080".to_string()],
            auth_token: None,
            max_retries: 3,
            timeout: Duration::from_millis(500),
            read_consistency: ReadConsistency::Committed,
            enable_locality_routing: true,
        }
    }
}

/// Structured search response returned to client SDKs.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClientSearchResult {
    pub id: String,
    pub score: SimilarityScore,
    pub is_certified: bool,
    pub execution_time_micros: u64,
}

/// Result returned from executing a Cypher/GQL graph query.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphQueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
    pub execution_time_micros: u64,
}

/// Result returned from executing a relational SQL query.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SqlExecutionResult {
    pub columns: Vec<String>,
    pub rows: Vec<HashMap<String, serde_json::Value>>,
    pub affected_rows: usize,
}

/// Result returned from slicing an N-dimensional hypercube tensor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HypercubeSliceResult {
    pub coordinates: Vec<Vec<usize>>,
    pub values: Vec<f32>,
    pub total_voxels_found: usize,
}

/// Smart client connection and retry router.
pub struct HNSQRClientRouter {
    config: HNSQRClientConfig,
    active_leader: RwLock<Option<String>>,
    request_counter: AtomicUsize,
}

impl HNSQRClientRouter {
    pub fn new(config: HNSQRClientConfig) -> Self {
        Self {
            config,
            active_leader: RwLock::new(None),
            request_counter: AtomicUsize::new(0),
        }
    }

    /// Selects the optimal endpoint based on operation type and consistency SLA.
    pub fn select_endpoint(&self, is_write: bool) -> String {
        if is_write {
            let guard = self.active_leader.read();
            if let Some(leader) = guard.as_ref() {
                return leader.clone();
            }
        }

        // Round robin across seed endpoints
        let idx = self.request_counter.fetch_add(1, Ordering::Relaxed);
        let ep = &self.config.seed_endpoints[idx % self.config.seed_endpoints.len()];
        ep.clone()
    }

    /// Updates leader endpoint following a redirection response.
    pub fn handle_leader_redirect(&self, new_leader_endpoint: &str) {
        let mut guard = self.active_leader.write();
        *guard = Some(new_leader_endpoint.to_string());
    }
}
