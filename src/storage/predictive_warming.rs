/* holosphere/src/storage/predictive_warming.rs */
//!▫~•◦-------------------------------‣
//! # Predictive Cache Warming & Proof Telemetry Replay
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Uses ProofTree leaf traversal frequency to persist coarse heat metadata
//! and execute phased startup cache warming:
//!   1. Manifest -> 2. ProofTree -> 3. Rivero -> 4. Hottest dense leaves -> 5. Remainder.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::HNSQRResult;

/// Heat summary describing access patterns across semantic proof regions.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProofHeatMap {
    pub leaf_access_counts: HashMap<usize, u64>,
    pub last_updated_epoch_ms: u64,
}

/// Predictive cache warming coordinator.
pub struct PredictiveWarmer {
    heat_map: RwLock<ProofHeatMap>,
}

impl Default for PredictiveWarmer {
    fn default() -> Self {
        Self::new()
    }
}

impl PredictiveWarmer {
    pub fn new() -> Self {
        Self {
            heat_map: RwLock::new(ProofHeatMap::default()),
        }
    }

    /// Records access to a ProofTree leaf.
    pub fn record_proof_access(&self, leaf_idx: usize) {
        let mut guard = self.heat_map.write();
        *guard.leaf_access_counts.entry(leaf_idx).or_insert(0) += 1;
        guard.last_updated_epoch_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
    }

    /// Computes prioritized list of leaf partitions to warm on startup.
    pub fn get_warm_priority_leaves(&self, top_n_leaves: usize) -> Vec<usize> {
        let guard = self.heat_map.read();
        let mut sorted: Vec<(usize, u64)> = guard
            .leaf_access_counts
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted
            .into_iter()
            .take(top_n_leaves)
            .map(|(leaf, _)| leaf)
            .collect()
    }

    /// Executes staged startup warming and returns elapsed startup time to SLA.
    pub fn execute_staged_warming<F1, F2, F3, F4>(
        &self,
        load_manifest: F1,
        load_proof_tree: F2,
        load_rivero: F3,
        warm_hot_leaves: F4,
    ) -> HNSQRResult<f64>
    where
        F1: FnOnce() -> HNSQRResult<()>,
        F2: FnOnce() -> HNSQRResult<()>,
        F3: FnOnce() -> HNSQRResult<()>,
        F4: FnOnce(&[usize]) -> HNSQRResult<()>,
    {
        let t0 = Instant::now();

        // 1. Manifest
        load_manifest()?;
        // 2. ProofTree
        load_proof_tree()?;
        // 3. Rivero
        load_rivero()?;
        // 4. Hottest dense leaves
        let hot_leaves = self.get_warm_priority_leaves(32);
        warm_hot_leaves(&hot_leaves)?;

        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    }
}
