/* holosphere/src/entity/context.rs */
//!▫~•◦-------------------------------‣
//! # Context Records and Structural Fingerprints
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides structured, comparable representations of hardware, workload,
//! software generation, and constraints for contextual reinforcement.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// 64-bit structural acceleration fingerprint of an execution context.
pub type ContextSignature = u64;

/// Structured record describing the environment in which an action was attempted.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextRecord {
    pub context_id: u64,
    pub hardware_class: String,
    pub workload_profile: String,
    pub topology: String,
    pub software_generation: u32,
    pub constraints: Vec<String>,
    pub signature: ContextSignature,
}

impl ContextRecord {
    /// Computes a deterministic 64-bit structural fingerprint for fast filtering.
    pub fn compute_signature(
        hardware_class: &str,
        workload_profile: &str,
        topology: &str,
        software_generation: u32,
    ) -> ContextSignature {
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(hardware_class.as_bytes());
        hasher.update(workload_profile.as_bytes());
        let h1 = hasher.finalize() as u64;

        let mut hasher2 = crc32fast::Hasher::new();
        hasher2.update(topology.as_bytes());
        hasher2.update(&software_generation.to_le_bytes());
        let h2 = hasher2.finalize() as u64;

        (h1 << 32) | h2
    }
}
