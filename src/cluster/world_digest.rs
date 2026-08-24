/* holosphere/src/cluster/world_digest.rs */
//!▫~•◦-------------------------------‣
//! # Canonical System-Wide World-State Digest
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the primary recovery and replication verification oracle for HoloSphere,
//! computing a deterministic, content-addressed cryptographic digest over all durable
//! semantic state at a specific committed LSN while strictly excluding rebuildable
//! physical acceleration artifacts.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Canonical, deterministic world-state digest at a committed log position $S_k$.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorldStateDigest {
    pub lsn: u64,
    pub entity_digest: [u8; 32],
    pub relation_digest: [u8; 32],
    pub experience_digest: [u8; 32],
    pub learning_digest: [u8; 32],
    pub schema_digest: [u8; 32],
    pub combined_digest: [u8; 32],
}

impl WorldStateDigest {
    /// Constructs and computes the combined world-state digest across all subsystem dimensions.
    pub fn compute(
        lsn: u64,
        entity_digest: [u8; 32],
        relation_digest: [u8; 32],
        experience_digest: [u8; 32],
        learning_digest: [u8; 32],
        schema_digest: [u8; 32],
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_WORLD_STATE_DIGEST_V1");
        hasher.update(&lsn.to_le_bytes());
        hasher.update(&entity_digest);
        hasher.update(&relation_digest);
        hasher.update(&experience_digest);
        hasher.update(&learning_digest);
        hasher.update(&schema_digest);

        let mut combined = [0u8; 32];
        combined.copy_from_slice(&hasher.finalize());

        Self {
            lsn,
            entity_digest,
            relation_digest,
            experience_digest,
            learning_digest,
            schema_digest,
            combined_digest: combined,
        }
    }
}
