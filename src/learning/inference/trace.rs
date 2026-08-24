/* holosphere/src/learning/inference/trace.rs */
//!▫~•◦-------------------------------‣
//! # Mechanically Traversable Inference Derivation Provenance
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the exact derivation trace documenting why a candidate was generated.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::entity::id::EntityId;
use crate::experience::id::AttemptId;
use crate::learning::inference::contract::{InferenceMethodId, InferenceSeed};
use crate::relation::id::RelationId;

/// Cryptographic semantic fingerprint identifying the exact algorithm version and contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SemanticFingerprint(pub [u8; 32]);

impl SemanticFingerprint {
    pub fn compute(method_id: InferenceMethodId, method_version: u32, parameters: &[u8]) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(&method_id.0.to_le_bytes());
        hasher.update(&method_version.to_le_bytes());
        hasher.update(parameters);
        let digest = hasher.finalize();
        let mut fp = [0u8; 32];
        fp.copy_from_slice(&digest);
        Self(fp)
    }
}

/// Durable audit trace capturing the exact inputs and parameters of an inference operation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InferenceTrace {
    pub method: InferenceMethodId,
    pub method_version: u32,
    pub source_entities: Vec<EntityId>,
    pub source_relations: Vec<RelationId>,
    pub source_attempts: Vec<AttemptId>,
    pub snapshot_lsn: u64,
    pub seed: InferenceSeed,
    pub parameter_digest: [u8; 32],
}
