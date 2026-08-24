/* holosphere/src/experience/context.rs */
//!▫~•◦-------------------------------‣
//! # Structured Execution Context & Cryptographic Fingerprinting
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the structured context dimensions and SHA-256 canonical fingerprinting.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

use crate::entity::id::ProvenanceId;
use crate::experience::id::ContextId;

/// Value representation for a single context dimension.
#[derive(Clone, Debug, PartialEq, PartialOrd, Serialize, Deserialize)]
pub enum ContextValue {
    String(Arc<str>),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

/// Key-value pair defining a single empirical dimension of an execution environment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DurableContextDimension {
    pub key: Arc<str>,
    pub value: ContextValue,
}

impl Eq for DurableContextDimension {}

impl PartialOrd for DurableContextDimension {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DurableContextDimension {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.key
            .cmp(&other.key)
            .then_with(|| match (&self.value, &other.value) {
                (ContextValue::String(a), ContextValue::String(b)) => a.cmp(b),
                (ContextValue::Integer(a), ContextValue::Integer(b)) => a.cmp(b),
                (ContextValue::Boolean(a), ContextValue::Boolean(b)) => a.cmp(b),
                (ContextValue::Float(a), ContextValue::Float(b)) => {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                }
                _ => std::cmp::Ordering::Equal,
            })
    }
}

/// Computes the SHA-256 canonical fingerprint and 64-bit fast hash of context dimensions.
pub fn compute_context_fingerprint(
    schema_version: u16,
    dimensions: &[DurableContextDimension],
) -> ([u8; 32], u64) {
    let mut sorted = dimensions.to_vec();
    sorted.sort_unstable(); // Sort by key ASC

    let mut sha = Sha256::new();
    sha.update(&schema_version.to_le_bytes());

    let mut crc = crc32fast::Hasher::new();
    crc.update(&schema_version.to_le_bytes());

    for dim in &sorted {
        sha.update(dim.key.as_bytes());
        crc.update(dim.key.as_bytes());

        match &dim.value {
            ContextValue::String(s) => {
                sha.update(&[0u8]);
                sha.update(s.as_bytes());
                crc.update(&[0u8]);
                crc.update(s.as_bytes());
            }
            ContextValue::Integer(i) => {
                sha.update(&[1u8]);
                sha.update(&i.to_le_bytes());
                crc.update(&[1u8]);
                crc.update(&i.to_le_bytes());
            }
            ContextValue::Float(f) => {
                sha.update(&[2u8]);
                sha.update(&f.to_bits().to_le_bytes());
                crc.update(&[2u8]);
                crc.update(&f.to_bits().to_le_bytes());
            }
            ContextValue::Boolean(b) => {
                sha.update(&[3u8]);
                sha.update(&[*b as u8]);
                crc.update(&[3u8]);
                crc.update(&[*b as u8]);
            }
        }
    }

    let hash_bytes: [u8; 32] = sha.finalize().into();
    let fast_hash = crc.finalize() as u64;

    (hash_bytes, fast_hash)
}

/// Structured context record representing the empirical operating environment.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextRecord {
    pub context_id: ContextId,
    pub schema_version: u16,
    pub dimensions: Vec<DurableContextDimension>,
    pub canonical_fingerprint: [u8; 32],
    pub fast_hash: u64,
    pub provenance_id: ProvenanceId,
}

impl ContextRecord {
    pub fn new(
        context_id: ContextId,
        schema_version: u16,
        mut dimensions: Vec<DurableContextDimension>,
        provenance_id: ProvenanceId,
    ) -> Self {
        dimensions.sort_unstable();
        let (canonical_fingerprint, fast_hash) =
            compute_context_fingerprint(schema_version, &dimensions);
        Self {
            context_id,
            schema_version,
            dimensions,
            canonical_fingerprint,
            fast_hash,
            provenance_id,
        }
    }
}
