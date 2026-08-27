/* holosphere/src/contextgraph/fingerprint.rs */
//!▫~•◦-------------------------------‣
//! # Deterministic Canonical Graph Fingerprint Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Produces cryptographic fingerprints of ContextGraph snapshots guaranteeing bit-exact
//! reproducibility across full vs incremental compilations and single vs multi-worker executions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use sha2::{Digest, Sha256};

use super::schema::{Entity, Relation};

pub struct GraphFingerprinter;

impl GraphFingerprinter {
    /// Computes canonical fingerprint of sorted entities and relations.
    #[must_use]
    pub fn compute_fingerprint(entities: &[Entity], relations: &[Relation]) -> [u8; 32] {
        let mut sorted_entities: Vec<&Entity> = entities.iter().collect();
        sorted_entities.sort_by(|a, b| a.id.cmp(&b.id));

        let mut sorted_relations: Vec<&Relation> = relations.iter().collect();
        sorted_relations.sort_by(|a, b| a.id.cmp(&b.id));

        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_CANONICAL_GRAPH_V1:\n");

        for entity in sorted_entities {
            hasher.update(b"ENTITY:");
            hasher.update(entity.id.as_str().as_bytes());
            hasher.update(b"|");
            hasher.update(entity.kind.as_str().as_bytes());
            hasher.update(b"|");
            hasher.update(entity.label.as_bytes());
            hasher.update(b"|");
            hasher.update(&entity.fingerprint);
            hasher.update(b"\n");
        }

        for relation in sorted_relations {
            hasher.update(b"RELATION:");
            hasher.update(relation.id.as_str().as_bytes());
            hasher.update(b"|");
            hasher.update(relation.kind.as_str().as_bytes());
            hasher.update(b"|");
            hasher.update(relation.origin.as_str().as_bytes());
            hasher.update(b"\n");
        }

        hasher.finalize().into()
    }
}
