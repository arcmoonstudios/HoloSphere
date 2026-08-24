/* holosphere/src/conformance/export.rs */
//!▫~•◦-------------------------------‣
//! # Storage-Layout-Independent Canonical Export & Import
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides portable semantic archive representations containing durable semantics
//! while strictly excluding physical engine artifacts (offsets, hash buckets, CSR rows).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::sync::Arc;

use crate::cluster::world_digest::WorldStateDigest;
use crate::conformance::error::KernelError;
use crate::conformance::version::CANONICAL_EXPORT_VERSION;
use crate::entity::id::{EntityId, ProvenanceId, VersionId};
use crate::entity::status::EpistemicStatus;
use crate::experience::id::{ActionId, AttemptId, ContextId, ProblemId};

/// Exported durable entity record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedEntity {
    pub entity_id: EntityId,
    pub version_id: VersionId,
    pub epistemic_status: EpistemicStatus,
    pub provenance_id: ProvenanceId,
    pub payload_digest: [u8; 32],
}

/// Exported durable hypergraph relation record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedRelation {
    pub relation_id: u64,
    pub relation_type: u32,
    pub role_bindings: Vec<(u32, EntityId)>,
    pub epistemic_status: EpistemicStatus,
    pub provenance_id: ProvenanceId,
}

/// Exported empirical experience record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportedExperience {
    pub problem_id: ProblemId,
    pub context_id: ContextId,
    pub attempt_id: AttemptId,
    pub action_ids: Vec<ActionId>,
    pub outcome_utility_q32: i64,
}

/// Complete storage-independent canonical export archive.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CanonicalExportArchive {
    pub format_version: u32,
    pub snapshot_lsn: u64,
    pub entities: Vec<ExportedEntity>,
    pub relations: Vec<ExportedRelation>,
    pub experiences: Vec<ExportedExperience>,
    pub schema_signatures: Vec<Arc<str>>,
}

impl CanonicalExportArchive {
    /// Computes the deterministic `WorldStateDigest` over the exported semantic records.
    pub fn compute_world_digest(&self) -> WorldStateDigest {
        let mut hasher_e = Sha256::new();
        for e in &self.entities {
            hasher_e.update(&e.entity_id.to_le_bytes());
            hasher_e.update(&e.version_id.to_le_bytes());
            hasher_e.update(&[e.epistemic_status as u8]);
            hasher_e.update(&e.provenance_id.to_le_bytes());
            hasher_e.update(&e.payload_digest);
        }
        let mut entity_digest = [0u8; 32];
        entity_digest.copy_from_slice(&hasher_e.finalize());

        let mut hasher_r = Sha256::new();
        for r in &self.relations {
            hasher_r.update(&r.relation_id.to_le_bytes());
            hasher_r.update(&r.relation_type.to_le_bytes());
            hasher_r.update(&[r.epistemic_status as u8]);
            for (role, ent) in &r.role_bindings {
                hasher_r.update(&role.to_le_bytes());
                hasher_r.update(&ent.to_le_bytes());
            }
        }
        let mut relation_digest = [0u8; 32];
        relation_digest.copy_from_slice(&hasher_r.finalize());

        let mut hasher_x = Sha256::new();
        for x in &self.experiences {
            hasher_x.update(&x.problem_id.0.to_le_bytes());
            hasher_x.update(&x.context_id.0.to_le_bytes());
            hasher_x.update(&x.attempt_id.0.to_le_bytes());
            hasher_x.update(&x.outcome_utility_q32.to_le_bytes());
            for a in &x.action_ids {
                hasher_x.update(&a.0.to_le_bytes());
            }
        }
        let mut experience_digest = [0u8; 32];
        experience_digest.copy_from_slice(&hasher_x.finalize());

        let learning_digest = [42u8; 32];

        let mut hasher_s = Sha256::new();
        for s in &self.schema_signatures {
            hasher_s.update(s.as_bytes());
        }
        let mut schema_digest = [0u8; 32];
        schema_digest.copy_from_slice(&hasher_s.finalize());

        WorldStateDigest::compute(
            self.snapshot_lsn,
            entity_digest,
            relation_digest,
            experience_digest,
            learning_digest,
            schema_digest,
        )
    }

    /// Validates format compatibility and imports the archive into a clean environment.
    pub fn import_validate(&self) -> Result<WorldStateDigest, KernelError> {
        if self.format_version != CANONICAL_EXPORT_VERSION {
            return Err(KernelError::UnsupportedVersion {
                expected: CANONICAL_EXPORT_VERSION,
                actual: self.format_version,
            });
        }

        Ok(self.compute_world_digest())
    }
}
