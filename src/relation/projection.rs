/* holosphere/src/relation/projection.rs */
//!▫~•◦-------------------------------‣
//! # Derived Binary CSR/CSC Graph Projection
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides derived CSR/CSC binary graph projections for relations declaring
//! a `BinaryProjectionSpec`. Canonical truth remains the N-ary hypergraph;
//! binary projections are strictly ephemeral acceleration structures.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use std::collections::HashMap;

use crate::entity::id::{EntityId, EntityIndex};
use crate::entity::provenance::ProvenanceRecord;
use crate::relation::id::{RelationId, RelationTypeId, RoleId};
use crate::relation::read::ResolvedRelationVersion;
use crate::relation::schema::{BinaryProjectionSpec, ProjectionDirection};

/// Lossless binary view of one canonical N-ary relation binding pair.
#[derive(Clone, Debug, PartialEq)]
pub struct ProjectedBinaryRelationEdge {
    pub relation_id: RelationId,
    pub relation_version_id: crate::relation::id::RelationVersionId,
    pub relation_type_id: RelationTypeId,
    pub schema_version: u16,
    pub source_role: RoleId,
    pub target_role: RoleId,
    pub source_entity_id: EntityId,
    pub target_entity_id: EntityId,
    pub valid_from_lsn: u64,
    pub valid_until_lsn: Option<u64>,
    pub epistemic_status: crate::entity::status::EpistemicStatus,
    pub lifecycle_status: crate::entity::status::LifecycleStatus,
    pub provenance: Option<ProvenanceRecord>,
}

/// Projects canonical truth into lineage-preserving binary edges. Multi-valued
/// roles produce the deterministic source × target cross-product.
pub fn project_resolved_relation(
    relation: &ResolvedRelationVersion,
    spec: BinaryProjectionSpec,
) -> Vec<ProjectedBinaryRelationEdge> {
    let mut sources: Vec<EntityId> = relation
        .bindings
        .iter()
        .filter(|binding| binding.role_id == spec.source_role)
        .map(|binding| binding.entity_id)
        .collect();
    let mut targets: Vec<EntityId> = relation
        .bindings
        .iter()
        .filter(|binding| binding.role_id == spec.target_role)
        .map(|binding| binding.entity_id)
        .collect();
    sources.sort_unstable();
    sources.dedup();
    targets.sort_unstable();
    targets.dedup();

    let mut projected = Vec::with_capacity(sources.len().saturating_mul(targets.len()));
    for source in sources {
        for &target in &targets {
            projected.push(ProjectedBinaryRelationEdge {
                relation_id: relation.relation_id,
                relation_version_id: relation.version_id,
                relation_type_id: relation.type_id,
                schema_version: relation.schema_version,
                source_role: spec.source_role,
                target_role: spec.target_role,
                source_entity_id: source,
                target_entity_id: target,
                valid_from_lsn: relation.valid_from_lsn,
                valid_until_lsn: relation.valid_until_lsn,
                epistemic_status: relation.epistemic_status,
                lifecycle_status: relation.lifecycle_status,
                provenance: relation.provenance.clone(),
            });
            if matches!(spec.direction, ProjectionDirection::Undirected) && source != target {
                projected.push(ProjectedBinaryRelationEdge {
                    relation_id: relation.relation_id,
                    relation_version_id: relation.version_id,
                    relation_type_id: relation.type_id,
                    schema_version: relation.schema_version,
                    source_role: spec.target_role,
                    target_role: spec.source_role,
                    source_entity_id: target,
                    target_entity_id: source,
                    valid_from_lsn: relation.valid_from_lsn,
                    valid_until_lsn: relation.valid_until_lsn,
                    epistemic_status: relation.epistemic_status,
                    lifecycle_status: relation.lifecycle_status,
                    provenance: relation.provenance.clone(),
                });
            }
        }
    }
    projected
}

/// Derived CSR (Compressed Sparse Row) binary graph projection for a single relation type.
#[derive(Clone, Debug, Default)]
pub struct BinaryCsrProjection {
    pub row_offsets: Vec<usize>,
    pub col_targets: Vec<EntityIndex>,
}

impl BinaryCsrProjection {
    /// Builds a CSR projection from a list of `(source, target)` directed edge pairs.
    pub fn build(num_nodes: usize, mut edges: Vec<(EntityIndex, EntityIndex)>) -> Self {
        edges.sort_unstable();
        edges.dedup();

        let mut row_offsets = vec![0usize; num_nodes + 1];
        for &(src, _) in &edges {
            if (src as usize) < num_nodes {
                row_offsets[src as usize + 1] += 1;
            }
        }

        // Prefix sum
        for i in 0..num_nodes {
            row_offsets[i + 1] += row_offsets[i];
        }

        let mut col_targets = vec![0; edges.len()];
        let mut cur_offsets = row_offsets.clone();

        for (src, dst) in edges {
            if (src as usize) < num_nodes {
                let pos = cur_offsets[src as usize];
                col_targets[pos] = dst;
                cur_offsets[src as usize] += 1;
            }
        }

        Self {
            row_offsets,
            col_targets,
        }
    }

    /// Queries outgoing neighbors for `node`.
    pub fn neighbors(&self, node: EntityIndex) -> &[EntityIndex] {
        let n = node as usize;
        if n + 1 < self.row_offsets.len() {
            let start = self.row_offsets[n];
            let end = self.row_offsets[n + 1];
            &self.col_targets[start..end]
        } else {
            &[]
        }
    }
}

/// Cache of derived binary graph projections keyed by `RelationTypeId`.
pub struct BinaryProjectionCache {
    projections: RwLock<HashMap<RelationTypeId, BinaryCsrProjection>>,
}

impl Default for BinaryProjectionCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryProjectionCache {
    pub fn new() -> Self {
        Self {
            projections: RwLock::new(HashMap::new()),
        }
    }

    pub fn insert(&self, type_id: RelationTypeId, proj: BinaryCsrProjection) {
        self.projections.write().insert(type_id, proj);
    }

    pub fn get(&self, type_id: RelationTypeId) -> Option<BinaryCsrProjection> {
        self.projections.read().get(&type_id).cloned()
    }

    pub fn clear(&self) {
        self.projections.write().clear();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::entity::id::DurableEvidenceRef;
    use crate::entity::status::{EpistemicStatus, LifecycleStatus};
    use crate::relation::id::DurableRoleBinding;

    #[test]
    fn lineage_preserving_projection_keeps_identity_and_provenance() {
        let provenance = ProvenanceRecord {
            source_uri: Arc::from("sensor://incident-7"),
            actor_id: Arc::from("collector-1"),
            extraction_method: Arc::from("direct"),
            commit_lsn: 42,
            timestamp_ms: 7,
            confidence: 1.0,
            evidence: vec![DurableEvidenceRef::Entity(11)],
            signature_hash: [9; 32],
        };
        let relation = ResolvedRelationVersion {
            relation_id: 700,
            version_id: 3,
            type_id: 77,
            schema_version: 1,
            valid_from_lsn: 42,
            valid_until_lsn: None,
            epistemic_status: EpistemicStatus::Observed,
            lifecycle_status: LifecycleStatus::Active,
            bindings: vec![
                DurableRoleBinding {
                    entity_id: 10,
                    role_id: 1,
                },
                DurableRoleBinding {
                    entity_id: 20,
                    role_id: 2,
                },
            ],
            provenance: Some(provenance.clone()),
        };

        let edges = project_resolved_relation(
            &relation,
            BinaryProjectionSpec {
                source_role: 1,
                target_role: 2,
                direction: ProjectionDirection::Directed,
            },
        );
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].relation_id, 700);
        assert_eq!(edges[0].relation_version_id, 3);
        assert_eq!(edges[0].source_entity_id, 10);
        assert_eq!(edges[0].target_entity_id, 20);
        assert_eq!(edges[0].valid_from_lsn, 42);
        assert_eq!(edges[0].provenance, Some(provenance));
    }
}
