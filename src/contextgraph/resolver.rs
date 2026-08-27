/* holosphere/src/contextgraph/resolver.rs */
//!▫~•◦-------------------------------‣
//! # Universal Reference Resolver & Ambiguity Preservation Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Statically resolves cross-entity references (code symbols, document citations,
//! artifact IDs, identity references) and preserves explicit ambiguity without guessing.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap};

use super::ir::UnresolvedReference;
use super::schema::{
    Entity, EntityId, Namespace, Relation, RelationId, RelationKind, RelationOrigin,
    RelationParticipant,
};
use crate::transport::model_gateway::VerificationState;

/// Universal Reference Resolver trait.
pub trait ReferenceResolver: Send + Sync {
    fn resolve(
        &self,
        reference: &UnresolvedReference,
        entities_by_id: &BTreeMap<EntityId, Entity>,
        entities_by_label: &HashMap<String, Vec<EntityId>>,
    ) -> Vec<Relation>;
}

/// General-purpose universal entity and symbol resolver.
pub struct UniversalReferenceResolver;

impl Default for UniversalReferenceResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl UniversalReferenceResolver {
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ReferenceResolver for UniversalReferenceResolver {
    fn resolve(
        &self,
        reference: &UnresolvedReference,
        entities_by_id: &BTreeMap<EntityId, Entity>,
        entities_by_label: &HashMap<String, Vec<EntityId>>,
    ) -> Vec<Relation> {
        let target_query = reference.target_ref.trim();
        let mut candidate_ids = Vec::new();

        // 1. Direct label lookup
        if let Some(ids) = entities_by_label.get(target_query) {
            candidate_ids.extend(ids.clone());
        }

        // 2. Suffix / Sub-path match (e.g. `HNSQRIndex::search` in `src/vector/index.rs::HNSQRIndex::search`)
        if candidate_ids.is_empty() {
            for (label, ids) in entities_by_label {
                if label.ends_with(target_query) || target_query.ends_with(label) {
                    candidate_ids.extend(ids.clone());
                }
            }
        }

        // Filter by expected kind if specified
        if let Some(expected) = &reference.expected_kind {
            candidate_ids.retain(|id| {
                entities_by_id
                    .get(id)
                    .map_or(false, |e| e.kind.as_str() == expected.as_str())
            });
        }

        candidate_ids.sort();
        candidate_ids.dedup();

        if candidate_ids.is_empty() {
            return Vec::new();
        }

        let origin = if candidate_ids.len() == 1 {
            RelationOrigin::Resolved
        } else {
            RelationOrigin::Ambiguous
        };
        let confidence = origin.default_confidence();

        let source_id = EntityId(reference.source_temp_id.clone());

        candidate_ids
            .into_iter()
            .map(|target_id| {
                let participants = vec![
                    RelationParticipant::new(source_id.clone(), "source"),
                    RelationParticipant::new(target_id, reference.role.clone()),
                ];
                let id = RelationId::compute(&reference.relation_kind, &participants, origin);
                Relation {
                    id,
                    kind: reference.relation_kind.clone(),
                    participants,
                    origin,
                    confidence,
                    provenance: Vec::new(),
                    verification_state: VerificationState::Verified,
                    attributes: BTreeMap::new(),
                }
            })
            .collect()
    }
}
