/* holosphere/src/relation/query.rs */
//!▫~•◦-------------------------------‣
//! # Hypergraph Relation Query & Pattern Matching
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the physical query execution engine for N-ary relations,
//! utilizing incidence postings and temporal filters.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::entity::id::EntityId;
use crate::entity::read::EntityReadSnapshot;
use crate::entity::status::EpistemicStatus;
use crate::relation::id::{RelationId, RelationTypeId, RoleId};
use crate::relation::read::{RelationReadSnapshot, ResolvedRelationVersion};
use crate::relation::schema::SchemaScope;
use thiserror::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyperPatternSemantics {
    Symmetric,
    RoleAware,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperPatternMember {
    pub alias: String,
    pub role_id: Option<RoleId>,
    pub entity_id: Option<EntityId>,
}

/// Validated N-ary pattern. Two members represent the degenerate binary case.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperPattern {
    pub type_id: Option<RelationTypeId>,
    pub members: Vec<HyperPatternMember>,
    pub semantics: HyperPatternSemantics,
    pub epistemic_in: Option<Vec<EpistemicStatus>>,
    pub scope: SchemaScope,
    pub as_of_lsn: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HyperPatternMatch {
    pub relation: ResolvedRelationVersion,
    pub bindings: Vec<(String, crate::relation::id::DurableRoleBinding)>,
}

#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum HyperPatternError {
    #[error("A hyperpattern requires at least two members")]
    ArityBelowTwo,
    #[error("Duplicate hyperpattern alias {0}")]
    DuplicateAlias(String),
    #[error("Symmetric hyperpattern member {0} must not declare a role")]
    RoleOnSymmetricMember(String),
    #[error("Role-aware hyperpattern member {0} must declare a role")]
    MissingRole(String),
}

impl HyperPattern {
    pub fn validate(&self) -> Result<(), HyperPatternError> {
        if self.members.len() < 2 {
            return Err(HyperPatternError::ArityBelowTwo);
        }
        let mut aliases = std::collections::HashSet::new();
        for member in &self.members {
            if !aliases.insert(member.alias.as_str()) {
                return Err(HyperPatternError::DuplicateAlias(member.alias.clone()));
            }
            match self.semantics {
                HyperPatternSemantics::Symmetric if member.role_id.is_some() => {
                    return Err(HyperPatternError::RoleOnSymmetricMember(
                        member.alias.clone(),
                    ));
                }
                HyperPatternSemantics::RoleAware if member.role_id.is_none() => {
                    return Err(HyperPatternError::MissingRole(member.alias.clone()));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Matches an already resolved relation with a canonical alias assignment.
    pub fn match_resolved(
        &self,
        relation: &ResolvedRelationVersion,
    ) -> Result<Option<HyperPatternMatch>, HyperPatternError> {
        self.validate()?;
        if self
            .type_id
            .is_some_and(|type_id| type_id != relation.type_id)
            || relation.bindings.len() != self.members.len()
        {
            return Ok(None);
        }
        if self
            .epistemic_in
            .as_ref()
            .is_some_and(|allowed| !allowed.contains(&relation.epistemic_status))
        {
            return Ok(None);
        }

        let mut remaining = relation.bindings.clone();
        match self.semantics {
            HyperPatternSemantics::Symmetric => {
                remaining.sort_unstable_by_key(|binding| (binding.entity_id, binding.role_id))
            }
            HyperPatternSemantics::RoleAware => remaining.sort_unstable(),
        }
        let mut assignments = Vec::with_capacity(self.members.len());
        let mut order: Vec<usize> = (0..self.members.len()).collect();
        order.sort_by_key(|&idx| self.members[idx].entity_id.is_none());
        for member_idx in order {
            let member = &self.members[member_idx];
            let position = remaining.iter().position(|binding| {
                member.entity_id.is_none_or(|id| id == binding.entity_id)
                    && match self.semantics {
                        HyperPatternSemantics::Symmetric => true,
                        HyperPatternSemantics::RoleAware => member.role_id == Some(binding.role_id),
                    }
            });
            let Some(position) = position else {
                return Ok(None);
            };
            assignments.push((member_idx, remaining.remove(position)));
        }
        assignments.sort_unstable_by_key(|(member_idx, _)| *member_idx);

        Ok(Some(HyperPatternMatch {
            relation: relation.clone(),
            bindings: assignments
                .into_iter()
                .map(|(idx, binding)| (self.members[idx].alias.clone(), binding))
                .collect(),
        }))
    }

    pub fn execute(
        &self,
        rel_snap: &RelationReadSnapshot,
        ent_snap: &EntityReadSnapshot,
    ) -> Result<Vec<HyperPatternMatch>, HyperPatternError> {
        self.validate()?;
        let candidates = RelationQuery {
            type_id: self.type_id,
            role_constraints: Vec::new(),
            epistemic_in: self.epistemic_in.clone(),
            scope: self.scope,
            as_of_lsn: self.as_of_lsn,
        }
        .execute(rel_snap, ent_snap);
        let mut matches = Vec::new();
        for relation in candidates {
            if let Some(matched) = self.match_resolved(&relation)? {
                matches.push(matched);
            }
        }
        Ok(matches)
    }
}

/// Query predicate matching N-ary hypergraph relations.
#[derive(Clone, Debug, Default)]
pub struct RelationQuery {
    pub type_id: Option<RelationTypeId>,
    pub role_constraints: Vec<(RoleId, EntityId)>,
    pub epistemic_in: Option<Vec<EpistemicStatus>>,
    pub scope: SchemaScope,
    pub as_of_lsn: Option<u64>,
}

impl RelationQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_type(mut self, type_id: RelationTypeId) -> Self {
        self.type_id = Some(type_id);
        self
    }

    pub fn with_role(mut self, role_id: RoleId, entity_id: EntityId) -> Self {
        self.role_constraints.push((role_id, entity_id));
        self
    }

    pub fn with_epistemic(mut self, allowed: Vec<EpistemicStatus>) -> Self {
        self.epistemic_in = Some(allowed);
        self
    }

    pub fn with_as_of(mut self, lsn: u64) -> Self {
        self.as_of_lsn = Some(lsn);
        self
    }

    /// Executes this query against a pinned `RelationReadSnapshot` and `EntityReadSnapshot`.
    pub fn execute(
        &self,
        rel_snap: &RelationReadSnapshot,
        ent_snap: &EntityReadSnapshot,
    ) -> Vec<ResolvedRelationVersion> {
        let query_lsn = self.as_of_lsn.unwrap_or(rel_snap.lsn);

        // 1. If we have role constraints and type_id, use incidence posting index
        let candidate_rel_ids: Vec<RelationId> = if let Some(type_id) = self.type_id {
            if !self.role_constraints.is_empty() {
                let mut postings_list = Vec::new();
                for &(role_id, entity_id) in &self.role_constraints {
                    if let Some((entity_idx, _)) = ent_snap.segment.arena.get_by_id(entity_id) {
                        let posting = rel_snap
                            .segment
                            .incidence
                            .lookup(type_id, role_id, entity_idx);
                        postings_list.push(posting);
                    } else {
                        return Vec::new(); // Entity doesn't exist in this snapshot
                    }
                }
                let intersected =
                    crate::relation::incidence::IncidenceIndex::intersect(&postings_list);
                intersected
                    .into_iter()
                    .filter_map(|rel_idx| rel_snap.segment.arena.index_to_id(rel_idx))
                    .collect()
            } else if self.as_of_lsn.is_some() {
                rel_snap.segment.arena.all_ids()
            } else {
                rel_snap.segment.arena.live_ids()
            }
        } else if self.as_of_lsn.is_some() {
            rel_snap.segment.arena.all_ids()
        } else {
            rel_snap.segment.arena.live_ids()
        };

        // 2. Resolve temporal version and verify all constraints
        let mut results = Vec::new();
        for rel_id in candidate_rel_ids {
            if let Some(resolved) = rel_snap.resolve_relation_at(rel_id, query_lsn, ent_snap) {
                if resolved.lifecycle_status == crate::entity::status::LifecycleStatus::Tombstoned {
                    continue;
                }
                // Check relation type if specified
                if let Some(type_id) = self.type_id {
                    if resolved.type_id != type_id {
                        continue;
                    }
                }

                // Check schema scope
                if let Some(schema) = rel_snap.segment.get_type_schema(resolved.type_id) {
                    match self.scope {
                        SchemaScope::AdmittedOnly => {
                            if schema.state != crate::relation::schema::RelationTypeState::Admitted
                            {
                                continue;
                            }
                        }
                        SchemaScope::IncludeProposed => {
                            if schema.state
                                == crate::relation::schema::RelationTypeState::Deprecated
                            {
                                continue;
                            }
                        }
                        SchemaScope::Historical => {}
                    }
                }

                // Check epistemic filter
                if let Some(allowed) = &self.epistemic_in {
                    if !allowed.contains(&resolved.epistemic_status) {
                        continue;
                    }
                }

                // Verify all role constraints
                let mut matches_all_roles = true;
                for &(role_id, entity_id) in &self.role_constraints {
                    let has_binding = resolved
                        .bindings
                        .iter()
                        .any(|b| b.role_id == role_id && b.entity_id == entity_id);
                    if !has_binding {
                        matches_all_roles = false;
                        break;
                    }
                }

                if matches_all_roles {
                    results.push(resolved);
                }
            }
        }

        // Canonical deterministic ordering: RelationId ASC
        results.sort_unstable_by_key(|r| r.relation_id);
        results
    }
}

#[cfg(test)]
mod hyperpattern_tests {
    use super::*;
    use crate::entity::status::LifecycleStatus;
    use crate::relation::id::DurableRoleBinding;

    fn ternary_relation() -> ResolvedRelationVersion {
        ResolvedRelationVersion {
            relation_id: 9,
            version_id: 1,
            type_id: 77,
            schema_version: 1,
            valid_from_lsn: 10,
            valid_until_lsn: None,
            epistemic_status: EpistemicStatus::Observed,
            lifecycle_status: LifecycleStatus::Active,
            bindings: vec![
                DurableRoleBinding {
                    entity_id: 30,
                    role_id: 3,
                },
                DurableRoleBinding {
                    entity_id: 10,
                    role_id: 1,
                },
                DurableRoleBinding {
                    entity_id: 20,
                    role_id: 2,
                },
            ],
            provenance: None,
        }
    }

    #[test]
    fn symmetric_pattern_has_no_distinguished_target() {
        let pattern = HyperPattern {
            type_id: Some(77),
            members: ["a", "b", "c"]
                .into_iter()
                .map(|alias| HyperPatternMember {
                    alias: alias.into(),
                    role_id: None,
                    entity_id: None,
                })
                .collect(),
            semantics: HyperPatternSemantics::Symmetric,
            epistemic_in: None,
            scope: SchemaScope::Historical,
            as_of_lsn: None,
        };
        let matched = pattern
            .match_resolved(&ternary_relation())
            .unwrap()
            .unwrap();
        let ids: Vec<_> = matched.bindings.iter().map(|(_, b)| b.entity_id).collect();
        assert_eq!(ids, vec![10, 20, 30]);
    }

    #[test]
    fn role_aware_pattern_preserves_semantic_roles() {
        let pattern = HyperPattern {
            type_id: Some(77),
            members: vec![
                HyperPatternMember {
                    alias: "trigger".into(),
                    role_id: Some(2),
                    entity_id: Some(20),
                },
                HyperPatternMember {
                    alias: "component".into(),
                    role_id: Some(1),
                    entity_id: None,
                },
                HyperPatternMember {
                    alias: "scope".into(),
                    role_id: Some(3),
                    entity_id: None,
                },
            ],
            semantics: HyperPatternSemantics::RoleAware,
            epistemic_in: None,
            scope: SchemaScope::Historical,
            as_of_lsn: None,
        };
        let matched = pattern
            .match_resolved(&ternary_relation())
            .unwrap()
            .unwrap();
        assert_eq!(matched.bindings[0].1.entity_id, 20);
        assert_eq!(matched.bindings[1].1.entity_id, 10);
        assert_eq!(matched.bindings[2].1.entity_id, 30);
    }
}
