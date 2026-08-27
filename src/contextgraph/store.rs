/* holosphere/src/contextgraph/store.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Ingestion & Hypergraph State Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Authoritative storage layer for universal entities, N-ary hypergraph relations,
//! LSN snapshots, inverted label indices, and atomic ContextGraphDelta transactions.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use super::fingerprint::GraphFingerprinter;
use super::schema::{
    ContextGraphDelta, Entity, EntityId, EntityKind, Namespace, Relation, RelationId,
};

/// Immutable state snapshot of the ContextGraph at a specific LSN.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ContextGraphStoreState {
    pub namespace: Namespace,
    pub commit_lsn: u64,
    pub entities: BTreeMap<EntityId, Entity>,
    pub relations: BTreeMap<RelationId, Relation>,
    pub entity_relations: BTreeMap<EntityId, Vec<RelationId>>,
    pub entities_by_kind: BTreeMap<EntityKind, Vec<EntityId>>,
    pub entities_by_label: HashMap<String, Vec<EntityId>>,
    pub entities_by_locator: HashMap<String, Vec<EntityId>>,
    pub canonical_fingerprint: [u8; 32],
}

pub struct ContextGraphStore {
    inner: Arc<RwLock<ContextGraphStoreState>>,
    next_lsn: AtomicU64,
}

impl Default for ContextGraphStore {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(ContextGraphStoreState::default())),
            next_lsn: AtomicU64::new(1),
        }
    }
}

impl ContextGraphStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// Atomically applies a ContextGraphDelta transaction and publishes the new LSN snapshot.
    pub fn commit_delta(&self, delta: ContextGraphDelta) -> u64 {
        let mut state = self.inner.write();
        let lsn = self.next_lsn.fetch_add(1, Ordering::SeqCst);

        state.namespace = delta.namespace;
        state.commit_lsn = lsn;

        // 1. Delete tombstoned relations
        for rel_id in &delta.delete_relations {
            if let Some(rel) = state.relations.remove(rel_id) {
                for p in &rel.participants {
                    if let Some(r_list) = state.entity_relations.get_mut(&p.entity_id) {
                        r_list.retain(|id| id != rel_id);
                    }
                }
            }
        }

        // 2. Delete tombstoned entities
        for entity_id in &delta.delete_entities {
            if let Some(entity) = state.entities.remove(entity_id) {
                if let Some(ids) = state.entities_by_label.get_mut(&entity.label) {
                    ids.retain(|id| id != entity_id);
                }
                if let Some(ids) = state.entities_by_kind.get_mut(&entity.kind) {
                    ids.retain(|id| id != entity_id);
                }
                if let Some(loc) = &entity.locator {
                    if let Some(ids) = state.entities_by_locator.get_mut(&loc.uri) {
                        ids.retain(|id| id != entity_id);
                    }
                }
                state.entity_relations.remove(entity_id);
            }
        }

        // 3. Insert / update entities
        for mut entity in delta.insert_entities {
            entity.valid_from_lsn = lsn;
            let id = entity.id.clone();

            // If entity already exists, clean up its prior index entries
            let old_meta = state.entities.get(&id).map(|e| {
                (
                    e.label.clone(),
                    e.kind.clone(),
                    e.locator.as_ref().map(|l| l.uri.clone()),
                )
            });

            if let Some((old_label, old_kind, old_loc_uri)) = old_meta {
                if old_label != entity.label {
                    if let Some(ids) = state.entities_by_label.get_mut(&old_label) {
                        ids.retain(|existing_id| existing_id != &id);
                    }
                }
                if old_kind != entity.kind {
                    if let Some(ids) = state.entities_by_kind.get_mut(&old_kind) {
                        ids.retain(|existing_id| existing_id != &id);
                    }
                }
                if let (Some(old_uri), Some(new_loc)) = (&old_loc_uri, &entity.locator) {
                    if old_uri != &new_loc.uri {
                        if let Some(ids) = state.entities_by_locator.get_mut(old_uri) {
                            ids.retain(|existing_id| existing_id != &id);
                        }
                    }
                }
            }

            let label_entry = state
                .entities_by_label
                .entry(entity.label.clone())
                .or_default();
            if !label_entry.contains(&id) {
                label_entry.push(id.clone());
            }

            let kind_entry = state
                .entities_by_kind
                .entry(entity.kind.clone())
                .or_default();
            if !kind_entry.contains(&id) {
                kind_entry.push(id.clone());
            }

            if let Some(loc) = &entity.locator {
                let loc_entry = state
                    .entities_by_locator
                    .entry(loc.uri.clone())
                    .or_default();
                if !loc_entry.contains(&id) {
                    loc_entry.push(id.clone());
                }
            }

            state.entities.insert(id, entity);
        }

        // 4. Insert / update relations
        for rel in delta.insert_relations {
            let rel_id = rel.id.clone();
            for p in &rel.participants {
                let r_list = state
                    .entity_relations
                    .entry(p.entity_id.clone())
                    .or_default();
                if !r_list.contains(&rel_id) {
                    r_list.push(rel_id.clone());
                }
            }
            state.relations.insert(rel_id, rel);
        }

        // 5. Recompute canonical fingerprint
        let entities_vec: Vec<Entity> = state.entities.values().cloned().collect();
        let relations_vec: Vec<Relation> = state.relations.values().cloned().collect();
        state.canonical_fingerprint =
            GraphFingerprinter::compute_fingerprint(&entities_vec, &relations_vec);

        lsn
    }

    #[must_use]
    pub fn snapshot(&self) -> ContextGraphStoreState {
        self.inner.read().clone()
    }

    #[must_use]
    pub fn get_entity(&self, id: &EntityId) -> Option<Entity> {
        self.inner.read().entities.get(id).cloned()
    }

    #[must_use]
    pub fn get_relations_for_entity(&self, id: &EntityId) -> Vec<Relation> {
        let state = self.inner.read();
        state
            .entity_relations
            .get(id)
            .map(|rel_ids| {
                rel_ids
                    .iter()
                    .filter_map(|rid| state.relations.get(rid).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[must_use]
    pub fn lookup_by_label(&self, label: &str) -> Vec<Entity> {
        let state = self.inner.read();
        state
            .entities_by_label
            .get(label)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| state.entities.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    #[must_use]
    pub fn lookup_by_kind(&self, kind: &EntityKind) -> Vec<Entity> {
        let state = self.inner.read();
        state
            .entities_by_kind
            .get(kind)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| state.entities.get(id).cloned())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Persists snapshot state to disk.
    pub fn save_snapshot_to_file(&self, path: impl AsRef<std::path::Path>) -> crate::HNSQRResult<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let state = self.inner.read();
        let bytes = serde_json::to_vec_pretty(&*state)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Loads store state from a serialized snapshot on disk.
    pub fn load_snapshot_from_file(path: impl AsRef<std::path::Path>) -> crate::HNSQRResult<Self> {
        let bytes = std::fs::read(path)?;
        let state: ContextGraphStoreState = serde_json::from_slice(&bytes)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))?;
        let next_lsn = state.commit_lsn + 1;
        Ok(Self {
            inner: Arc::new(RwLock::new(state)),
            next_lsn: AtomicU64::new(next_lsn),
        })
    }
}
