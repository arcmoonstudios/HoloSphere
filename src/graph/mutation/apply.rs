/* hnsqr/src/graph/mutation/apply.rs */
//!▫~•◦-------------------------------‣
//! # Graph Mutation Applier — State-Machine Application Layer
//!▫~•◦-------------------------------------------------------------------‣
//!
//! `GraphMutationApplier` is called from `ShardStateMachine::apply` after a
//! `DataMutation::Graph` entry is committed by Raft quorum.  It owns the
//! mutable `GraphGeneration` and the label/rel-type catalogs.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::RwLock;

use crate::graph::catalog::labels::{LABEL_FAST_SLOTS, LabelCatalog};
use crate::graph::catalog::relationships::RelTypeCatalog;
use crate::graph::mutation::command::GraphMutation;
use crate::graph::storage::edge_delta::EdgeRecord;
use crate::graph::storage::generation::GraphGeneration;
use crate::graph::storage::node_arena::GraphNodeRecord;
use crate::{HNSQRError, HNSQRResult, NodeIndex};

/// Owns the mutable graph generation and catalogs; applies committed mutations.
pub struct GraphMutationApplier {
    generation: Arc<RwLock<GraphGeneration>>,
    /// Retained for catalog snapshot serialisation and label-resolution queries.
    label_catalog: Arc<LabelCatalog>,
    rel_catalog: Arc<RelTypeCatalog>,
    /// External ID → internal `NodeIndex` lookup.
    node_id_map: RwLock<HashMap<String, NodeIndex>>,
    /// `RelationshipId` → internal delta record offset.
    rel_id_map: RwLock<HashMap<u64, u32>>,
}

impl GraphMutationApplier {
    pub fn new(
        generation: Arc<RwLock<GraphGeneration>>,
        label_catalog: Arc<LabelCatalog>,
        rel_catalog: Arc<RelTypeCatalog>,
    ) -> Self {
        Self {
            generation,
            label_catalog,
            rel_catalog,
            node_id_map: RwLock::new(HashMap::new()),
            rel_id_map: RwLock::new(HashMap::new()),
        }
    }

    pub fn label_catalog(&self) -> Arc<LabelCatalog> {
        self.label_catalog.clone()
    }
    pub fn rel_catalog(&self) -> Arc<RelTypeCatalog> {
        self.rel_catalog.clone()
    }
    pub fn generation(&self) -> Arc<RwLock<GraphGeneration>> {
        self.generation.clone()
    }

    /// Captures a fully materialized immutable snapshot of the graph at generation/lsn k.
    pub fn snapshot(&self, lsn: u64) -> crate::graph::storage::snapshot::ImmutableGraphSnapshot {
        let (nodes, live_nodes, edge_records, live_edges, gen_id, properties) = {
            let gen_guard = self.generation.read();
            let (n, ln) = gen_guard.nodes.snapshot();
            let (e, le) = if let Some(delta) = &gen_guard.edge_delta {
                delta.snapshot()
            } else {
                (Vec::new(), Vec::new())
            };
            (
                n,
                ln,
                e,
                le,
                gen_guard.generation,
                gen_guard.properties.clone(),
            )
        };

        crate::graph::storage::snapshot::ImmutableGraphSnapshot {
            generation: gen_id,
            lsn,
            label_catalog: self.label_catalog.snapshot(),
            rel_type_catalog: self.rel_catalog.snapshot(),
            nodes,
            live_nodes,
            edge_records,
            live_edges,
            properties,
            node_id_map: self.node_id_map.read().clone(),
            rel_id_map: self.rel_id_map.read().clone(),
        }
    }

    /// Applies a committed `GraphMutation`.  Called from the state machine apply loop.
    pub fn apply(&self, mutation: &GraphMutation) -> HNSQRResult<()> {
        match mutation {
            GraphMutation::CreateNode {
                external_id,
                labels,
                properties,
                vector_slot,
            } => self.create_node(external_id, labels, properties.clone(), *vector_slot),
            GraphMutation::DeleteNode { external_id } => self.delete_node(external_id),
            GraphMutation::SetNodeLabels {
                external_id,
                labels,
            } => self.set_node_labels(external_id, labels, true),
            GraphMutation::RemoveNodeLabels {
                external_id,
                labels,
            } => self.set_node_labels(external_id, labels, false),
            GraphMutation::PatchNodeProperties {
                external_id,
                properties,
            } => self.patch_node_properties(external_id, properties.clone()),
            GraphMutation::CreateRelationship {
                relationship_id,
                src_external_id,
                dst_external_id,
                rel_type,
                properties,
                weight,
            } => self.create_relationship(
                *relationship_id,
                src_external_id,
                dst_external_id,
                *rel_type,
                properties.clone(),
                *weight,
            ),
            GraphMutation::DeleteRelationship { relationship_id } => {
                self.delete_relationship(*relationship_id)
            }
            GraphMutation::PatchRelationshipProperties {
                relationship_id,
                properties,
            } => self.patch_rel_properties(*relationship_id, properties.clone()),
            GraphMutation::Batch(mutations) => {
                // Pre-validate the entire batch before applying any mutation.
                for m in mutations {
                    self.prevalidate(m)?;
                }
                for m in mutations {
                    self.apply(m)?;
                }
                Ok(())
            }
        }
    }

    pub fn prevalidate(&self, mutation: &GraphMutation) -> HNSQRResult<()> {
        match mutation {
            GraphMutation::CreateRelationship {
                src_external_id,
                dst_external_id,
                ..
            } => {
                let map = self.node_id_map.read();
                if !map.contains_key(src_external_id.as_str()) {
                    return Err(HNSQRError::NodeNotFound(src_external_id.clone()));
                }
                if !map.contains_key(dst_external_id.as_str()) {
                    return Err(HNSQRError::NodeNotFound(dst_external_id.clone()));
                }
                Ok(())
            }
            GraphMutation::SetNodeLabels { external_id, .. }
            | GraphMutation::RemoveNodeLabels { external_id, .. }
            | GraphMutation::PatchNodeProperties { external_id, .. } => {
                let map = self.node_id_map.read();
                if !map.contains_key(external_id.as_str()) {
                    return Err(HNSQRError::NodeNotFound(external_id.clone()));
                }
                Ok(())
            }
            GraphMutation::PatchRelationshipProperties {
                relationship_id, ..
            } => {
                let rels = self.rel_id_map.read();
                if !rels.contains_key(relationship_id) {
                    return Err(HNSQRError::InvalidRequest(format!(
                        "Relationship {relationship_id} not found"
                    )));
                }
                Ok(())
            }
            GraphMutation::Batch(inner) => {
                for m in inner {
                    self.prevalidate(m)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn create_node(
        &self,
        external_id: &str,
        labels: &[u32],
        properties: crate::graph::mutation::command::GraphProperties,
        vector_slot: Option<NodeIndex>,
    ) -> HNSQRResult<()> {
        // Idempotent: if already exists, skip.
        if self.node_id_map.read().contains_key(external_id) {
            return Ok(());
        }

        // Build label bitmask.
        let mut label_fast_mask = 0u64;
        for &label_id in labels {
            if label_id < LABEL_FAST_SLOTS {
                label_fast_mask |= 1u64 << label_id;
            }
            // Overflow labels are handled in the property store for now.
        }

        let record = GraphNodeRecord {
            label_fast_mask,
            out_ref: crate::graph::storage::edge_delta::NULL_EDGE,
            in_ref: crate::graph::storage::edge_delta::NULL_EDGE,
            out_degree: 0,
            in_degree: 0,
            vector_slot: vector_slot.unwrap_or(u32::MAX),
            label_overflow_ref: crate::graph::storage::node_arena::NULL_OVERFLOW_REF,
        };

        let graph_gen = self.generation.read();
        let node_idx = graph_gen.nodes.alloc(record);
        drop(graph_gen);

        // Store properties.
        if !properties.is_empty() {
            let graph_gen = self.generation.read();
            for (key, value) in properties {
                graph_gen.properties.set_node_property(node_idx, key, value);
            }
        }

        self.node_id_map
            .write()
            .insert(external_id.to_string(), node_idx);
        Ok(())
    }

    fn delete_node(&self, external_id: &str) -> HNSQRResult<()> {
        let node_idx = {
            let mut map = self.node_id_map.write();
            match map.remove(external_id) {
                Some(idx) => idx,
                None => return Ok(()), // Already deleted — idempotent.
            }
        };
        let graph_gen = self.generation.read();
        graph_gen.nodes.delete(node_idx);
        graph_gen.properties.delete_node_properties(node_idx);
        // Note: incident edges become orphaned; they will be skipped at query time
        // because the node record is tombstoned.  Full cascade requires scanning
        // the adjacency, which is deferred to a planned compaction operation.
        Ok(())
    }

    fn set_node_labels(&self, external_id: &str, labels: &[u32], add: bool) -> HNSQRResult<()> {
        let map = self.node_id_map.read();
        let &node_idx = map
            .get(external_id)
            .ok_or_else(|| HNSQRError::NodeNotFound(external_id.to_string()))?;
        drop(map);

        let graph_gen = self.generation.read();
        if let Some(mut record) = graph_gen.nodes.get(node_idx) {
            for &label_id in labels {
                if label_id < LABEL_FAST_SLOTS {
                    if add {
                        record.label_fast_mask |= 1u64 << label_id;
                    } else {
                        record.label_fast_mask &= !(1u64 << label_id);
                    }
                }
            }
            graph_gen.nodes.update(node_idx, record);
        }
        Ok(())
    }

    fn patch_node_properties(
        &self,
        external_id: &str,
        properties: crate::graph::mutation::command::GraphProperties,
    ) -> HNSQRResult<()> {
        let map = self.node_id_map.read();
        let &node_idx = map
            .get(external_id)
            .ok_or_else(|| HNSQRError::NodeNotFound(external_id.to_string()))?;
        drop(map);
        let graph_gen = self.generation.read();
        graph_gen
            .properties
            .patch_node_properties(node_idx, properties);
        Ok(())
    }

    fn create_relationship(
        &self,
        relationship_id: u64,
        src_external_id: &str,
        dst_external_id: &str,
        rel_type: u16,
        properties: crate::graph::mutation::command::GraphProperties,
        weight: f32,
    ) -> HNSQRResult<()> {
        // Idempotent.
        if self.rel_id_map.read().contains_key(&relationship_id) {
            return Ok(());
        }

        let (src_idx, dst_idx) = {
            let map = self.node_id_map.read();
            let src = *map
                .get(src_external_id)
                .ok_or_else(|| HNSQRError::NodeNotFound(src_external_id.to_string()))?;
            let dst = *map
                .get(dst_external_id)
                .ok_or_else(|| HNSQRError::NodeNotFound(dst_external_id.to_string()))?;
            (src, dst)
        };

        let graph_gen = self.generation.read();
        let delta = graph_gen.edge_delta.as_ref().ok_or_else(|| {
            HNSQRError::Internal("Graph generation is sealed; cannot write".to_string())
        })?;

        // Build the edge record with zeroed next pointers before linking both adjacency chains.
        let mut edge = EdgeRecord::new(rel_type, src_idx, dst_idx, weight, 0);

        // Thread the new edge into the source node's out-chain.
        if let Some(mut src_rec) = graph_gen.nodes.get(src_idx) {
            edge.next_src = src_rec.out_ref;
            let delta_id = delta.append(edge);
            src_rec.out_ref = delta_id;
            src_rec.out_degree = src_rec.out_degree.saturating_add(1);
            graph_gen.nodes.update(src_idx, src_rec);

            // Thread into destination node's in-chain.
            if let Some(mut dst_rec) = graph_gen.nodes.get(dst_idx) {
                // We need to patch next_dst on the just-appended record.
                let mut committed_edge = delta.get(delta_id).unwrap();
                committed_edge.next_dst = dst_rec.in_ref;
                delta.update(delta_id, committed_edge);
                dst_rec.in_ref = delta_id;
                dst_rec.in_degree = dst_rec.in_degree.saturating_add(1);
                graph_gen.nodes.update(dst_idx, dst_rec);
            }

            // Store properties.
            if !properties.is_empty() {
                for (key, value) in properties {
                    graph_gen.properties.set_rel_property(delta_id, key, value);
                }
            }

            self.rel_id_map.write().insert(relationship_id, delta_id);
        } else {
            return Err(HNSQRError::NodeNotFound(src_external_id.to_string()));
        }
        Ok(())
    }

    fn delete_relationship(&self, relationship_id: u64) -> HNSQRResult<()> {
        let delta_id = {
            let mut map = self.rel_id_map.write();
            match map.remove(&relationship_id) {
                Some(id) => id,
                None => return Ok(()),
            }
        };
        let graph_gen = self.generation.read();
        if let Some(delta) = &graph_gen.edge_delta {
            delta.delete(delta_id);
            graph_gen.properties.delete_rel_properties(delta_id);
        }
        Ok(())
    }

    fn patch_rel_properties(
        &self,
        relationship_id: u64,
        properties: crate::graph::mutation::command::GraphProperties,
    ) -> HNSQRResult<()> {
        let map = self.rel_id_map.read();
        let &delta_id = map.get(&relationship_id).ok_or_else(|| {
            HNSQRError::Internal(format!("Relationship {relationship_id} not found"))
        })?;
        drop(map);
        let graph_gen = self.generation.read();
        graph_gen
            .properties
            .patch_rel_properties(delta_id, properties);
        Ok(())
    }

    /// Returns the internal `NodeIndex` for an external node ID.
    pub fn resolve_node(&self, external_id: &str) -> Option<NodeIndex> {
        self.node_id_map.read().get(external_id).copied()
    }

    /// Returns the current node count.
    pub fn node_count(&self) -> usize {
        self.generation.read().node_count()
    }

    /// Returns the current edge count.
    pub fn edge_count(&self) -> usize {
        self.generation.read().edge_count()
    }
}
