/* hnsqr/src/graph/storage/properties.rs */
//!▫~•◦-------------------------------‣
//! # Graph Property Store — Columnar Property Values for Nodes and Edges
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Properties are stored in a simple append-only map rather than column
//! blocks at this stage.  The interface is intentionally narrow so it can
//! be replaced with a columnar layout when profiling shows it necessary.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::HashMap;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::NodeIndex;
use crate::graph::catalog::PropertyKey;

/// Supported property value types for graph nodes and relationships.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphPropertyValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
    IntList(Vec<i64>),
    FloatList(Vec<f64>),
    TextList(Vec<String>),
}

/// Thread-safe columnar-adjacent property store.
///
/// Both node and edge properties are keyed by `(entity_id, PropertyKey)`.
/// At this prototype stage entity IDs for nodes and edges occupy separate
/// namespaces; the caller is responsible for routing to the right store.
pub struct GraphPropertyStore {
    /// node_id → HashMap<PropertyKey, GraphPropertyValue>
    node_props: RwLock<HashMap<NodeIndex, HashMap<PropertyKey, GraphPropertyValue>>>,
    /// rel_id → HashMap<PropertyKey, GraphPropertyValue>
    rel_props: RwLock<HashMap<u32, HashMap<PropertyKey, GraphPropertyValue>>>,
}

impl Default for GraphPropertyStore {
    fn default() -> Self {
        Self {
            node_props: RwLock::new(HashMap::new()),
            rel_props: RwLock::new(HashMap::new()),
        }
    }
}

impl GraphPropertyStore {
    // ── Node properties ────────────────────────────────────────────────

    pub fn set_node_property(&self, node: NodeIndex, key: PropertyKey, value: GraphPropertyValue) {
        self.node_props
            .write()
            .entry(node)
            .or_default()
            .insert(key, value);
    }

    pub fn get_node_property(
        &self,
        node: NodeIndex,
        key: PropertyKey,
    ) -> Option<GraphPropertyValue> {
        self.node_props
            .read()
            .get(&node)
            .and_then(|m| m.get(&key))
            .cloned()
    }

    pub fn get_node_properties(&self, node: NodeIndex) -> HashMap<PropertyKey, GraphPropertyValue> {
        self.node_props
            .read()
            .get(&node)
            .cloned()
            .unwrap_or_default()
    }

    pub fn delete_node_properties(&self, node: NodeIndex) {
        self.node_props.write().remove(&node);
    }

    pub fn patch_node_properties(
        &self,
        node: NodeIndex,
        patch: HashMap<PropertyKey, GraphPropertyValue>,
    ) {
        let mut guard = self.node_props.write();
        let entry = guard.entry(node).or_default();
        for (k, v) in patch {
            entry.insert(k, v);
        }
    }

    // ── Edge properties ────────────────────────────────────────────────

    pub fn set_rel_property(&self, rel_id: u32, key: PropertyKey, value: GraphPropertyValue) {
        self.rel_props
            .write()
            .entry(rel_id)
            .or_default()
            .insert(key, value);
    }

    pub fn get_rel_property(&self, rel_id: u32, key: PropertyKey) -> Option<GraphPropertyValue> {
        self.rel_props
            .read()
            .get(&rel_id)
            .and_then(|m| m.get(&key))
            .cloned()
    }

    pub fn delete_rel_properties(&self, rel_id: u32) {
        self.rel_props.write().remove(&rel_id);
    }

    pub fn patch_rel_properties(
        &self,
        rel_id: u32,
        patch: HashMap<PropertyKey, GraphPropertyValue>,
    ) {
        let mut guard = self.rel_props.write();
        let entry = guard.entry(rel_id).or_default();
        for (k, v) in patch {
            entry.insert(k, v);
        }
    }
}
