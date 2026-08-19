/* hnsqr/src/graph/mutation/command.rs */
//!▫~•◦-------------------------------‣
//! # Graph Mutation Commands
//!▫~•◦-------------------------------------------------------------------‣

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::graph::catalog::labels::LabelId;
use crate::graph::catalog::properties::PropertyKey;
use crate::graph::catalog::relationships::RelTypeId;
use crate::graph::storage::properties::GraphPropertyValue;
use crate::NodeIndex;

/// Stable external identity for a relationship, assigned at creation.
pub type RelationshipId = u64;

/// Property map carried alongside a graph mutation.
pub type GraphProperties = HashMap<PropertyKey, GraphPropertyValue>;

/// Complete command set for graph topology and property mutations.
///
/// Corresponds to the GraphQuery CREATE / DELETE / MERGE / SET primitives.
/// All variants are Raft-replicated before touching local state.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum GraphMutation {
    // ── Node operations ────────────────────────────────────────────────

    /// Create a new graph node, optionally binding it to a vector slot.
    CreateNode {
        /// Stable external identifier used by the client.
        external_id: String,
        labels: Vec<LabelId>,
        properties: GraphProperties,
        /// `Some(slot)` binds this node to an existing HNSQR vector.
        vector_slot: Option<NodeIndex>,
    },

    /// Delete a node and all its incident edges (cascade).
    DeleteNode {
        external_id: String,
    },

    /// Add labels to an existing node (idempotent).
    SetNodeLabels {
        external_id: String,
        labels: Vec<LabelId>,
    },

    /// Remove labels from an existing node (idempotent).
    RemoveNodeLabels {
        external_id: String,
        labels: Vec<LabelId>,
    },

    /// Patch (upsert) properties on a node.
    PatchNodeProperties {
        external_id: String,
        properties: GraphProperties,
    },

    // ── Relationship operations ─────────────────────────────────────────

    /// Create a directed relationship between two nodes.
    CreateRelationship {
        relationship_id: RelationshipId,
        src_external_id: String,
        dst_external_id: String,
        rel_type: RelTypeId,
        properties: GraphProperties,
        weight: f32,
    },

    /// Delete a specific relationship.
    DeleteRelationship {
        relationship_id: RelationshipId,
    },

    /// Patch properties on an existing relationship.
    PatchRelationshipProperties {
        relationship_id: RelationshipId,
        properties: GraphProperties,
    },

    // ── Batch ───────────────────────────────────────────────────────────

    /// Atomic batch of graph mutations applied in order.
    /// Fails entirely if any constituent mutation fails pre-validation.
    Batch(Vec<GraphMutation>),
}
