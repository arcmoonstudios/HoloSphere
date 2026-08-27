/* holosphere/src/contextgraph/schema.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Schema — Domain-Neutral Entities, N-ary Relations & Epistemics
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Universal schema for multi-domain knowledge graphs: code, documents, runtime systems,
//! Git commits, datasets, measurements, and organizational entities with first-class epistemics.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::BTreeMap;
use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::transport::model_gateway::{EvidenceClass, VerificationState};

/// Logical namespace isolating repositories, workspaces, environments, or domains.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Namespace(pub String);

impl Namespace {
    #[must_use]
    pub fn new(ns: impl Into<String>) -> Self {
        Self(ns.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for Namespace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Namespace {
    fn default() -> Self {
        Self("default".to_string())
    }
}

/// Extensible taxonomy of entity types across software, documents, git, events, systems.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityKind(pub String);

impl EntityKind {
    #[must_use]
    pub fn new(kind: impl Into<String>) -> Self {
        Self(kind.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    // Common standard kinds
    #[must_use]
    pub fn code_function() -> Self {
        Self("code:function".to_string())
    }
    #[must_use]
    pub fn code_struct() -> Self {
        Self("code:struct".to_string())
    }
    #[must_use]
    pub fn code_trait() -> Self {
        Self("code:trait".to_string())
    }
    #[must_use]
    pub fn code_module() -> Self {
        Self("code:module".to_string())
    }
    #[must_use]
    pub fn code_file() -> Self {
        Self("code:file".to_string())
    }
    #[must_use]
    pub fn code_rationale() -> Self {
        Self("code:rationale".to_string())
    }
    #[must_use]
    pub fn document_section() -> Self {
        Self("document:section".to_string())
    }
    #[must_use]
    pub fn document_claim() -> Self {
        Self("document:claim".to_string())
    }
    #[must_use]
    pub fn git_commit() -> Self {
        Self("git:commit".to_string())
    }
    #[must_use]
    pub fn test_case() -> Self {
        Self("test:test_case".to_string())
    }
    #[must_use]
    pub fn system_service() -> Self {
        Self("system:service".to_string())
    }
}

impl fmt::Display for EntityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Extensible taxonomy of relation kinds across multi-domain graphs.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RelationKind(pub String);

impl RelationKind {
    #[must_use]
    pub fn new(kind: impl Into<String>) -> Self {
        Self(kind.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    // Common standard relations
    #[must_use]
    pub fn calls() -> Self {
        Self("calls".to_string())
    }
    #[must_use]
    pub fn defines() -> Self {
        Self("defines".to_string())
    }
    #[must_use]
    pub fn contains() -> Self {
        Self("contains".to_string())
    }
    #[must_use]
    pub fn implements() -> Self {
        Self("implements".to_string())
    }
    #[must_use]
    pub fn imports() -> Self {
        Self("imports".to_string())
    }
    #[must_use]
    pub fn depends_on() -> Self {
        Self("depends_on".to_string())
    }
    #[must_use]
    pub fn tests() -> Self {
        Self("tests".to_string())
    }
    #[must_use]
    pub fn explains() -> Self {
        Self("explains".to_string())
    }
    #[must_use]
    pub fn justifies() -> Self {
        Self("justifies".to_string())
    }
    #[must_use]
    pub fn mentions() -> Self {
        Self("mentions".to_string())
    }
    #[must_use]
    pub fn supports() -> Self {
        Self("supports".to_string())
    }
    #[must_use]
    pub fn contradicts() -> Self {
        Self("contradicts".to_string())
    }
    #[must_use]
    pub fn caused_by() -> Self {
        Self("caused_by".to_string())
    }
    #[must_use]
    pub fn derived_from() -> Self {
        Self("derived_from".to_string())
    }
}

impl fmt::Display for RelationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Epistemic origin of a relation or entity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RelationOrigin {
    /// Deterministically extracted from syntax / structured AST / source proof.
    Extracted,
    /// Statically or logically resolved against an authoritative registry.
    Resolved,
    /// Inferred via heuristic or topological reasoning.
    Inferred,
    /// Generated by synthesis or model.
    Generated,
    /// Directly observed via telemetry or empirical execution.
    Observed,
    /// Ambiguous reference with multiple candidate targets preserved without guessing.
    Ambiguous,
}

impl RelationOrigin {
    #[must_use]
    pub const fn default_confidence(self) -> f32 {
        match self {
            Self::Extracted | Self::Observed => 1.0,
            Self::Resolved => 0.95,
            Self::Inferred => 0.75,
            Self::Generated => 0.70,
            Self::Ambiguous => 0.50,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Extracted => "extracted",
            Self::Resolved => "resolved",
            Self::Inferred => "inferred",
            Self::Generated => "generated",
            Self::Observed => "observed",
            Self::Ambiguous => "ambiguous",
        }
    }
}

impl fmt::Display for RelationOrigin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Universal URI / span locator pointing to where the entity was discovered.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ResourceLocator {
    pub uri: String,
    pub start_line: Option<usize>,
    pub end_line: Option<usize>,
    pub start_byte: Option<usize>,
    pub end_byte: Option<usize>,
}

impl ResourceLocator {
    #[must_use]
    pub fn file(path: impl AsRef<std::path::Path>, start_line: usize, end_line: usize) -> Self {
        let uri = format!(
            "file:///{}",
            path.as_ref().to_string_lossy().replace('\\', "/")
        );
        Self {
            uri,
            start_line: Some(start_line),
            end_line: Some(end_line),
            start_byte: None,
            end_byte: None,
        }
    }

    #[must_use]
    pub fn uri(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            start_line: None,
            end_line: None,
            start_byte: None,
            end_byte: None,
        }
    }
}

impl fmt::Display for ResourceLocator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let (Some(s), Some(e)) = (self.start_line, self.end_line) {
            write!(f, "{}#L{}-L{}", self.uri, s, e)
        } else {
            write!(f, "{}", self.uri)
        }
    }
}

/// Provenance metadata tracking adapter versions and compiler passes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceRef {
    pub adapter_name: String,
    pub adapter_version: String,
    pub compiler_version: String,
    pub source_fingerprint: String,
    pub commit_lsn: u64,
}

/// Deterministic stable identifier for any ContextGraph entity.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct EntityId(pub String);

impl EntityId {
    #[must_use]
    pub fn compute(
        namespace: &Namespace,
        kind: &EntityKind,
        label: &str,
        locator_uri: Option<&str>,
        content_fingerprint: &[u8; 32],
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_ENTITY_V1:");
        hasher.update(namespace.as_str().as_bytes());
        hasher.update(b"|");
        hasher.update(kind.as_str().as_bytes());
        hasher.update(b"|");
        hasher.update(label.as_bytes());
        hasher.update(b"|");
        hasher.update(locator_uri.unwrap_or("").as_bytes());
        hasher.update(b"|");
        hasher.update(content_fingerprint);

        let hex = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        Self(format!("ent_{hex}"))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EntityId({})", self.0)
    }
}

impl fmt::Display for EntityId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Deterministic identifier for an N-ary ContextGraph relationship.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RelationId(pub String);

impl RelationId {
    #[must_use]
    pub fn compute(
        kind: &RelationKind,
        participants: &[RelationParticipant],
        origin: RelationOrigin,
    ) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"HOLOSPHERE_RELATION_V1:");
        hasher.update(kind.as_str().as_bytes());
        hasher.update(b":");
        hasher.update(origin.as_str().as_bytes());
        hasher.update(b":");

        let mut sorted_participants = participants.to_vec();
        sorted_participants.sort_by(|a, b| {
            a.entity_id
                .cmp(&b.entity_id)
                .then_with(|| a.role.cmp(&b.role))
        });

        for p in sorted_participants {
            hasher.update(p.entity_id.as_str().as_bytes());
            hasher.update(b"=");
            hasher.update(p.role.as_bytes());
            hasher.update(b";");
        }

        let hex = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        Self(format!("rel_{hex}"))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for RelationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RelationId({})", self.0)
    }
}

impl fmt::Display for RelationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// N-ary participant in a hypergraph relationship with a semantic role.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct RelationParticipant {
    pub entity_id: EntityId,
    pub role: String,
}

impl RelationParticipant {
    #[must_use]
    pub fn new(entity_id: EntityId, role: impl Into<String>) -> Self {
        Self {
            entity_id,
            role: role.into(),
        }
    }

    #[must_use]
    pub fn source(entity_id: EntityId) -> Self {
        Self::new(entity_id, "source")
    }
    #[must_use]
    pub fn target(entity_id: EntityId) -> Self {
        Self::new(entity_id, "target")
    }
    #[must_use]
    pub fn caller(entity_id: EntityId) -> Self {
        Self::new(entity_id, "caller")
    }
    #[must_use]
    pub fn callee(entity_id: EntityId) -> Self {
        Self::new(entity_id, "callee")
    }
}

/// Universal domain-neutral graph Entity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    pub id: EntityId,
    pub kind: EntityKind,
    pub label: String,
    pub namespace: Namespace,
    pub locator: Option<ResourceLocator>,
    pub attributes: BTreeMap<String, serde_json::Value>,
    pub provenance: Vec<ProvenanceRef>,
    pub evidence_class: EvidenceClass,
    pub verification_state: VerificationState,
    pub fingerprint: [u8; 32],
    pub valid_from_lsn: u64,
}

/// Universal domain-neutral N-ary Relation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Relation {
    pub id: RelationId,
    pub kind: RelationKind,
    pub participants: Vec<RelationParticipant>,
    pub origin: RelationOrigin,
    pub confidence: f32,
    pub provenance: Vec<ProvenanceRef>,
    pub verification_state: VerificationState,
    pub attributes: BTreeMap<String, serde_json::Value>,
}

impl Relation {
    /// Helper to create standard binary directed relations.
    #[must_use]
    pub fn binary(
        source: &EntityId,
        target: &EntityId,
        kind: RelationKind,
        origin: RelationOrigin,
    ) -> Self {
        let participants = vec![
            RelationParticipant::source(source.clone()),
            RelationParticipant::target(target.clone()),
        ];
        let id = RelationId::compute(&kind, &participants, origin);
        Self {
            id,
            kind,
            participants,
            origin,
            confidence: origin.default_confidence(),
            provenance: Vec::new(),
            verification_state: VerificationState::Verified,
            attributes: BTreeMap::new(),
        }
    }

    /// Helper to create caller -> callee relations.
    #[must_use]
    pub fn call(caller: &EntityId, callee: &EntityId, origin: RelationOrigin) -> Self {
        let participants = vec![
            RelationParticipant::caller(caller.clone()),
            RelationParticipant::callee(callee.clone()),
        ];
        let id = RelationId::compute(&RelationKind::calls(), &participants, origin);
        Self {
            id,
            kind: RelationKind::calls(),
            participants,
            origin,
            confidence: origin.default_confidence(),
            provenance: Vec::new(),
            verification_state: VerificationState::Verified,
            attributes: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn primary_source(&self) -> Option<&EntityId> {
        self.participants
            .iter()
            .find(|p| p.role == "source" || p.role == "caller")
            .map(|p| &p.entity_id)
    }

    #[must_use]
    pub fn primary_target(&self) -> Option<&EntityId> {
        self.participants
            .iter()
            .find(|p| p.role == "target" || p.role == "callee")
            .map(|p| &p.entity_id)
    }
}

/// Atomic mutation transaction payload committed to HoloSphere ContextGraph.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextGraphDelta {
    pub namespace: Namespace,
    pub insert_entities: Vec<Entity>,
    pub delete_entities: Vec<EntityId>,
    pub insert_relations: Vec<Relation>,
    pub delete_relations: Vec<RelationId>,
    pub touched_locators: Vec<String>,
}
