/* holosphere/src/contextgraph/manifest.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Manifest & Source Fingerprint Cache
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Tracks content fingerprints, adapter metadata, emitted Entity IDs and Relation IDs
//! per source locator for efficient, fine-grained incremental compilation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::schema::{EntityId, Namespace, RelationId};
use crate::HNSQRResult;

/// Recorded state for a single ingested source item (file, URL, git commit).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceManifestEntry {
    pub locator_uri: String,
    pub source_type: String,
    pub content_fingerprint: [u8; 32],
    pub adapter_name: String,
    pub adapter_version: String,
    pub emitted_entity_ids: Vec<EntityId>,
    pub emitted_relation_ids: Vec<RelationId>,
    pub timestamp_secs: u64,
}

/// Namespace-scoped universal manifest.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextGraphManifest {
    pub namespace: Namespace,
    pub sources: BTreeMap<String, SourceManifestEntry>,
    pub canonical_graph_fingerprint: Option<[u8; 32]>,
    pub commit_lsn: u64,
}

/// Diff categorization of sources against recorded manifest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SourceDiff {
    pub added: Vec<String>,
    pub modified: Vec<String>,
    pub deleted: Vec<String>,
    pub unchanged: Vec<String>,
}

impl SourceDiff {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.modified.is_empty() && self.deleted.is_empty()
    }
}

impl ContextGraphManifest {
    #[must_use]
    pub fn new(namespace: Namespace) -> Self {
        Self {
            namespace,
            sources: BTreeMap::new(),
            canonical_graph_fingerprint: None,
            commit_lsn: 0,
        }
    }

    /// Computes incremental source diff against current fingerprints.
    #[must_use]
    pub fn diff(&self, current_fingerprints: &BTreeMap<String, [u8; 32]>) -> SourceDiff {
        let mut diff = SourceDiff::default();
        let current_keys: BTreeSet<&String> = current_fingerprints.keys().collect();

        for recorded_uri in self.sources.keys() {
            if !current_keys.contains(recorded_uri) {
                diff.deleted.push(recorded_uri.clone());
            }
        }

        for (uri, current_fp) in current_fingerprints {
            if let Some(recorded) = self.sources.get(uri) {
                if &recorded.content_fingerprint == current_fp {
                    diff.unchanged.push(uri.clone());
                } else {
                    diff.modified.push(uri.clone());
                }
            } else {
                diff.added.push(uri.clone());
            }
        }

        diff.added.sort();
        diff.modified.sort();
        diff.deleted.sort();
        diff.unchanged.sort();
        diff
    }

    pub fn apply_update(&mut self, entries: Vec<SourceManifestEntry>, deleted_uris: &[String]) {
        for uri in deleted_uris {
            self.sources.remove(uri);
        }
        for entry in entries {
            self.sources.insert(entry.locator_uri.clone(), entry);
        }
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> HNSQRResult<()> {
        if let Some(parent) = path.as_ref().parent() {
            std::fs::create_dir_all(parent)?;
        }
        let serialized = serde_json::to_string_pretty(self)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))?;
        std::fs::write(path, serialized)?;
        Ok(())
    }

    pub fn load_from_file(path: impl AsRef<Path>) -> HNSQRResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let manifest = serde_json::from_str(&content)
            .map_err(|e| crate::HNSQRError::SerializationError(e.to_string()))?;
        Ok(manifest)
    }
}
