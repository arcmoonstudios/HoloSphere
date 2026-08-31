/* holosphere/src/contextgraph/compiler.rs */
//!▫~•◦-------------------------------‣
//! # Universal ContextGraph Compiler Pipeline
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Orchestrates adapter detection, content fingerprinting, AST/text extraction,
//! universal multi-pass reference resolution, and atomic delta creation.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::adapter::SourceInput;
use super::adapters::AdapterRegistry;
use super::fingerprint::GraphFingerprinter;
use super::invalidation::InvalidationGraph;
use super::ir::{ExtractedEntity, ExtractionBatch, UnresolvedReference};
use super::manifest::{ContextGraphManifest, SourceManifestEntry};
use super::resolver::{ReferenceResolver, UniversalReferenceResolver};
use super::schema::{
    ContextGraphDelta, Entity, EntityId, Namespace, ProvenanceRef, Relation, RelationId,
    RelationParticipant,
};
use crate::{HNSQRError, HNSQRResult};

/// Output of a ContextCompiler execution.
#[derive(Clone, Debug)]
pub struct ContextCompilationOutput {
    pub namespace: Namespace,
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub manifest: ContextGraphManifest,
    pub canonical_fingerprint: [u8; 32],
    pub duration_ms: u64,
    pub report: CompilationReport,
}

/// Evidence-bounded coverage report for a single static ContextGraph compilation.
///
/// It reports what the compiler observed; it never claims runtime or whole-program
/// completeness.  The reference counters partition every adapter-emitted unresolved
/// reference exactly once: resolved, ambiguous, or unmatched.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompilationReport {
    pub source_count: usize,
    pub sources_by_type: BTreeMap<String, usize>,
    pub entity_count: usize,
    pub relation_count: usize,
    pub reference_count: usize,
    pub resolved_reference_count: usize,
    pub ambiguous_reference_count: usize,
    pub unmatched_reference_count: usize,
}

impl CompilationReport {
    #[must_use]
    pub fn references_accounted_for(&self) -> bool {
        self.reference_count
            == self.resolved_reference_count
                + self.ambiguous_reference_count
                + self.unmatched_reference_count
    }
}

impl ContextCompilationOutput {
    #[must_use]
    pub fn into_delta(self) -> ContextGraphDelta {
        let touched_locators = self.manifest.sources.keys().cloned().collect();
        ContextGraphDelta {
            namespace: self.namespace,
            insert_entities: self.entities,
            delete_entities: Vec::new(),
            insert_relations: self.relations,
            delete_relations: Vec::new(),
            touched_locators,
        }
    }
}

pub struct ContextCompiler {
    adapters: Arc<AdapterRegistry>,
    resolver: Arc<dyn ReferenceResolver>,
}

impl Default for ContextCompiler {
    fn default() -> Self {
        Self {
            adapters: Arc::new(AdapterRegistry::default()),
            resolver: Arc::new(UniversalReferenceResolver::new()),
        }
    }
}

impl ContextCompiler {
    #[must_use]
    pub fn new(adapters: Arc<AdapterRegistry>, resolver: Arc<dyn ReferenceResolver>) -> Self {
        Self { adapters, resolver }
    }

    /// Compiles a set of source inputs into a fully resolved ContextGraph delta.
    pub fn compile(
        &self,
        namespace: &Namespace,
        sources: &[SourceInput],
    ) -> HNSQRResult<ContextCompilationOutput> {
        let start = Instant::now();

        // 1. Parallel Extraction
        let adapters = self.adapters.clone();
        let extracted_batches: Vec<HNSQRResult<(String, String, ExtractionBatch)>> = sources
            .par_iter()
            .map(|source| {
                let adapter = adapters.find_adapter(source).ok_or_else(|| {
                    HNSQRError::InvalidRequest(format!(
                        "No adapter detected for {}",
                        source.locator
                    ))
                })?;

                let batch = adapter.extract(source, namespace)?;
                Ok((
                    adapter.name().to_string(),
                    adapter.version().to_string(),
                    batch,
                ))
            })
            .collect();

        // 2. Deterministic Assembly
        let mut sorted_batches = Vec::new();
        for res in extracted_batches {
            sorted_batches.push(res?);
        }
        sorted_batches.sort_by(|a, b| a.2.source.locator.cmp(&b.2.source.locator));

        let mut all_entities = Vec::new();
        let mut all_relations = Vec::new();
        let mut all_unresolved = Vec::new();
        let mut temp_id_to_final_id: HashMap<String, EntityId> = HashMap::new();
        let mut manifest_entries = Vec::new();

        for (adapter_name, adapter_version, batch) in sorted_batches {
            let mut file_entity_ids = Vec::new();
            let mut file_relation_ids = Vec::new();

            let prov = ProvenanceRef {
                adapter_name: adapter_name.clone(),
                adapter_version: adapter_version.clone(),
                compiler_version: "2.0.0".to_string(),
                source_fingerprint: format!("{:x?}", batch.source.content_hash),
                commit_lsn: 0,
            };

            for ext_ent in batch.entities {
                let locator_uri = ext_ent.locator.as_ref().map(|l| l.uri.as_str());
                let final_id = EntityId::compute(
                    namespace,
                    &ext_ent.kind,
                    &ext_ent.label,
                    locator_uri,
                    &ext_ent.fingerprint,
                );

                temp_id_to_final_id.insert(ext_ent.temp_id.clone(), final_id.clone());
                file_entity_ids.push(final_id.clone());

                all_entities.push(Entity {
                    id: final_id,
                    kind: ext_ent.kind,
                    label: ext_ent.label,
                    namespace: namespace.clone(),
                    locator: ext_ent.locator,
                    attributes: ext_ent.attributes,
                    provenance: vec![prov.clone()],
                    evidence_class: ext_ent.evidence_class,
                    verification_state: ext_ent.verification_state,
                    fingerprint: ext_ent.fingerprint,
                    valid_from_lsn: 0,
                });
            }

            for ext_rel in batch.relations {
                let mut participants = Vec::new();
                for (temp_ref, role) in ext_rel.participants {
                    if let Some(target_id) = temp_id_to_final_id.get(&temp_ref) {
                        participants.push(RelationParticipant::new(target_id.clone(), role));
                    }
                }
                if !participants.is_empty() {
                    let rel_id = RelationId::compute(&ext_rel.kind, &participants, ext_rel.origin);
                    file_relation_ids.push(rel_id.clone());
                    all_relations.push(Relation {
                        id: rel_id,
                        kind: ext_rel.kind,
                        participants,
                        origin: ext_rel.origin,
                        confidence: ext_rel.confidence,
                        provenance: vec![prov.clone()],
                        verification_state:
                            crate::transport::model_gateway::VerificationState::Verified,
                        attributes: ext_rel.attributes,
                    });
                }
            }

            all_unresolved.extend(batch.unresolved);

            manifest_entries.push(SourceManifestEntry {
                locator_uri: batch.source.locator.clone(),
                source_type: batch.source.source_type.clone(),
                content_fingerprint: batch.source.content_hash,
                adapter_name,
                adapter_version,
                emitted_entity_ids: file_entity_ids,
                emitted_relation_ids: file_relation_ids,
                timestamp_secs: 0,
            });
        }

        // 3. Multi-Pass Reference Resolution
        let mut entities_by_id: BTreeMap<EntityId, Entity> = BTreeMap::new();
        let mut entities_by_label: HashMap<String, Vec<EntityId>> = HashMap::new();

        for ent in &all_entities {
            entities_by_id.insert(ent.id.clone(), ent.clone());
            entities_by_label
                .entry(ent.label.clone())
                .or_default()
                .push(ent.id.clone());
        }

        let reference_count = all_unresolved.len();
        let mut resolved_reference_count = 0;
        let mut ambiguous_reference_count = 0;
        let mut unmatched_reference_count = 0;
        for unres in all_unresolved {
            let src_final_id = temp_id_to_final_id
                .get(&unres.source_temp_id)
                .cloned()
                .unwrap_or_else(|| EntityId(unres.source_temp_id.clone()));

            let resolved_unres = UnresolvedReference {
                source_temp_id: src_final_id.0,
                ..unres
            };

            let resolved_rels =
                self.resolver
                    .resolve(&resolved_unres, &entities_by_id, &entities_by_label);
            if resolved_rels.is_empty() {
                unmatched_reference_count += 1;
            } else if resolved_rels
                .iter()
                .any(|relation| relation.origin == super::schema::RelationOrigin::Ambiguous)
            {
                ambiguous_reference_count += 1;
            } else {
                resolved_reference_count += 1;
            }
            all_relations.extend(resolved_rels);
        }

        // 4. Canonical Sorting & Deduplication
        all_entities.sort_by(|a, b| a.id.cmp(&b.id));
        all_entities.dedup_by(|a, b| a.id == b.id);

        all_relations.sort_by(|a, b| a.id.cmp(&b.id));
        all_relations.dedup_by(|a, b| a.id == b.id);

        let canonical_fingerprint =
            GraphFingerprinter::compute_fingerprint(&all_entities, &all_relations);

        let mut manifest = ContextGraphManifest::new(namespace.clone());
        manifest.apply_update(manifest_entries, &[]);
        manifest.canonical_graph_fingerprint = Some(canonical_fingerprint);

        let duration_ms = start.elapsed().as_millis() as u64;
        let mut sources_by_type = BTreeMap::new();
        for source in manifest.sources.values() {
            *sources_by_type
                .entry(source.source_type.clone())
                .or_insert(0) += 1;
        }
        let report = CompilationReport {
            source_count: manifest.sources.len(),
            sources_by_type,
            entity_count: all_entities.len(),
            relation_count: all_relations.len(),
            reference_count,
            resolved_reference_count,
            ambiguous_reference_count,
            unmatched_reference_count,
        };
        debug_assert!(report.references_accounted_for());

        Ok(ContextCompilationOutput {
            namespace: namespace.clone(),
            entities: all_entities,
            relations: all_relations,
            manifest,
            canonical_fingerprint,
            duration_ms,
            report,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn report_partitions_resolved_ambiguous_and_unmatched_references() {
        let compiler = ContextCompiler::default();
        let namespace = Namespace::new("test:coverage");
        let sources = vec![SourceInput::from_text(
            "fn target() {}\nfn first() { target(); missing(); }\nfn second() { target(); }",
            "memory://coverage.rs",
            "rust",
        )];

        let output = compiler.compile(&namespace, &sources).unwrap();
        assert_eq!(output.report.source_count, 1);
        assert_eq!(output.report.sources_by_type.get("rust"), Some(&1));
        assert!(output.report.entity_count > 0);
        assert!(output.report.relation_count > 0);
        assert!(output.report.references_accounted_for());
        assert!(output.report.resolved_reference_count >= 1);
        assert!(output.report.unmatched_reference_count >= 1);
    }
}
