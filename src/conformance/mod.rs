/* holosphere/src/conformance/mod.rs */
//!▫~•◦-------------------------------‣
//! # Semantic Kernel v1 Conformance & Compatibility Subsystem
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Enforces immutable versioning, storage-independent canonical export/import,
//! typed public error taxonomy, and deterministic golden fixture conformance.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

pub mod corpus;
pub mod error;
pub mod export;
pub mod version;

pub use corpus::create_v1_golden_fixture;
pub use error::KernelError;
pub use export::{CanonicalExportArchive, ExportedEntity, ExportedExperience, ExportedRelation};
pub use version::{
    CANONICAL_EXPORT_VERSION, ENTITY_SCHEMA_VERSION, EXPERIENCE_SCHEMA_VERSION,
    INFERENCE_TRACE_VERSION, LEARNING_SCHEMA_VERSION, RAFT_LOG_RECORD_VERSION,
    RELATION_SCHEMA_VERSION, SEMANTIC_KERNEL_VERSION, SNAPSHOT_FORMAT_VERSION,
    SYNTHESIS_TRACE_VERSION, WORLD_DIGEST_VERSION,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_v1_conformance_golden_fixture_export_import_roundtrip() {
        let fixture = create_v1_golden_fixture();
        let digest_export = fixture.compute_world_digest();

        let imported_digest = fixture.import_validate().expect("import validate");
        assert_eq!(digest_export, imported_digest);
        assert_eq!(digest_export.lsn, 10_000);
    }

    #[test]
    fn test_v1_conformance_unsupported_version_fails_closed() {
        let mut corrupted_fixture = create_v1_golden_fixture();
        corrupted_fixture.format_version = 999;

        let result = corrupted_fixture.import_validate();
        assert_eq!(
            result,
            Err(KernelError::UnsupportedVersion {
                expected: CANONICAL_EXPORT_VERSION,
                actual: 999,
            })
        );
    }
}
