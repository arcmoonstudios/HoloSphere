//! Immutable, dataset-backed proof artifacts for benchmark-only setup.
//!
//! Proof-tree construction is indexing work. It is kept out of benchmark processes
//! so query latency represents only query execution.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::{HNSQRError, HNSQRResult};

use super::SemanticProofTree;

pub const PROOF_BENCHMARK_ARTIFACT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofBenchmarkArtifact {
    pub version: u32,
    pub source_real_dimension: usize,
    pub vector_count: usize,
    pub tree: SemanticProofTree,
}

impl ProofBenchmarkArtifact {
    pub fn new(
        source_real_dimension: usize,
        tree: SemanticProofTree,
    ) -> HNSQRResult<Self> {
        let vector_count = tree.total_vectors();
        Ok(Self {
            version: PROOF_BENCHMARK_ARTIFACT_VERSION,
            source_real_dimension,
            vector_count,
            tree,
        })
    }

    pub fn save(&self, path: impl AsRef<Path>) -> HNSQRResult<()> {
        let bytes = bincode::serialize(self).map_err(|error| {
            HNSQRError::InvalidConfig(format!(
                "cannot serialize proof benchmark artifact: {error}"
            ))
        })?;
        std::fs::write(path, bytes).map_err(|error| {
            HNSQRError::InvalidConfig(format!("cannot write proof benchmark artifact: {error}"))
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
        source_real_dimension: usize,
        vector_count: usize,
    ) -> HNSQRResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|error| {
            HNSQRError::InvalidConfig(format!(
                "cannot read proof benchmark artifact '{}': {error}",
                path.display()
            ))
        })?;
        let artifact: Self = bincode::deserialize(&bytes).map_err(|error| {
            HNSQRError::InvalidConfig(format!(
                "cannot decode proof benchmark artifact '{}': {error}",
                path.display()
            ))
        })?;
        if artifact.version != PROOF_BENCHMARK_ARTIFACT_VERSION
            || artifact.source_real_dimension != source_real_dimension
            || artifact.vector_count != vector_count
            || artifact.tree.total_vectors() != vector_count
        {
            return Err(HNSQRError::InvalidConfig(format!(
                "proof benchmark artifact '{}' does not match the requested real dimension ({source_real_dimension}) and vector count ({vector_count})",
                path.display()
            )));
        }
        Ok(artifact)
    }
}

#[must_use]
pub fn proof_benchmark_artifact_filename(
    source_real_dimension: usize,
    vector_count: usize,
) -> String {
    format!(
        "gate-b-proof-v{PROOF_BENCHMARK_ARTIFACT_VERSION}-d{source_real_dimension}-n{vector_count}.bin"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_rejects_a_mismatched_workload() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("proof.bin");
        let artifact =
            ProofBenchmarkArtifact::new(128, SemanticProofTree::empty(64)).unwrap();
        artifact.save(&path).unwrap();

        assert!(ProofBenchmarkArtifact::load(&path, 128, 0).is_ok());
        assert!(ProofBenchmarkArtifact::load(&path, 128, 1).is_err());
        assert_eq!(
            proof_benchmark_artifact_filename(1536, 25_000),
            "gate-b-proof-v1-d1536-n25000.bin"
        );
    }
}
