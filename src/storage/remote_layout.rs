/* hnsqr/src/storage/remote_layout.rs */
//!▫~•◦-------------------------------‣
//! # Proof-Aware Remote Layout & Leaf-Locality Block Packing
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Organizes remote exact-vector byte blocks along ProofTree leaf partitions
//! so that evaluating an unresolved leaf candidate requires minimal contiguous
//! range fetches from cloud object storage, maximizing useful/fetched byte efficiency.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::proof::tree::SemanticProofTree;
use crate::{NodeIndex, VectorEmbedding};

/// Chunk size configuration for remote range fetching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum RemoteChunkSize {
    K64 = 65536,
    K128 = 131072,
    #[default]
    K256 = 262144,
    K512 = 524288,
    M1 = 1048576,
    M2 = 2097152,
    M4 = 4194304,
}

/// Layout mapping proof-tree leaf partitions to byte offsets in remote segment files.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofLeafBlockMapping {
    pub leaf_node_index: usize,
    pub slot_indices: Vec<NodeIndex>,
    pub byte_offset: u64,
    pub byte_length: usize,
}

/// Pack builder organizing dense vector storage into leaf-contiguous blocks.
pub struct ProofAwareLayoutBuilder;

impl ProofAwareLayoutBuilder {
    /// Packs dense vectors into proof-tree locality order and produces chunk layout metadata.
    pub fn build_leaf_locality_layout(
        proof_tree: &SemanticProofTree,
        vectors: &[VectorEmbedding],
        chunk_size: RemoteChunkSize,
    ) -> (Vec<u8>, Vec<ProofLeafBlockMapping>) {
        let mut byte_buffer = Vec::new();
        let mut mappings = Vec::new();

        for (leaf_idx, node) in proof_tree.nodes.iter().enumerate() {
            if !node.is_leaf() {
                continue;
            }
            let start_offset = byte_buffer.len() as u64;
            let mut leaf_slots = Vec::new();
            let member_end = (node.member_start + node.member_len) as usize;
            let slots_slice = &proof_tree.leaf_slots
                [node.member_start as usize..member_end.min(proof_tree.leaf_slots.len())];

            for &slot in slots_slice {
                if let Some(vec) = vectors.get(slot as usize) {
                    leaf_slots.push(slot);
                    for c in vec.complex_data() {
                        byte_buffer.extend_from_slice(&c.re.to_le_bytes());
                        byte_buffer.extend_from_slice(&c.im.to_le_bytes());
                    }
                }
            }

            let length = (byte_buffer.len() as u64 - start_offset) as usize;
            mappings.push(ProofLeafBlockMapping {
                leaf_node_index: leaf_idx,
                slot_indices: leaf_slots,
                byte_offset: start_offset,
                byte_length: length,
            });
        }

        // Align total buffer to chunk boundary
        let rem = byte_buffer.len() % (chunk_size as usize);
        if rem != 0 {
            byte_buffer.resize(byte_buffer.len() + ((chunk_size as usize) - rem), 0);
        }

        (byte_buffer, mappings)
    }
}

/// Telemetry tracking remote range request amplification and efficiency.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RemoteAmplificationMetrics {
    pub total_range_requests: u64,
    pub total_fetched_bytes: u64,
    pub total_useful_bytes: u64,
    pub lutz_eliminations_prior_to_fetch: u64,
}

impl RemoteAmplificationMetrics {
    pub fn useful_bytes_ratio(&self) -> f64 {
        if self.total_fetched_bytes == 0 {
            return 1.0;
        }
        (self.total_useful_bytes as f64 / self.total_fetched_bytes as f64).clamp(0.0, 1.0)
    }
}
