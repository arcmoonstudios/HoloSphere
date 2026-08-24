/* holosphere/src/entity/eligibility.rs */
//!▫~•◦-------------------------------‣
//! # Canonical Eligibility Object & Set Operations
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides the canonical representation of candidate qualification universes
//! E = {e | P(e, S_k) = true} at a pinned snapshot generation.
//!
//! ## Invariant Guarantees
//! - Generation Lock: An `EligibilitySet` is valid ONLY for the generation in which
//!   its `EntityIndex`es were produced.
//! - Exactness: Logical membership is strictly invariant to whether the set is
//!   represented as a `DenseBitmap` or `SparseIndices`.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::entity::id::EntityIndex;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum EligibilityError {
    #[error(
        "Generation mismatch: eligibility generation {eligibility_gen} != snapshot generation {snapshot_gen}"
    )]
    GenerationMismatch {
        eligibility_gen: u64,
        snapshot_gen: u64,
    },
    #[error(
        "Snapshot LSN mismatch: eligibility LSN {eligibility_lsn} != snapshot LSN {snapshot_lsn}"
    )]
    LsnMismatch {
        eligibility_lsn: u64,
        snapshot_lsn: u64,
    },
}

/// Common trait for querying eligibility membership over `EntityIndex`.
pub trait EligibilityView {
    fn contains(&self, entity: EntityIndex) -> bool;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn generation(&self) -> u64;
    fn snapshot_lsn(&self) -> u64;
}

/// Physical representation of the eligible candidate set.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EligibilityRepresentation {
    /// Roaring dense bitmap for high-density candidate universes.
    DenseBitmap(RoaringBitmap),
    /// Sorted, deduplicated vector of generation-local row indices for sparse sets.
    SparseIndices(Vec<EntityIndex>),
}

/// Canonical eligibility set $E$ produced by graph, temporal, or epistemic filters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EligibilitySet {
    pub snapshot_lsn: u64,
    pub entity_generation: u64,
    pub live_population: usize,
    pub eligible_count: usize,
    pub representation: EligibilityRepresentation,
}

impl EligibilitySet {
    /// Creates an eligibility set from a RoaringBitmap.
    pub fn from_dense(
        snapshot_lsn: u64,
        entity_generation: u64,
        live_population: usize,
        bitmap: RoaringBitmap,
    ) -> Self {
        let eligible_count = bitmap.len() as usize;
        Self {
            snapshot_lsn,
            entity_generation,
            live_population,
            eligible_count,
            representation: EligibilityRepresentation::DenseBitmap(bitmap),
        }
    }

    /// Creates an eligibility set from sparse sorted entity indices.
    pub fn from_sparse(
        snapshot_lsn: u64,
        entity_generation: u64,
        live_population: usize,
        mut indices: Vec<EntityIndex>,
    ) -> Self {
        indices.sort_unstable();
        indices.dedup();
        let eligible_count = indices.len();
        Self {
            snapshot_lsn,
            entity_generation,
            live_population,
            eligible_count,
            representation: EligibilityRepresentation::SparseIndices(indices),
        }
    }

    /// Creates an empty eligibility set for generation `entity_generation` at `snapshot_lsn`.
    pub fn empty(snapshot_lsn: u64, entity_generation: u64, live_population: usize) -> Self {
        Self {
            snapshot_lsn,
            entity_generation,
            live_population,
            eligible_count: 0,
            representation: EligibilityRepresentation::SparseIndices(Vec::new()),
        }
    }

    /// Converts to dense `RoaringBitmap` representation.
    pub fn to_dense_bitmap(&self) -> RoaringBitmap {
        match &self.representation {
            EligibilityRepresentation::DenseBitmap(b) => b.clone(),
            EligibilityRepresentation::SparseIndices(indices) => {
                let mut b = RoaringBitmap::new();
                for &idx in indices {
                    b.insert(idx);
                }
                b
            }
        }
    }

    /// Converts to sorted `Vec<EntityIndex>` sparse representation.
    pub fn to_sparse_indices(&self) -> Vec<EntityIndex> {
        match &self.representation {
            EligibilityRepresentation::DenseBitmap(b) => b.iter().collect(),
            EligibilityRepresentation::SparseIndices(indices) => indices.clone(),
        }
    }

    /// Selectivity ratio $|E| / N \in [0.0, 1.0]$.
    pub fn selectivity(&self) -> f64 {
        if self.live_population == 0 {
            0.0
        } else {
            (self.eligible_count as f64) / (self.live_population as f64)
        }
    }

    /// Validates that this eligibility set matches the target snapshot's LSN and generation.
    pub fn validate_against_snapshot(
        &self,
        snapshot_lsn: u64,
        snapshot_gen: u64,
    ) -> Result<(), EligibilityError> {
        if self.entity_generation != snapshot_gen {
            return Err(EligibilityError::GenerationMismatch {
                eligibility_gen: self.entity_generation,
                snapshot_gen,
            });
        }
        if self.snapshot_lsn != snapshot_lsn {
            return Err(EligibilityError::LsnMismatch {
                eligibility_lsn: self.snapshot_lsn,
                snapshot_lsn,
            });
        }
        Ok(())
    }
}

impl EligibilityView for EligibilitySet {
    #[inline(always)]
    fn contains(&self, entity: EntityIndex) -> bool {
        match &self.representation {
            EligibilityRepresentation::DenseBitmap(b) => b.contains(entity),
            EligibilityRepresentation::SparseIndices(indices) => {
                indices.binary_search(&entity).is_ok()
            }
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.eligible_count
    }

    #[inline(always)]
    fn generation(&self) -> u64 {
        self.entity_generation
    }

    #[inline(always)]
    fn snapshot_lsn(&self) -> u64 {
        self.snapshot_lsn
    }
}
