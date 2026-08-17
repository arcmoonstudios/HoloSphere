/* hnsqr/src/rivero_witness.rs */
//!▫~•◦-------------------------------‣
//! # Bounded Deterministic Witness-Neighbor Resolution Engine
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides deterministic reciprocal witness neighbor selection, rank-ordered edge pruning,
//! and strict two-hop graph expansion bounds for the Rivero candidate resolution path.
//!
//! ## Key Capabilities
//! - **Bounded Degree Ceilings:** Enforces fixed physical maximum degree limits ($\le 64$) for layer-0 adjacency buffers.
//! - **Two-Hop Bounded Expansion:** Guarantees strict mathematical ceilings on total edge scans independently of corpus size.
//! - **Deterministic Reciprocal Pruning:** Atomically maintains exact symmetric nearest-witness connectivity with zero thread-order drift.
//!
//! ### Architectural Notes
//! Interfaces directly with `RiveroTerritoryIndex` during candidate generation and `RiveroBulkBuilder`
//! during multi-threaded index construction.
//!
//! #### Example
//! ```rust
//! use hnsqr::rivero_witness::{ScoredWitness, select_top};
//!
//! let mut candidates = vec![
//!     ScoredWitness { index: 1, similarity: 0.95 },
//!     ScoredWitness { index: 2, similarity: 0.88 },
//! ];
//! let top_witnesses = select_top(&mut candidates, 1);
//! ```
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use smallvec::SmallVec;

use crate::{NodeIndex, SimilarityScore};

/// Default maximum witness neighbors retained per strict Rivero node.
pub const RIVERO_WITNESS_DEFAULT_DEGREE: usize = 64;
/// Default number of exactly-ranked Rivero seeds expanded by strict search.
pub const RIVERO_WITNESS_DEFAULT_SEEDS: usize = 48;
/// Default number of newly discovered first-hop seeds expanded once more.
pub const RIVERO_WITNESS_DEFAULT_SECOND_SEEDS: usize = 16;
/// Physical ceiling imposed by the inline layer-zero adjacency buffer.
pub const RIVERO_WITNESS_MAX_DEGREE: usize = 64;
/// Witnesses stored inline per live node before spilling to a node-local allocation.
pub const RIVERO_WITNESS_INLINE_DEGREE: usize = 16;
/// Physical ceiling on query seeds, keeping witness work mechanically bounded.
pub const RIVERO_WITNESS_MAX_SEEDS: usize = 64;

/// A candidate witness paired with its exact symmetric similarity to an owner.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ScoredWitness {
    pub index: NodeIndex,
    pub similarity: SimilarityScore,
}

impl ScoredWitness {
    const EMPTY: Self = Self {
        index: NodeIndex::MAX,
        similarity: f32::NEG_INFINITY,
    };
}

#[inline]
pub(crate) const fn bounded_degree(requested: usize) -> usize {
    if requested > RIVERO_WITNESS_MAX_DEGREE {
        RIVERO_WITNESS_MAX_DEGREE
    } else {
        requested
    }
}

#[inline]
pub(crate) const fn bounded_seeds(requested: usize) -> usize {
    if requested > RIVERO_WITNESS_MAX_SEEDS {
        RIVERO_WITNESS_MAX_SEEDS
    } else {
        requested
    }
}

/// Maximum layer-zero adjacency entries inspected by one witness expansion.
#[inline]
pub const fn witness_edge_scan_bound(seeds: usize, degree: usize) -> usize {
    bounded_seeds(seeds) * bounded_degree(degree)
}

/// Maximum adjacency entries inspected across two fixed witness hops.
#[inline]
pub const fn witness_two_hop_edge_scan_bound(
    first_hop_seeds: usize,
    second_hop_seeds: usize,
    degree: usize,
) -> usize {
    (bounded_seeds(first_hop_seeds) + bounded_seeds(second_hop_seeds)) * bounded_degree(degree)
}

#[inline]
fn witness_order(lhs: &ScoredWitness, rhs: &ScoredWitness) -> std::cmp::Ordering {
    rhs.similarity
        .total_cmp(&lhs.similarity)
        .then_with(|| lhs.index.cmp(&rhs.index))
}

/// Retains the deterministic exact-similarity top degree from a bounded route.
pub fn select_top(
    scored: &mut [ScoredWitness],
    requested_degree: usize,
) -> SmallVec<[ScoredWitness; RIVERO_WITNESS_INLINE_DEGREE]> {
    let degree = bounded_degree(requested_degree);
    if degree == 0 {
        return SmallVec::new();
    }

    scored.sort_unstable_by(witness_order);
    let mut selected = SmallVec::new();
    for candidate in scored.iter().copied() {
        if selected
            .iter()
            .any(|existing: &ScoredWitness| existing.index == candidate.index)
        {
            continue;
        }
        selected.push(candidate);
        if selected.len() == degree {
            break;
        }
    }
    selected
}

/// Inserts a reciprocal edge and deterministically prunes the owner to its exact
/// top-degree live witnesses. The callback scores the owner's existing edges and
/// returns `None` for stale entries.
pub(crate) fn insert_reciprocal(
    current: &mut SmallVec<[ScoredWitness; RIVERO_WITNESS_INLINE_DEGREE]>,
    incoming: ScoredWitness,
    requested_degree: usize,
) -> bool {
    let degree = bounded_degree(requested_degree);
    if degree == 0 {
        current.clear();
        return false;
    }

    let mut ranked = [ScoredWitness::EMPTY; RIVERO_WITNESS_MAX_DEGREE + 1];
    let mut len = 0usize;
    for existing in current.iter().copied() {
        if existing.index == incoming.index || len == RIVERO_WITNESS_MAX_DEGREE {
            continue;
        }
        ranked[len] = existing;
        len += 1;
    }
    ranked[len] = incoming;
    len += 1;
    ranked[..len].sort_unstable_by(witness_order);

    current.clear();
    current.extend_from_slice(&ranked[..len.min(degree)]);
    current.iter().any(|edge| edge.index == incoming.index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_selection_is_bounded_and_deterministic() {
        let mut candidates = vec![
            ScoredWitness {
                index: 9,
                similarity: 0.8,
            },
            ScoredWitness {
                index: 3,
                similarity: 0.9,
            },
            ScoredWitness {
                index: 2,
                similarity: 0.9,
            },
        ];
        let selected = select_top(&mut candidates, 2);
        assert_eq!(
            selected.iter().map(|edge| edge.index).collect::<Vec<_>>(),
            [2, 3]
        );
    }

    #[test]
    fn reciprocal_pruning_replaces_the_exact_worst_edge() {
        let mut current = SmallVec::from_slice(&[
            ScoredWitness {
                index: 1,
                similarity: 0.9,
            },
            ScoredWitness {
                index: 2,
                similarity: 0.8,
            },
            ScoredWitness {
                index: 3,
                similarity: 0.7,
            },
        ]);
        let incoming = ScoredWitness {
            index: 4,
            similarity: 0.85,
        };
        let retained = insert_reciprocal(&mut current, incoming, 3);
        assert!(retained);
        assert_eq!(
            current.iter().map(|edge| edge.index).collect::<Vec<_>>(),
            [1, 4, 2]
        );
        assert_eq!(current.len(), 3);
    }

    #[test]
    fn production_two_hop_bound_is_exact_and_corpus_independent() {
        assert_eq!(
            witness_two_hop_edge_scan_bound(
                RIVERO_WITNESS_DEFAULT_SEEDS,
                RIVERO_WITNESS_DEFAULT_SECOND_SEEDS,
                RIVERO_WITNESS_DEFAULT_DEGREE,
            ),
            4_096
        );
        assert_eq!(bounded_seeds(usize::MAX), RIVERO_WITNESS_MAX_SEEDS);
        assert_eq!(bounded_degree(usize::MAX), RIVERO_WITNESS_MAX_DEGREE);
    }
}
