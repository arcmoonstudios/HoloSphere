/* holosphere/src/learning/integrity/dependency.rs */
//!▫~•◦-------------------------------‣
//! # Circular Epistemic Support Guards
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Enforces acyclicity across the derivation and reinforcement topology, ensuring
//! that a hypothesis cannot increase its epistemic confidence from evidence derived
//! directly or indirectly from itself.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::id::EntityId;
use crate::learning::integrity::lineage::EpistemicLineageGraph;

/// Result of evaluating whether an evidential claim introduces circular self-reinforcement.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircularityCheck {
    /// Valid acyclic dependency structure.
    Acyclic,
    /// Direct or transitive cycle detected: hypothesis depends on itself.
    CircularDependencyDetected {
        target: EntityId,
        reinforcing_source: EntityId,
    },
}

/// Evaluates whether using `evidence_source` to reinforce `target_hypothesis` creates a cycle.
pub fn check_epistemic_circularity(
    lineage: &EpistemicLineageGraph,
    target_hypothesis: EntityId,
    evidence_source: EntityId,
) -> CircularityCheck {
    if lineage.would_create_cycle(target_hypothesis, evidence_source) {
        CircularityCheck::CircularDependencyDetected {
            target: target_hypothesis,
            reinforcing_source: evidence_source,
        }
    } else {
        CircularityCheck::Acyclic
    }
}
