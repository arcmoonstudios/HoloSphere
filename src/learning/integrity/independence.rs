/* holosphere/src/learning/integrity/independence.rs */
//!▫~•◦-------------------------------‣
//! # Evidence Independence & Anti-Multiplication Accounting
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Prevents artificial confidence inflation by ensuring that copied beliefs,
//! downstream citations, or multi-agent echoes collapse to their true number of
//! independent empirical roots.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::entity::id::EntityId;
use crate::learning::collective::AgentId;
use crate::learning::integrity::lineage::{EmpiricalRootId, EpistemicLineageGraph};

/// Detailed accounting report of independent empirical evidence vs copied assertions.
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct EvidenceIndependenceReport {
    /// Total raw assertions or agent belief statements submitted.
    pub total_assertions: usize,
    /// Number of distinct agents submitting assertions.
    pub reporting_agents_count: usize,
    /// Exact count of truly independent empirical observation roots.
    pub independent_root_count: usize,
    /// Set of unique empirical roots backing the claim.
    pub empirical_roots: Vec<EmpiricalRootId>,
}

/// Computes the genuine independent empirical support for a collection of beliefs.
pub fn evaluate_evidence_independence(
    lineage: &EpistemicLineageGraph,
    claims: &[(AgentId, EntityId)],
) -> EvidenceIndependenceReport {
    let mut unique_agents = HashSet::new();
    let mut unique_roots = HashSet::new();

    for (agent, entity) in claims {
        unique_agents.insert(*agent);
        let roots = lineage.get_empirical_roots(*entity);
        unique_roots.extend(roots);
    }

    let mut roots_vec: Vec<EmpiricalRootId> = unique_roots.into_iter().collect();
    roots_vec.sort();

    EvidenceIndependenceReport {
        total_assertions: claims.len(),
        reporting_agents_count: unique_agents.len(),
        independent_root_count: roots_vec.len(),
        empirical_roots: roots_vec,
    }
}
