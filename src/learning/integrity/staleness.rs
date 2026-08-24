/* holosphere/src/learning/integrity/staleness.rs */
//!▫~•◦-------------------------------‣
//! # Synthesis Dependency Digests & Staleness Guards
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Validates candidate resolution proposals prior to materialization against the
//! world's current LSN and dependency digest, preventing silent application of
//! obsolete hypotheses.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

use crate::entity::id::VersionId;

/// Complete dependency digest captured when a resolution candidate is synthesized.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SynthesisDependencyDigest {
    pub snapshot_lsn: u64,
    pub problem_version: VersionId,
    pub context_fingerprint: [u8; 32],
    pub precedent_digest: [u8; 32],
    pub relation_digest: [u8; 32],
    pub policy_version: u32,
}

/// Outcome of evaluating proposal freshness against current world state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProposalStalenessCheck {
    /// Proposal dependencies are fresh and valid for materialization.
    Fresh,
    /// Problem version or state has moved past the synthesized snapshot.
    StaleProblemVersion {
        expected: VersionId,
        actual: VersionId,
    },
    /// Context properties have mutated since synthesis.
    StaleContextFingerprint,
    /// Precedent evidence has updated (new contradictions or outcomes).
    StalePrecedents,
    /// Synthesis policy version has changed.
    StalePolicyVersion { expected: u32, actual: u32 },
}

impl SynthesisDependencyDigest {
    /// Validates the digest against current live parameters.
    pub fn validate(
        &self,
        current_problem_version: VersionId,
        current_context_fingerprint: &[u8; 32],
        current_precedent_digest: &[u8; 32],
        current_policy_version: u32,
    ) -> ProposalStalenessCheck {
        if self.problem_version != current_problem_version {
            return ProposalStalenessCheck::StaleProblemVersion {
                expected: self.problem_version,
                actual: current_problem_version,
            };
        }
        if self.context_fingerprint != *current_context_fingerprint {
            return ProposalStalenessCheck::StaleContextFingerprint;
        }
        if self.precedent_digest != *current_precedent_digest {
            return ProposalStalenessCheck::StalePrecedents;
        }
        if self.policy_version != current_policy_version {
            return ProposalStalenessCheck::StalePolicyVersion {
                expected: self.policy_version,
                actual: current_policy_version,
            };
        }
        ProposalStalenessCheck::Fresh
    }
}
