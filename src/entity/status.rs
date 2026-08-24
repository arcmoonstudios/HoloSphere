/* holosphere/src/entity/status.rs */
//!▫~•◦-------------------------------‣
//! # Epistemic and Lifecycle Status Types
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Separates epistemic certainty (Observed, Asserted, Inferred, Provisional, Contradicted)
//! from temporal lifecycle state (Active, Superseded, Tombstoned).
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};

/// Epistemic justification and confidence status for an entity, relation, or claim.
///
/// Canonical epistemic state belongs to specific claims, relations, or versions.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpistemicStatus {
    /// Sourced directly from immutable ground-truth external observations (logs, telemetry, sensors, ingested documents).
    Observed = 0,
    /// Explicitly asserted by an authenticated subject, user, or authoritative external agent.
    Asserted = 1,
    /// Derived deterministically by an automated inference engine or structural composition rule.
    Inferred = 2,
    /// Hypothesized or unverified candidate structure (e.g. analogical transfer, initial induction).
    Provisional = 3,
    /// Actively falsified or contradicted by newer evidence or counter-observations.
    Contradicted = 4,
}

impl EpistemicStatus {
    #[inline(always)]
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Observed,
            1 => Self::Asserted,
            2 => Self::Inferred,
            3 => Self::Provisional,
            4 => Self::Contradicted,
            _ => Self::Provisional,
        }
    }

    #[inline(always)]
    pub fn is_verified(self) -> bool {
        matches!(self, Self::Observed | Self::Asserted | Self::Inferred)
    }

    #[inline(always)]
    pub fn is_provisional(self) -> bool {
        matches!(self, Self::Provisional)
    }

    #[inline(always)]
    pub fn is_contradicted(self) -> bool {
        matches!(self, Self::Contradicted)
    }
}

/// Operational lifecycle state of an entity, relation instance, or version snapshot.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LifecycleStatus {
    /// Current, active, and live in the canonical topology.
    Active = 0,
    /// Valid historical state that has been superseded by a newer version in the lineage.
    Superseded = 1,
    /// Soft-deleted / tombstoned entity whose slot is preserved to maintain index stability.
    Tombstoned = 2,
}

impl LifecycleStatus {
    #[inline(always)]
    pub fn from_u8(val: u8) -> Self {
        match val {
            0 => Self::Active,
            1 => Self::Superseded,
            2 => Self::Tombstoned,
            _ => Self::Active,
        }
    }

    #[inline(always)]
    pub fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    #[inline(always)]
    pub fn is_superseded(self) -> bool {
        matches!(self, Self::Superseded)
    }

    #[inline(always)]
    pub fn is_tombstoned(self) -> bool {
        matches!(self, Self::Tombstoned)
    }
}
