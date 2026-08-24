/* holosphere/src/entity/epistemic.rs */
//!▫~•◦-------------------------------‣
//! # Epistemic & Lifecycle Adjudication Matrix
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Enforces deterministic state transition rules. Ensures no inferred or
//! provisional statement can silently mutate into an observation, and no
//! tombstoned entity can be silently resurrected without an explicit new version.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use crate::entity::status::{EpistemicStatus, LifecycleStatus};
use thiserror::Error;

/// Errors encountered during invalid epistemic state transitions.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum EpistemicTransitionError {
    #[error(
        "Forbidden transition from {from:?} to {to:?}: inferred/provisional knowledge cannot mutate into Observed without a new version"
    )]
    CannotMutateToObserved {
        from: EpistemicStatus,
        to: EpistemicStatus,
    },
    #[error(
        "Forbidden transition from {from:?} to {to:?}: contradicted state is terminal for this claim"
    )]
    CannotTransitionFromContradicted {
        from: EpistemicStatus,
        to: EpistemicStatus,
    },
    #[error("Forbidden transition from {from:?} to {to:?}")]
    InvalidTransition {
        from: EpistemicStatus,
        to: EpistemicStatus,
    },
}

/// Validates an epistemic state transition according to the strict canonical matrix.
///
/// ## Transition Rules
/// - `Observed -> Contradicted`: ALLOWED
/// - `Asserted -> Contradicted`: ALLOWED
/// - `Provisional -> Inferred`: ALLOWED
/// - `Provisional -> Contradicted`: ALLOWED
/// - `Inferred -> Contradicted`: ALLOWED
/// - `Inferred -> Observed`: FORBIDDEN (Must create a new `Observed` version)
/// - `Provisional -> Observed`: FORBIDDEN
/// - `Contradicted -> *`: FORBIDDEN
pub fn validate_epistemic_transition(
    from: EpistemicStatus,
    to: EpistemicStatus,
) -> Result<(), EpistemicTransitionError> {
    if from == to {
        return Ok(());
    }

    if to == EpistemicStatus::Observed {
        return Err(EpistemicTransitionError::CannotMutateToObserved { from, to });
    }

    match from {
        EpistemicStatus::Observed => {
            if to == EpistemicStatus::Contradicted {
                Ok(())
            } else {
                Err(EpistemicTransitionError::InvalidTransition { from, to })
            }
        }
        EpistemicStatus::Asserted => {
            if to == EpistemicStatus::Contradicted {
                Ok(())
            } else {
                Err(EpistemicTransitionError::InvalidTransition { from, to })
            }
        }
        EpistemicStatus::Provisional => {
            if to == EpistemicStatus::Inferred || to == EpistemicStatus::Contradicted {
                Ok(())
            } else {
                Err(EpistemicTransitionError::InvalidTransition { from, to })
            }
        }
        EpistemicStatus::Inferred => {
            if to == EpistemicStatus::Contradicted {
                Ok(())
            } else {
                Err(EpistemicTransitionError::InvalidTransition { from, to })
            }
        }
        EpistemicStatus::Contradicted => {
            Err(EpistemicTransitionError::CannotTransitionFromContradicted { from, to })
        }
    }
}

/// Errors encountered during invalid lifecycle state transitions.
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum LifecycleTransitionError {
    #[error(
        "Forbidden transition from {from:?} to {to:?}: tombstoned entity cannot be silently resurrected"
    )]
    CannotResurrectTombstone {
        from: LifecycleStatus,
        to: LifecycleStatus,
    },
    #[error("Forbidden transition from {from:?} to {to:?}")]
    InvalidTransition {
        from: LifecycleStatus,
        to: LifecycleStatus,
    },
}

/// Validates a lifecycle state transition.
pub fn validate_lifecycle_transition(
    from: LifecycleStatus,
    to: LifecycleStatus,
) -> Result<(), LifecycleTransitionError> {
    if from == to {
        return Ok(());
    }

    match from {
        LifecycleStatus::Active => {
            if to == LifecycleStatus::Superseded || to == LifecycleStatus::Tombstoned {
                Ok(())
            } else {
                Err(LifecycleTransitionError::InvalidTransition { from, to })
            }
        }
        LifecycleStatus::Superseded => {
            if to == LifecycleStatus::Tombstoned {
                Ok(())
            } else {
                Err(LifecycleTransitionError::InvalidTransition { from, to })
            }
        }
        LifecycleStatus::Tombstoned => {
            Err(LifecycleTransitionError::CannotResurrectTombstone { from, to })
        }
    }
}
