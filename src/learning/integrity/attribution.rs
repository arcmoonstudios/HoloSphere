/* holosphere/src/learning/integrity/attribution.rs */
//!▫~•◦-------------------------------‣
//! # Multi-Action Plan Credit Attribution & Ablation Guards
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Enforces rigorous causal credit attribution for compound action plans,
//! preventing accidental standalone credit assignment to individual steps
//! when only a joint combination $[A, B]$ was empirically tested.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

use crate::experience::id::{ActionId, AttemptId};

/// Justified method of attributing utility to actions in a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlanAttributionMethod {
    /// Credit is assigned solely to the joint multi-action plan as a compound whole.
    JointPlanOnly,
    /// Credit is assigned to a step because standalone ablation testing proved its contribution.
    AblationTested,
    /// Credit is assigned because independent standalone single-action trials confirmed efficacy.
    StandaloneEmpiricalConfirmed,
}

/// Evidential record specifying what an attempt outcome actually proves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlanAttributionRecord {
    pub attempt_id: AttemptId,
    pub plan_actions: Vec<ActionId>,
    pub attribution_method: PlanAttributionMethod,
    /// The exact actions or compound plans that are legitimately credited by this attempt.
    pub justified_credited_actions: Vec<ActionId>,
    /// Set of actions for which individual standalone credit is strictly withheld.
    pub withheld_standalone_actions: Vec<ActionId>,
}

/// Derives the legitimate causal attribution from an attempt's action invocations.
pub fn compute_plan_attribution(
    attempt_id: AttemptId,
    invoked_actions: &[ActionId],
    standalone_evidence_actions: &HashSet<ActionId>,
) -> PlanAttributionRecord {
    let mut unique_actions: Vec<ActionId> = invoked_actions.to_vec();
    unique_actions.sort();
    unique_actions.dedup();

    if unique_actions.len() <= 1 {
        // Single action attempt: standalone credit is fully justified
        return PlanAttributionRecord {
            attempt_id,
            plan_actions: unique_actions.clone(),
            attribution_method: PlanAttributionMethod::StandaloneEmpiricalConfirmed,
            justified_credited_actions: unique_actions,
            withheld_standalone_actions: Vec::new(),
        };
    }

    // Multi-action compound plan
    let mut justified = Vec::new();
    let mut withheld = Vec::new();

    for &a in &unique_actions {
        if standalone_evidence_actions.contains(&a) {
            justified.push(a);
        } else {
            withheld.push(a);
        }
    }

    let method = if withheld.is_empty() {
        PlanAttributionMethod::AblationTested
    } else {
        PlanAttributionMethod::JointPlanOnly
    };

    PlanAttributionRecord {
        attempt_id,
        plan_actions: unique_actions,
        attribution_method: method,
        justified_credited_actions: justified,
        withheld_standalone_actions: withheld,
    }
}
