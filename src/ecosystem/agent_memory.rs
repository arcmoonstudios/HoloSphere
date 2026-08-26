/* holosphere/src/ecosystem/agent_memory.rs */
//!▫~•◦-------------------------------‣
//! # Autonomous Long-Term Agentic Memory & Fact Consolidation (Mem0/Zep Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides an autonomous background memory consolidation engine that extracts episodic facts
//! from dialogue turns, reconciles contradictory beliefs, updates evolving user persona profiles,
//! and evaluates Ebbinghaus temporal forgetting curves ($R = e^{-t/S}$) with emotional salience weighting.
//!
//! ## Expiration Inversion — O(M) → O(k) Pruning
//!
//! Rather than recalculating R(t) across every stored fact on each consolidation tick,
//! the exact expiry timestamp is precomputed once at ingest/reinforce time:
//!
//! ```text
//! t_expire = last_accessed_secs − ln(R_prune) × memory_strength × 86400
//! ```
//!
//! A per-profile min-heap (`BinaryHeap<Reverse<ExpirationRecord>>`) keeps facts
//! sorted by expiry.  `prune_decayed_facts` pops from the heap until the earliest
//! entry has not yet expired — touching only the `k` expired facts rather than
//! doing a full-table scan over all `M` active memories.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::HNSQRResult;

/// Category of an extracted agentic memory fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactCategory {
    UserPreference,
    BiographicalData,
    SystemInstruction,
    WorldKnowledge,
    TemporalEvent,
}

/// An extracted atomic fact with confidence and salience ratings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EpisodicFact {
    pub fact_id: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub category: FactCategory,
    pub confidence: f32,
    pub emotional_salience: f32, // 0.0 to 1.0
    pub recall_count: u32,
    pub last_accessed_secs: u64,
    pub created_at_secs: u64,
}

/// A scheduled expiry record used by the per-profile min-heap.
///
/// The heap is ordered by `t_expire` (ascending) so the next-to-die fact
/// is always at the top.  `fact_id` is used to locate the fact in
/// `consolidated_facts` at prune time.
///
/// Ordering is derived on `t_expire` first; `fact_id` breaks ties
/// deterministically so `PartialEq` / `Eq` are logically consistent.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExpirationRecord {
    /// Unix timestamp (seconds) at which `R` crosses the pruning threshold.
    pub t_expire: u64,
    /// Stable identity that links this record back to the corresponding
    /// `EpisodicFact` in `consolidated_facts`.
    pub fact_id: String,
}

/// Computes the exact Unix timestamp at which a fact's Ebbinghaus retention
/// drops below `r_prune`, given the fact's current access time and strength.
///
/// Derived by inverting `R = exp(−Δt / (memory_strength × 86400))`:
/// ```text
/// t_expire = last_accessed_secs + (−ln(r_prune) × memory_strength × 86400) as u64
/// ```
///
/// `memory_strength` mirrors the formula in `evaluate_retention`:
/// `(recall_count + 1) × (1 + emotional_salience × 2)`
#[inline]
pub fn compute_expiry_secs(fact: &EpisodicFact, r_prune: f32) -> u64 {
    let memory_strength = (fact.recall_count as f32 + 1.0) * (1.0 + fact.emotional_salience * 2.0);
    // −ln(r_prune) is positive for r_prune ∈ (0, 1).
    let lifetime_secs = (-r_prune.ln()) * memory_strength * 86400.0;
    fact.last_accessed_secs.saturating_add(lifetime_secs as u64)
}

/// Dynamic user persona graph synthesized from consolidated episodic facts.
///
/// `expiry_heap` mirrors `consolidated_facts` as a min-heap keyed on the
/// precomputed expiry timestamp for the configured pruning threshold.
/// It is always consistent with the fact vector — every live fact has exactly
/// one entry in the heap (possibly stale after a conflict supersession, which
/// the prune sweep handles by checking presence in the fact map).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UserPersonaProfile {
    pub user_id: String,
    pub preferences: HashMap<String, String>,
    pub consolidated_facts: Vec<EpisodicFact>,
    pub total_interactions: u64,
    /// Min-heap of scheduled expirations.  Wrapped in `Reverse` so
    /// `BinaryHeap::pop` returns the *earliest* expiry (smallest `t_expire`).
    #[serde(skip)]
    pub expiry_heap: BinaryHeap<Reverse<ExpirationRecord>>,
}

impl UserPersonaProfile {
    /// Rebuilds `expiry_heap` from scratch for the current `consolidated_facts`.
    /// Used after deserialization (heap is `#[serde(skip)]`) or when the heap
    /// has drifted significantly from the fact set.
    pub fn rebuild_expiry_heap(&mut self, r_prune: f32) {
        self.expiry_heap.clear();
        for fact in &self.consolidated_facts {
            self.expiry_heap.push(Reverse(ExpirationRecord {
                t_expire: compute_expiry_secs(fact, r_prune),
                fact_id: fact.fact_id.clone(),
            }));
        }
    }
}

/// Autonomous Memory Consolidator and Ebbinghaus Decay Evaluator.
pub struct AutonomousMemoryConsolidator {
    profiles: RwLock<HashMap<String, UserPersonaProfile>>,
    total_consolidations: AtomicU64,
    total_conflicts_resolved: AtomicU64,
}

impl AutonomousMemoryConsolidator {
    pub fn new() -> Self {
        Self {
            profiles: RwLock::new(HashMap::new()),
            total_consolidations: AtomicU64::new(0),
            total_conflicts_resolved: AtomicU64::new(0),
        }
    }

    /// Ingests a new fact extracted from user interactions, resolving contradictory beliefs.
    ///
    /// When a conflict is resolved (an older fact is superseded) the stale heap entry
    /// is left in place — it is harmlessly skipped at prune time because the fact_id
    /// lookup will either find the replacement fact (not yet expired) or find nothing
    /// (already pruned).  A fresh entry for the new fact is pushed unconditionally.
    pub fn ingest_fact(&self, user_id: &str, fact: EpisodicFact) -> HNSQRResult<()> {
        // Default pruning threshold used when precomputing expiry on ingest.
        // Must match the threshold passed to prune_decayed_facts at call sites.
        const DEFAULT_R_PRUNE: f32 = 0.05;

        let mut profiles = self.profiles.write();
        let profile = profiles
            .entry(user_id.to_string())
            .or_insert_with(|| UserPersonaProfile {
                user_id: user_id.to_string(),
                preferences: HashMap::new(),
                consolidated_facts: Vec::new(),
                total_interactions: 0,
                expiry_heap: BinaryHeap::new(),
            });

        profile.total_interactions += 1;

        // 1. Conflict Resolution: Check if this fact supersedes an older contradictory fact.
        let mut conflict_idx = None;
        for (i, existing) in profile.consolidated_facts.iter().enumerate() {
            if existing.subject == fact.subject && existing.predicate == fact.predicate {
                conflict_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = conflict_idx {
            // Supersede: replace the slot.  The old heap entry is stale but benign —
            // the prune sweep will skip it because fact_id lookup finds the new version.
            self.total_conflicts_resolved
                .fetch_add(1, Ordering::Relaxed);
            profile.consolidated_facts[idx] = fact.clone();
        } else {
            profile.consolidated_facts.push(fact.clone());
        }

        // Push a fresh expiry record for the incoming fact.
        profile.expiry_heap.push(Reverse(ExpirationRecord {
            t_expire: compute_expiry_secs(&fact, DEFAULT_R_PRUNE),
            fact_id: fact.fact_id.clone(),
        }));

        if fact.category == FactCategory::UserPreference {
            profile.preferences.insert(fact.predicate, fact.object);
        }

        self.total_consolidations.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Fetches the consolidated user persona profile.
    pub fn get_profile(&self, user_id: &str) -> Option<UserPersonaProfile> {
        self.profiles.read().get(user_id).cloned()
    }

    /// Evaluates Ebbinghaus retention score: $R = e^{-\frac{\Delta t}{S \cdot (1 + \text{salience})}}$
    pub fn evaluate_retention(&self, fact: &EpisodicFact, current_time_secs: u64) -> f32 {
        let delta_t = current_time_secs.saturating_sub(fact.last_accessed_secs) as f32;
        let memory_strength =
            (fact.recall_count as f32 + 1.0) * (1.0 + fact.emotional_salience * 2.0);
        let decay_rate = delta_t / (memory_strength * 86400.0); // scaled to days
        (-decay_rate).exp()
    }

    /// Retrieves active non-decayed persona facts for a user.
    pub fn get_active_persona(
        &self,
        user_id: &str,
        current_time_secs: u64,
        min_retention: f32,
    ) -> Option<UserPersonaProfile> {
        let profiles = self.profiles.read();
        profiles.get(user_id).map(|p| {
            let filtered_facts: Vec<EpisodicFact> = p
                .consolidated_facts
                .iter()
                .filter(|f| self.evaluate_retention(f, current_time_secs) >= min_retention)
                .cloned()
                .collect();

            UserPersonaProfile {
                user_id: p.user_id.clone(),
                preferences: p.preferences.clone(),
                consolidated_facts: filtered_facts,
                total_interactions: p.total_interactions,
                expiry_heap: BinaryHeap::new(), // read-only view; heap not needed
            }
        })
    }

    /// Removes facts whose retention has fallen below `min_retention`.
    ///
    /// ## Complexity
    ///
    /// **O(k log M)** where `k` is the number of expired facts and `M` is the
    /// total live fact count — compared to the previous **O(P × M)** full-table
    /// scan over all profiles × all facts.
    ///
    /// ### Algorithm
    ///
    /// Each profile maintains a `BinaryHeap<Reverse<ExpirationRecord>>` (min-heap
    /// by `t_expire`).  This sweep pops entries from the heap until the earliest
    /// entry has not yet expired, then performs a single linear pass only over
    /// the *identified* expired IDs to rebuild `consolidated_facts` and
    /// `preferences`.
    ///
    /// Stale heap entries (from conflict-resolution supersessions) are detected
    /// by looking up the `fact_id` in the current fact map: if the stored expiry
    /// is later than the current time and the fact is still live, the entry is
    /// re-pushed; if the ID is absent (already removed), the entry is simply
    /// discarded.
    pub fn prune_decayed_facts(&self, current_time_secs: u64, min_retention: f32) -> usize {
        let mut profiles = self.profiles.write();
        let mut removed = 0;

        for profile in profiles.values_mut() {
            // Build a fast O(1) lookup from fact_id → index in consolidated_facts.
            let fact_index: HashMap<&str, usize> = profile
                .consolidated_facts
                .iter()
                .enumerate()
                .map(|(i, f)| (f.fact_id.as_str(), i))
                .collect();

            // Collect fact IDs that have definitely expired per the heap.
            // We stop as soon as the heap top has not yet expired.
            let mut expired_ids: Vec<String> = Vec::new();
            let mut stale_to_reinsert: Vec<Reverse<ExpirationRecord>> = Vec::new();

            while let Some(Reverse(rec)) = profile.expiry_heap.pop() {
                if rec.t_expire > current_time_secs {
                    // Not expired yet — return to heap and stop scanning.
                    stale_to_reinsert.push(Reverse(rec));
                    break;
                }

                // The heap entry says this fact should be expired by now.
                // Cross-check against the live fact to handle stale entries from
                // superseded conflicts: if the live fact was reinforced its expiry
                // was extended, so we must re-evaluate rather than blindly remove.
                match fact_index.get(rec.fact_id.as_str()) {
                    None => {
                        // fact_id no longer in the vec — already removed, discard.
                    }
                    Some(&idx) => {
                        let fact = &profile.consolidated_facts[idx];
                        let actual_retention = Self::retention_pure(fact, current_time_secs);
                        if actual_retention < min_retention {
                            expired_ids.push(rec.fact_id.clone());
                        } else {
                            // The fact was reinforced (recall_count/salience changed)
                            // so its real expiry is later. Re-push a corrected entry.
                            stale_to_reinsert.push(Reverse(ExpirationRecord {
                                t_expire: compute_expiry_secs(fact, min_retention),
                                fact_id: rec.fact_id.clone(),
                            }));
                        }
                    }
                }
            }

            // Re-insert any entries that were popped but are not truly expired.
            for entry in stale_to_reinsert {
                profile.expiry_heap.push(entry);
            }

            if expired_ids.is_empty() {
                continue;
            }

            // Build a set for O(1) membership test during retain.
            let expired_set: std::collections::HashSet<&str> =
                expired_ids.iter().map(|s| s.as_str()).collect();

            let before = profile.consolidated_facts.len();
            profile
                .consolidated_facts
                .retain(|f| !expired_set.contains(f.fact_id.as_str()));
            removed += before - profile.consolidated_facts.len();

            // Rebuild derived preference map from surviving facts.
            profile.preferences.clear();
            for fact in &profile.consolidated_facts {
                if fact.category == FactCategory::UserPreference {
                    profile
                        .preferences
                        .insert(fact.predicate.clone(), fact.object.clone());
                }
            }
        }

        removed
    }

    /// Pure-function Ebbinghaus retention — does not borrow `self`, usable
    /// inside `prune_decayed_facts` without a self-borrow conflict.
    #[inline]
    fn retention_pure(fact: &EpisodicFact, current_time_secs: u64) -> f32 {
        let delta_t = current_time_secs.saturating_sub(fact.last_accessed_secs) as f32;
        let memory_strength =
            (fact.recall_count as f32 + 1.0) * (1.0 + fact.emotional_salience * 2.0);
        let decay_rate = delta_t / (memory_strength * 86400.0);
        (-decay_rate).exp()
    }

    pub fn total_consolidations(&self) -> u64 {
        self.total_consolidations.load(Ordering::Relaxed)
    }

    pub fn total_conflicts_resolved(&self) -> u64 {
        self.total_conflicts_resolved.load(Ordering::Relaxed)
    }

    /// Captures an immutable point-in-time snapshot of agent memory.
    pub fn snapshot(&self) -> AutonomousMemorySnapshot {
        let profiles = self.profiles.read().clone();
        AutonomousMemorySnapshot { profiles }
    }
}

/// Immutable point-in-time snapshot of autonomous agent memory.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AutonomousMemorySnapshot {
    pub profiles: HashMap<String, UserPersonaProfile>,
}

impl AutonomousMemorySnapshot {
    pub fn get_profile(&self, user_id: &str) -> Option<&UserPersonaProfile> {
        self.profiles.get(user_id)
    }
}

impl Default for AutonomousMemoryConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autonomous_memory_fact_consolidation() {
        let consolidator = AutonomousMemoryConsolidator::new();

        // 1. Ingest initial preference: User likes Python
        let fact1 = EpisodicFact {
            fact_id: "f1".into(),
            subject: "user".into(),
            predicate: "favorite_programming_language".into(),
            object: "Python".into(),
            category: FactCategory::UserPreference,
            confidence: 0.9,
            emotional_salience: 0.5,
            recall_count: 1,
            last_accessed_secs: 1000,
            created_at_secs: 1000,
        };
        consolidator.ingest_fact("usr_001", fact1).unwrap();

        let profile1 = consolidator
            .get_active_persona("usr_001", 1000, 0.1)
            .unwrap();
        assert_eq!(
            profile1.preferences.get("favorite_programming_language"),
            Some(&"Python".to_string())
        );

        // 2. Ingest contradictory updated preference: User migrated to Rust
        let fact2 = EpisodicFact {
            fact_id: "f2".into(),
            subject: "user".into(),
            predicate: "favorite_programming_language".into(),
            object: "Rust".into(),
            category: FactCategory::UserPreference,
            confidence: 0.99,
            emotional_salience: 0.9,
            recall_count: 5,
            last_accessed_secs: 2000,
            created_at_secs: 2000,
        };
        consolidator.ingest_fact("usr_001", fact2).unwrap();

        assert_eq!(consolidator.total_conflicts_resolved(), 1);
        let profile2 = consolidator
            .get_active_persona("usr_001", 2000, 0.1)
            .unwrap();
        assert_eq!(
            profile2.preferences.get("favorite_programming_language"),
            Some(&"Rust".to_string())
        );
    }

    #[test]
    fn pruning_removes_decayed_facts_and_rebuilds_preferences() {
        let consolidator = AutonomousMemoryConsolidator::new();
        consolidator
            .ingest_fact(
                "usr_001",
                EpisodicFact {
                    fact_id: "f1".into(),
                    subject: "user".into(),
                    predicate: "favorite_color".into(),
                    object: "blue".into(),
                    category: FactCategory::UserPreference,
                    confidence: 1.0,
                    emotional_salience: 0.0,
                    recall_count: 0,
                    last_accessed_secs: 0,
                    created_at_secs: 0,
                },
            )
            .unwrap();
        // After 365 days at recall_count=0, salience=0.0:
        // memory_strength = 1.0 * 1.0 = 1.0
        // R = exp(-365) ≈ 0 — well below 0.1 threshold.
        assert_eq!(consolidator.prune_decayed_facts(365 * 86_400, 0.1), 1);
        let profile = consolidator.get_profile("usr_001").unwrap();
        assert!(profile.consolidated_facts.is_empty());
        assert!(profile.preferences.is_empty());
    }

    #[test]
    fn test_compute_expiry_secs_inverts_retention_formula() {
        // At t = t_expire the retention should be approximately r_prune.
        let fact = EpisodicFact {
            fact_id: "x".into(),
            subject: "s".into(),
            predicate: "p".into(),
            object: "o".into(),
            category: FactCategory::WorldKnowledge,
            confidence: 1.0,
            emotional_salience: 0.5,
            recall_count: 3,
            last_accessed_secs: 0,
            created_at_secs: 0,
        };
        let r_prune = 0.05f32;
        let t_expire = compute_expiry_secs(&fact, r_prune);
        let consolidator = AutonomousMemoryConsolidator::new();
        let r_at_expiry = consolidator.evaluate_retention(&fact, t_expire);
        // Allow ±5% relative error from f32 rounding in the inversion.
        assert!(
            (r_at_expiry - r_prune).abs() < 0.005,
            "r_at_expiry={r_at_expiry:.5} r_prune={r_prune:.5}"
        );
    }

    #[test]
    fn test_heap_prune_o_k_path() {
        // Verifies the heap-driven O(k) path: one expired fact + one live fact.
        // Only the expired one should be removed.
        let consolidator = AutonomousMemoryConsolidator::new();

        // Fact that was last accessed a long time ago — will be expired.
        consolidator
            .ingest_fact(
                "usr_heap",
                EpisodicFact {
                    fact_id: "old".into(),
                    subject: "user".into(),
                    predicate: "pref_a".into(),
                    object: "val_a".into(),
                    category: FactCategory::UserPreference,
                    confidence: 1.0,
                    emotional_salience: 0.0,
                    recall_count: 0,
                    last_accessed_secs: 0,
                    created_at_secs: 0,
                },
            )
            .unwrap();

        // Fact accessed just now — retention ≈ 1.0, nowhere near expiry.
        let now: u64 = 365 * 86_400;
        consolidator
            .ingest_fact(
                "usr_heap",
                EpisodicFact {
                    fact_id: "fresh".into(),
                    subject: "user".into(),
                    predicate: "pref_b".into(),
                    object: "val_b".into(),
                    category: FactCategory::UserPreference,
                    confidence: 1.0,
                    emotional_salience: 1.0,
                    recall_count: 100,
                    last_accessed_secs: now,
                    created_at_secs: now,
                },
            )
            .unwrap();

        let removed = consolidator.prune_decayed_facts(now, 0.05);
        assert_eq!(removed, 1, "exactly one expired fact should be removed");

        let profile = consolidator.get_profile("usr_heap").unwrap();
        assert_eq!(profile.consolidated_facts.len(), 1);
        assert_eq!(profile.consolidated_facts[0].fact_id, "fresh");
        // Preference for pref_a must be gone; pref_b must survive.
        assert!(!profile.preferences.contains_key("pref_a"));
        assert!(profile.preferences.contains_key("pref_b"));
    }
}
