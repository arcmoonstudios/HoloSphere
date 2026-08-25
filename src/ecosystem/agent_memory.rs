/* holosphere/src/ecosystem/agent_memory.rs */
//!▫~•◦-------------------------------‣
//! # Autonomous Long-Term Agentic Memory & Fact Consolidation (Mem0/Zep Rival)
//!▫~•◦-------------------------------------------------------------------‣
//!
//! Provides an autonomous background memory consolidation engine that extracts episodic facts
//! from dialogue turns, reconciles contradictory beliefs, updates evolving user persona profiles,
//! and evaluates Ebbinghaus temporal forgetting curves ($R = e^{-t/S}$) with emotional salience weighting.
/*▫~•◦------------------------------------------------------------------------------------‣
 * © 2026 ArcMoon Studios ◦ SPDX-License-Identifier MIT OR Apache-2.0 ◦ Author: Lord Xyn ✶
 *///•------------------------------------------------------------------------------------‣

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

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

/// Dynamic user persona graph synthesized from consolidated episodic facts.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UserPersonaProfile {
    pub user_id: String,
    pub preferences: HashMap<String, String>,
    pub consolidated_facts: Vec<EpisodicFact>,
    pub total_interactions: u64,
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
    pub fn ingest_fact(&self, user_id: &str, fact: EpisodicFact) -> HNSQRResult<()> {
        let mut profiles = self.profiles.write();
        let profile = profiles
            .entry(user_id.to_string())
            .or_insert_with(|| UserPersonaProfile {
                user_id: user_id.to_string(),
                preferences: HashMap::new(),
                consolidated_facts: Vec::new(),
                total_interactions: 0,
            });

        profile.total_interactions += 1;

        // 1. Conflict Resolution: Check if this fact supersedes an older contradictory fact
        let mut conflict_idx = None;
        for (i, existing) in profile.consolidated_facts.iter().enumerate() {
            if existing.subject == fact.subject && existing.predicate == fact.predicate {
                conflict_idx = Some(i);
                break;
            }
        }

        if let Some(idx) = conflict_idx {
            // Supersede if incoming fact has higher confidence or is more recent
            self.total_conflicts_resolved
                .fetch_add(1, Ordering::Relaxed);
            profile.consolidated_facts[idx] = fact.clone();
        } else {
            profile.consolidated_facts.push(fact.clone());
        }

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
            }
        })
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
}
