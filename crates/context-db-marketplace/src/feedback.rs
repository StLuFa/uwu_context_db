//! Authenticated marketplace feedback and publisher reputation updates.

use crate::types::*;
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AdoptionOutcome {
    Adopted { confidence_gain: f32 },
    Rejected { reason: String },
    Outdated,
    Contradicted,
}

/// The complete signed feedback payload. The publisher is intentionally absent:
/// it is always resolved from the trusted registry by `entry_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackPayload {
    pub entry_id: MarketId,
    pub consumer: AgentId,
    pub outcome: AdoptionOutcome,
    pub evidence: Option<agent_context_db_core::ContextUri>,
    pub timestamp: DateTime<Utc>,
    pub replay_id: Uuid,
}

#[derive(Debug, Clone)]
pub struct MarketFeedback {
    pub payload: FeedbackPayload,
    pub public_key: String,
    pub signature: String,
}

pub trait FeedbackRegistry: Send + Sync {
    fn publisher_for(&self, entry_id: &MarketId) -> Option<AgentId>;
    fn has_consumption(&self, entry_id: &MarketId, consumer: &AgentId) -> bool;
}

pub trait FeedbackSignatureVerifier: Send + Sync {
    /// Verifies the signature and binds `public_key` to `payload.consumer`.
    fn verify(&self, payload: &FeedbackPayload, public_key: &str, signature: &str) -> bool;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeedbackError {
    UnknownEntry,
    NoConsumption,
    InvalidSignature,
    Stale,
    FutureDated,
    Replay,
    InvalidConfidenceGain,
}

pub struct ReputationEngine {
    kpis: parking_lot::RwLock<HashMap<AgentId, ReputationKpi>>,
    bonds: parking_lot::RwLock<HashMap<AgentId, ReputationBond>>,
    replay_ids: parking_lot::Mutex<HashSet<Uuid>>,
    max_feedback_age: Duration,
    max_future_skew: Duration,
}

impl Default for ReputationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ReputationEngine {
    pub fn new() -> Self {
        Self {
            kpis: parking_lot::RwLock::new(HashMap::new()),
            bonds: parking_lot::RwLock::new(HashMap::new()),
            replay_ids: parking_lot::Mutex::new(HashSet::new()),
            max_feedback_age: Duration::minutes(10),
            max_future_skew: Duration::seconds(30),
        }
    }

    pub fn register(&self, agent: AgentId) {
        self.kpis
            .write()
            .entry(agent.clone())
            .or_insert_with(|| ReputationKpi {
                agent: agent.clone(),
                last_active: Utc::now(),
                ..Default::default()
            });
        self.bonds
            .write()
            .entry(agent.clone())
            .or_insert_with(|| ReputationBond::new(agent));
    }

    /// The only feedback ingestion path. Validation finishes before the replay id
    /// is committed and before any publisher KPI can be changed.
    pub fn submit_feedback(
        &self,
        feedback: MarketFeedback,
        registry: &dyn FeedbackRegistry,
        verifier: &dyn FeedbackSignatureVerifier,
        now: DateTime<Utc>,
    ) -> Result<AgentId, FeedbackError> {
        let payload = &feedback.payload;
        if payload.timestamp < now - self.max_feedback_age {
            return Err(FeedbackError::Stale);
        }
        if payload.timestamp > now + self.max_future_skew {
            return Err(FeedbackError::FutureDated);
        }
        if !verifier.verify(payload, &feedback.public_key, &feedback.signature) {
            return Err(FeedbackError::InvalidSignature);
        }
        if let AdoptionOutcome::Adopted { confidence_gain } = payload.outcome
            && (!confidence_gain.is_finite() || !(0.0..=1.0).contains(&confidence_gain))
        {
            return Err(FeedbackError::InvalidConfidenceGain);
        }
        let publisher = registry
            .publisher_for(&payload.entry_id)
            .ok_or(FeedbackError::UnknownEntry)?;
        if !registry.has_consumption(&payload.entry_id, &payload.consumer) {
            return Err(FeedbackError::NoConsumption);
        }
        let mut replay_ids = self.replay_ids.lock();
        if !replay_ids.insert(payload.replay_id) {
            return Err(FeedbackError::Replay);
        }

        if !self.kpis.read().contains_key(&publisher) {
            // A trusted registry may learn a publisher before the reputation cache does.
            self.register(publisher.clone());
        }
        let mut kpis = self.kpis.write();
        let kpi = kpis.get_mut(&publisher).expect("publisher was registered");
        kpi.last_active = now;
        match payload.outcome {
            AdoptionOutcome::Adopted { .. } => {
                kpi.entries_published = kpi.entries_published.saturating_add(1);
                kpi.adoption_rate = (kpi.adoption_rate * 0.9 + 0.1).min(1.0);
            }
            AdoptionOutcome::Rejected { .. } | AdoptionOutcome::Contradicted => {
                kpi.downvote_count = kpi.downvote_count.saturating_add(1);
                kpi.adoption_rate = (kpi.adoption_rate * 0.9).max(0.0);
                if matches!(payload.outcome, AdoptionOutcome::Contradicted) {
                    kpi.contradiction_count = kpi.contradiction_count.saturating_add(1);
                }
            }
            AdoptionOutcome::Outdated => {}
        }
        kpi.recompute();
        if matches!(payload.outcome, AdoptionOutcome::Contradicted)
            && let Some(bond) = self.bonds.write().get_mut(&publisher)
        {
            bond.demote(BondLevel::Contributor);
        }
        Ok(publisher)
    }

    pub fn record_immune_contribution(&self, agent: &AgentId) {
        if let Some(kpi) = self.kpis.write().get_mut(agent) {
            kpi.immune_contributions = kpi.immune_contributions.saturating_add(1);
            kpi.recompute();
        }
    }

    pub fn get_kpi(&self, agent: &AgentId) -> Option<ReputationKpi> {
        self.kpis.read().get(agent).cloned()
    }

    pub fn recalc_bonds(&self) {
        let kpis = self.kpis.read();
        let mut bonds = self.bonds.write();
        for (agent, kpi) in kpis.iter() {
            if let Some(bond) = bonds.get_mut(agent) {
                bond.promote_if_qualified(kpi);
            }
        }
    }

    pub fn top_agents(&self, n: usize) -> Vec<(AgentId, f32)> {
        let mut sorted = self
            .kpis
            .read()
            .iter()
            .map(|(a, k)| (a.clone(), k.composite))
            .collect::<Vec<_>>();
        sorted.sort_by(|a, b| b.1.total_cmp(&a.1));
        sorted.truncate(n);
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Registry {
        entry: MarketId,
        publisher: AgentId,
        consumer: AgentId,
    }
    impl FeedbackRegistry for Registry {
        fn publisher_for(&self, id: &MarketId) -> Option<AgentId> {
            (*id == self.entry).then(|| self.publisher.clone())
        }
        fn has_consumption(&self, id: &MarketId, consumer: &AgentId) -> bool {
            *id == self.entry && consumer == &self.consumer
        }
    }
    struct Verifier;
    impl FeedbackSignatureVerifier for Verifier {
        fn verify(&self, p: &FeedbackPayload, key: &str, signature: &str) -> bool {
            key == p.consumer.as_str() && signature == "valid"
        }
    }
    fn feedback(entry: MarketId, consumer: &str, now: DateTime<Utc>) -> MarketFeedback {
        MarketFeedback {
            payload: FeedbackPayload {
                entry_id: entry,
                consumer: AgentId::new(consumer),
                outcome: AdoptionOutcome::Adopted {
                    confidence_gain: 0.1,
                },
                evidence: None,
                timestamp: now,
                replay_id: Uuid::new_v4(),
            },
            public_key: consumer.into(),
            signature: "valid".into(),
        }
    }

    #[test]
    fn authenticated_consumption_updates_only_registry_publisher_and_rejects_replay() {
        let now = Utc::now();
        let entry = MarketId::new();
        let publisher = AgentId::new("publisher");
        let consumer = AgentId::new("consumer");
        let registry = Registry {
            entry,
            publisher: publisher.clone(),
            consumer: consumer.clone(),
        };
        let engine = ReputationEngine::new();
        engine.register(consumer.clone());
        let signed = feedback(entry, "consumer", now);
        assert_eq!(
            engine.submit_feedback(signed.clone(), &registry, &Verifier, now),
            Ok(publisher.clone())
        );
        assert!(engine.get_kpi(&publisher).unwrap().adoption_rate > 0.0);
        assert_eq!(engine.get_kpi(&consumer).unwrap().adoption_rate, 0.0);
        assert_eq!(
            engine.submit_feedback(signed, &registry, &Verifier, now),
            Err(FeedbackError::Replay)
        );
    }

    #[test]
    fn rejects_unconsumed_bad_signature_and_stale_feedback() {
        let now = Utc::now();
        let entry = MarketId::new();
        let registry = Registry {
            entry,
            publisher: AgentId::new("publisher"),
            consumer: AgentId::new("other"),
        };
        let engine = ReputationEngine::new();
        assert_eq!(
            engine.submit_feedback(feedback(entry, "consumer", now), &registry, &Verifier, now),
            Err(FeedbackError::NoConsumption)
        );
        let mut bad = feedback(entry, "other", now);
        bad.signature = "bad".into();
        assert_eq!(
            engine.submit_feedback(bad, &registry, &Verifier, now),
            Err(FeedbackError::InvalidSignature)
        );
        assert_eq!(
            engine.submit_feedback(
                feedback(entry, "other", now - Duration::minutes(11)),
                &registry,
                &Verifier,
                now
            ),
            Err(FeedbackError::Stale)
        );
    }
}
