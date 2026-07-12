use crate::planner::MeshQueryPlan;
use crate::privacy::DpPolicy;
use crate::types::{FederatedQueryIntent, KnowledgeNetworkError, Result};
use agent_context_db_marketplace::{AgentId, DiscoveryQuery, FederatedDiscoveryHit};
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    time::{Duration, Instant},
};

#[derive(Debug, Clone)]
pub struct GovernancePolicy {
    pub min_query_quality: f32,
    pub require_license_compatibility: bool,
    pub require_k_anonymity_for_private_queries: bool,
    pub max_fetch_peers: usize,
    pub allow_training_candidates: bool,
    pub max_queries_per_window: u64,
    pub quota_window: Duration,
    pub max_concurrent_queries: usize,
    pub max_tracked_actors: usize,
}

impl Default for GovernancePolicy {
    fn default() -> Self {
        Self {
            min_query_quality: 0.0,
            require_license_compatibility: true,
            require_k_anonymity_for_private_queries: true,
            max_fetch_peers: 32,
            allow_training_candidates: true,
            max_queries_per_window: 1_000,
            quota_window: Duration::from_secs(60),
            max_concurrent_queries: 64,
            max_tracked_actors: 10_000,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GovernanceEngine {
    pub policy: GovernancePolicy,
    quotas: std::sync::Arc<Mutex<HashMap<AgentId, QuotaState>>>,
}

#[derive(Debug, Clone)]
struct QuotaState {
    window_started: Instant,
    accepted: u64,
    in_flight: usize,
}

pub struct QueryPermit {
    actor: AgentId,
    quotas: std::sync::Arc<Mutex<HashMap<AgentId, QuotaState>>>,
}

impl Drop for QueryPermit {
    fn drop(&mut self) {
        if let Some(state) = self.quotas.lock().get_mut(&self.actor) {
            state.in_flight = state.in_flight.saturating_sub(1);
        }
    }
}

impl GovernanceEngine {
    pub fn validate(&self) -> Result<()> {
        if self.policy.max_fetch_peers == 0
            || self.policy.max_queries_per_window == 0
            || self.policy.quota_window.is_zero()
            || self.policy.max_concurrent_queries == 0
            || self.policy.max_tracked_actors == 0
        {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "governance quotas and fanout limits must be positive".into(),
            ));
        }
        Ok(())
    }

    pub fn acquire_query(&self, actor: &AgentId) -> Result<QueryPermit> {
        self.validate()?;
        let now = Instant::now();
        let mut quotas = self.quotas.lock();
        quotas.retain(|_, state| {
            state.in_flight > 0
                || now.duration_since(state.window_started) < self.policy.quota_window
        });
        if !quotas.contains_key(actor) && quotas.len() >= self.policy.max_tracked_actors {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "federation actor quota capacity exhausted".into(),
            ));
        }
        let state = quotas.entry(actor.clone()).or_insert(QuotaState {
            window_started: now,
            accepted: 0,
            in_flight: 0,
        });
        if now.duration_since(state.window_started) >= self.policy.quota_window {
            state.window_started = now;
            state.accepted = 0;
        }
        if state.accepted >= self.policy.max_queries_per_window {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "federation query rate quota exhausted".into(),
            ));
        }
        if state.in_flight >= self.policy.max_concurrent_queries {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "federation concurrent query quota exhausted".into(),
            ));
        }
        state.accepted += 1;
        state.in_flight += 1;
        drop(quotas);
        Ok(QueryPermit {
            actor: actor.clone(),
            quotas: self.quotas.clone(),
        })
    }

    pub fn authorize_query(
        &self,
        actor: &AgentId,
        query: &DiscoveryQuery,
        intent: FederatedQueryIntent,
        dp_policy: &DpPolicy,
    ) -> Result<()> {
        if actor.as_str().is_empty() {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "anonymous federation actor".into(),
            ));
        }
        if query.min_quality < self.policy.min_query_quality {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "query quality threshold below governance policy".into(),
            ));
        }
        if self.policy.require_license_compatibility && !query.license_compatible {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "license-compatible discovery required".into(),
            ));
        }
        if self.policy.require_k_anonymity_for_private_queries
            && matches!(intent, FederatedQueryIntent::PrivacyCritical)
            && dp_policy.min_k_anonymity < 2
        {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "privacy-critical query requires k-anonymity".into(),
            ));
        }
        if matches!(intent, FederatedQueryIntent::TrainingCandidate)
            && !self.policy.allow_training_candidates
        {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "training candidate discovery disabled".into(),
            ));
        }
        Ok(())
    }

    pub fn authorize_plan(&self, plan: &MeshQueryPlan) -> Result<()> {
        if plan.fetch_peers.len() > self.policy.max_fetch_peers {
            return Err(KnowledgeNetworkError::PolicyDenied(
                "fetch peer fanout exceeds governance policy".into(),
            ));
        }
        Ok(())
    }

    pub fn filter_hits(&self, hits: Vec<FederatedDiscoveryHit>) -> Vec<FederatedDiscoveryHit> {
        hits.into_iter()
            .filter(|hit| {
                !self.policy.require_license_compatibility
                    || hit.publication.license.derivative_allowed
            })
            .collect()
    }
}
