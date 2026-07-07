use crate::planner::MeshQueryPlan;
use crate::privacy::DpPolicy;
use crate::types::{FederatedQueryIntent, KnowledgeNetworkError, Result};
use agent_context_db_marketplace_types::{AgentId, DiscoveryQuery, FederatedDiscoveryHit};

#[derive(Debug, Clone)]
pub struct GovernancePolicy {
    pub min_query_quality: f32,
    pub require_license_compatibility: bool,
    pub require_k_anonymity_for_private_queries: bool,
    pub max_fetch_peers: usize,
    pub allow_training_candidates: bool,
}

impl Default for GovernancePolicy {
    fn default() -> Self {
        Self {
            min_query_quality: 0.0,
            require_license_compatibility: true,
            require_k_anonymity_for_private_queries: true,
            max_fetch_peers: 32,
            allow_training_candidates: true,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GovernanceEngine {
    pub policy: GovernancePolicy,
}

impl GovernanceEngine {
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
