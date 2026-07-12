use crate::learning::RouteOutcomeLearning;
use crate::types::FederatedQueryIntent;
use agent_context_db_marketplace::AgentId;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederationNodeConfig {
    pub id: AgentId,
    pub roles: Vec<crate::types::FederationRole>,
    pub endpoint: String,
    pub region: String,
    pub max_in_flight: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FederationTopologyConfig {
    pub local_node: AgentId,
    pub nodes: Vec<FederationNodeConfig>,
    pub required_regions: usize,
}

impl FederationTopologyConfig {
    pub fn validate(&self) -> crate::types::Result<()> {
        if self.nodes.len() < 2 {
            return Err(crate::types::KnowledgeNetworkError::PolicyDenied(
                "federation topology requires at least two nodes".into(),
            ));
        }
        let mut ids = HashSet::new();
        let mut endpoints = HashSet::new();
        let mut regions = HashSet::new();
        for node in &self.nodes {
            if node.id.as_str().is_empty()
                || node.endpoint.trim().is_empty()
                || node.region.trim().is_empty()
                || node.roles.is_empty()
                || node.max_in_flight == 0
            {
                return Err(crate::types::KnowledgeNetworkError::PolicyDenied(
                    "topology nodes require id, endpoint, region, role, and positive capacity"
                        .into(),
                ));
            }
            if !ids.insert(node.id.clone()) || !endpoints.insert(node.endpoint.clone()) {
                return Err(crate::types::KnowledgeNetworkError::PolicyDenied(
                    "topology node ids and endpoints must be unique".into(),
                ));
            }
            regions.insert(node.region.as_str());
        }
        if !ids.contains(&self.local_node) {
            return Err(crate::types::KnowledgeNetworkError::PolicyDenied(
                "local node is absent from federation topology".into(),
            ));
        }
        if self.required_regions == 0 || regions.len() < self.required_regions {
            return Err(crate::types::KnowledgeNetworkError::PolicyDenied(
                "federation topology does not satisfy required region diversity".into(),
            ));
        }
        Ok(())
    }

    pub fn capacities(&self) -> HashMap<AgentId, usize> {
        self.nodes
            .iter()
            .map(|node| (node.id.clone(), node.max_in_flight))
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct PeerConcurrencyLimits {
    limits: Arc<HashMap<AgentId, Arc<Semaphore>>>,
}

impl PeerConcurrencyLimits {
    pub fn from_topology(config: &FederationTopologyConfig) -> crate::types::Result<Self> {
        config.validate()?;
        Ok(Self {
            limits: Arc::new(
                config
                    .nodes
                    .iter()
                    .map(|node| {
                        (
                            node.id.clone(),
                            Arc::new(Semaphore::new(node.max_in_flight)),
                        )
                    })
                    .collect(),
            ),
        })
    }

    pub async fn acquire(&self, peer: &AgentId) -> crate::types::Result<OwnedSemaphorePermit> {
        self.limits
            .get(peer)
            .ok_or_else(|| {
                crate::types::KnowledgeNetworkError::PolicyDenied(format!(
                    "peer {} is absent from topology limits",
                    peer.as_str()
                ))
            })?
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| {
                crate::types::KnowledgeNetworkError::Transport(format!(
                    "peer {} concurrency limiter closed",
                    peer.as_str()
                ))
            })
    }
}

#[derive(Debug, Clone)]
pub struct TopologyOptimizer {
    pub min_peer_diversity: usize,
    pub learning_weight: f32,
}

impl Default for TopologyOptimizer {
    fn default() -> Self {
        Self {
            min_peer_diversity: 3,
            learning_weight: 0.18,
        }
    }
}

impl TopologyOptimizer {
    pub fn optimize_peer_order(
        &self,
        peers: &mut Vec<AgentId>,
        intent: FederatedQueryIntent,
        learning: &RouteOutcomeLearning,
    ) {
        let mut seen = HashSet::new();
        peers.retain(|peer| seen.insert(peer.clone()));
        peers.sort_by(|a, b| {
            let score_a = self.peer_score(a, intent, learning);
            let score_b = self.peer_score(b, intent, learning);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.as_str().cmp(b.as_str()))
        });
    }

    pub fn peer_score(
        &self,
        peer: &AgentId,
        intent: FederatedQueryIntent,
        learning: &RouteOutcomeLearning,
    ) -> f32 {
        let learning_bonus = learning.route_bonus(peer) * self.learning_weight;
        let intent_bonus = match intent {
            FederatedQueryIntent::FastLookup => 0.04,
            FederatedQueryIntent::HighPrecision => 0.06,
            FederatedQueryIntent::HighRecall | FederatedQueryIntent::CrossDomainSynthesis => 0.02,
            FederatedQueryIntent::PrivacyCritical => 0.08,
            FederatedQueryIntent::TrainingCandidate | FederatedQueryIntent::CorroborationCheck => {
                0.05
            }
        };
        (0.5 + learning_bonus + intent_bonus).clamp(0.0, 1.0)
    }
}
