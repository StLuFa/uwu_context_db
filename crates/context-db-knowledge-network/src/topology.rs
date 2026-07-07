use crate::learning::RouteOutcomeLearning;
use crate::types::FederatedQueryIntent;
use agent_context_db_marketplace_types::AgentId;
use std::collections::HashSet;

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
