use crate::capability::PeerCandidate;
use agent_context_db_marketplace::{AgentId, BondLevel};
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct PeerTrustProfile {
    pub peer: AgentId,
    pub bond_level: BondLevel,
    pub reputation: f32,
    pub latency_score: f32,
    pub failure_rate: f32,
}

impl Default for PeerTrustProfile {
    fn default() -> Self {
        Self {
            peer: AgentId::default(),
            bond_level: BondLevel::Observer,
            reputation: 0.5,
            latency_score: 0.5,
            failure_rate: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PeerRouteScore {
    pub peer: AgentId,
    pub capability_match: f32,
    pub trust: f32,
    pub reputation: f32,
    pub latency_score: f32,
    pub budget_score: f32,
    pub diversity_score: f32,
    pub final_score: f32,
}

#[derive(Debug, Clone)]
pub struct TrustRouterConfig {
    pub min_reputation: f32,
    pub min_bond_level: BondLevel,
}

impl Default for TrustRouterConfig {
    fn default() -> Self {
        Self {
            min_reputation: 0.25,
            min_bond_level: BondLevel::Observer,
        }
    }
}

pub struct TrustRouter {
    config: TrustRouterConfig,
    profiles: RwLock<HashMap<AgentId, PeerTrustProfile>>,
    blacklist: RwLock<HashSet<AgentId>>,
}

impl TrustRouter {
    pub fn new(config: TrustRouterConfig) -> Self {
        Self {
            config,
            profiles: RwLock::new(HashMap::new()),
            blacklist: RwLock::new(HashSet::new()),
        }
    }

    pub fn upsert_profile(&self, profile: PeerTrustProfile) {
        self.profiles.write().insert(profile.peer.clone(), profile);
    }

    pub fn route(&self, candidates: Vec<PeerCandidate>, limit: usize) -> Vec<PeerRouteScore> {
        let profiles = self.profiles.read();
        let blacklist = self.blacklist.read();
        let mut scores: Vec<_> =
            candidates
                .into_iter()
                .filter(|candidate| !blacklist.contains(&candidate.peer))
                .filter_map(|candidate| {
                    let profile = profiles.get(&candidate.peer).cloned().unwrap_or_else(|| {
                        PeerTrustProfile {
                            peer: candidate.peer.clone(),
                            reputation: candidate.reputation_hint,
                            ..Default::default()
                        }
                    });
                    if profile.reputation < self.config.min_reputation
                        || profile.bond_level < self.config.min_bond_level
                    {
                        return None;
                    }
                    let trust = match profile.bond_level {
                        BondLevel::Observer => 0.35,
                        BondLevel::Contributor => 0.55,
                        BondLevel::Validator => 0.80,
                        BondLevel::Authority => 1.0,
                    };
                    let budget_score = 1.0;
                    let diversity_score = 0.75;
                    let final_score = candidate.capability_match * 0.30
                        + trust * 0.20
                        + profile.reputation * 0.15
                        + profile.latency_score * 0.15
                        + budget_score * 0.10
                        + diversity_score * 0.10
                        - profile.failure_rate * 0.20;
                    Some(PeerRouteScore {
                        peer: candidate.peer,
                        capability_match: candidate.capability_match,
                        trust,
                        reputation: profile.reputation,
                        latency_score: profile.latency_score,
                        budget_score,
                        diversity_score,
                        final_score: final_score.clamp(0.0, 1.0),
                    })
                })
                .collect();
        scores.sort_by(|a, b| {
            b.final_score
                .partial_cmp(&a.final_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(limit);
        scores
    }
}

impl Default for TrustRouter {
    fn default() -> Self {
        Self::new(TrustRouterConfig::default())
    }
}
