use crate::capability::{CapabilitySketch, PeerCandidate};
use crate::types::PrivateQuerySketch;
use agent_context_db_marketplace_types::AgentId;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct SemanticCapabilityNode {
    pub peer: AgentId,
    pub domains: HashSet<String>,
    pub capability_score: f32,
    pub freshness_epoch: u64,
}

#[derive(Debug, Default)]
pub struct SemanticCapabilityGraph {
    nodes: RwLock<HashMap<AgentId, SemanticCapabilityNode>>,
    domain_edges: RwLock<HashMap<String, HashSet<AgentId>>>,
}

impl SemanticCapabilityGraph {
    pub fn observe_capability(
        &self,
        sketch: &CapabilitySketch,
        domains: impl IntoIterator<Item = String>,
    ) {
        let domains = domains.into_iter().collect::<HashSet<_>>();
        let node = SemanticCapabilityNode {
            peer: sketch.peer.clone(),
            domains: domains.clone(),
            capability_score: sketch.reputation_hint.clamp(0.0, 1.0),
            freshness_epoch: sketch.freshness_epoch,
        };
        self.nodes.write().insert(sketch.peer.clone(), node);
        let mut edges = self.domain_edges.write();
        for domain in domains {
            edges.entry(domain).or_default().insert(sketch.peer.clone());
        }
    }

    pub fn boost_candidates(
        &self,
        query_domains: &[String],
        _query: &PrivateQuerySketch,
        candidates: &mut [PeerCandidate],
    ) {
        if query_domains.is_empty() {
            return;
        }
        let edges = self.domain_edges.read();
        let mut semantic_peers = HashSet::new();
        for domain in query_domains {
            if let Some(peers) = edges.get(domain) {
                semantic_peers.extend(peers.iter().cloned());
            }
        }
        if semantic_peers.is_empty() {
            return;
        }
        for candidate in candidates {
            if semantic_peers.contains(&candidate.peer) {
                candidate.capability_match = (candidate.capability_match + 0.12).clamp(0.0, 1.0);
            }
        }
    }

    pub fn related_peers(&self, query_domains: &[String], limit: usize) -> Vec<AgentId> {
        let edges = self.domain_edges.read();
        let mut scored = HashMap::<AgentId, usize>::new();
        for domain in query_domains {
            if let Some(peers) = edges.get(domain) {
                for peer in peers {
                    *scored.entry(peer.clone()).or_default() += 1;
                }
            }
        }
        let mut peers = scored.into_iter().collect::<Vec<_>>();
        peers.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.as_str().cmp(b.0.as_str())));
        peers
            .into_iter()
            .take(limit)
            .map(|(peer, _)| peer)
            .collect()
    }
}
