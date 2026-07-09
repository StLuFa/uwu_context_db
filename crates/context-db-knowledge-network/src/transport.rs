use crate::planner::MeshQueryPlan;
use crate::types::{FetchResponse, PrivateQuerySketch, ProbeResponse, Result};
use agent_context_db_marketplace::{AgentId, FederatedDiscoveryHit, PublicationMetadata};
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::HashMap;

#[async_trait]
pub trait MeshTransport: Send + Sync {
    async fn probe(
        &self,
        peer: &AgentId,
        sketch: &PrivateQuerySketch,
        plan: &MeshQueryPlan,
    ) -> Result<ProbeResponse>;
    async fn fetch(
        &self,
        peer: &AgentId,
        sketch: &PrivateQuerySketch,
        plan: &MeshQueryPlan,
    ) -> Result<FetchResponse>;
    async fn gossip_capability(&self, _peer: &AgentId, _payload: Vec<u8>) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
pub struct InMemoryMeshTransport {
    publications: RwLock<HashMap<AgentId, Vec<PublicationMetadata>>>,
}

impl InMemoryMeshTransport {
    pub fn publish_for_peer(&self, peer: AgentId, publications: Vec<PublicationMetadata>) {
        self.publications.write().insert(peer, publications);
    }

    fn score(publication: &PublicationMetadata, sketch: &PrivateQuerySketch) -> f32 {
        let quality_gate = sketch.quality_bucket_min as f32 / 10.0;
        if publication.quality_score < quality_gate {
            return 0.0;
        }
        let domain_signal = if sketch.domain_bloom.iter().any(|v| *v > 0) {
            0.15
        } else {
            0.0
        };
        (publication.quality_score * 0.75
            + domain_signal
            + publication.corroboration.level as u8 as f32 * 0.025)
            .clamp(0.0, 1.0)
    }
}

#[async_trait]
impl MeshTransport for InMemoryMeshTransport {
    async fn probe(
        &self,
        peer: &AgentId,
        sketch: &PrivateQuerySketch,
        _plan: &MeshQueryPlan,
    ) -> Result<ProbeResponse> {
        let pubs = self.publications.read();
        let matches = pubs
            .get(peer)
            .map(|items| {
                items
                    .iter()
                    .filter(|p| Self::score(p, sketch) > 0.0)
                    .count()
            })
            .unwrap_or(0);
        let max_score = pubs
            .get(peer)
            .and_then(|items| {
                items
                    .iter()
                    .map(|p| Self::score(p, sketch))
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            })
            .unwrap_or(0.0);
        Ok(ProbeResponse {
            peer: peer.clone(),
            noisy_match_count: matches as u32,
            noisy_max_score: max_score,
            expected_latency_ms: 50,
        })
    }

    async fn fetch(
        &self,
        peer: &AgentId,
        sketch: &PrivateQuerySketch,
        plan: &MeshQueryPlan,
    ) -> Result<FetchResponse> {
        let mut hits = self
            .publications
            .read()
            .get(peer)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|publication| {
                let score = Self::score(&publication, sketch);
                (score > 0.0).then(|| FederatedDiscoveryHit {
                    publication,
                    relevance: score,
                    reputation_bonus: 0.5,
                    corroboration_bonus: 0.5,
                    lineage_independence: 0.75,
                    noisy_score: score,
                    final_score: score,
                    source_peer: Some(peer.clone()),
                })
            })
            .collect::<Vec<_>>();
        hits.sort_by(|a, b| {
            b.noisy_score
                .partial_cmp(&a.noisy_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(plan.top_k_per_peer);
        Ok(FetchResponse {
            peer: peer.clone(),
            noisy_total_available: hits.len() as u32,
            hits,
        })
    }
}
