use crate::types::{FederationRole, PrivateQuerySketch, Result};
use agent_context_db_marketplace_types::{AgentId, MarketEntryType};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoisyCentroid {
    pub vector: Vec<f32>,
    pub weight_bucket: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoisyHistogram {
    pub buckets: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilitySketch {
    pub peer: AgentId,
    pub role: FederationRole,
    pub domain_bloom: Vec<u8>,
    pub embedding_centroids: Vec<NoisyCentroid>,
    pub entry_type_histogram: NoisyHistogram,
    pub quality_histogram: NoisyHistogram,
    pub license_mask: u64,
    pub freshness_epoch: u64,
    pub reputation_hint: f32,
    pub signed_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
}

impl CapabilitySketch {
    pub fn simple(peer: AgentId, domains: &[String], reputation_hint: f32) -> Self {
        let mut domain_bloom = vec![0u8; 32];
        for domain in domains {
            let idx = domain
                .bytes()
                .fold(0usize, |acc, b| acc.wrapping_add(b as usize))
                % domain_bloom.len();
            domain_bloom[idx] = 1;
        }
        Self {
            peer,
            role: FederationRole::LeafAgent,
            domain_bloom,
            embedding_centroids: Vec::new(),
            entry_type_histogram: NoisyHistogram {
                buckets: vec![1; 5],
            },
            quality_histogram: NoisyHistogram {
                buckets: vec![1; 10],
            },
            license_mask: u64::MAX,
            freshness_epoch: Utc::now().timestamp() as u64,
            reputation_hint,
            signed_at: Utc::now(),
            expires_at: Utc::now() + chrono::Duration::hours(1),
        }
    }
}

#[derive(Debug, Clone)]
pub struct PeerCandidate {
    pub peer: AgentId,
    pub capability_match: f32,
    pub reputation_hint: f32,
    pub role: FederationRole,
}

#[async_trait]
pub trait CapabilityIndex: Send + Sync {
    async fn upsert(&self, sketch: CapabilitySketch) -> Result<()>;
    async fn candidate_peers(
        &self,
        query: &PrivateQuerySketch,
        limit: usize,
    ) -> Result<Vec<PeerCandidate>>;
}

#[derive(Default)]
pub struct InMemoryCapabilityIndex {
    sketches: RwLock<HashMap<AgentId, CapabilitySketch>>,
}

#[async_trait]
impl CapabilityIndex for InMemoryCapabilityIndex {
    async fn upsert(&self, sketch: CapabilitySketch) -> Result<()> {
        self.sketches.write().insert(sketch.peer.clone(), sketch);
        Ok(())
    }

    async fn candidate_peers(
        &self,
        query: &PrivateQuerySketch,
        limit: usize,
    ) -> Result<Vec<PeerCandidate>> {
        let mut candidates: Vec<_> = self
            .sketches
            .read()
            .values()
            .filter(|sketch| sketch.expires_at > Utc::now())
            .map(|sketch| {
                let overlap = sketch
                    .domain_bloom
                    .iter()
                    .zip(query.domain_bloom.iter())
                    .filter(|(a, b)| **a > 0 && **b > 0)
                    .count() as f32;
                let denom = query.domain_bloom.iter().filter(|v| **v > 0).count().max(1) as f32;
                PeerCandidate {
                    peer: sketch.peer.clone(),
                    capability_match: (overlap / denom).clamp(0.0, 1.0),
                    reputation_hint: sketch.reputation_hint,
                    role: sketch.role,
                }
            })
            .collect();
        candidates.sort_by(|a, b| {
            b.capability_match
                .partial_cmp(&a.capability_match)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(limit);
        Ok(candidates)
    }
}

pub fn entry_type_mask(types: &[MarketEntryType]) -> u32 {
    types.iter().fold(0u32, |acc, ty| acc | (1 << (*ty as u32)))
}
