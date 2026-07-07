use crate::types::{KnowledgeNetworkError, PrivateQuerySketch, Result};
use agent_context_db_marketplace_types::{AgentId, DiscoveryQuery};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DpPolicy {
    pub enabled: bool,
    pub query_epsilon: f32,
    pub response_epsilon: f32,
    pub relay_epsilon: f32,
    pub delta: f32,
    pub clip_norm: f32,
    pub projected_dims: usize,
    pub min_k_anonymity: usize,
}

impl Default for DpPolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            query_epsilon: 1.0,
            response_epsilon: 0.5,
            relay_epsilon: 0.1,
            delta: 1e-6,
            clip_norm: 1.0,
            projected_dims: 128,
            min_k_anonymity: 3,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PrivacyCost {
    pub query_epsilon: f32,
    pub response_epsilon: f32,
    pub relay_epsilon: f32,
    pub delta: f32,
}

impl PrivacyCost {
    pub fn for_query(policy: &DpPolicy) -> Self {
        Self {
            query_epsilon: policy.query_epsilon,
            response_epsilon: policy.response_epsilon,
            relay_epsilon: policy.relay_epsilon,
            delta: policy.delta,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BudgetScope {
    Query,
    Peer(AgentId),
    Domain(String),
    Path(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyReceipt {
    pub id: Uuid,
    pub actor: AgentId,
    pub scope: BudgetScope,
    pub cost: PrivacyCost,
    pub reserved_at: DateTime<Utc>,
}

#[async_trait]
pub trait PrivacyBudgetLedger: Send + Sync {
    async fn reserve(
        &self,
        actor: &AgentId,
        scope: BudgetScope,
        cost: PrivacyCost,
    ) -> Result<PrivacyReceipt>;
    async fn commit(&self, receipt: &PrivacyReceipt) -> Result<()>;
    async fn refund(&self, receipt: &PrivacyReceipt) -> Result<()>;
}

#[derive(Debug)]
pub struct InMemoryPrivacyBudgetLedger {
    windows: RwLock<HashMap<String, (DateTime<Utc>, f32)>>,
    max_epsilon_per_window: f32,
    window: Duration,
}

impl InMemoryPrivacyBudgetLedger {
    pub fn new(max_epsilon_per_window: f32, window: Duration) -> Self {
        Self {
            windows: RwLock::new(HashMap::new()),
            max_epsilon_per_window,
            window,
        }
    }

    fn key(actor: &AgentId, scope: &BudgetScope) -> String {
        format!("{}::{scope:?}", actor.as_str())
    }
}

impl Default for InMemoryPrivacyBudgetLedger {
    fn default() -> Self {
        Self::new(16.0, Duration::hours(24))
    }
}

#[async_trait]
impl PrivacyBudgetLedger for InMemoryPrivacyBudgetLedger {
    async fn reserve(
        &self,
        actor: &AgentId,
        scope: BudgetScope,
        cost: PrivacyCost,
    ) -> Result<PrivacyReceipt> {
        let key = Self::key(actor, &scope);
        let mut windows = self.windows.write();
        let now = Utc::now();
        let entry = windows.entry(key).or_insert((now, 0.0));
        if now - entry.0 > self.window {
            *entry = (now, 0.0);
        }
        let total = cost.query_epsilon + cost.response_epsilon + cost.relay_epsilon;
        if entry.1 + total > self.max_epsilon_per_window {
            return Err(KnowledgeNetworkError::PrivacyBudgetExhausted(
                actor.to_string(),
            ));
        }
        entry.1 += total;
        Ok(PrivacyReceipt {
            id: Uuid::new_v4(),
            actor: actor.clone(),
            scope,
            cost,
            reserved_at: now,
        })
    }

    async fn commit(&self, _receipt: &PrivacyReceipt) -> Result<()> {
        Ok(())
    }

    async fn refund(&self, receipt: &PrivacyReceipt) -> Result<()> {
        let key = Self::key(&receipt.actor, &receipt.scope);
        if let Some((_start, spent)) = self.windows.write().get_mut(&key) {
            let total = receipt.cost.query_epsilon
                + receipt.cost.response_epsilon
                + receipt.cost.relay_epsilon;
            *spent = (*spent - total).max(0.0);
        }
        Ok(())
    }
}

pub trait DpMechanism: Send + Sync {
    fn protect_embedding(&self, embedding: &[f32], policy: &DpPolicy) -> Vec<f32>;
    fn protect_count(&self, count: u32, epsilon: f32) -> u32;
    fn protect_score(&self, score: f32, epsilon: f32) -> f32;
}

#[derive(Debug, Default)]
pub struct DeterministicDpMechanism;

impl DeterministicDpMechanism {
    fn noise(seed: impl Hash, scale: f32) -> f32 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        seed.hash(&mut hasher);
        let unit = (hasher.finish() % 10_000) as f32 / 10_000.0;
        (unit - 0.5) * 2.0 * scale
    }
}

impl DpMechanism for DeterministicDpMechanism {
    fn protect_embedding(&self, embedding: &[f32], policy: &DpPolicy) -> Vec<f32> {
        let dims = policy.projected_dims.min(embedding.len()).max(1);
        let norm = embedding
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-6);
        let clip = (policy.clip_norm / norm).min(1.0);
        (0..dims)
            .map(|i| {
                embedding.get(i).copied().unwrap_or_default() * clip
                    + Self::noise(i, 1.0 / policy.query_epsilon.max(0.01))
            })
            .collect()
    }

    fn protect_count(&self, count: u32, epsilon: f32) -> u32 {
        (count as f32 + Self::noise(count, 1.0 / epsilon.max(0.01)))
            .max(0.0)
            .round() as u32
    }

    fn protect_score(&self, score: f32, epsilon: f32) -> f32 {
        (score + Self::noise(score.to_bits(), 0.25 / epsilon.max(0.01))).clamp(0.0, 1.0)
    }
}

pub struct PrivacyGuard {
    pub policy: DpPolicy,
    pub mechanism: Arc<dyn DpMechanism>,
    pub budget_ledger: Arc<dyn PrivacyBudgetLedger>,
}

impl PrivacyGuard {
    pub fn new(policy: DpPolicy, budget_ledger: Arc<dyn PrivacyBudgetLedger>) -> Self {
        Self {
            policy,
            mechanism: Arc::new(DeterministicDpMechanism),
            budget_ledger,
        }
    }

    pub async fn protect_query(
        &self,
        actor: &AgentId,
        query: &DiscoveryQuery,
    ) -> Result<(PrivateQuerySketch, PrivacyReceipt)> {
        let cost = PrivacyCost::for_query(&self.policy);
        let receipt = self
            .budget_ledger
            .reserve(actor, BudgetScope::Query, cost)
            .await?;
        let projected = if self.policy.enabled {
            self.mechanism
                .protect_embedding(&query.query_embedding, &self.policy)
        } else {
            query.query_embedding.clone()
        };
        let embedding_lsh = projected
            .iter()
            .take(16)
            .map(|v| u64::from(v.to_bits()))
            .collect();
        let mut domain_bloom = vec![0u8; 32];
        for domain in &query.domains {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            domain.hash(&mut hasher);
            let idx = (hasher.finish() as usize) % domain_bloom.len();
            domain_bloom[idx] = 1;
        }
        let entry_type_mask = query
            .entry_types
            .iter()
            .fold(0u32, |acc, ty| acc | (1 << (*ty as u32)));
        Ok((
            PrivateQuerySketch {
                sketch_id: Uuid::new_v4(),
                embedding_lsh,
                projected_noisy_embedding: projected,
                domain_bloom,
                entry_type_mask,
                quality_bucket_min: (query.min_quality.clamp(0.0, 1.0) * 10.0) as u8,
                issued_at: Utc::now(),
            },
            receipt,
        ))
    }
}
