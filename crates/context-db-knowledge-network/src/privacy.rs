use crate::types::{KnowledgeNetworkError, PrivateQuerySketch, Result};
use agent_context_db_marketplace::{AgentId, DiscoveryQuery};
use async_trait::async_trait;
use chrono::{DateTime, Duration, Utc};
use parking_lot::RwLock;
use rand::rngs::OsRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use uuid::Uuid;

const RDP_ORDERS: [f64; 8] = [1.25, 1.5, 2.0, 3.0, 5.0, 8.0, 16.0, 32.0];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DpMechanismKind {
    Gaussian,
    Laplace,
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RdpCharge {
    pub orders: Vec<f64>,
    pub epsilons: Vec<f64>,
}

impl RdpCharge {
    pub fn gaussian(sensitivity: f64, sigma: f64) -> Self {
        let sigma = sigma.max(1e-12);
        let sensitivity_sq = sensitivity.max(0.0).powi(2);
        let epsilons = RDP_ORDERS
            .iter()
            .map(|alpha| alpha * sensitivity_sq / (2.0 * sigma.powi(2)))
            .collect::<Vec<_>>();
        Self {
            orders: RDP_ORDERS.to_vec(),
            epsilons,
        }
    }

    pub fn zero() -> Self {
        Self {
            orders: RDP_ORDERS.to_vec(),
            epsilons: vec![0.0; RDP_ORDERS.len()],
        }
    }

    pub fn epsilon_delta(&self, delta: f64) -> f64 {
        let log_delta = (1.0 / delta.max(1e-300)).ln();
        self.orders
            .iter()
            .zip(&self.epsilons)
            .filter(|(alpha, _)| **alpha > 1.0)
            .map(|(alpha, rdp)| rdp + log_delta / (alpha - 1.0))
            .fold(f64::INFINITY, f64::min)
    }

    fn add_assign(&mut self, other: &RdpCharge) {
        if self.orders != other.orders {
            return;
        }
        for (left, right) in self.epsilons.iter_mut().zip(&other.epsilons) {
            *left += right;
        }
    }

    fn sub_assign(&mut self, other: &RdpCharge) {
        if self.orders != other.orders {
            return;
        }
        for (left, right) in self.epsilons.iter_mut().zip(&other.epsilons) {
            *left = (*left - right).max(0.0);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyCost {
    pub query_epsilon: f32,
    pub response_epsilon: f32,
    pub relay_epsilon: f32,
    pub delta: f32,
    pub mechanism: DpMechanismKind,
    pub clip_norm: f32,
    pub noise_sigma: f32,
    pub rdp: RdpCharge,
}

impl PrivacyCost {
    pub fn for_query(policy: &DpPolicy) -> Self {
        let sigma = gaussian_sigma(policy.query_epsilon, policy.delta, policy.clip_norm);
        let rdp = if policy.enabled {
            RdpCharge::gaussian(policy.clip_norm as f64, sigma as f64)
        } else {
            RdpCharge::zero()
        };
        Self {
            query_epsilon: policy.query_epsilon,
            response_epsilon: policy.response_epsilon,
            relay_epsilon: policy.relay_epsilon,
            delta: policy.delta,
            mechanism: DpMechanismKind::Gaussian,
            clip_norm: policy.clip_norm,
            noise_sigma: sigma,
            rdp,
        }
    }

    pub fn nominal_epsilon(&self) -> f32 {
        self.query_epsilon + self.response_epsilon + self.relay_epsilon
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
    pub committed_at: Option<DateTime<Utc>>,
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

#[derive(Debug, Clone)]
struct BudgetWindow {
    started_at: DateTime<Utc>,
    rdp_spent: RdpCharge,
    reserved: HashMap<Uuid, PrivacyCost>,
    committed: HashSet<Uuid>,
}

impl BudgetWindow {
    fn new(started_at: DateTime<Utc>) -> Self {
        Self {
            started_at,
            rdp_spent: RdpCharge::zero(),
            reserved: HashMap::new(),
            committed: HashSet::new(),
        }
    }
}

#[derive(Debug)]
pub struct InMemoryPrivacyBudgetLedger {
    windows: RwLock<HashMap<String, BudgetWindow>>,
    max_epsilon_per_window: f32,
    max_delta_per_window: f32,
    window: Duration,
}

impl InMemoryPrivacyBudgetLedger {
    pub fn new(max_epsilon_per_window: f32, window: Duration) -> Self {
        Self {
            windows: RwLock::new(HashMap::new()),
            max_epsilon_per_window,
            max_delta_per_window: 1e-6,
            window,
        }
    }

    pub fn with_delta(mut self, max_delta_per_window: f32) -> Self {
        self.max_delta_per_window = max_delta_per_window;
        self
    }

    pub fn spent_epsilon_delta(&self, actor: &AgentId, scope: &BudgetScope) -> Option<f64> {
        self.windows
            .read()
            .get(&Self::key(actor, scope))
            .map(|window| {
                window
                    .rdp_spent
                    .epsilon_delta(self.max_delta_per_window as f64)
            })
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
        let window = windows.entry(key).or_insert_with(|| BudgetWindow::new(now));
        if now - window.started_at > self.window {
            *window = BudgetWindow::new(now);
        }

        let mut projected = window.rdp_spent.clone();
        projected.add_assign(&cost.rdp);
        let projected_epsilon = projected.epsilon_delta(self.max_delta_per_window as f64);
        if !projected_epsilon.is_finite() || projected_epsilon > self.max_epsilon_per_window as f64
        {
            return Err(KnowledgeNetworkError::PrivacyBudgetExhausted(format!(
                "{} projected epsilon {:.4} exceeds window {:.4}",
                actor, projected_epsilon, self.max_epsilon_per_window
            )));
        }

        let receipt = PrivacyReceipt {
            id: Uuid::new_v4(),
            actor: actor.clone(),
            scope,
            cost,
            reserved_at: now,
            committed_at: None,
        };
        window.rdp_spent = projected;
        window.reserved.insert(receipt.id, receipt.cost.clone());
        Ok(receipt)
    }

    async fn commit(&self, receipt: &PrivacyReceipt) -> Result<()> {
        let key = Self::key(&receipt.actor, &receipt.scope);
        let mut windows = self.windows.write();
        let window = windows.get_mut(&key).ok_or_else(|| {
            KnowledgeNetworkError::PrivacyBudgetExhausted("unknown privacy receipt".into())
        })?;
        if !window.reserved.contains_key(&receipt.id) {
            return Err(KnowledgeNetworkError::PrivacyBudgetExhausted(
                "privacy receipt was not reserved".into(),
            ));
        }
        window.committed.insert(receipt.id);
        Ok(())
    }

    async fn refund(&self, receipt: &PrivacyReceipt) -> Result<()> {
        let key = Self::key(&receipt.actor, &receipt.scope);
        if let Some(window) = self.windows.write().get_mut(&key) {
            if window.committed.contains(&receipt.id) {
                return Ok(());
            }
            if let Some(cost) = window.reserved.remove(&receipt.id) {
                window.rdp_spent.sub_assign(&cost.rdp);
            }
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
pub struct SamplingDpMechanism;

impl SamplingDpMechanism {
    fn gaussian_noise(sigma: f32) -> f32 {
        if sigma <= 0.0 || !sigma.is_finite() {
            return 0.0;
        }
        let normal = Normal::new(0.0, sigma as f64).expect("positive finite sigma");
        normal.sample(&mut OsRng) as f32
    }

    fn laplace_noise(scale: f32) -> f32 {
        if scale <= 0.0 || !scale.is_finite() {
            return 0.0;
        }
        // Inverse-CDF Laplace sampler using OS randomness through rand_distr Uniform.
        let u = rand::random::<f32>().clamp(f32::MIN_POSITIVE, 1.0 - f32::EPSILON) - 0.5;
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

impl DpMechanism for SamplingDpMechanism {
    fn protect_embedding(&self, embedding: &[f32], policy: &DpPolicy) -> Vec<f32> {
        let dims = policy.projected_dims.min(embedding.len()).max(1);
        let clipped = clip_l2(&embedding[..dims], policy.clip_norm.max(0.0));
        let sigma = gaussian_sigma(policy.query_epsilon, policy.delta, policy.clip_norm);
        clipped
            .into_iter()
            .map(|value| value + Self::gaussian_noise(sigma))
            .collect()
    }

    fn protect_count(&self, count: u32, epsilon: f32) -> u32 {
        let scale = 1.0 / epsilon.max(1e-6);
        (count as f32 + Self::laplace_noise(scale)).max(0.0).round() as u32
    }

    fn protect_score(&self, score: f32, epsilon: f32) -> f32 {
        let scale = 1.0 / epsilon.max(1e-6);
        (score + Self::laplace_noise(scale)).clamp(0.0, 1.0)
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
            mechanism: Arc::new(SamplingDpMechanism),
            budget_ledger,
        }
    }

    pub fn with_mechanism(mut self, mechanism: Arc<dyn DpMechanism>) -> Self {
        self.mechanism = mechanism;
        self
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

fn gaussian_sigma(epsilon: f32, delta: f32, sensitivity: f32) -> f32 {
    let epsilon = epsilon.max(1e-6) as f64;
    let delta = delta.clamp(1e-12, 0.999_999) as f64;
    let sensitivity = sensitivity.max(1e-6) as f64;
    (sensitivity * (2.0 * (1.25 / delta).ln()).sqrt() / epsilon) as f32
}

fn clip_l2(values: &[f32], clip_norm: f32) -> Vec<f32> {
    if values.is_empty() {
        return vec![];
    }
    let norm = values.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm <= clip_norm || norm <= 1e-12 {
        return values.to_vec();
    }
    let factor = clip_norm / norm;
    values.iter().map(|value| value * factor).collect()
}

pub fn has_k_anonymity(peer_count: usize, policy: &DpPolicy) -> bool {
    !policy.enabled || peer_count >= policy.min_k_anonymity.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query() -> DiscoveryQuery {
        DiscoveryQuery {
            query_embedding: vec![10.0, 0.0, 0.0, 0.0],
            domains: vec!["rust".into()],
            entry_types: vec![],
            min_quality: 0.5,
            min_corroboration_level: agent_context_db_marketplace::CorroborationLevel::Unverified,
            license_compatible: true,
        }
    }

    #[test]
    fn gaussian_embedding_noise_is_not_deterministic() {
        let mechanism = SamplingDpMechanism;
        let policy = DpPolicy {
            query_epsilon: 1.0,
            delta: 1e-6,
            clip_norm: 1.0,
            projected_dims: 4,
            ..Default::default()
        };
        let first = mechanism.protect_embedding(&[0.25, 0.25, 0.25, 0.25], &policy);
        let second = mechanism.protect_embedding(&[0.25, 0.25, 0.25, 0.25], &policy);
        assert_ne!(first, second);
    }

    #[test]
    fn clip_l2_limits_embedding_norm_before_noise() {
        let clipped = clip_l2(&[3.0, 4.0], 1.0);
        let norm = clipped.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn k_anonymity_requires_minimum_peer_count() {
        let policy = DpPolicy {
            min_k_anonymity: 3,
            ..Default::default()
        };
        assert!(!has_k_anonymity(2, &policy));
        assert!(has_k_anonymity(3, &policy));
    }

    #[tokio::test]
    async fn rdp_ledger_rejects_composed_budget_over_window() {
        let ledger = InMemoryPrivacyBudgetLedger::new(2.0, Duration::hours(1));
        let actor = AgentId::new("agent-a");
        let policy = DpPolicy {
            query_epsilon: 1.0,
            delta: 1e-6,
            clip_norm: 1.0,
            ..Default::default()
        };
        let cost = PrivacyCost::for_query(&policy);
        let mut accepted = 0;
        loop {
            match ledger
                .reserve(&actor, BudgetScope::Query, cost.clone())
                .await
            {
                Ok(receipt) => {
                    ledger.commit(&receipt).await.unwrap();
                    accepted += 1;
                    if accepted > 100 {
                        panic!("RDP ledger accepted too many composed queries");
                    }
                }
                Err(_) => break,
            }
        }
        assert!(accepted >= 1);
    }

    #[tokio::test]
    async fn refund_removes_uncommitted_rdp_charge() {
        let ledger = InMemoryPrivacyBudgetLedger::new(2.0, Duration::hours(1));
        let actor = AgentId::new("agent-a");
        let cost = PrivacyCost::for_query(&DpPolicy::default());
        let receipt = ledger
            .reserve(&actor, BudgetScope::Query, cost.clone())
            .await
            .unwrap();
        let spent_before = ledger
            .spent_epsilon_delta(&actor, &BudgetScope::Query)
            .unwrap();
        ledger.refund(&receipt).await.unwrap();
        let spent_after = ledger
            .spent_epsilon_delta(&actor, &BudgetScope::Query)
            .unwrap();
        assert!(spent_before > spent_after);
    }

    #[tokio::test]
    async fn protect_query_reserves_rdp_receipt() {
        let guard = PrivacyGuard::new(
            DpPolicy::default(),
            Arc::new(InMemoryPrivacyBudgetLedger::default()),
        );
        let (_sketch, receipt) = guard
            .protect_query(&AgentId::new("agent-a"), &query())
            .await
            .unwrap();
        assert_eq!(receipt.cost.mechanism, DpMechanismKind::Gaussian);
        assert!(
            receipt
                .cost
                .rdp
                .epsilon_delta(receipt.cost.delta as f64)
                .is_finite()
        );
    }
}
