use agent_context_db_marketplace::{AgentId, FederatedDiscoveryHit, PublicationMetadata};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use uuid::Uuid;

pub type Result<T> = std::result::Result<T, KnowledgeNetworkError>;

#[derive(Debug, Error)]
pub enum KnowledgeNetworkError {
    #[error("privacy budget exhausted: {0}")]
    PrivacyBudgetExhausted(String),
    #[error("policy denied: {0}")]
    PolicyDenied(String),
    #[error("transport: {0}")]
    Transport(String),
    #[error("planner: {0}")]
    Planner(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederationRole {
    LeafAgent,
    TrustedPeer,
    DomainRouter,
    BridgeNode,
    RegistryMirror,
    ReputationAnchor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederatedQueryIntent {
    FastLookup,
    HighPrecision,
    HighRecall,
    PrivacyCritical,
    CrossDomainSynthesis,
    TrainingCandidate,
    CorroborationCheck,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederationReturnMode {
    FastPartial,
    Balanced,
    Exhaustive,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateQuerySketch {
    pub sketch_id: Uuid,
    pub embedding_lsh: Vec<u64>,
    pub projected_noisy_embedding: Vec<f32>,
    pub domain_bloom: Vec<u8>,
    pub entry_type_mask: u32,
    pub quality_bucket_min: u8,
    pub issued_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshDiscoveryOpts {
    pub intent: FederatedQueryIntent,
    pub return_mode: FederationReturnMode,
    pub max_peers: usize,
    pub probe_peers: usize,
    pub fetch_peers: usize,
    pub final_top_k: usize,
    pub deadline_ms: u64,
    pub peer_timeout_ms: u64,
    pub max_concurrency: usize,
}

impl MeshDiscoveryOpts {
    pub fn validate(&self) -> Result<()> {
        if self.max_peers == 0 {
            return Err(KnowledgeNetworkError::Planner(
                "max_peers must be greater than zero".into(),
            ));
        }
        if self.probe_peers == 0 || self.fetch_peers == 0 {
            return Err(KnowledgeNetworkError::Planner(
                "probe_peers and fetch_peers must be greater than zero".into(),
            ));
        }
        if self.probe_peers > self.max_peers || self.fetch_peers > self.max_peers {
            return Err(KnowledgeNetworkError::Planner(
                "probe_peers and fetch_peers must not exceed max_peers".into(),
            ));
        }
        if self.final_top_k == 0 {
            return Err(KnowledgeNetworkError::Planner(
                "final_top_k must be greater than zero".into(),
            ));
        }
        if self.deadline_ms == 0 || self.peer_timeout_ms == 0 {
            return Err(KnowledgeNetworkError::Planner(
                "deadline_ms and peer_timeout_ms must be greater than zero".into(),
            ));
        }
        if self.peer_timeout_ms > self.deadline_ms {
            return Err(KnowledgeNetworkError::Planner(
                "peer_timeout_ms must not exceed deadline_ms".into(),
            ));
        }
        if self.max_concurrency == 0 || self.max_concurrency > self.max_peers {
            return Err(KnowledgeNetworkError::Planner(
                "max_concurrency must be between 1 and max_peers".into(),
            ));
        }
        Ok(())
    }
}

impl Default for MeshDiscoveryOpts {
    fn default() -> Self {
        Self {
            intent: FederatedQueryIntent::FastLookup,
            return_mode: FederationReturnMode::FastPartial,
            max_peers: 16,
            probe_peers: 12,
            fetch_peers: 6,
            final_top_k: 20,
            deadline_ms: 800,
            peer_timeout_ms: 250,
            max_concurrency: 8,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbeResponse {
    pub peer: AgentId,
    pub noisy_match_count: u32,
    pub noisy_max_score: f32,
    pub expected_latency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FetchResponse {
    pub peer: AgentId,
    pub hits: Vec<FederatedDiscoveryHit>,
    pub noisy_total_available: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressiveMeshResult {
    pub phase: MeshResultPhase,
    pub hits: Vec<FederatedDiscoveryHit>,
    pub confidence: f32,
    pub missing_peer_count: usize,
    pub failed_peer_count: usize,
    pub timed_out_peer_count: usize,
    pub cancelled_peer_count: usize,
    pub privacy_receipts: Vec<crate::privacy::PrivacyReceipt>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshResultPhase {
    PartialFast,
    Refined,
    FinalStable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivatePeerPublication {
    pub peer: AgentId,
    pub publication: PublicationMetadata,
    pub noisy_score: f32,
}
