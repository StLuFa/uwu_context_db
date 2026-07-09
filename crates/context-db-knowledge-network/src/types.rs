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
