//! Shared marketplace and federated discovery data transfer types.
//!
//! This crate intentionally contains no consolidation product payloads. It is the
//! narrow boundary shared by local marketplace code and KnowledgeNetwork.

use agent_context_db_core::{ContentType, ContextUri, EpistemicType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MarketId(pub Uuid);

impl MarketId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for MarketId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BondLevel {
    Observer = 0,
    Contributor = 1,
    Validator = 2,
    Authority = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketEntryType {
    Fact,
    Skill,
    Procedure,
    Antibody,
    ErrorPattern,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CorroborationLevel {
    Unverified = 0,
    SingleSession = 1,
    CrossSession = 2,
    CrossAgent = 3,
    Established = 4,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorroborationProof {
    pub corroborators: Vec<AgentId>,
    pub total_count: usize,
    pub independent_sources: usize,
    pub level: CorroborationLevel,
}

impl CorroborationProof {
    pub fn new() -> Self {
        Self {
            corroborators: vec![],
            total_count: 0,
            independent_sources: 0,
            level: CorroborationLevel::Unverified,
        }
    }

    pub fn can_publish(&self) -> bool {
        self.level >= CorroborationLevel::CrossSession
    }

    pub fn add_corroboration(&mut self, agent: AgentId, session_count: usize) {
        if !self.corroborators.contains(&agent) {
            self.corroborators.push(agent);
            self.independent_sources += 1;
        }
        self.total_count += session_count;
        self.level = if self.independent_sources >= 5 && self.total_count >= 10 {
            CorroborationLevel::Established
        } else if self.independent_sources >= 2 {
            CorroborationLevel::CrossAgent
        } else if self.total_count >= 2 {
            CorroborationLevel::CrossSession
        } else if self.total_count >= 1 {
            CorroborationLevel::SingleSession
        } else {
            CorroborationLevel::Unverified
        };
    }
}

impl Default for CorroborationProof {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseInfo {
    pub scope: LicenseScope,
    pub attribution_required: bool,
    pub commercial_use: bool,
    pub derivative_allowed: bool,
}

impl Default for LicenseInfo {
    fn default() -> Self {
        Self {
            scope: LicenseScope::Open,
            attribution_required: true,
            commercial_use: false,
            derivative_allowed: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LicenseScope {
    Open,
    NonCommercial,
    TenantOnly,
    RequiresApproval,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LineageNode {
    pub market_id: MarketId,
    pub publisher: AgentId,
    pub action: LineageAction,
    pub parent_ids: Vec<MarketId>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LineageAction {
    Origin,
    Derived,
    Merged,
    Corrected,
    Deprecated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationMetadata {
    pub id: MarketId,
    pub publisher: AgentId,
    pub domain: String,
    pub entry_type: MarketEntryType,
    pub source_uri: Option<ContextUri>,
    pub quality_score: f32,
    pub corroboration: CorroborationProof,
    pub license: LicenseInfo,
    pub epistemic_type: EpistemicType,
    pub content_type: ContentType,
    pub half_life_days: Option<f64>,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryQuery {
    pub query_embedding: Vec<f32>,
    pub domains: Vec<String>,
    pub entry_types: Vec<MarketEntryType>,
    pub min_quality: f32,
    pub min_corroboration_level: CorroborationLevel,
    pub license_compatible: bool,
}

impl Default for DiscoveryQuery {
    fn default() -> Self {
        Self {
            query_embedding: Vec::new(),
            domains: Vec::new(),
            entry_types: Vec::new(),
            min_quality: 0.0,
            min_corroboration_level: CorroborationLevel::Unverified,
            license_compatible: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchTier {
    Local,
    Cache,
    Federation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedDiscoveryHit {
    pub publication: PublicationMetadata,
    pub relevance: f32,
    pub reputation_bonus: f32,
    pub corroboration_bonus: f32,
    pub lineage_independence: f32,
    pub noisy_score: f32,
    pub final_score: f32,
    pub source_peer: Option<AgentId>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedDiscoveryResult {
    pub hits: Vec<FederatedDiscoveryHit>,
    pub total_available: usize,
    pub domains_covered: Vec<String>,
    pub avg_quality: f32,
    pub search_tier: SearchTier,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub agent_id: AgentId,
    pub domains: Vec<String>,
    pub bond_level: BondLevel,
    pub last_seen: DateTime<Utc>,
}
