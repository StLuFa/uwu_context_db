//! 市场核心数据类型 — C2D 边界 + 声誉 KPI + 血统 DAG。

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeProvenancePayload {
    pub publisher: AgentId,
    pub content: String,
    pub evidence_uris: Vec<ContextUri>,
    pub evidence_chain_hash: String,
    pub quality_score: f32,
    pub confidence: f32,
    pub epistemic_type: EpistemicType,
    pub content_type: ContentType,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeProvenance {
    pub publisher: AgentId,
    pub public_key: String,
    pub signature: String,
    pub evidence_chain_hash: String,
    pub signed_at: DateTime<Utc>,
}

pub fn evidence_chain_hash(evidence_uris: &[ContextUri]) -> String {
    let mut values = evidence_uris
        .iter()
        .map(|uri| uri.as_str().to_string())
        .collect::<Vec<_>>();
    values.sort();
    blake3::hash(values.join("\n").as_bytes())
        .to_hex()
        .to_string()
}

pub fn provenance_payload_hash(
    payload: &KnowledgeProvenancePayload,
) -> std::result::Result<String, serde_json::Error> {
    let bytes = serde_json::to_vec(payload)?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    pub provenance: Option<KnowledgeProvenance>,
    pub license: LicenseInfo,
    pub epistemic_type: EpistemicType,
    pub content_type: ContentType,
    pub half_life: Option<agent_context_db_core::HalfLife>,
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
            min_quality: 0.7,
            min_corroboration_level: CorroborationLevel::CrossSession,
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

/// 市场条目 — Agent 发布到市场的知识晶体。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEntry {
    pub id: MarketId,
    pub publisher: AgentId,
    pub domain: String,
    pub entry_type: MarketEntryType,
    /// 精炼后的知识原则（不是原始 session！）
    pub principle: String,
    /// 支撑证据的 URI（不包含内容，只含引用）
    pub evidence_uris: Vec<ContextUri>,
    /// 质量分
    pub quality_score: f32,
    /// 置信度
    pub confidence: f32,
    /// 确认证明：哪些 Agent 独立确认了此知识
    pub corroboration: CorroborationProof,
    /// 发布者签名与证据链哈希，用于离线验证知识来源与抗篡改。
    pub provenance: Option<KnowledgeProvenance>,
    /// 许可证
    pub license: KnowledgeLicense,
    pub epistemic_type: EpistemicType,
    pub content_type: ContentType,
    pub half_life: Option<agent_context_db_core::HalfLife>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 过期时间（半衰期驱动）
    pub expires_at: Option<DateTime<Utc>>,
}

impl MarketEntry {
    pub fn provenance_payload(&self) -> KnowledgeProvenancePayload {
        KnowledgeProvenancePayload {
            publisher: self.publisher.clone(),
            content: self.principle.clone(),
            evidence_chain_hash: evidence_chain_hash(&self.evidence_uris),
            evidence_uris: self.evidence_uris.clone(),
            quality_score: self.quality_score,
            confidence: self.confidence,
            epistemic_type: self.epistemic_type,
            content_type: self.content_type,
            created_at: self.created_at,
        }
    }

    pub fn publication_metadata(&self) -> PublicationMetadata {
        PublicationMetadata {
            id: self.id,
            publisher: self.publisher.clone(),
            domain: self.domain.clone(),
            entry_type: self.entry_type,
            source_uri: self.evidence_uris.first().cloned(),
            quality_score: self.quality_score,
            corroboration: self.corroboration.clone(),
            provenance: self.provenance.clone(),
            license: self.license.clone().into(),
            epistemic_type: self.epistemic_type,
            content_type: self.content_type,
            half_life: self.half_life,
            created_at: self.created_at,
            expires_at: self.expires_at,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReputationKpi {
    pub agent: AgentId,
    pub entries_published: u32,
    pub last_active: DateTime<Utc>,
    pub adoption_rate: f32,
    pub avg_quality_score: f32,
    pub corroboration_rate: f32,
    pub contradiction_count: u32,
    pub downvote_count: u32,
    pub immune_contributions: u32,
    pub composite: f32,
}

impl ReputationKpi {
    pub fn recompute(&mut self) {
        self.composite = (self.adoption_rate * 0.25
            + self.avg_quality_score * 0.25
            + self.corroboration_rate * 0.20
            + (1.0 - (self.contradiction_count as f32 / 10.0).min(1.0)) * 0.15
            + (1.0 - (self.downvote_count as f32 / 10.0).min(1.0)) * 0.10
            + (self.immune_contributions as f32 / 10.0).min(1.0) * 0.05)
            .clamp(0.0, 1.0);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationBond {
    pub agent: AgentId,
    pub bond_level: BondLevel,
    pub accumulated_since: DateTime<Utc>,
    pub decay_factor: f32,
}

impl ReputationBond {
    pub fn new(agent: AgentId) -> Self {
        Self {
            agent,
            bond_level: BondLevel::Observer,
            accumulated_since: Utc::now(),
            decay_factor: 0.5,
        }
    }

    /// 当前声誉加成（0-0.2）。
    pub fn current_bonus(&self, now: DateTime<Utc>) -> f32 {
        let days = (now - self.accumulated_since).num_hours() as f32 / 24.0;
        let decay = (-days / 30.0 * self.decay_factor).exp();
        let base = match self.bond_level {
            BondLevel::Observer => 0.0,
            BondLevel::Contributor => 0.05,
            BondLevel::Validator => 0.12,
            BondLevel::Authority => 0.20,
        };
        base * decay
    }

    pub fn promote_if_qualified(&mut self, kpi: &ReputationKpi) {
        self.bond_level = if kpi.composite >= 0.9 && kpi.entries_published >= 50 {
            BondLevel::Authority
        } else if kpi.entries_published >= 10 && kpi.adoption_rate >= 0.8 {
            BondLevel::Validator
        } else if kpi.entries_published >= 3 && kpi.adoption_rate >= 0.6 {
            BondLevel::Contributor
        } else {
            BondLevel::Observer
        };
    }

    pub fn demote(&mut self, to: BondLevel) {
        self.bond_level = to;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KnowledgeLicense {
    /// 自由使用，无需署名。
    PublicDomain,
    /// 需署名的自由使用。
    Attribution,
    /// 仅限同一 tenant 内使用。
    TenantOnly,
    /// 需发布者显式授权。
    RequiresApproval,
}

impl From<KnowledgeLicense> for LicenseInfo {
    fn from(value: KnowledgeLicense) -> Self {
        match value {
            KnowledgeLicense::PublicDomain => Self {
                scope: LicenseScope::Open,
                attribution_required: false,
                commercial_use: true,
                derivative_allowed: true,
            },
            KnowledgeLicense::Attribution => Self {
                scope: LicenseScope::Open,
                attribution_required: true,
                commercial_use: true,
                derivative_allowed: true,
            },
            KnowledgeLicense::TenantOnly => Self {
                scope: LicenseScope::TenantOnly,
                attribution_required: true,
                commercial_use: false,
                derivative_allowed: true,
            },
            KnowledgeLicense::RequiresApproval => Self {
                scope: LicenseScope::RequiresApproval,
                attribution_required: true,
                commercial_use: false,
                derivative_allowed: false,
            },
        }
    }
}
