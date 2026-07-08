//! 市场核心数据类型 — C2D 边界 + 声誉 KPI + 血统 DAG。

use agent_context_db_core::{ContentType, ContextUri, EpistemicType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub use agent_context_db_marketplace_types::{
    AgentId, BondLevel, CorroborationLevel, CorroborationProof, KnowledgeProvenance,
    KnowledgeProvenancePayload, LicenseInfo, LicenseScope, LineageAction, LineageNode,
    MarketEntryType, MarketId, PublicationMetadata, ThreatSeverity,
};

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
    pub half_life_days: Option<f64>,
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
            evidence_chain_hash: agent_context_db_marketplace_types::evidence_chain_hash(
                &self.evidence_uris,
            ),
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
            half_life_days: self.half_life_days,
            created_at: self.created_at,
            expires_at: None,
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
