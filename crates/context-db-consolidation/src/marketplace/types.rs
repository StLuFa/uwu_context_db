//! 市场核心数据类型 — C2D 边界 + 声誉 KPI + 血统 DAG。

use crate::ConsolidationProduct;
use agent_context_db_core::{ContentType, ContextUri, EpistemicType};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ===========================================================================
// MarketId
// ===========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MarketId(pub Uuid);

impl MarketId {
    pub fn new() -> Self { Self(Uuid::new_v4()) }
}

/// Agent 标识（复用 core 的 agent scope URI）。
pub type AgentId = String;

// ===========================================================================
// ShareLevel — 隐私硬边界
// ===========================================================================

/// 共享层级 — 定义"什么可以离开 Agent 的本地环境"。
///
/// **硬规则**：原始 session 和原始 embedding 永不超过 Private 层。
/// 只共享 ConsolidationProduct（精炼原则）和 Antibody（攻击签名）。
#[derive(Debug, Clone)]
pub enum ShareLevel {
    /// 绝不对外共享（raw sessions, embeddings, personal data）。
    Private,
    /// 可发布到市场（ConsolidationProduct + 证据计数 + 确认证明）。
    Marketable {
        product: ConsolidationProduct,
        evidence_count: usize,
    },
    /// 免疫共享（攻击特征签名，非原始 prompt）。
    Immune {
        antibody_signature: Vec<f32>,
        severity: ThreatSeverity,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

// ===========================================================================
// MarketEntry
// ===========================================================================

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
    /// 知识使用许可
    pub license: KnowledgeLicense,
    /// 认识论类型
    pub epistemic_type: EpistemicType,
    /// 内容类型
    pub content_type: ContentType,
    /// 半衰期（天）
    pub half_life_days: Option<f64>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
    /// 过期时间（半衰期驱动）
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketEntryType {
    Fact,
    Skill,
    Procedure,
    Antibody,
    ErrorPattern,
}

// ===========================================================================
// CorroborationProof + CorroborationLadder
// ===========================================================================

/// 确认证明 — 独立来源的确认记录。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorroborationProof {
    /// 确认此知识的 Agent 列表（含独立确认的 session 数）
    pub corroborators: Vec<(AgentId, usize)>,
    /// 总确认次数
    pub total_count: usize,
    /// 独立来源数
    pub independent_sources: usize,
    /// 确认等级
    pub level: CorroborationLevel,
}

/// 确认阶梯 — 独立来源累积确认等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CorroborationLevel {
    /// 0 个独立源（不能进入市场）
    Unverified = 0,
    /// 1 个 session 产出（仅本地使用）
    SingleSession = 1,
    /// ≥2 个独立 session 确认（可发布）
    CrossSession = 2,
    /// ≥2 个 agent 独立确认（市场推荐）
    CrossAgent = 3,
    /// ≥5 个 agent + ≥10 session（权威知识）
    Established = 4,
}

impl CorroborationProof {
    pub fn new() -> Self {
        Self { corroborators: vec![], total_count: 0, independent_sources: 0, level: CorroborationLevel::Unverified }
    }

    /// 添加一个确认。
    pub fn add_corroboration(&mut self, agent: AgentId, session_count: usize) {
        self.total_count += 1;
        self.corroborators.push((agent, session_count));
        self.independent_sources = self.corroborators.len();
        self.recalc_level();
    }

    fn recalc_level(&mut self) {
        let agents = self.independent_sources;
        let total = self.total_count;
        self.level = if agents >= 5 && total >= 10 { CorroborationLevel::Established }
        else if agents >= 2 { CorroborationLevel::CrossAgent }
        else if total >= 2 { CorroborationLevel::CrossSession }
        else if total >= 1 { CorroborationLevel::SingleSession }
        else { CorroborationLevel::Unverified };
    }

    pub fn can_publish(&self) -> bool {
        self.level >= CorroborationLevel::CrossSession
    }
}

// ===========================================================================
// ReputationKpi — KPI 驱动多维 KPI
// ===========================================================================

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
    /// KPI 权重（可配置）。
    const W_ADOPTION: f32 = 0.30;
    const W_QUALITY: f32 = 0.25;
    const W_CORROBORATION: f32 = 0.20;
    const W_CONTRADICTION_PENALTY: f32 = 0.15;
    const W_DOWNVOTE_PENALTY: f32 = 0.10;

    pub fn recompute(&mut self) {
        self.composite = Self::W_ADOPTION * self.adoption_rate
            + Self::W_QUALITY * self.avg_quality_score
            + Self::W_CORROBORATION * self.corroboration_rate
            - Self::W_CONTRADICTION_PENALTY * (self.contradiction_count as f32 / self.entries_published.max(1) as f32)
            - Self::W_DOWNVOTE_PENALTY * (self.downvote_count as f32 / self.entries_published.max(1) as f32);
        self.composite = self.composite.clamp(0.0, 1.0);
    }
}

// ===========================================================================
// ReputationBond — 声誉债券
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationBond {
    pub agent: AgentId,
    pub bond_level: BondLevel,
    pub accumulated_since: DateTime<Utc>,
    pub decay_factor: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BondLevel {
    Observer = 0,
    Contributor = 1,
    Validator = 2,
    Authority = 3,
}

impl ReputationBond {
    pub fn new(agent: AgentId) -> Self {
        Self { agent, bond_level: BondLevel::Observer, accumulated_since: Utc::now(), decay_factor: 0.5 }
    }

    /// 当前声誉加成（0-0.2）。
    pub fn current_bonus(&self, now: DateTime<Utc>) -> f32 {
        let days = (now - self.accumulated_since).num_hours() as f32 / 24.0;
        let decay = (-days / 30.0 * self.decay_factor).exp();
        match self.bond_level {
            BondLevel::Authority => 0.20 * decay,
            BondLevel::Validator => 0.10 * decay,
            BondLevel::Contributor => 0.05 * decay,
            BondLevel::Observer => 0.0,
        }
    }

    pub fn promote(&mut self, kpi: &ReputationKpi) {
        self.bond_level = if kpi.entries_published >= 50 && kpi.contradiction_count == 0 {
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

// ===========================================================================
// KnowledgeLicense
// ===========================================================================

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

// ===========================================================================
// LineageNode — 血统 DAG知识血统 DAG
// ===========================================================================

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
    Adopted { by: AgentId },
    BuiltUpon { by: AgentId, new_id: MarketId },
    Contradicted { by: AgentId, reason: String },
    Resolved { by: AgentId, merged_id: MarketId },
    Superseded { by: MarketId },
}
