//! PublishGate — 质量门控 + 确认阶梯 + 声誉债券 + C2D 边界。

use crate::marketplace::types::*;
use crate::marketplace::registry::FederatedRegistry;
use crate::ConsolidationProduct;

/// 发布门控 — 检查 knowledge 是否可以进入市场。
pub struct PublishGate {
    /// 最低质量分。
    min_quality: f32,
    /// 最低确认等级。
    min_corroboration: CorroborationLevel,
}

impl PublishGate {
    pub fn new() -> Self {
        Self { min_quality: 0.80, min_corroboration: CorroborationLevel::CrossSession }
    }

    /// 尝试发布一个 ConsolidationProduct。
    ///
    /// 检查：
    /// 1. quality_score ≥ 0.80
    /// 2. corroboration ≥ CrossSession (≥2 independent sessions)
    /// 3. ShareLevel 不是 Private
    /// 4. 内容非空
    pub fn try_publish(
        &self,
        product: &ConsolidationProduct,
        corroboration: &CorroborationProof,
        publisher: &str,
        domain: &str,
    ) -> Result<MarketEntry, PublishError> {
        // 1. 质量
        if product.quality_score < self.min_quality {
            return Err(PublishError::QualityTooLow(product.quality_score));
        }

        // 2. 确认
        if !corroboration.can_publish() {
            return Err(PublishError::InsufficientCorroboration(corroboration.level));
        }

        // 3. 内容
        if product.content.is_empty() {
            return Err(PublishError::EmptyContent);
        }

        let entry_type = match product.content_type {
            agent_context_db_core::ContentType::Skill => MarketEntryType::Skill,
            agent_context_db_core::ContentType::Procedure => MarketEntryType::Procedure,
            agent_context_db_core::ContentType::Error => MarketEntryType::ErrorPattern,
            _ => MarketEntryType::Fact,
        };

        let entry = MarketEntry {
            id: MarketId::new(),
            publisher: publisher.to_string(),
            domain: domain.to_string(),
            entry_type,
            principle: product.content.clone(),
            evidence_uris: product.evidence_uris.clone(),
            quality_score: product.quality_score,
            confidence: product.confidence,
            corroboration: corroboration.clone(),
            license: KnowledgeLicense::Attribution,
            epistemic_type: product.epistemic_type,
            content_type: product.content_type,
            half_life_days: product.metadata.half_life_days,
            created_at: chrono::Utc::now(),
            expires_at: product.metadata.half_life_days.map(|d| {
                chrono::Utc::now() + chrono::Duration::days(d as i64)
            }),
        };

        Ok(entry)
    }

    /// 带声誉加成的评分（声誉债券越高，初始可信度加成越大）。
    pub fn boosted_quality(&self, base_quality: f32, bond: &ReputationBond) -> f32 {
        let bonus = bond.current_bonus(chrono::Utc::now());
        (base_quality + bonus).min(1.0)
    }
}

#[derive(Debug, Clone)]
pub enum PublishError {
    QualityTooLow(f32),
    InsufficientCorroboration(CorroborationLevel),
    EmptyContent,
    NotOwner,
}

impl std::fmt::Display for PublishError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PublishError::QualityTooLow(q) => write!(f, "quality {:.2} below threshold", q),
            PublishError::InsufficientCorroboration(l) => write!(f, "corroboration level {:?} insufficient", l),
            PublishError::EmptyContent => write!(f, "empty content"),
            PublishError::NotOwner => write!(f, "not the owner"),
        }
    }
}
