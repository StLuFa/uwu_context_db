//! 三维混合检索 — Relevance + Recency + EpistemicConfidence。

use crate::config::HybridRetrievalConfig;
use agent_context_db_core::{ContentType, Result};
use chrono::{DateTime, Utc};

/// 检索权重配置。
#[derive(Debug, Clone)]
pub struct RetrievalWeights {
    pub relevance: f32,
    pub recency: f32,
    pub confidence: f32,
}

impl Default for RetrievalWeights {
    fn default() -> Self {
        Self {
            relevance: 0.4,
            recency: 0.3,
            confidence: 0.3,
        }
    }
}

/// 三维混合评分器。
pub struct HybridRetriever {
    weights: RetrievalWeights,
}

impl HybridRetriever {
    pub fn new(config: HybridRetrievalConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            weights: RetrievalWeights {
                relevance: config.relevance_weight,
                recency: config.recency_weight,
                confidence: config.confidence_weight,
            },
        })
    }
    pub fn with_default() -> Result<Self> {
        Self::new(HybridRetrievalConfig::default())
    }

    /// 三维加权评分 — Relevance + Recency + EpistemicConfidence。
    pub fn score(
        &self,
        relevance: f32,
        last_access: DateTime<Utc>,
        epistemic_type: ContentType,
        quality_score: f32,
    ) -> f32 {
        let recency = Self::recency_decay(last_access);
        let confidence = Self::epistemic_weight(epistemic_type) * quality_score;

        self.weights.relevance * relevance
            + self.weights.recency * recency
            + self.weights.confidence * confidence
    }

    /// 时间衰减 — 指数衰减，30 天半衰。
    fn recency_decay(last_access: DateTime<Utc>) -> f32 {
        let days = (Utc::now() - last_access).num_hours() as f32 / 24.0;
        (-days / 30.0).exp().clamp(0.05, 1.0)
    }

    /// 认识论类型 → 基础置信度权重。
    fn epistemic_weight(ct: ContentType) -> f32 {
        match ct {
            ContentType::Fact => 1.0,
            ContentType::Error => 0.9,
            ContentType::Skill => 0.85,
            ContentType::Procedure => 0.7,
            ContentType::Preference => 0.6,
            ContentType::Heuristic => 0.5,
            ContentType::Reflection => 0.4,
            ContentType::Belief => 0.3,
            ContentType::Hypothesis => 0.15,
            _ => 0.1,
        }
    }
}
