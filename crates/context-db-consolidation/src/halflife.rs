//! 半衰期预测 — LLM 驱动的知识有效期预测 + 到期复审。
//!
//! 理论根基：放射性衰变 + Anki SM-2 间隔重复。
//! 巩固时让 LLM 评估 domain volatility / specificity / technological context。

use agent_context_db_core::{LlmClient, LlmOpts};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::sync::Arc;

/// 半衰期预测结果。
#[derive(Debug, Clone)]
pub struct HalfLifePrediction {
    pub half_life_days: f64,
    pub confidence: f32,
    pub reasoning: String,
}

/// LLM 驱动的半衰期预测器。
pub struct HalfLifePredictor {
    llm: Arc<dyn LlmClient>,
}

#[derive(Debug, Deserialize)]
struct LlmHalfLifeResponse {
    half_life_days: f64,
    confidence: f32,
    reasoning: String,
}

impl HalfLifePredictor {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }

    /// 预测知识的半衰期。
    ///
    /// 策略：
    /// 1. 优先用 LLM 评估 domain volatility + specificity + tech context
    /// 2. LLM 不可用时回退到启发性规则
    pub async fn predict(&self, content: &str, _domain_hint: Option<&str>) -> HalfLifePrediction {
        let prompt = format!(
            r#"Predict the knowledge half-life (in days) for this insight:

"{content}"

Consider these factors:
- Domain volatility: framework-specific knowledge decays faster than mathematical truths
- Specificity: specific API calls/function names change fast, general principles slow
- Technological context: is this tied to a specific version/era?

Return a JSON object with these fields:
{{"half_life_days": <number, -1 for near-infinite timeless truths>,
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence explaining the decision>"}}"#
        );

        let opts = LlmOpts {
            max_tokens: Some(256),
            temperature: Some(0.0),
            ..Default::default()
        };

        match self.llm.complete(&prompt, &opts).await {
            Ok(response) => {
                // Try to parse LLM JSON response
                if let Ok(parsed) = serde_json::from_str::<LlmHalfLifeResponse>(&response) {
                    return HalfLifePrediction {
                        half_life_days: parsed.half_life_days.max(1.0),
                        confidence: parsed.confidence.clamp(0.0, 1.0),
                        reasoning: parsed.reasoning,
                    };
                }
                // Parse failed — fall through to heuristic
            }
            Err(_) => {
                // LLM unavailable — fall through to heuristic
            }
        }

        // Heuristic fallback
        Self::heuristic_predict(content)
    }

    /// 启发性规则回退（不依赖 LLM）。
    fn heuristic_predict(content: &str) -> HalfLifePrediction {
        let has_code = content.contains('(') && content.contains(')')
            || content.contains("::")
            || content.contains("fn ");
        let has_principle = content.contains("原则")
            || content.contains("principle")
            || content.contains("always")
            || content.contains("never")
            || content.contains("定义");

        let (days, reasoning) = if has_code && !has_principle {
            (
                60.0,
                "contains specific API/function references — likely version-dependent",
            )
        } else if has_principle {
            (365.0, "contains general principles — slow to change")
        } else {
            (180.0, "mixed content — moderate decay")
        };

        HalfLifePrediction {
            half_life_days: days,
            confidence: 0.6,
            reasoning: reasoning.to_string(),
        }
    }

    /// 查找已过半衰期的条目。
    pub fn find_expired(&self, created_at: DateTime<Utc>, half_life_days: f64) -> bool {
        let age_days = (Utc::now() - created_at).num_hours() as f64 / 24.0;
        age_days > half_life_days
    }
}

// ===========================================================================
// Anki SM-2 风格的 retrieval-induced revival（强化学习式）
// ===========================================================================

/// 检索命中且被采纳时，重置 stability 并延长有效期。
///
/// SM-2 风格: stability_new = stability * (1 + reinforcements * 0.5)
pub fn reinforce_on_adoption(current_stability: f64, reinforcements: u32) -> (f64, u32) {
    let new_stability = current_stability * (1.0 + reinforcements as f64 * 0.5);
    (new_stability, reinforcements.saturating_add(1))
}
