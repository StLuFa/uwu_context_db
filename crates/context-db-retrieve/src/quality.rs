//! 检索质量闸门（F20 幻觉检测 + F17 压缩感知加载）。
//!
//! - [`HallucinationDetector`]：对检索结果做一致性/幻觉检测
//! - [`CompressionAwareLoader`]：根据 context window 压力动态调整 L0/L1/L2

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri, LlmClient, LlmOpts, Result};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// F20 幻觉检测
// ═══════════════════════════════════════════════════════════════════════════

/// 检索命中质量评估结果。
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// 原始命中数
    pub total: usize,
    /// 通过检测的命中数
    pub verified: usize,
    /// 被标记为幻觉/不一致的 URI
    pub rejected: Vec<ContextUri>,
    /// 逐条评分
    pub scores: Vec<HitQualityScore>,
    /// 整体可信度
    pub overall_confidence: f32,
}

#[derive(Debug, Clone)]
pub struct HitQualityScore {
    pub uri: ContextUri,
    /// 相关性 0-1
    pub relevance: f32,
    /// 一致性 0-1（与查询的语义一致性）
    pub consistency: f32,
    /// 是否通过
    pub passed: bool,
    /// 拒绝原因
    pub reason: Option<String>,
}

/// 幻觉检测器 —— 检索质量闸门。
///
/// 在 Rerank 之后、返回结果之前插入，过滤低质量命中。
pub struct HallucinationDetector {
    llm: Arc<dyn LlmClient>,
    /// 最低通过阈值
    threshold: f32,
}

impl HallucinationDetector {
    pub fn new(llm: Arc<dyn LlmClient>, threshold: f32) -> Self {
        Self { llm, threshold }
    }

    /// 对一批检索命中做质量评估。
    pub async fn evaluate(
        &self,
        query: &str,
        hits: &[crate::RetrievalHit],
    ) -> Result<QualityReport> {
        let mut scores = Vec::with_capacity(hits.len());
        let mut rejected = Vec::new();

        for hit in hits {
            let content_text = hit.content.sparse_text().to_string();

            let prompt = format!(
                r#"You are a retrieval quality auditor. Evaluate if this retrieved content is consistent with and relevant to the query.

Query: "{query}"
Retrieved content: "{content_text}"

Return a JSON object with:
- "relevant": true/false — is the content genuinely relevant?
- "consistent": true/false — is the content internally consistent (no fabrication)?
- "reason": brief explanation if rejected
"#
            );

            let opts = LlmOpts {
                max_tokens: Some(256),
                temperature: Some(0.0),
                ..Default::default()
            };

            let response = match self.llm.complete(&prompt, &opts).await {
                Ok(r) => r,
                Err(_) => {
                    // LLM 不可用时默认通过
                    scores.push(HitQualityScore {
                        uri: hit.uri.clone(),
                        relevance: hit.relevance,
                        consistency: 0.5,
                        passed: true,
                        reason: None,
                    });
                    continue;
                }
            };

            #[derive(serde::Deserialize)]
            struct AuditResult {
                relevant: bool,
                consistent: bool,
                reason: Option<String>,
            }

            let audit: AuditResult = serde_json::from_str(&response).unwrap_or(AuditResult {
                relevant: true,
                consistent: true,
                reason: None,
            });

            let consistency = if audit.consistent { 0.9 } else { 0.2 };
            let passed = audit.relevant && audit.consistent && hit.relevance >= self.threshold;

            if !passed {
                rejected.push(hit.uri.clone());
            }

            scores.push(HitQualityScore {
                uri: hit.uri.clone(),
                relevance: hit.relevance,
                consistency,
                passed,
                reason: if passed { None } else { audit.reason },
            });
        }

        let total = hits.len();
        let verified = total - rejected.len();
        let overall = if total > 0 {
            verified as f32 / total as f32
        } else {
            1.0
        };

        Ok(QualityReport {
            total,
            verified,
            rejected,
            scores,
            overall_confidence: overall,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// F17 压缩感知加载
// ═══════════════════════════════════════════════════════════════════════════

/// Context window 压力等级。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PressureLevel {
    /// 宽裕（< 30% 占用）→ L2
    Relaxed,
    /// 适中（30-60%）→ L1
    Moderate,
    /// 紧张（60-85%）→ L0
    Tight,
    /// 危急（> 85%）→ 只返回 URI
    Critical,
}

impl PressureLevel {
    pub fn from_ratio(used: usize, total: usize) -> Self {
        if total == 0 {
            return Self::Relaxed;
        }
        let ratio = used as f32 / total as f32;
        match ratio {
            r if r < 0.3 => Self::Relaxed,
            r if r < 0.6 => Self::Moderate,
            r if r < 0.85 => Self::Tight,
            _ => Self::Critical,
        }
    }

    /// 该压力下应加载的内容级别。
    pub fn default_level(self) -> ContentLevel {
        match self {
            Self::Relaxed => ContentLevel::L2,
            Self::Moderate => ContentLevel::L1,
            Self::Tight => ContentLevel::L0,
            Self::Critical => ContentLevel::L0,
        }
    }
}

/// 压缩感知加载器 —— 根据 context window 压力动态调整。
pub struct CompressionAwareLoader {
    /// token 窗口总容量
    window_total: usize,
    /// 当前已用
    window_used: usize,
}

impl CompressionAwareLoader {
    pub fn new(window_total: usize, window_used: usize) -> Self {
        Self { window_total, window_used }
    }

    pub fn pressure(&self) -> PressureLevel {
        PressureLevel::from_ratio(self.window_used, self.window_total)
    }

    /// 计算剩余 budget。
    pub fn remaining(&self) -> usize {
        self.window_total.saturating_sub(self.window_used)
    }

    /// 为一批 URI 分配加载级别。
    ///
    /// 高相关性命中优先获得更高级别。
    pub fn allocate_levels(
        &self,
        hits: &[crate::RetrievalHit],
    ) -> Vec<(ContextUri, ContentLevel)> {
        let pressure = self.pressure();
        let remaining = self.remaining();

        let mut plan = Vec::with_capacity(hits.len());
        let mut budget = remaining;
        let base_level = pressure.default_level();

        let mut sorted: Vec<&crate::RetrievalHit> = hits.iter().collect();
        sorted.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap_or(std::cmp::Ordering::Equal));

        for hit in sorted {
            let level = if budget > 2000 && base_level == ContentLevel::L2 {
                ContentLevel::L2
            } else if budget > 200 && base_level != ContentLevel::L0 {
                ContentLevel::L1
            } else {
                ContentLevel::L0
            };

            let cost = match level {
                ContentLevel::L2 => 8000,
                ContentLevel::L1 => 2000,
                ContentLevel::L0 => 100,
            };

            if budget >= cost {
                budget = budget.saturating_sub(cost);
                plan.push((hit.uri.clone(), level));
            } else if budget >= 100 {
                plan.push((hit.uri.clone(), ContentLevel::L0));
                budget = budget.saturating_sub(100);
            } else {
                break;
            }
        }

        plan
    }

    /// 更新已用 token 数。
    pub fn consume(&mut self, tokens: usize) {
        self.window_used = self.window_used.saturating_add(tokens);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_from_ratio() {
        assert_eq!(PressureLevel::from_ratio(10, 100), PressureLevel::Relaxed);
        assert_eq!(PressureLevel::from_ratio(40, 100), PressureLevel::Moderate);
        assert_eq!(PressureLevel::from_ratio(70, 100), PressureLevel::Tight);
        assert_eq!(PressureLevel::from_ratio(90, 100), PressureLevel::Critical);
    }

    #[test]
    fn allocate_respects_budget() {
        let loader = CompressionAwareLoader::new(10000, 9000); // 只剩 1000 tokens

        let hits = vec![
            crate::RetrievalHit {
                uri: ContextUri::parse("uwu://t/x").unwrap(),
                level: ContentLevel::L0,
                content: ContentPayload::Abstract("a".into()),
                relevance: 0.9,
                parent_chain: vec![],
                memory_class: None,
            },
        ];

        let plan = loader.allocate_levels(&hits);
        // Tight pressure + small budget → L0 only
        assert_eq!(plan[0].1, ContentLevel::L0);
    }
}
