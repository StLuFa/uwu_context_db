//! 多视角巩固 — 多视角分析 + 知识缺口发现 + 合成。
//!
//! 对同一主题从 4 个认知视角分别收集证据，再合成为高质量巩固产物。

use agent_context_db_core::{ContentType, ContextEntry, ContextUri, EpistemicType, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ===========================================================================
// 视角定义
// ===========================================================================

/// 认知分析视角（STORM 四视角）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Perspective {
    /// 因果关系视角 — "为什么 A 导致 B"
    Causal,
    /// 时序演变视角 — "A 如何随时间变化"
    Temporal,
    /// 对比分析视角 — "A 与 B 的异同"
    Comparative,
    /// 反例证伪视角 — "什么情况下 A 不成立"
    Counterexample,
}

impl Perspective {
    pub fn all() -> Vec<Perspective> {
        vec![
            Perspective::Causal,
            Perspective::Temporal,
            Perspective::Comparative,
            Perspective::Counterexample,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Perspective::Causal => "causal",
            Perspective::Temporal => "temporal",
            Perspective::Comparative => "comparative",
            Perspective::Counterexample => "counterexample",
        }
    }

    /// 每个视角的分析提示词前缀。
    pub fn prompt_prefix(&self) -> &'static str {
        match self {
            Perspective::Causal => "Analyze the causal relationships: what causes what, and why?",
            Perspective::Temporal => "Trace the temporal evolution: how did this change over time?",
            Perspective::Comparative => {
                "Compare and contrast: what are the similarities and differences?"
            }
            Perspective::Counterexample => {
                "Find counterexamples: under what conditions does this NOT hold?"
            }
        }
    }
}

// ===========================================================================
// 视角分析结果
// ===========================================================================

/// 单个视角的分析结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerspectiveView {
    pub perspective: Perspective,
    pub summary: String,
    pub key_insights: Vec<String>,
    pub confidence: f32,
    pub evidence_uris: Vec<ContextUri>,
    pub gaps: Vec<String>, // 发现的待探索问题
}

/// 多视角合成产物。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiPerspectiveProduct {
    pub topic_uri: ContextUri,
    pub topic_summary: String,
    pub views: Vec<PerspectiveView>,
    pub synthesized: String,
    pub discovered_gaps: Vec<KnowledgeGap>,
    pub overall_confidence: f32,
    pub epistemic_type: EpistemicType,
}

/// 知识缺口。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGap {
    pub description: String,
    pub severity: f32,
    pub source_perspective: Perspective,
    pub suggested_exploration: String,
}

// ===========================================================================
// 多视角巩固器
// ===========================================================================

/// 多视角巩固器 — 对同一主题从多个认知视角分别分析，再合成。
pub struct MultiPerspectiveConsolidator {
    perspectives: Vec<Perspective>,
    min_confidence: f32,
}

impl MultiPerspectiveConsolidator {
    pub fn new() -> Self {
        Self {
            perspectives: Perspective::all(),
            min_confidence: 0.3,
        }
    }

    pub fn with_perspectives(mut self, perspectives: Vec<Perspective>) -> Self {
        self.perspectives = perspectives;
        self
    }

    /// 多视角巩固主流程。
    pub async fn consolidate(
        &self,
        topic_uri: &ContextUri,
        topic: &str,
        evidence: &[ContextEntry],
    ) -> MultiPerspectiveProduct {
        // 1. 每个视角分别分析
        let mut views = Vec::new();
        for perspective in &self.perspectives {
            let view = self.analyze_from(*perspective, topic, evidence);
            views.push(view);
        }

        // 2. 发现知识缺口
        let gaps = self.identify_gaps(&views);

        // 3. 多视角合成
        let synthesized = self.synthesize(topic, &views, &gaps);

        // 4. 计算综合置信度
        let overall_confidence =
            views.iter().map(|v| v.confidence).sum::<f32>() / views.len().max(1) as f32;

        MultiPerspectiveProduct {
            topic_uri: topic_uri.clone(),
            topic_summary: topic.to_string(),
            views,
            synthesized,
            discovered_gaps: gaps,
            overall_confidence,
            epistemic_type: EpistemicType::Fact,
        }
    }

    /// 从单个视角分析证据。
    fn analyze_from(
        &self,
        perspective: Perspective,
        topic: &str,
        evidence: &[ContextEntry],
    ) -> PerspectiveView {
        let content_summaries: Vec<String> =
            evidence.iter().map(|e| e.l0_text().to_string()).collect();

        let summary = format!(
            "[{} perspective on '{}']: analyzed {} evidence items",
            perspective.name(),
            topic,
            evidence.len()
        );

        // 基于证据数量和内容长度估算置信度
        let confidence = if evidence.is_empty() {
            0.1
        } else {
            (evidence.len() as f32 / 5.0).min(1.0) * 0.7
                + content_summaries
                    .iter()
                    .map(|s| (s.len() as f32 / 500.0).min(1.0))
                    .sum::<f32>()
                    / evidence.len().max(1) as f32
                    * 0.3
        };

        // 发现知识缺口
        let gaps = if evidence.len() < 3 {
            vec![format!(
                "[{}] insufficient evidence for '{}': need more data points",
                perspective.name(),
                topic
            )]
        } else {
            vec![]
        };

        PerspectiveView {
            perspective,
            summary,
            key_insights: content_summaries.into_iter().take(3).collect(),
            confidence,
            evidence_uris: evidence.iter().map(|e| e.uri.clone()).collect(),
            gaps,
        }
    }

    /// 识别跨视角的知识缺口。
    fn identify_gaps(&self, views: &[PerspectiveView]) -> Vec<KnowledgeGap> {
        let mut gaps = Vec::new();

        for view in views {
            for gap_text in &view.gaps {
                gaps.push(KnowledgeGap {
                    description: gap_text.clone(),
                    severity: 1.0 - view.confidence,
                    source_perspective: view.perspective,
                    suggested_exploration: format!("explore '{}' from additional angles", gap_text),
                });
            }
        }

        // 多视角交叉缺口：某视角有高置信度但其他视角缺失
        let high_conf: Vec<_> = views.iter().filter(|v| v.confidence > 0.7).collect();
        let low_conf: Vec<_> = views.iter().filter(|v| v.confidence < 0.3).collect();

        for hc in &high_conf {
            for lc in &low_conf {
                gaps.push(KnowledgeGap {
                    description: format!(
                        "high confidence in {} ({:.1}) but low in {} ({:.1})",
                        hc.perspective.name(),
                        hc.confidence,
                        lc.perspective.name(),
                        lc.confidence
                    ),
                    severity: hc.confidence - lc.confidence,
                    source_perspective: lc.perspective,
                    suggested_exploration: format!(
                        "apply {} perspective analysis to strengthen {} view",
                        hc.perspective.name(),
                        lc.perspective.name()
                    ),
                });
            }
        }

        gaps.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap());
        gaps
    }

    /// 多视角合成 — 从多个视角结果生成统一摘要。
    fn synthesize(&self, topic: &str, views: &[PerspectiveView], gaps: &[KnowledgeGap]) -> String {
        let mut synthesis = format!("# Multi-Perspective Analysis: {topic}\n\n");

        // 各视角摘要
        synthesis.push_str("## Perspective Views\n\n");
        for view in views {
            synthesis.push_str(&format!(
                "### {name} (confidence: {conf:.2})\n{summary}\n",
                name = view.perspective.name(),
                conf = view.confidence,
                summary = view.summary,
            ));
            if !view.key_insights.is_empty() {
                synthesis.push_str("Key insights:\n");
                for insight in &view.key_insights {
                    synthesis.push_str(&format!("- {insight}\n"));
                }
            }
            synthesis.push('\n');
        }

        // 知识缺口
        if !gaps.is_empty() {
            synthesis.push_str("## Discovered Knowledge Gaps\n\n");
            for gap in gaps.iter().take(5) {
                synthesis.push_str(&format!(
                    "- [{sev:.2}] {desc} (explore: {exp})\n",
                    sev = gap.severity,
                    desc = gap.description,
                    exp = gap.suggested_exploration,
                ));
            }
            synthesis.push('\n');
        }

        // 综合判断
        let avg_conf = views.iter().map(|v| v.confidence).sum::<f32>() / views.len().max(1) as f32;
        synthesis.push_str(&format!(
            "## Synthesis\nOverall confidence: {avg_conf:.2} across {} perspectives\n",
            views.len()
        ));

        synthesis
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_perspectives_present() {
        let all = Perspective::all();
        assert_eq!(all.len(), 4);
    }

    #[tokio::test]
    async fn consolidator_basic() {
        let consolidator = MultiPerspectiveConsolidator::new();
        let uri = ContextUri::parse("uwu://t/a/x/fact/test").unwrap();
        let product = consolidator.consolidate(&uri, "test topic", &[]).await;
        assert_eq!(product.views.len(), 4);
        assert!(product.overall_confidence < 0.5); // 无证据 → 低置信度
    }
}
