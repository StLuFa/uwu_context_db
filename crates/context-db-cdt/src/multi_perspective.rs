//! 多视角巩固 — 多视角分析 + 知识缺口发现 + 合成。
//!
//! 对同一主题从 4 个认知视角分别收集证据，再合成为高质量巩固产物。

use agent_context_db_core::{
    ContextEntry, ContextUri, EpistemicType, LlmClient, LlmOpts, LlmTaskKind, PromptOptimization,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    /// Empty for schema-valid/local views; populated when model output fails validation.
    pub validation_errors: Vec<String>,
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
    llm: Option<Arc<dyn LlmClient>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPerspectiveView {
    summary: String,
    key_insights: Vec<String>,
    gaps: Vec<String>,
    confidence: f32,
}

impl Default for MultiPerspectiveConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiPerspectiveConsolidator {
    pub fn new() -> Self {
        Self {
            perspectives: Perspective::all(),
            llm: None,
        }
    }

    pub fn with_perspectives(mut self, perspectives: Vec<Perspective>) -> Self {
        self.perspectives = perspectives;
        self
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    /// 多视角巩固主流程。
    pub async fn consolidate(
        &self,
        topic_uri: &ContextUri,
        topic: &str,
        evidence: &[ContextEntry],
    ) -> MultiPerspectiveProduct {
        // 1. 多视角分析：有 LLM 时使用 batch_complete 一次提交全部视角 prompt。
        let views = match &self.llm {
            Some(llm) => self
                .analyze_batch_with_llm(llm.as_ref(), topic, evidence)
                .await
                .unwrap_or_else(|| self.analyze_batch_locally(topic, evidence)),
            None => self.analyze_batch_locally(topic, evidence),
        };

        // 2. 发现知识缺口
        let gaps = self.identify_gaps(&views);

        // 3. 多视角合成
        let synthesized = match &self.llm {
            Some(llm) => self
                .synthesize_with_llm(llm.as_ref(), topic, &views, &gaps)
                .await
                .unwrap_or_else(|| self.synthesize(topic, &views, &gaps)),
            None => self.synthesize(topic, &views, &gaps),
        };

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

    fn analyze_batch_locally(
        &self,
        topic: &str,
        evidence: &[ContextEntry],
    ) -> Vec<PerspectiveView> {
        self.perspectives
            .iter()
            .map(|perspective| self.analyze_from(*perspective, topic, evidence))
            .collect()
    }

    async fn analyze_batch_with_llm(
        &self,
        llm: &dyn LlmClient,
        topic: &str,
        evidence: &[ContextEntry],
    ) -> Option<Vec<PerspectiveView>> {
        let evidence_text = evidence_prompt(evidence);
        let prompts = self
            .perspectives
            .iter()
            .map(|perspective| perspective_prompt(*perspective, topic, &evidence_text))
            .collect::<Vec<_>>();
        let opts = LlmOpts {
            max_tokens: Some(700),
            temperature: Some(0.2),
            task: LlmTaskKind::Synthesis,
            prompt: PromptOptimization::default()
                .force_cache()
                .target_tokens(1_500),
            ..Default::default()
        };
        let responses = llm.batch_complete(&prompts, &opts).await.ok()?;
        if responses.len() != self.perspectives.len() {
            return None;
        }

        Some(
            self.perspectives
                .iter()
                .zip(responses)
                .map(|(perspective, response)| {
                    self.view_from_llm_response(*perspective, topic, evidence, &response)
                })
                .collect(),
        )
    }

    fn view_from_llm_response(
        &self,
        perspective: Perspective,
        topic: &str,
        evidence: &[ContextEntry],
        response: &str,
    ) -> PerspectiveView {
        let fallback = self.analyze_from(perspective, topic, evidence);
        let trimmed = response.trim();
        if trimmed.is_empty() {
            return fallback;
        }

        if let Ok(raw) = serde_json::from_str::<RawPerspectiveView>(trimmed) {
            return PerspectiveView {
                perspective,
                summary: raw.summary,
                key_insights: raw.key_insights.into_iter().take(8).collect(),
                // Model confidence is optional and never allowed to exceed the evidence-derived
                // ceiling. Missing confidence means no additional epistemic signal.
                confidence: if raw.confidence.is_finite() {
                    raw.confidence.clamp(0.0, fallback.confidence)
                } else {
                    0.0
                },
                evidence_uris: evidence.iter().map(|entry| entry.uri.clone()).collect(),
                gaps: raw.gaps.into_iter().take(8).collect(),
                validation_errors: Vec::new(),
            };
        }

        PerspectiveView {
            // Invalid model text is never promoted into summary or insights.
            confidence: (fallback.confidence * 0.5).clamp(0.0, 1.0),
            validation_errors: vec![
                "LLM perspective response failed strict JSON schema validation".into(),
            ],
            ..fallback
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
            validation_errors: Vec::new(),
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

    async fn synthesize_with_llm(
        &self,
        llm: &dyn LlmClient,
        topic: &str,
        views: &[PerspectiveView],
        gaps: &[KnowledgeGap],
    ) -> Option<String> {
        let prompt = synthesis_prompt(topic, views, gaps);
        let opts = LlmOpts {
            max_tokens: Some(1200),
            temperature: Some(0.2),
            task: LlmTaskKind::Synthesis,
            prompt: PromptOptimization::default()
                .force_cache()
                .target_tokens(2_500),
            ..Default::default()
        };
        llm.complete(&prompt, &opts)
            .await
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
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

fn evidence_prompt(evidence: &[ContextEntry]) -> String {
    if evidence.is_empty() {
        return "(no evidence)".into();
    }
    evidence
        .iter()
        .take(12)
        .map(|entry| format!("- {}: {}", entry.uri, entry.l0_text()))
        .collect::<Vec<_>>()
        .join("\n")
}

fn perspective_prompt(perspective: Perspective, topic: &str, evidence_text: &str) -> String {
    format!(
        "{}\nTopic: {topic}\nEvidence:\n{evidence_text}\n\n\
         Return exactly one JSON object matching this schema, with no markdown or extra fields: \
         {{\"summary\":string,\"key_insights\":string[],\"gaps\":string[],\"confidence\":number}}. \
         All fields are required. Use only the evidence above; gaps must be concrete exploration questions for missing evidence.",
        perspective.prompt_prefix()
    )
}

fn synthesis_prompt(topic: &str, views: &[PerspectiveView], gaps: &[KnowledgeGap]) -> String {
    let views_text = views
        .iter()
        .map(|view| {
            format!(
                "## {} confidence {:.2}\n{}\n{}",
                view.perspective.name(),
                view.confidence,
                view.summary,
                view.key_insights
                    .iter()
                    .map(|item| format!("- {item}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        })
        .collect::<Vec<_>>()
        .join("\n\n");
    let gaps_text = gaps
        .iter()
        .take(8)
        .map(|gap| format!("- {:.2}: {}", gap.severity, gap.description))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "Synthesize these STORM-style perspectives for topic `{topic}` into a coherent report.\n\n\
         Perspectives:\n{views_text}\n\nKnowledge gaps:\n{gaps_text}\n\n\
         Return markdown with these sections: Evidence-grounded synthesis, Cross-perspective tensions, Confidence calibration, Next exploration steps. \
         Every strong claim must name the supporting perspective; unresolved gaps must remain explicit."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_perspectives_present() {
        let all = Perspective::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn llm_json_response_becomes_structured_perspective_view() {
        let consolidator = MultiPerspectiveConsolidator::new();
        let response = r#"{"summary":"causal summary","key_insights":["a causes b"],"gaps":["need counterexample"],"confidence":0.81}"#;
        let view = consolidator.view_from_llm_response(Perspective::Causal, "topic", &[], response);
        assert_eq!(view.summary, "causal summary");
        assert_eq!(view.key_insights, vec!["a causes b".to_string()]);
        assert_eq!(view.gaps, vec!["need counterexample".to_string()]);
        assert_eq!(view.confidence, 0.1);
    }

    #[test]
    fn malformed_response_reduces_fallback_confidence() {
        let consolidator = MultiPerspectiveConsolidator::new();
        let fallback = consolidator.analyze_from(Perspective::Causal, "topic", &[]);
        let view = consolidator.view_from_llm_response(
            Perspective::Causal,
            "topic",
            &[],
            "provider error: upstream unavailable",
        );
        assert!(view.confidence < fallback.confidence);
        assert_eq!(view.summary, fallback.summary);
        assert_eq!(view.key_insights, fallback.key_insights);
        assert_eq!(view.validation_errors.len(), 1);
    }

    #[test]
    fn unknown_or_missing_json_fields_fail_strict_schema_without_reward() {
        let consolidator = MultiPerspectiveConsolidator::new();
        for response in [
            r#"{\"summary\":\"x\",\"key_insights\":[],\"gaps\":[],\"confidence\":0.9,\"extra\":true}"#,
            r#"{\"summary\":\"x\",\"confidence\":0.9}"#,
        ] {
            let fallback = consolidator.analyze_from(Perspective::Causal, "topic", &[]);
            let view =
                consolidator.view_from_llm_response(Perspective::Causal, "topic", &[], response);
            assert!(view.confidence < fallback.confidence);
            assert!(!view.validation_errors.is_empty());
        }
    }

    #[tokio::test]
    async fn consolidator_basic() {
        let consolidator = MultiPerspectiveConsolidator::new();
        let uri = ContextUri::parse("uwu://t/a/memory/fact/test").unwrap();
        let product = consolidator.consolidate(&uri, "test topic", &[]).await;
        assert_eq!(product.views.len(), 4);
        assert!(product.overall_confidence < 0.5); // 无证据 → 低置信度
    }
}
