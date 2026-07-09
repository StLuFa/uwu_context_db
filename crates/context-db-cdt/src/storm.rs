//! STORM — 多视角提问、证据组织和知识综合。
//!
//! STORM 在 CDT 中承担“把分散记忆组织成结构化知识”的角色：
//! 为主题生成多角色问题，按问题收集证据，再合成 outline/report，并输出 consolidation signal。

use crate::consolidation::{CdtConsolidationSignal, CdtSignalSource};
use crate::multi_perspective::{MultiPerspectiveConsolidator, Perspective};
use agent_context_db_core::{ContentType, ContextEntry, ContextUri, EpistemicType, LlmClient};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StormQuestion {
    pub perspective: Perspective,
    pub question: String,
    pub priority: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StormSection {
    pub title: String,
    pub perspective: Perspective,
    pub evidence_uris: Vec<ContextUri>,
    pub claims: Vec<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StormReport {
    pub topic_uri: ContextUri,
    pub topic: String,
    pub questions: Vec<StormQuestion>,
    pub sections: Vec<StormSection>,
    pub synthesis: String,
    pub unresolved_questions: Vec<String>,
    pub confidence: f32,
}

#[derive(Debug, Clone)]
pub struct StormSynthesizer {
    perspectives: Vec<Perspective>,
    max_questions_per_perspective: usize,
    llm: Option<Arc<dyn LlmClient>>,
}

impl StormSynthesizer {
    pub fn new() -> Self {
        Self {
            perspectives: Perspective::all(),
            max_questions_per_perspective: 2,
            llm: None,
        }
    }

    pub fn with_perspectives(mut self, perspectives: Vec<Perspective>) -> Self {
        self.perspectives = perspectives;
        self
    }

    pub fn with_max_questions(mut self, max_questions_per_perspective: usize) -> Self {
        self.max_questions_per_perspective = max_questions_per_perspective.max(1);
        self
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub async fn synthesize(
        &self,
        topic_uri: &ContextUri,
        topic: &str,
        evidence: &[ContextEntry],
    ) -> StormReport {
        let questions = self.generate_questions(topic, evidence);
        let evidence_by_perspective = self.organize_evidence(evidence, &questions);
        let mut sections = Vec::new();
        for perspective in &self.perspectives {
            let selected = evidence_by_perspective
                .get(perspective)
                .cloned()
                .unwrap_or_default();
            sections.push(self.build_section(*perspective, &selected));
        }

        let mut multi = MultiPerspectiveConsolidator::new().with_perspectives(self.perspectives.clone());
        if let Some(llm) = &self.llm {
            multi = multi.with_llm(llm.clone());
        }
        let multi = multi.consolidate(topic_uri, topic, evidence).await;
        let unresolved_questions = multi
            .discovered_gaps
            .iter()
            .map(|gap| gap.suggested_exploration.clone())
            .collect::<Vec<_>>();
        let confidence = if sections.is_empty() {
            0.0
        } else {
            sections.iter().map(|s| s.confidence).sum::<f32>() / sections.len() as f32
        };
        let synthesis = self.compose_report(topic, &sections, &multi.synthesized, &unresolved_questions);

        StormReport {
            topic_uri: topic_uri.clone(),
            topic: topic.into(),
            questions,
            sections,
            synthesis,
            unresolved_questions,
            confidence: confidence.max(multi.overall_confidence * 0.8).clamp(0.0, 1.0),
        }
    }

    pub fn consolidation_signal(&self, report: &StormReport) -> CdtConsolidationSignal {
        CdtConsolidationSignal {
            uri: report.topic_uri.clone(),
            content_type: ContentType::Fact,
            epistemic_type: EpistemicType::Fact,
            content: report.synthesis.clone(),
            quality_score: report.confidence,
            confidence: report.confidence,
            evidence_uris: report
                .sections
                .iter()
                .flat_map(|section| section.evidence_uris.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect(),
            contradiction_uris: vec![],
            source: CdtSignalSource::StormSynthesis,
            tags: vec!["storm".into(), "synthesis".into()],
        }
    }

    fn generate_questions(&self, topic: &str, evidence: &[ContextEntry]) -> Vec<StormQuestion> {
        let evidence_count = evidence.len().max(1) as f32;
        let mut questions = Vec::new();
        for perspective in &self.perspectives {
            let templates = question_templates(*perspective, topic);
            for (idx, question) in templates
                .into_iter()
                .take(self.max_questions_per_perspective)
                .enumerate()
            {
                questions.push(StormQuestion {
                    perspective: *perspective,
                    question,
                    priority: (0.7 + idx as f32 * 0.05 + (1.0 / evidence_count) * 0.1).clamp(0.0, 1.0),
                });
            }
        }
        questions.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
        questions
    }

    fn organize_evidence(
        &self,
        evidence: &[ContextEntry],
        questions: &[StormQuestion],
    ) -> HashMap<Perspective, Vec<ContextEntry>> {
        let mut map: HashMap<Perspective, Vec<ContextEntry>> = HashMap::new();
        for question in questions {
            for entry in evidence {
                if evidence_matches(entry, question) {
                    map.entry(question.perspective).or_default().push(entry.clone());
                }
            }
        }
        for perspective in &self.perspectives {
            map.entry(*perspective).or_insert_with(|| evidence.iter().take(3).cloned().collect());
        }
        map
    }

    fn build_section(&self, perspective: Perspective, evidence: &[ContextEntry]) -> StormSection {
        let claims = evidence
            .iter()
            .take(5)
            .map(|entry| entry.l0_text().to_string())
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>();
        let confidence = if claims.is_empty() {
            0.1
        } else {
            (0.35 + claims.len() as f32 * 0.12).clamp(0.0, 1.0)
        };
        StormSection {
            title: format!("{} view", perspective.name()),
            perspective,
            evidence_uris: evidence.iter().map(|entry| entry.uri.clone()).collect(),
            claims,
            confidence,
        }
    }

    fn compose_report(
        &self,
        topic: &str,
        sections: &[StormSection],
        multi_synthesis: &str,
        unresolved: &[String],
    ) -> String {
        let mut out = format!("# STORM Synthesis: {topic}\n\n");
        for section in sections {
            out.push_str(&format!(
                "## {} (confidence {:.2})\n",
                section.title, section.confidence
            ));
            for claim in &section.claims {
                out.push_str(&format!("- {claim}\n"));
            }
            out.push('\n');
        }
        out.push_str("## Integrated View\n");
        out.push_str(multi_synthesis);
        out.push('\n');
        if !unresolved.is_empty() {
            out.push_str("\n## Open Questions\n");
            for question in unresolved.iter().take(5) {
                out.push_str(&format!("- {question}\n"));
            }
        }
        out
    }
}

impl Default for StormSynthesizer {
    fn default() -> Self {
        Self::new()
    }
}

fn question_templates(perspective: Perspective, topic: &str) -> Vec<String> {
    match perspective {
        Perspective::Causal => vec![
            format!("What causes `{topic}` to succeed or fail?"),
            format!("Which prerequisites change the outcome of `{topic}`?"),
        ],
        Perspective::Temporal => vec![
            format!("How does `{topic}` evolve across attempts?"),
            format!("Which steps must happen before `{topic}` is reliable?"),
        ],
        Perspective::Comparative => vec![
            format!("How does `{topic}` differ between successful and failed trajectories?"),
            format!("Which alternative strategies compete with `{topic}`?"),
        ],
        Perspective::Counterexample => vec![
            format!("When does `{topic}` not hold?"),
            format!("What evidence would falsify the current understanding of `{topic}`?"),
        ],
    }
}

fn evidence_matches(entry: &ContextEntry, question: &StormQuestion) -> bool {
    let text = entry.l0_text().to_ascii_lowercase();
    match question.perspective {
        Perspective::Causal => text.contains("because") || text.contains("failed") || text.contains("caused"),
        Perspective::Temporal => text.contains("step") || text.contains("after") || text.contains("before"),
        Perspective::Comparative => text.contains("success") || text.contains("failed") || text.contains("alternative"),
        Perspective::Counterexample => text.contains("error") || text.contains("not") || text.contains("avoid"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::TenantId;
    use uuid::Uuid;

    #[tokio::test]
    async fn storm_builds_report_and_signal() {
        let topic_uri = ContextUri::parse("uwu://t/agent/storm/fact/deploy").unwrap();
        let evidence = vec![
            ContextEntry::new_text(
                ContextUri::parse("uwu://t/agent/storm/error/e1").unwrap(),
                TenantId(Uuid::new_v4()),
                "deploy failed because registry timed out",
            ),
            ContextEntry::new_text(
                ContextUri::parse("uwu://t/agent/storm/procedure/p1").unwrap(),
                TenantId(Uuid::new_v4()),
                "step one build before pushing image",
            ),
        ];
        let storm = StormSynthesizer::new();
        let report = storm.synthesize(&topic_uri, "deploy reliability", &evidence).await;
        let signal = storm.consolidation_signal(&report);
        assert!(!report.questions.is_empty());
        assert_eq!(report.sections.len(), 4);
        assert!(report.synthesis.contains("STORM Synthesis"));
        assert_eq!(signal.source, CdtSignalSource::StormSynthesis);
        assert!(!signal.evidence_uris.is_empty());
    }
}
