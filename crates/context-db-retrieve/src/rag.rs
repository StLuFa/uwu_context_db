//! Calibrated answer synthesis on top of retrieval results.
//!
//! The synthesizer is deliberately evidence-first: it builds citations from
//! retrieved memories, checks coverage/quality/conflict pressure, and abstains
//! before calling an LLM when the evidence set cannot support a trustworthy
//! answer.

use agent_context_db_core::{ContentPayload, LlmClient, LlmOpts, LlmTaskKind, PromptOptimization};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::sync::Arc;

use crate::{RetrievalHit, RetrievalResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnswerDecision {
    Answer,
    Abstain,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerCitation {
    pub index: usize,
    pub uri: agent_context_db_core::ContextUri,
    pub relevance: f32,
    pub quality: f32,
    pub excerpt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnswerConfidence {
    pub evidence_score: f32,
    pub coverage_score: f32,
    pub consistency_score: f32,
    pub final_score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibratedAnswer {
    pub decision: AnswerDecision,
    pub answer: Option<String>,
    pub citations: Vec<AnswerCitation>,
    pub confidence: AnswerConfidence,
    pub abstention_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AnswerSynthesisConfig {
    pub max_citations: usize,
    pub min_evidence_score: f32,
    pub min_coverage_score: f32,
    pub min_consistency_score: f32,
    pub min_final_confidence: f32,
    pub max_prompt_chars: usize,
}

impl Default for AnswerSynthesisConfig {
    fn default() -> Self {
        Self {
            max_citations: 8,
            min_evidence_score: 0.45,
            min_coverage_score: 0.32,
            min_consistency_score: 0.55,
            min_final_confidence: 0.48,
            max_prompt_chars: 12_000,
        }
    }
}

pub struct CalibratedAnswerSynthesizer {
    llm: Option<Arc<dyn LlmClient>>,
    config: AnswerSynthesisConfig,
}

impl Default for CalibratedAnswerSynthesizer {
    fn default() -> Self {
        Self {
            llm: None,
            config: AnswerSynthesisConfig::default(),
        }
    }
}

impl CalibratedAnswerSynthesizer {
    pub fn new(config: AnswerSynthesisConfig) -> Self {
        Self { llm: None, config }
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub async fn synthesize(
        &self,
        question: &str,
        retrieval: &RetrievalResult,
    ) -> CalibratedAnswer {
        let citations = self.select_citations(&retrieval.hits);
        let confidence = confidence_for(question, &citations);
        if let Some(reason) = self.abstention_reason(&citations, &confidence) {
            return CalibratedAnswer {
                decision: AnswerDecision::Abstain,
                answer: None,
                citations,
                confidence,
                abstention_reason: Some(reason),
            };
        }

        let answer = match &self.llm {
            Some(llm) => self.llm_answer(llm, question, &citations).await,
            None => extractive_answer(question, &citations),
        };

        CalibratedAnswer {
            decision: AnswerDecision::Answer,
            answer: Some(answer),
            citations,
            confidence,
            abstention_reason: None,
        }
    }

    fn select_citations(&self, hits: &[RetrievalHit]) -> Vec<AnswerCitation> {
        let mut scored = hits
            .iter()
            .map(|hit| {
                let quality = hit.metadata.quality_score.unwrap_or(0.5).clamp(0.0, 1.0);
                let score = hit.relevance.clamp(0.0, 1.0) * 0.65 + quality * 0.35;
                (hit, quality, score)
            })
            .collect::<Vec<_>>();
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));
        scored
            .into_iter()
            .take(self.config.max_citations)
            .enumerate()
            .map(|(index, (hit, quality, _))| AnswerCitation {
                index: index + 1,
                uri: hit.uri.clone(),
                relevance: hit.relevance.clamp(0.0, 1.0),
                quality,
                excerpt: excerpt(&hit.content, 420),
            })
            .collect()
    }

    fn abstention_reason(
        &self,
        citations: &[AnswerCitation],
        confidence: &AnswerConfidence,
    ) -> Option<String> {
        if citations.is_empty() {
            return Some("no retrieved evidence is available".into());
        }
        if confidence.evidence_score < self.config.min_evidence_score {
            return Some(format!(
                "retrieved evidence is too weak ({:.2} < {:.2})",
                confidence.evidence_score, self.config.min_evidence_score
            ));
        }
        if confidence.coverage_score < self.config.min_coverage_score {
            return Some(format!(
                "retrieved evidence does not cover enough of the question ({:.2} < {:.2})",
                confidence.coverage_score, self.config.min_coverage_score
            ));
        }
        if confidence.consistency_score < self.config.min_consistency_score {
            return Some(format!(
                "retrieved evidence contains unresolved conflicts ({:.2} < {:.2})",
                confidence.consistency_score, self.config.min_consistency_score
            ));
        }
        if confidence.final_score < self.config.min_final_confidence {
            return Some(format!(
                "calibrated confidence is below answer threshold ({:.2} < {:.2})",
                confidence.final_score, self.config.min_final_confidence
            ));
        }
        None
    }

    async fn llm_answer(
        &self,
        llm: &Arc<dyn LlmClient>,
        question: &str,
        citations: &[AnswerCitation],
    ) -> String {
        let mut evidence = String::new();
        for citation in citations {
            evidence.push_str(&format!(
                "[{}] {}\nquality={:.2} relevance={:.2}\n{}\n\n",
                citation.index,
                citation.uri,
                citation.quality,
                citation.relevance,
                citation.excerpt
            ));
        }
        if evidence.len() > self.config.max_prompt_chars {
            evidence.truncate(self.config.max_prompt_chars);
        }
        let prompt = format!(
            "Answer the question using only the cited evidence. Cite claims with [n]. If the evidence is insufficient, say that directly.\n\nQuestion:\n{question}\n\nEvidence:\n{evidence}\n\nAnswer:"
        );
        llm.complete(
            &prompt,
            &LlmOpts {
                max_tokens: Some(700),
                temperature: Some(0.0),
                task: LlmTaskKind::Synthesis,
                prompt: PromptOptimization::default()
                    .force_cache()
                    .target_tokens(3_000),
                ..Default::default()
            },
        )
        .await
        .unwrap_or_else(|_| extractive_answer(question, citations))
    }
}

fn confidence_for(question: &str, citations: &[AnswerCitation]) -> AnswerConfidence {
    let evidence_score = if citations.is_empty() {
        0.0
    } else {
        citations
            .iter()
            .map(|c| c.relevance * 0.65 + c.quality * 0.35)
            .sum::<f32>()
            / citations.len() as f32
    };
    let coverage_score = lexical_coverage(question, citations);
    let consistency_score = consistency_score(citations);
    let final_score =
        (evidence_score * 0.45 + coverage_score * 0.30 + consistency_score * 0.25).clamp(0.0, 1.0);
    AnswerConfidence {
        evidence_score,
        coverage_score,
        consistency_score,
        final_score,
    }
}

fn lexical_coverage(question: &str, citations: &[AnswerCitation]) -> f32 {
    let question_tokens = tokens(question);
    if question_tokens.is_empty() {
        return 0.5;
    }
    let evidence_text = citations
        .iter()
        .map(|c| c.excerpt.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let evidence_tokens = tokens(&evidence_text);
    let covered = question_tokens
        .iter()
        .filter(|token| evidence_tokens.contains(*token))
        .count();
    (covered as f32 / question_tokens.len() as f32).clamp(0.0, 1.0)
}

fn consistency_score(citations: &[AnswerCitation]) -> f32 {
    let mut penalty = 0.0f32;
    for i in 0..citations.len() {
        for j in (i + 1)..citations.len() {
            if has_contradiction_marker(&citations[i].excerpt, &citations[j].excerpt) {
                penalty += 0.50;
            }
        }
    }
    (1.0 - penalty.min(0.85)).clamp(0.0, 1.0)
}

fn extractive_answer(_question: &str, citations: &[AnswerCitation]) -> String {
    let mut answer = String::new();
    for citation in citations.iter().take(3) {
        if !answer.is_empty() {
            answer.push('\n');
        }
        answer.push_str(&format!("[{}] {}", citation.index, citation.excerpt));
    }
    answer
}

fn excerpt(content: &ContentPayload, max_chars: usize) -> String {
    let text = content.sparse_text().trim();
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    text.chars().take(max_chars).collect::<String>()
}

fn tokens(text: &str) -> std::collections::HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|token| token.len() >= 3)
        .map(|token| token.to_ascii_lowercase())
        .collect()
}

fn has_contradiction_marker(a: &str, b: &str) -> bool {
    let a = format!(" {} ", a.to_ascii_lowercase());
    let b = format!(" {} ", b.to_ascii_lowercase());
    let pairs = [
        (" enabled ", " disabled "),
        (" allow ", " deny "),
        (" true ", " false "),
        (" required ", " optional "),
        (" must ", " must not "),
        (" should ", " should not "),
        (" can ", " cannot "),
        ("支持", "不支持"),
        ("允许", "禁止"),
        ("必须", "不能"),
    ];
    pairs.iter().any(|(positive, negative)| {
        (a.contains(positive) && b.contains(negative))
            || (a.contains(negative) && b.contains(positive))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentLevel, ContextMeta, ContextUri};

    fn hit(uri: &str, text: &str, relevance: f32, quality: f32) -> RetrievalHit {
        RetrievalHit {
            uri: ContextUri::parse(uri).unwrap(),
            level: ContentLevel::L0,
            content: ContentPayload::Text {
                sparse: text.into(),
                dense: text.into(),
                full: text.into(),
            },
            relevance,
            parent_chain: vec![],
            content_type: None,
            metadata: ContextMeta {
                quality_score: Some(quality),
                ..Default::default()
            },
            created_at: None,
            updated_at: None,
        }
    }

    #[tokio::test]
    async fn synthesizer_answers_with_citations_when_evidence_is_strong() {
        let result = RetrievalResult {
            hits: vec![hit(
                "uwu://t/a/memory/fact/cache",
                "cache writes require durable invalidation evidence and audit logs",
                0.92,
                0.86,
            )],
            trace: Default::default(),
            tokens_used: 20,
        };
        let answer = CalibratedAnswerSynthesizer::default()
            .synthesize("cache writes require audit evidence", &result)
            .await;
        assert_eq!(answer.decision, AnswerDecision::Answer);
        assert_eq!(answer.citations.len(), 1);
        assert!(answer.answer.unwrap().contains("[1]"));
    }

    #[tokio::test]
    async fn synthesizer_abstains_on_conflicting_evidence() {
        let result = RetrievalResult {
            hits: vec![
                hit(
                    "uwu://t/a/memory/fact/a",
                    "cache must be enabled for writes",
                    0.9,
                    0.8,
                ),
                hit(
                    "uwu://t/a/memory/fact/b",
                    "cache must not be enabled for writes",
                    0.9,
                    0.8,
                ),
            ],
            trace: Default::default(),
            tokens_used: 20,
        };
        let answer = CalibratedAnswerSynthesizer::default()
            .synthesize("cache enabled writes", &result)
            .await;
        assert_eq!(answer.decision, AnswerDecision::Abstain);
        assert!(answer.abstention_reason.unwrap().contains("conflicts"));
    }
}
