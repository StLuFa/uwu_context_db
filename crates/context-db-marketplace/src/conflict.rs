//! ConflictResolver — 多维证据对比 + LLM 多智能体辩论仲裁。

use crate::types::*;
use agent_context_db_core::{JsonSchema, LlmClient, LlmOpts, LlmTaskKind, PromptOptimization};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;

/// 冲突 — 两个 MarketEntry 对同一事实给出矛盾结论。
#[derive(Debug, Clone)]
pub struct MarketConflict {
    pub entry_a: MarketEntry,
    pub entry_b: MarketEntry,
    pub conflict_type: ConflictType,
    pub similarity: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictType {
    /// 直接语义矛盾（A说X对，B说X错）。
    DirectContradiction,
    /// 部分重叠但结论不同。
    Overlapping,
    /// 同一证据得出不同结论。
    SameEvidenceDifferentConclusion,
}

/// 仲裁结果。
#[derive(Debug, Clone, PartialEq)]
pub enum MarketConflictResolution {
    /// 保留 A，让 B 过期。
    KeepA { reason: String },
    /// 保留 B，让 A 过期。
    KeepB { reason: String },
    /// 两者合并。
    Fuse {
        merged_content: String,
        reason: String,
    },
    /// 交由人工审议。
    DeferToHuman { reason: String },
    /// 两者都保留（不同上下文适用）。
    KeepBoth { reason: String },
}

#[derive(Debug, Clone)]
pub struct DebateConfig {
    pub rounds: usize,
    pub judge_temperature: f32,
    pub advocate_temperature: f32,
    pub max_tokens_per_turn: u32,
    pub min_judge_confidence: f32,
}

impl Default for DebateConfig {
    fn default() -> Self {
        Self {
            rounds: 2,
            judge_temperature: 0.15,
            advocate_temperature: 0.45,
            max_tokens_per_turn: 512,
            min_judge_confidence: 0.62,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DebateRole {
    AdvocateA,
    AdvocateB,
    Judge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateTurn {
    pub round: usize,
    pub role: DebateRole,
    pub argument: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateVerdict {
    pub resolution: String,
    pub reason: String,
    pub merged: Option<String>,
    pub confidence: f32,
    pub hidden_assumptions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateReport {
    pub turns: Vec<DebateTurn>,
    pub verdict: DebateVerdict,
}

/// 冲突仲裁器。
pub struct ConflictResolver {
    llm: Option<Arc<dyn LlmClient>>,
    debate_config: DebateConfig,
}

impl ConflictResolver {
    pub fn new() -> Self {
        Self {
            llm: None,
            debate_config: DebateConfig::default(),
        }
    }

    pub fn with_llm(mut self, llm: Arc<dyn LlmClient>) -> Self {
        self.llm = Some(llm);
        self
    }

    pub fn with_debate_config(mut self, config: DebateConfig) -> Self {
        self.debate_config = config;
        self
    }

    /// 检测两个条目是否存在冲突。
    pub fn detect(&self, a: &MarketEntry, b: &MarketEntry) -> Option<MarketConflict> {
        if a.domain != b.domain || a.entry_type != b.entry_type {
            return None;
        }

        let words_a: HashSet<&str> = a.principle.split_whitespace().collect();
        let words_b: HashSet<&str> = b.principle.split_whitespace().collect();
        let intersection = words_a.intersection(&words_b).count();
        let union = words_a.union(&words_b).count();
        let jaccard = if union > 0 {
            intersection as f32 / union as f32
        } else {
            0.0
        };

        if jaccard > 0.8 {
            return None;
        }

        if jaccard > 0.4 && has_contradiction_marker(&a.principle, &b.principle) {
            let common_evidence = a.evidence_uris.iter().any(|u| b.evidence_uris.contains(u));
            let conflict_type = if common_evidence {
                ConflictType::SameEvidenceDifferentConclusion
            } else if jaccard > 0.6 {
                ConflictType::DirectContradiction
            } else {
                ConflictType::Overlapping
            };

            return Some(MarketConflict {
                entry_a: a.clone(),
                entry_b: b.clone(),
                conflict_type,
                similarity: jaccard,
            });
        }

        None
    }

    /// 批量检测。
    pub fn detect_all(&self, entries: &[MarketEntry]) -> Vec<MarketConflict> {
        let mut conflicts = Vec::new();
        for i in 0..entries.len() {
            for j in (i + 1)..entries.len() {
                if let Some(c) = self.detect(&entries[i], &entries[j]) {
                    conflicts.push(c);
                }
            }
        }
        conflicts
    }

    /// 仲裁冲突。证据强弱悬殊时直接裁决；其余高价值冲突走多智能体辩论。
    pub async fn arbitrate(&self, conflict: &MarketConflict) -> MarketConflictResolution {
        if let Some(resolution) = strong_evidence_resolution(conflict) {
            return resolution;
        }

        if self.llm.is_some() {
            let (resolution, _) = self.debate_arbitrate(conflict).await;
            return resolution;
        }

        defer_tie_resolution(conflict)
    }

    /// 多智能体辩论仲裁：A/B advocate 分别陈述和反驳，judge 最后给结构化裁决。
    pub async fn debate_arbitrate(
        &self,
        conflict: &MarketConflict,
    ) -> (MarketConflictResolution, DebateReport) {
        let Some(llm) = &self.llm else {
            let resolution = defer_tie_resolution(conflict);
            return (resolution.clone(), fallback_report(resolution));
        };

        let mut turns = Vec::new();
        for round in 0..self.debate_config.rounds.max(1) {
            let prompts = vec![
                debate_prompt(conflict, DebateRole::AdvocateA, round, &turns),
                debate_prompt(conflict, DebateRole::AdvocateB, round, &turns),
            ];
            let responses = llm
                .batch_complete(
                    &prompts,
                    &LlmOpts {
                        max_tokens: Some(self.debate_config.max_tokens_per_turn),
                        temperature: Some(self.debate_config.advocate_temperature),
                        task: LlmTaskKind::Arbitration,
                        prompt: PromptOptimization::default()
                            .force_cache()
                            .target_tokens(2_000),
                        ..Default::default()
                    },
                )
                .await
                .unwrap_or_default();

            turns.push(DebateTurn {
                round,
                role: DebateRole::AdvocateA,
                argument: responses.first().cloned().unwrap_or_default(),
            });
            turns.push(DebateTurn {
                round,
                role: DebateRole::AdvocateB,
                argument: responses.get(1).cloned().unwrap_or_default(),
            });
        }

        let verdict = judge_verdict(llm, conflict, &turns, &self.debate_config).await;
        let resolution = resolution_from_verdict(&verdict, self.debate_config.min_judge_confidence);
        (resolution, DebateReport { turns, verdict })
    }
}

impl Default for ConflictResolver {
    fn default() -> Self {
        Self::new()
    }
}

fn strong_evidence_resolution(conflict: &MarketConflict) -> Option<MarketConflictResolution> {
    let a = &conflict.entry_a;
    let b = &conflict.entry_b;
    let a_corrob = a.corroboration.independent_sources;
    let b_corrob = b.corroboration.independent_sources;

    if a_corrob > b_corrob * 2 && a.quality_score > b.quality_score + 0.2 {
        return Some(MarketConflictResolution::KeepA {
            reason: format!(
                "{} corroborators vs {}, quality {:.2} vs {:.2}",
                a_corrob, b_corrob, a.quality_score, b.quality_score
            ),
        });
    }
    if b_corrob > a_corrob * 2 && b.quality_score > a.quality_score + 0.2 {
        return Some(MarketConflictResolution::KeepB {
            reason: format!(
                "{} corroborators vs {}, quality {:.2} vs {:.2}",
                b_corrob, a_corrob, b.quality_score, a.quality_score
            ),
        });
    }
    None
}

fn defer_tie_resolution(conflict: &MarketConflict) -> MarketConflictResolution {
    let a = &conflict.entry_a;
    let b = &conflict.entry_b;
    MarketConflictResolution::DeferToHuman {
        reason: format!(
            "Evidence tie: A({}ev,{}cor,{:.2}q) vs B({}ev,{}cor,{:.2}q)",
            a.evidence_uris.len(),
            a.corroboration.independent_sources,
            a.quality_score,
            b.evidence_uris.len(),
            b.corroboration.independent_sources,
            b.quality_score
        ),
    }
}

fn debate_prompt(
    conflict: &MarketConflict,
    role: DebateRole,
    round: usize,
    prior: &[DebateTurn],
) -> String {
    let stance = match role {
        DebateRole::AdvocateA => "support Entry A and identify weaknesses in Entry B",
        DebateRole::AdvocateB => "support Entry B and identify weaknesses in Entry A",
        DebateRole::Judge => "judge",
    };
    let history = prior
        .iter()
        .map(|turn| format!("round {} {:?}: {}", turn.round, turn.role, turn.argument))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        r#"You are participating in a multi-agent knowledge-market debate.
Your role: {stance}.
Round: {round}.

Entry A by {a_pub}: {a_principle}
Evidence A: {a_ev} URIs, corroborators={a_cor}, quality={a_q:.2}, confidence={a_c:.2}

Entry B by {b_pub}: {b_principle}
Evidence B: {b_ev} URIs, corroborators={b_cor}, quality={b_q:.2}, confidence={b_c:.2}

Conflict type: {conflict_type:?}, lexical similarity={similarity:.3}

Prior debate:
{history}

Give a concise argument. Surface hidden assumptions and evidence gaps."#,
        a_pub = conflict.entry_a.publisher,
        a_principle = conflict.entry_a.principle,
        a_ev = conflict.entry_a.evidence_uris.len(),
        a_cor = conflict.entry_a.corroboration.independent_sources,
        a_q = conflict.entry_a.quality_score,
        a_c = conflict.entry_a.confidence,
        b_pub = conflict.entry_b.publisher,
        b_principle = conflict.entry_b.principle,
        b_ev = conflict.entry_b.evidence_uris.len(),
        b_cor = conflict.entry_b.corroboration.independent_sources,
        b_q = conflict.entry_b.quality_score,
        b_c = conflict.entry_b.confidence,
        conflict_type = conflict.conflict_type,
        similarity = conflict.similarity,
    )
}

async fn judge_verdict(
    llm: &Arc<dyn LlmClient>,
    conflict: &MarketConflict,
    turns: &[DebateTurn],
    config: &DebateConfig,
) -> DebateVerdict {
    let transcript = turns
        .iter()
        .map(|turn| format!("round {} {:?}: {}", turn.round, turn.role, turn.argument))
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = format!(
        r#"Judge this knowledge conflict after a structured debate.

Entry A: {a}
Entry B: {b}
Conflict type: {typ:?}

Debate transcript:
{transcript}

Return JSON only:
{{"resolution":"keep_a|keep_b|fuse|defer|keep_both","reason":"...","merged":"... or null","confidence":0.0,"hidden_assumptions":["..."]}}"#,
        a = conflict.entry_a.principle,
        b = conflict.entry_b.principle,
        typ = conflict.conflict_type,
    );
    let schema = JsonSchema::new(serde_json::json!({
        "type": "object",
        "properties": {
            "resolution": {"type":"string"},
            "reason": {"type":"string"},
            "merged": {"type":["string", "null"]},
            "confidence": {"type":"number"},
            "hidden_assumptions": {"type":"array", "items":{"type":"string"}}
        },
        "required": ["resolution", "reason", "confidence"]
    }));
    let opts = LlmOpts {
        max_tokens: Some(config.max_tokens_per_turn),
        temperature: Some(config.judge_temperature),
        task: LlmTaskKind::Arbitration,
        prompt: PromptOptimization::default()
            .force_cache()
            .target_tokens(3_000),
        ..Default::default()
    };
    let raw = match llm.complete_json(&prompt, &schema, &opts).await {
        Ok(value) => value,
        Err(_) => llm.complete(&prompt, &opts).await.unwrap_or_default(),
    };
    parse_verdict(&raw).unwrap_or_else(|| DebateVerdict {
        resolution: "defer".into(),
        reason: "judge response could not be parsed".into(),
        merged: None,
        confidence: 0.0,
        hidden_assumptions: Vec::new(),
    })
}

fn parse_verdict(raw: &str) -> Option<DebateVerdict> {
    let value: serde_json::Value = serde_json::from_str(raw)
        .or_else(|_| serde_json::from_str(&extract_json_object(raw)))
        .ok()?;
    Some(DebateVerdict {
        resolution: value["resolution"].as_str()?.to_ascii_lowercase(),
        reason: value["reason"].as_str().unwrap_or("debate judged").into(),
        merged: value["merged"].as_str().map(ToOwned::to_owned),
        confidence: value["confidence"].as_f64().unwrap_or(0.0).clamp(0.0, 1.0) as f32,
        hidden_assumptions: value["hidden_assumptions"]
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(ToOwned::to_owned))
                    .collect()
            })
            .unwrap_or_default(),
    })
}

fn extract_json_object(text: &str) -> String {
    let trimmed = text.trim();
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            return trimmed[start..=end].to_string();
        }
    }
    trimmed.to_string()
}

fn resolution_from_verdict(
    verdict: &DebateVerdict,
    min_confidence: f32,
) -> MarketConflictResolution {
    if verdict.confidence < min_confidence {
        return MarketConflictResolution::DeferToHuman {
            reason: format!(
                "debate confidence {:.2} below threshold {:.2}: {}",
                verdict.confidence, min_confidence, verdict.reason
            ),
        };
    }
    match verdict.resolution.as_str() {
        "keep_a" => MarketConflictResolution::KeepA {
            reason: verdict.reason.clone(),
        },
        "keep_b" => MarketConflictResolution::KeepB {
            reason: verdict.reason.clone(),
        },
        "fuse" => MarketConflictResolution::Fuse {
            merged_content: verdict.merged.clone().unwrap_or_default(),
            reason: verdict.reason.clone(),
        },
        "keep_both" => MarketConflictResolution::KeepBoth {
            reason: verdict.reason.clone(),
        },
        _ => MarketConflictResolution::DeferToHuman {
            reason: verdict.reason.clone(),
        },
    }
}

fn fallback_report(resolution: MarketConflictResolution) -> DebateReport {
    let reason = match resolution {
        MarketConflictResolution::KeepA { reason }
        | MarketConflictResolution::KeepB { reason }
        | MarketConflictResolution::Fuse { reason, .. }
        | MarketConflictResolution::DeferToHuman { reason }
        | MarketConflictResolution::KeepBoth { reason } => reason,
    };
    DebateReport {
        turns: Vec::new(),
        verdict: DebateVerdict {
            resolution: "defer".into(),
            reason,
            merged: None,
            confidence: 0.0,
            hidden_assumptions: Vec::new(),
        },
    }
}

fn has_contradiction_marker(a: &str, b: &str) -> bool {
    let negation_words = [
        "not",
        "no",
        "never",
        "impossible",
        "cannot",
        "can't",
        "don't",
        "false",
        "wrong",
        "incorrect",
        "不",
        "没有",
        "错误",
    ];
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_neg = negation_words.iter().any(|n| a_lower.contains(n));
    let b_neg = negation_words.iter().any(|n| b_lower.contains(n));
    a_neg != b_neg
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentType, EpistemicType, LlmError};
    use async_trait::async_trait;

    struct DebateLlm;

    #[async_trait]
    impl LlmClient for DebateLlm {
        async fn complete(&self, _prompt: &str, _opts: &LlmOpts) -> Result<String, LlmError> {
            Ok("argument".into())
        }

        async fn batch_complete(
            &self,
            prompts: &[String],
            _opts: &LlmOpts,
        ) -> Result<Vec<String>, LlmError> {
            Ok(prompts
                .iter()
                .enumerate()
                .map(|(idx, _)| format!("argument {idx}"))
                .collect())
        }

        async fn complete_json(
            &self,
            _prompt: &str,
            _schema: &JsonSchema,
            _opts: &LlmOpts,
        ) -> Result<String, LlmError> {
            Ok(r#"{"resolution":"keep_a","reason":"A has stronger operational evidence","merged":null,"confidence":0.82,"hidden_assumptions":["evidence URIs are trustworthy"]}"#.into())
        }
    }

    fn entry(principle: &str, quality: f32) -> MarketEntry {
        MarketEntry {
            id: MarketId::new(),
            publisher: AgentId::new("agent"),
            domain: "retrieval".into(),
            entry_type: MarketEntryType::Fact,
            principle: principle.into(),
            evidence_uris: Vec::new(),
            quality_score: quality,
            confidence: quality,
            corroboration: CorroborationProof::new(),
            provenance: None,
            license: KnowledgeLicense::Attribution,
            epistemic_type: EpistemicType::Fact,
            content_type: ContentType::Fact,
            half_life_days: None,
            created_at: chrono::Utc::now(),
            expires_at: None,
        }
    }

    #[tokio::test]
    async fn debate_arbitration_uses_advocates_and_judge() {
        let conflict = MarketConflict {
            entry_a: entry("bounded graph traversal improves recall", 0.7),
            entry_b: entry("bounded graph traversal does not improve recall", 0.7),
            conflict_type: ConflictType::DirectContradiction,
            similarity: 0.7,
        };
        let resolver = ConflictResolver::new()
            .with_llm(Arc::new(DebateLlm))
            .with_debate_config(DebateConfig {
                rounds: 2,
                ..Default::default()
            });

        let (resolution, report) = resolver.debate_arbitrate(&conflict).await;

        assert!(matches!(resolution, MarketConflictResolution::KeepA { .. }));
        assert_eq!(report.turns.len(), 4);
        assert!(report.verdict.confidence > 0.8);
    }
}
