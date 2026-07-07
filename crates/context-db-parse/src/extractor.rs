//! `MemoryExtractorImpl`：8 类记忆提取 + LLM 去重。
//!
//! 依赖 core 的 `LlmClient` 端口进行语义处理。

use agent_context_db_core::{ContextUri, LlmClient, LlmOpts, MemoryClass, Result};
use async_trait::async_trait;
use std::sync::Arc;

use crate::{CandidateAction, DedupDecision, MemoryCandidate, MemoryExtractor};

/// 基于 `LlmClient` 的记忆提取器实现。
pub struct MemoryExtractorImpl {
    llm: Arc<dyn LlmClient>,
}

impl MemoryExtractorImpl {
    pub fn new(llm: Arc<dyn LlmClient>) -> Self {
        Self { llm }
    }
}

#[async_trait]
impl MemoryExtractor for MemoryExtractorImpl {
    async fn extract(&self, archive: &ContextUri) -> Result<Vec<MemoryCandidate>> {
        // 构建提取 prompt
        let prompt = format!(
            r#"You are a memory extraction system. From the conversation archive at "{archive}",
extract structured memories across 8 categories.

Return a JSON array of objects with:
- "class": one of ["profile", "preferences", "entities", "events", "cases", "patterns", "tools", "skills"]
- "content": a concise description (1-2 sentences)
- "confidence": a float 0.0-1.0

Only include entries with confidence > 0.5.
"#
        );

        let opts = LlmOpts {
            max_tokens: Some(2048),
            temperature: Some(0.1),
            ..Default::default()
        };

        let response = self.llm.complete(&prompt, &opts).await.map_err(|e| {
            agent_context_db_core::ContextError::Storage(format!("llm extract: {e}"))
        })?;

        // 解析 JSON
        let candidates: Vec<MemoryCandidate> =
            serde_json::from_str(&response).unwrap_or_else(|_| {
                vec![MemoryCandidate {
                    class: MemoryClass::Cases,
                    content: response.chars().take(200).collect(),
                    source_uri: archive.clone(),
                    confidence: 0.5,
                }]
            });

        Ok(candidates
            .into_iter()
            .filter(|c| c.confidence >= 0.5)
            .map(|mut c| {
                c.source_uri = archive.clone();
                c
            })
            .collect())
    }

    async fn deduplicate(&self, candidates: Vec<MemoryCandidate>) -> Result<Vec<DedupDecision>> {
        if candidates.is_empty() {
            return Ok(vec![]);
        }

        let candidates_json = serde_json::to_string(&candidates).unwrap_or_default();

        let prompt = format!(
            r#"You are a memory deduplication system. Given these memory candidates, decide for each whether to:
- "skip": the memory already exists (duplicate)
- "create": create a new memory entry
- "merge": merge with an existing memory (provide merge_target)

Candidates: {candidates_json}

Return a JSON array of objects with: "candidate_index", "action", "reason", "merge_target" (optional).
"#
        );

        let opts = LlmOpts {
            max_tokens: Some(1024),
            temperature: Some(0.0),
            ..Default::default()
        };

        let response =
            self.llm.complete(&prompt, &opts).await.map_err(|e| {
                agent_context_db_core::ContextError::Storage(format!("llm dedup: {e}"))
            })?;

        // 解析或 fallback
        #[derive(serde::Deserialize)]
        struct RawDecision {
            candidate_index: usize,
            action: String,
            reason: Option<String>,
            merge_target: Option<String>,
        }

        let raw: Vec<RawDecision> = serde_json::from_str(&response).unwrap_or_default();

        if raw.is_empty() {
            // Fallback: 全部创建
            return Ok(candidates
                .into_iter()
                .map(|c| DedupDecision {
                    candidate: c,
                    action: CandidateAction::Create,
                    merge_target: None,
                    reason: "new".into(),
                })
                .collect());
        }

        Ok(raw
            .into_iter()
            .filter_map(|r| {
                let idx = r.candidate_index;
                let candidate = candidates.get(idx)?.clone();
                let action = match r.action.as_str() {
                    "skip" => CandidateAction::Skip,
                    "create" => CandidateAction::Create,
                    "merge" => CandidateAction::Merge,
                    "delete" => CandidateAction::Delete,
                    _ => CandidateAction::Create,
                };
                Some(DedupDecision {
                    candidate,
                    action,
                    merge_target: r.merge_target.and_then(|s| ContextUri::parse(s).ok()),
                    reason: r.reason.unwrap_or_default(),
                })
            })
            .collect())
    }
}
