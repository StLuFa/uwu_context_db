//! `MemoryExtractorImpl`：8 类记忆提取 + 历史召回驱动的严格 LLM 去重。

use agent_context_db_core::{
    ContextUri, LlmClient, LlmOpts, LlmTaskKind, PromptOptimization, Result, VectorIndex,
};
use async_trait::async_trait;
use std::sync::Arc;

use crate::{CandidateAction, DedupDecision, MemoryCandidate, MemoryExtractor};

/// 基于 `LlmClient` 的记忆提取器实现。
pub struct MemoryExtractorImpl {
    llm: Arc<dyn LlmClient>,
    history: Arc<dyn VectorIndex>,
    history_collection: String,
    history_top_k: usize,
}

impl MemoryExtractorImpl {
    /// 构造启用历史存量召回的提取器。历史索引是必需依赖，避免退化为仅批内去重。
    pub fn new(
        llm: Arc<dyn LlmClient>,
        history: Arc<dyn VectorIndex>,
        history_collection: impl Into<String>,
        history_top_k: usize,
    ) -> Result<Self> {
        if history_top_k == 0 {
            return Err(agent_context_db_core::ContextError::Unsupported(
                "memory dedup history_top_k must be greater than zero".into(),
            ));
        }
        Ok(Self {
            llm,
            history,
            history_collection: history_collection.into(),
            history_top_k,
        })
    }
}

#[derive(serde::Deserialize)]
struct RawDecision {
    candidate_index: usize,
    action: String,
    reason: Option<String>,
    merge_target: Option<String>,
}

fn contract_error(message: impl Into<String>) -> agent_context_db_core::ContextError {
    agent_context_db_core::ContextError::Storage(format!(
        "llm dedup contract violation: {}",
        message.into()
    ))
}

fn validate_decisions(
    raw: Vec<RawDecision>,
    candidates: Vec<MemoryCandidate>,
) -> Result<Vec<DedupDecision>> {
    if raw.len() != candidates.len() {
        return Err(contract_error(format!(
            "expected {} decisions, received {}",
            candidates.len(),
            raw.len()
        )));
    }
    let mut decisions: Vec<Option<DedupDecision>> = vec![None; candidates.len()];
    for item in raw {
        let candidate = candidates.get(item.candidate_index).ok_or_else(|| {
            contract_error(format!("index {} is out of range", item.candidate_index))
        })?;
        if decisions[item.candidate_index].is_some() {
            return Err(contract_error(format!(
                "duplicate index {}",
                item.candidate_index
            )));
        }
        let action = match item.action.as_str() {
            "skip" => CandidateAction::Skip,
            "create" => CandidateAction::Create,
            "merge" => CandidateAction::Merge,
            invalid => {
                return Err(contract_error(format!(
                    "invalid action {invalid:?} at index {}",
                    item.candidate_index
                )));
            }
        };
        let merge_target = match (action, item.merge_target) {
            (CandidateAction::Merge, Some(uri)) => Some(ContextUri::parse(uri).map_err(|e| {
                contract_error(format!(
                    "invalid merge_target at index {}: {e}",
                    item.candidate_index
                ))
            })?),
            (CandidateAction::Merge, None) => {
                return Err(contract_error(format!(
                    "merge at index {} has no merge_target",
                    item.candidate_index
                )));
            }
            (_, Some(_)) => {
                return Err(contract_error(format!(
                    "non-merge at index {} has merge_target",
                    item.candidate_index
                )));
            }
            (_, None) => None,
        };
        decisions[item.candidate_index] = Some(DedupDecision {
            candidate: candidate.clone(),
            action,
            merge_target,
            reason: item.reason.unwrap_or_default(),
        });
    }
    decisions
        .into_iter()
        .enumerate()
        .map(|(index, decision)| {
            decision.ok_or_else(|| contract_error(format!("missing index {index}")))
        })
        .collect()
}

#[async_trait]
impl MemoryExtractor for MemoryExtractorImpl {
    async fn extract(&self, archive: &ContextUri) -> Result<Vec<MemoryCandidate>> {
        let prompt = format!(
            r#"You are a memory extraction system. From the conversation archive at "{archive}",
extract structured memories across all supported content categories.

Return a JSON array of objects with "content_type", "content", and "confidence".
Only include entries with confidence >= 0.5."#
        );
        let opts = LlmOpts {
            max_tokens: Some(2048),
            temperature: Some(0.1),
            task: LlmTaskKind::Extraction,
            prompt: PromptOptimization::default()
                .force_cache()
                .target_tokens(2_500),
            ..Default::default()
        };
        let response = self.llm.complete(&prompt, &opts).await.map_err(|e| {
            agent_context_db_core::ContextError::Storage(format!("llm extract: {e}"))
        })?;
        let candidates: Vec<MemoryCandidate> = serde_json::from_str(&response)?;
        Ok(candidates
            .into_iter()
            .filter(|candidate| candidate.confidence >= 0.5)
            .map(|mut candidate| {
                candidate.source_uri = archive.clone();
                candidate
            })
            .collect())
    }

    async fn deduplicate(&self, candidates: Vec<MemoryCandidate>) -> Result<Vec<DedupDecision>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let candidates_json = serde_json::to_string(&candidates)?;
        let mut history_by_candidate = Vec::with_capacity(candidates.len());
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let embedding = self.llm.embed(&candidate.content).await.map_err(|e| {
                agent_context_db_core::ContextError::Storage(format!(
                    "memory history embedding for candidate {candidate_index}: {e}"
                ))
            })?;
            let hits = self
                .history
                .search(
                    &self.history_collection,
                    embedding.vector,
                    self.history_top_k,
                    Some(serde_json::json!({"content_type": candidate.content_type})),
                )
                .await?;
            history_by_candidate.push(serde_json::json!({
                "candidate_index": candidate_index,
                "matches": hits.into_iter().map(|hit| serde_json::json!({
                    "uri": hit.uri,
                    "score": hit.score,
                    "memory": hit.payload,
                })).collect::<Vec<_>>()
            }));
        }
        let history_json = serde_json::to_string(&history_by_candidate)?;
        let prompt = format!(
            r#"Compare every memory candidate against both the batch and recalled stored memories.
Actions are exactly "skip", "create", or "merge". A merge requires a recalled memory URI as merge_target.
Candidates: {candidates_json}
Recalled stored memories by candidate: {history_json}
Return one JSON object for every candidate index, exactly once, with "candidate_index", "action", "reason", and optional "merge_target". Do not omit, duplicate, or invent indexes."#
        );
        let opts = LlmOpts {
            max_tokens: Some(1024),
            temperature: Some(0.0),
            task: LlmTaskKind::Deduplication,
            prompt: PromptOptimization::default()
                .force_cache()
                .target_tokens(1_800),
            ..Default::default()
        };
        let response =
            self.llm.complete(&prompt, &opts).await.map_err(|e| {
                agent_context_db_core::ContextError::Storage(format!("llm dedup: {e}"))
            })?;
        let raw: Vec<RawDecision> = serde_json::from_str(&response)
            .map_err(|e| contract_error(format!("invalid JSON: {e}")))?;
        validate_decisions(raw, candidates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{
        ContentType, EmbeddingVector, IndexHit, IndexPoint, JsonSchema, LlmError,
    };
    use std::sync::Mutex;

    struct MockLlm {
        response: String,
        prompts: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn complete(
            &self,
            prompt: &str,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            self.prompts.lock().unwrap().push(prompt.to_string());
            Ok(self.response.clone())
        }

        async fn complete_json(
            &self,
            prompt: &str,
            _schema: &JsonSchema,
            _opts: &LlmOpts,
        ) -> std::result::Result<String, LlmError> {
            self.complete(prompt, _opts).await
        }

        async fn embed(&self, _text: &str) -> std::result::Result<EmbeddingVector, LlmError> {
            Ok(EmbeddingVector::new(vec![1.0, 0.0], "test", 1))
        }
    }

    struct HistoryIndex {
        searches: Mutex<usize>,
    }

    #[async_trait]
    impl VectorIndex for HistoryIndex {
        async fn upsert(&self, _collection: &str, _point: IndexPoint) -> Result<()> {
            Ok(())
        }
        async fn search(
            &self,
            _collection: &str,
            _query: Vec<f32>,
            _top_k: usize,
            _filter: Option<serde_json::Value>,
        ) -> Result<Vec<IndexHit>> {
            *self.searches.lock().unwrap() += 1;
            Ok(vec![IndexHit {
                uri: ContextUri::parse("uwu://t/user/u/memories/preferences/existing").unwrap(),
                score: 0.99,
                payload: serde_json::json!({"content": "dark mode"}),
            }])
        }
        async fn delete(&self, _collection: &str, _uri: &ContextUri) -> Result<()> {
            Ok(())
        }
    }

    fn candidate(content: &str) -> MemoryCandidate {
        MemoryCandidate {
            content_type: ContentType::Preference,
            content: content.into(),
            source_uri: ContextUri::parse("uwu://t/user/u/sessions/s1").unwrap(),
            confidence: 0.9,
        }
    }

    async fn run(response: &str, candidates: Vec<MemoryCandidate>) -> Result<Vec<DedupDecision>> {
        let llm = Arc::new(MockLlm {
            response: response.into(),
            prompts: Mutex::new(Vec::new()),
        });
        let history = Arc::new(HistoryIndex {
            searches: Mutex::new(0),
        });
        let extractor = MemoryExtractorImpl::new(llm.clone(), history.clone(), "memories", 5)?;
        let result = extractor.deduplicate(candidates).await;
        assert_eq!(*history.searches.lock().unwrap(), 2);
        assert!(llm.prompts.lock().unwrap()[0].contains("existing"));
        result
    }

    #[tokio::test]
    async fn contract_accepts_exactly_one_decision_per_index_in_input_order() {
        let decisions = run(
            r#"[{"candidate_index":1,"action":"create"},{"candidate_index":0,"action":"skip"}]"#,
            vec![candidate("a"), candidate("b")],
        )
        .await
        .unwrap();
        assert_eq!(decisions.len(), 2);
        assert_eq!(decisions[0].candidate.content, "a");
        assert_eq!(decisions[0].action, CandidateAction::Skip);
    }

    #[tokio::test]
    async fn contract_rejects_missing_duplicate_out_of_range_and_invalid_action() {
        for response in [
            r#"[{"candidate_index":0,"action":"create"}]"#,
            r#"[{"candidate_index":0,"action":"create"},{"candidate_index":0,"action":"skip"}]"#,
            r#"[{"candidate_index":0,"action":"create"},{"candidate_index":2,"action":"skip"}]"#,
            r#"[{"candidate_index":0,"action":"delete"},{"candidate_index":1,"action":"create"}]"#,
        ] {
            assert!(
                run(response, vec![candidate("a"), candidate("b")])
                    .await
                    .is_err()
            );
        }
    }

    #[tokio::test]
    async fn malformed_response_fails_instead_of_creating_every_candidate() {
        assert!(
            run("not json", vec![candidate("a"), candidate("b")])
                .await
                .is_err()
        );
    }
}
