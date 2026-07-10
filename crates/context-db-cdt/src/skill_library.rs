//! SkillLibrary — embedding 检索 + deposit（课程驱动）。

use agent_context_db_core::{ContextUri, VectorIndex};
use agent_context_db_version::ReplaySkillCandidate;
use std::sync::Arc;

/// Skill 条目。
#[derive(Debug, Clone)]
pub struct SkillEntry {
    pub uri: ContextUri,
    pub name: String,
    pub description: String,
    pub precondition: String,
    pub success_rate: f32,
    pub embedding: Vec<f32>,
}

/// Skill 库 — Validated skill 带 embedding，可检索复用。
pub struct SkillLibrary {
    index: Arc<dyn VectorIndex>,
}

impl SkillLibrary {
    pub fn new(index: Arc<dyn VectorIndex>) -> Self {
        Self { index }
    }

    /// 新任务时检索 top-K 相似 skill。
    pub async fn retrieve(&self, task_embedding: &[f32], k: usize) -> Vec<SkillEntry> {
        // 使用向量索引搜索相似 skill
        let hits = self
            .index
            .search("skills", task_embedding.to_vec(), k, None)
            .await
            .unwrap_or_default();
        hits.into_iter()
            .map(|h| SkillEntry {
                uri: h.uri.clone(),
                name: h.uri.to_string(),
                description: String::new(),
                precondition: String::new(),
                success_rate: h.score,
                embedding: vec![],
            })
            .collect()
    }

    /// 执行成功后存入 skill library。
    pub async fn deposit(&self, skill: &SkillEntry) {
        self.deposit_with_payload(
            skill.uri.clone(),
            skill.embedding.clone(),
            serde_json::json!({
                "name": skill.name,
                "description": skill.description,
                "precondition": skill.precondition,
                "success_rate": skill.success_rate,
            }),
        )
        .await;
    }

    /// 将睡眠期经验重放产出的 skill candidate 写入技能索引。
    pub async fn deposit_replay_candidate(
        &self,
        candidate: &ReplaySkillCandidate,
        embedding: Vec<f32>,
    ) {
        self.deposit_with_payload(
            candidate.uri.clone(),
            embedding,
            serde_json::json!({
                "name": candidate.name,
                "description": candidate.description,
                "precondition": candidate.precondition,
                "success_rate": candidate.success_rate,
                "evidence": candidate.evidence,
                "source": "dream_replay",
            }),
        )
        .await;
    }

    pub async fn deposit_replay_candidates(
        &self,
        candidates: &[(ReplaySkillCandidate, Vec<f32>)],
    ) -> usize {
        let mut written = 0;
        for (candidate, embedding) in candidates {
            self.deposit_replay_candidate(candidate, embedding.clone())
                .await;
            written += 1;
        }
        written
    }

    async fn deposit_with_payload(
        &self,
        uri: ContextUri,
        vector: Vec<f32>,
        payload: serde_json::Value,
    ) {
        let point = agent_context_db_core::IndexPoint {
            uri,
            vector,
            embedding_model_id: None,
            embedding_dim: None,
            embedding_version: None,
            payload,
        };
        let _ = self.index.upsert("skills", point).await;
    }
}
