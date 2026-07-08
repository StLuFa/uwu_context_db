//! SkillLibrary — embedding 检索 + deposit（课程驱动）。

use agent_context_db_core::{ContextUri, VectorIndex};
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
        let point = agent_context_db_core::IndexPoint {
            uri: skill.uri.clone(),
            vector: skill.embedding.clone(),
            embedding_model_id: None,
            embedding_dim: None,
            embedding_version: None,
            payload: serde_json::json!({
                "name": skill.name,
                "precondition": skill.precondition,
                "success_rate": skill.success_rate,
            }),
        };
        let _ = self.index.upsert("skills", point).await;
    }
}
