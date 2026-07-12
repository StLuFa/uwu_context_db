//! SkillLibrary — embedding 检索 + deposit（课程驱动）。

use crate::config::SkillLibraryConfig;
use agent_context_db_core::{ContextError, ContextUri, Result, VectorIndex};
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
    scope: ContextUri,
    collection: String,
    config: SkillLibraryConfig,
}

impl SkillLibrary {
    pub fn new(
        index: Arc<dyn VectorIndex>,
        scope: ContextUri,
        config: SkillLibraryConfig,
    ) -> Result<Self> {
        config.validate()?;
        let segments = scope.segments();
        if segments.len() < 3 || segments[1] != "agent" {
            return Err(ContextError::InvalidUri(format!(
                "skill scope must be uwu://<tenant>/agent/<id>: {scope}"
            )));
        }
        let namespace = blake3::hash(scope.as_str().as_bytes()).to_hex();
        Ok(Self {
            index,
            scope,
            collection: format!("skills_{}", &namespace[..config.collection_hash_chars]),
            config,
        })
    }

    fn validate_writeback(&self, success_rate: f32) -> Result<()> {
        if success_rate.is_finite()
            && (self.config.min_writeback_success_rate..=1.0).contains(&success_rate)
        {
            Ok(())
        } else {
            Err(ContextError::Unsupported(format!(
                "skill success rate must be finite and within {}..=1 (got {success_rate})",
                self.config.min_writeback_success_rate
            )))
        }
    }

    fn owns(&self, uri: &ContextUri) -> bool {
        let root = self.scope.as_str().trim_end_matches('/');
        uri.as_str() == root || uri.as_str().starts_with(&format!("{root}/"))
    }

    /// 新任务时检索 top-K 相似 skill。
    pub async fn retrieve(&self, task_embedding: &[f32]) -> Result<Vec<SkillEntry>> {
        let hits = self
            .index
            .search(
                &self.collection,
                task_embedding.to_vec(),
                self.config.default_retrieval_limit,
                None,
            )
            .await?;
        let skills = hits
            .into_iter()
            .filter_map(|hit| {
                if !self.owns(&hit.uri) {
                    tracing::warn!("cross-namespace skill ignored");
                    return None;
                }
                let payload = hit.payload.as_object()?;
                let name = payload.get("name")?.as_str()?.trim();
                let description = payload.get("description")?.as_str()?.trim();
                let precondition = payload.get("precondition")?.as_str()?.trim();
                let success_rate = payload.get("success_rate")?.as_f64()? as f32;
                let embedding = payload
                    .get("embedding")?
                    .as_array()?
                    .iter()
                    .map(|value| value.as_f64().map(|number| number as f32))
                    .collect::<Option<Vec<_>>>()?;
                if name.is_empty()
                    || description.is_empty()
                    || !success_rate.is_finite()
                    || !(0.0..=1.0).contains(&success_rate)
                    || embedding.is_empty()
                {
                    tracing::warn!(uri = %hit.uri, "invalid skill payload ignored");
                    return None;
                }
                Some(SkillEntry {
                    uri: hit.uri,
                    name: name.to_string(),
                    description: description.to_string(),
                    precondition: precondition.to_string(),
                    success_rate,
                    embedding,
                })
            })
            .collect::<Vec<_>>();
        Ok(skills)
    }

    /// 执行成功后存入 skill library。
    pub async fn deposit(&self, skill: &SkillEntry) -> Result<()> {
        self.validate_writeback(skill.success_rate)?;
        self.deposit_with_payload(
            skill.uri.clone(),
            skill.embedding.clone(),
            serde_json::json!({
                "name": skill.name,
                "description": skill.description,
                "precondition": skill.precondition,
                "success_rate": skill.success_rate,
                "embedding": skill.embedding,
            }),
        )
        .await
    }

    /// 将睡眠期经验重放产出的 skill candidate 写入技能索引。
    pub async fn deposit_replay_candidate(
        &self,
        candidate: &ReplaySkillCandidate,
        embedding: Vec<f32>,
    ) -> Result<()> {
        self.validate_writeback(candidate.success_rate)?;
        self.deposit_with_payload(
            candidate.uri.clone(),
            embedding.clone(),
            serde_json::json!({
                "name": candidate.name,
                "description": candidate.description,
                "precondition": candidate.precondition,
                "success_rate": candidate.success_rate,
                "embedding": embedding,
                "evidence": candidate.evidence,
                "source": "dream_replay",
            }),
        )
        .await
    }

    pub async fn deposit_replay_candidates(
        &self,
        candidates: &[(ReplaySkillCandidate, Vec<f32>)],
    ) -> Result<usize> {
        let mut written = 0;
        for (candidate, embedding) in candidates {
            self.deposit_replay_candidate(candidate, embedding.clone())
                .await?;
            written += 1;
        }
        Ok(written)
    }

    async fn deposit_with_payload(
        &self,
        uri: ContextUri,
        vector: Vec<f32>,
        payload: serde_json::Value,
    ) -> Result<()> {
        if !self.owns(&uri) {
            return Err(ContextError::PermissionDenied(format!(
                "skill URI {uri} is outside namespace {}",
                self.scope
            )));
        }
        let point = agent_context_db_core::IndexPoint {
            uri,
            vector,
            embedding_model_id: None,
            embedding_dim: None,
            embedding_version: None,
            payload,
        };
        self.index.upsert(&self.collection, point).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContextError, IndexHit, IndexPoint, Result};
    use async_trait::async_trait;
    use parking_lot::Mutex;

    #[derive(Default)]
    struct RoundTripIndex {
        point: Mutex<Option<IndexPoint>>,
    }

    #[async_trait]
    impl VectorIndex for RoundTripIndex {
        async fn upsert(&self, _collection: &str, point: IndexPoint) -> Result<()> {
            *self.point.lock() = Some(point);
            Ok(())
        }

        async fn search(
            &self,
            _collection: &str,
            _query: Vec<f32>,
            _top_k: usize,
            _filter: Option<serde_json::Value>,
        ) -> Result<Vec<IndexHit>> {
            let point = self
                .point
                .lock()
                .clone()
                .ok_or_else(|| ContextError::NotFound("skill".into()))?;
            Ok(vec![IndexHit {
                uri: point.uri,
                score: 0.99,
                payload: point.payload,
            }])
        }

        async fn delete(&self, _collection: &str, _uri: &ContextUri) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn deposit_and_retrieve_round_trip_complete_payload() {
        let library = SkillLibrary::new(
            Arc::new(RoundTripIndex::default()),
            ContextUri::parse("uwu://tenant/agent/a").unwrap(),
            SkillLibraryConfig::default(),
        )
        .unwrap();
        let skill = SkillEntry {
            uri: ContextUri::parse("uwu://tenant/agent/a/memory/skill/deploy").unwrap(),
            name: "deploy".into(),
            description: "deploy after tests pass".into(),
            precondition: "tests pass".into(),
            success_rate: 0.91,
            embedding: vec![0.1, 0.2, 0.3],
        };
        library.deposit(&skill).await.unwrap();
        let result = library.retrieve(&[0.1, 0.2, 0.3]).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].name, skill.name);
        assert_eq!(result[0].description, skill.description);
        assert_eq!(result[0].precondition, skill.precondition);
        assert_eq!(result[0].embedding, skill.embedding);
        assert!((result[0].success_rate - skill.success_rate).abs() < f32::EPSILON);
    }
}
