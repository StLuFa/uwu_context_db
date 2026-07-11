//! `TrajectoryExtractorImpl`：会话→Trajectory→Experience 两层归纳。
//!
//! 使用 `LlmClient` 从会话归档中提取：
//! - Trajectory（做了什么、怎么做、结果如何）
//! - Experience（多条轨迹归纳为可复用经验）

use agent_context_db_core::{ContentLevel, ContextUri, FsOps, LlmClient, LlmOpts, Result};
use async_trait::async_trait;
use std::sync::Arc;
use uuid::Uuid;

use crate::{Experience, Trajectory, TrajectoryExtractor};

/// 基于 `LlmClient` 的轨迹提取器实现。
pub struct TrajectoryExtractorImpl {
    llm: Arc<dyn LlmClient>,
    fs: Arc<dyn FsOps>,
}

impl TrajectoryExtractorImpl {
    pub fn new(llm: Arc<dyn LlmClient>, fs: Arc<dyn FsOps>) -> Self {
        Self { llm, fs }
    }
}

#[async_trait]
impl TrajectoryExtractor for TrajectoryExtractorImpl {
    async fn extract_trajectory(&self, archive: &ContextUri) -> Result<Trajectory> {
        // 读取归档内容
        let content = self.fs.read(archive, ContentLevel::L1).await?;

        let text = match &content {
            agent_context_db_core::ContentPayload::Text {
                dense,
                full,
                sparse,
            } => {
                if !dense.trim().is_empty() {
                    dense.clone()
                } else if !full.trim().is_empty() {
                    full.clone()
                } else {
                    sparse.clone()
                }
            }
            _ => {
                return Err(agent_context_db_core::ContextError::Unsupported(format!(
                    "trajectory archive {archive} is not text"
                )));
            }
        };
        if text.trim().is_empty() {
            return Err(agent_context_db_core::ContextError::Storage(format!(
                "trajectory archive {archive} is empty"
            )));
        }

        let prompt = format!(
            r#"Analyze this conversation archive and extract a structured trajectory.

Archive content:
{text}

Return a JSON object with these fields:
- "did_what": what the agent/user accomplished (one sentence)
- "how": the approach/method used (one sentence)
- "result": the outcome or result (one sentence)
"#
        );

        let opts = LlmOpts {
            max_tokens: Some(512),
            temperature: Some(0.1),
            ..Default::default()
        };

        let response = self.llm.complete(&prompt, &opts).await.map_err(|e| {
            agent_context_db_core::ContextError::Storage(format!("llm trajectory: {e}"))
        })?;

        #[derive(serde::Deserialize)]
        struct RawTrajectory {
            did_what: String,
            how: String,
            result: String,
        }

        let raw: RawTrajectory = serde_json::from_str(&response).map_err(|error| {
            agent_context_db_core::ContextError::Storage(format!(
                "llm trajectory returned invalid JSON: {error}"
            ))
        })?;
        if raw.did_what.trim().is_empty()
            || raw.how.trim().is_empty()
            || raw.result.trim().is_empty()
        {
            return Err(agent_context_db_core::ContextError::Storage(
                "llm trajectory returned empty required fields".into(),
            ));
        }

        let traj_uri = archive.join("trajectory.json");

        Ok(Trajectory {
            uri: traj_uri,
            session_id: Uuid::new_v4(),
            did_what: raw.did_what,
            how: raw.how,
            result: raw.result,
            state_snapshot_uri: None,
            created_at: chrono::Utc::now(),
        })
    }

    async fn induce_experience(&self, trajectories: Vec<ContextUri>) -> Result<Experience> {
        // 读取每条轨迹的内容
        let mut traj_texts = Vec::new();
        if trajectories.is_empty() {
            return Err(agent_context_db_core::ContextError::Storage(
                "cannot induce experience from an empty trajectory list".into(),
            ));
        }
        for uri in &trajectories {
            let content = self.fs.read(uri, ContentLevel::L1).await?;
            let agent_context_db_core::ContentPayload::Text {
                dense,
                full,
                sparse,
            } = content
            else {
                return Err(agent_context_db_core::ContextError::Unsupported(format!(
                    "trajectory {uri} is not text"
                )));
            };
            let text = if !dense.trim().is_empty() {
                dense
            } else if !full.trim().is_empty() {
                full
            } else {
                sparse
            };
            if text.trim().is_empty() {
                return Err(agent_context_db_core::ContextError::Storage(format!(
                    "trajectory {uri} is empty"
                )));
            }
            traj_texts.push(text);
        }

        let combined = traj_texts.join("\n---\n");

        let prompt = format!(
            r#"Analyze these related trajectories and induce a reusable experience.

Trajectories:
{combined}

Return a JSON object with:
- "situation": common scenario/context across trajectories
- "approach": generalizable approach that worked
- "reflect": what was learned, what could be improved
"#
        );

        let opts = LlmOpts {
            max_tokens: Some(768),
            temperature: Some(0.2),
            ..Default::default()
        };

        let response = self.llm.complete(&prompt, &opts).await.map_err(|e| {
            agent_context_db_core::ContextError::Storage(format!("llm experience: {e}"))
        })?;

        #[derive(serde::Deserialize)]
        struct RawExperience {
            situation: String,
            approach: String,
            reflect: String,
        }

        let raw: RawExperience = serde_json::from_str(&response).map_err(|error| {
            agent_context_db_core::ContextError::Storage(format!(
                "llm experience returned invalid JSON: {error}"
            ))
        })?;
        if raw.situation.trim().is_empty()
            || raw.approach.trim().is_empty()
            || raw.reflect.trim().is_empty()
        {
            return Err(agent_context_db_core::ContextError::Storage(
                "llm experience returned empty required fields".into(),
            ));
        }

        let first = trajectories.first().ok_or_else(|| {
            agent_context_db_core::ContextError::Storage(
                "cannot induce experience from an empty trajectory list".into(),
            )
        })?;
        let parent = first.parent().ok_or_else(|| {
            agent_context_db_core::ContextError::InvalidUri(format!(
                "trajectory URI has no parent: {first}"
            ))
        })?;
        let exp_uri = parent.join("experience.json");

        Ok(Experience {
            uri: exp_uri,
            situation: raw.situation,
            approach: raw.approach,
            reflect: raw.reflect,
            related_trajectories: trajectories,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::{ContentPayload, Page, PageRequest};

    /// Mock FsOps that returns canned content.
    struct MockFs(String);
    #[async_trait]
    impl FsOps for MockFs {
        async fn ls(
            &self,
            _: &ContextUri,
            _: PageRequest,
        ) -> Result<Page<agent_context_db_core::DirEntry>> {
            Ok(Page::new(vec![], None))
        }
        async fn find(
            &self,
            _: &agent_context_db_core::FindPattern,
            _: PageRequest,
        ) -> Result<Page<ContextUri>> {
            Ok(Page::new(vec![], None))
        }
        async fn grep(
            &self,
            _: &str,
            _: &ContextUri,
        ) -> Result<Vec<agent_context_db_core::GrepHit>> {
            Ok(vec![])
        }
        async fn tree(
            &self,
            r: &ContextUri,
            _: usize,
            _: PageRequest,
        ) -> Result<Page<agent_context_db_core::TreeNode>> {
            Ok(Page::new(
                vec![agent_context_db_core::TreeNode {
                    uri: r.clone(),
                    is_dir: true,
                    children: vec![],
                }],
                None,
            ))
        }
        async fn read(&self, _: &ContextUri, _: ContentLevel) -> Result<ContentPayload> {
            Ok(ContentPayload::Text {
                sparse: self.0.clone(),
                dense: self.0.clone(),
                full: self.0.clone(),
            })
        }
    }

    /// Mock LlmClient that returns canned JSON.
    struct MockLlm;
    #[async_trait]
    impl LlmClient for MockLlm {
        async fn complete(
            &self,
            _: &str,
            _: &LlmOpts,
        ) -> std::result::Result<String, agent_context_db_core::LlmError> {
            Ok(r#"{"did_what": "deployed app", "how": "used docker compose", "result": "deployment succeeded"}"#.into())
        }
        async fn embed(
            &self,
            _: &str,
        ) -> std::result::Result<
            agent_context_db_core::EmbeddingVector,
            agent_context_db_core::LlmError,
        > {
            Ok(agent_context_db_core::EmbeddingVector::new(
                vec![1.0],
                "test",
                1,
            ))
        }
        async fn complete_json(
            &self,
            _: &str,
            _: &agent_context_db_core::JsonSchema,
            _: &LlmOpts,
        ) -> std::result::Result<String, agent_context_db_core::LlmError> {
            Ok(r#"{"did_what": "deployed app", "how": "used docker compose", "result": "deployment succeeded"}"#.into())
        }
    }

    #[tokio::test]
    async fn extract_trajectory_from_archive() {
        let llm = Arc::new(MockLlm);
        let fs = Arc::new(MockFs(
            "user: deploy the app\nassistant: running docker compose...".into(),
        ));
        let extractor = TrajectoryExtractorImpl::new(llm, fs);

        let archive = ContextUri::parse("uwu://t1/sessions/s1/archive/0/messages.jsonl").unwrap();
        let traj = extractor.extract_trajectory(&archive).await.unwrap();

        assert_eq!(traj.did_what, "deployed app");
        assert_eq!(traj.how, "used docker compose");
        assert_eq!(traj.result, "deployment succeeded");
    }

    #[tokio::test]
    async fn induce_experience_from_trajectories() {
        let llm = Arc::new(MockLlm);
        let fs = Arc::new(MockFs("trajectory content".into()));
        let extractor = TrajectoryExtractorImpl::new(llm, fs);

        let uris = vec![
            ContextUri::parse("uwu://t1/trajectories/t1/trajectory.json").unwrap(),
            ContextUri::parse("uwu://t1/trajectories/t2/trajectory.json").unwrap(),
        ];

        let exp = extractor.induce_experience(uris.clone()).await.unwrap();
        assert!(exp.related_trajectories.len() == 2);
        assert!(!exp.situation.is_empty());
    }
}
