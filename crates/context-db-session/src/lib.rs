//! # agent-context-db-session (L4 会话层)
//!
//! 两阶段 commit 的会话压缩：
//! - Phase1 同步：归档消息 + 清空当前窗口 + 返回 task_id
//! - Phase2 异步：生成 L0/L1 + 提取记忆 + 写 memory_diff.json
//!
//! ## 解耦约束
//!
//! - 仅依赖 core 的类型与 `ContentRepo` 端口（写归档）；不依赖 parse/compressor。
//! - `SessionCompressor` 是端口（零实现），编排由 composition root 装配。

pub mod compressor;

pub use compressor::{
    MemoryExtractorShim, SemanticProcessorShim, SessionCompressorImpl, ShimAction,
    ShimCandidateAction,
};

use agent_context_db_core::{ContextUri, MemoryClass, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 两阶段 commit 会话压缩器。
#[async_trait]
pub trait SessionCompressor: Send + Sync {
    /// Phase1 同步：归档消息 + 清空当前 + 返回 task_id。
    async fn commit_phase1(&self, session: &SessionHandle) -> Result<CommitTaskId>;
    /// Phase2 异步：生成 L0/L1 + 提取记忆 + 写 memory_diff.json。
    async fn commit_phase2(&self, task_id: CommitTaskId) -> Result<DoneMarker>;
    /// 查询异步任务状态。
    async fn poll_task(&self, task_id: CommitTaskId) -> Result<TaskStatus>;
}

#[derive(Debug, Clone)]
pub struct SessionHandle {
    pub session_id: Uuid,
    pub user_id: String,
    pub agent_id: String,
    pub messages: Vec<SessionMessage>,
    pub compression_index: u64,
    pub archive_dir: ContextUri,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub role: Role,
    pub content: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Role {
    User,
    Assistant,
    Tool,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommitTaskId(pub Uuid);

impl CommitTaskId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for CommitTaskId {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub enum TaskStatus {
    Pending,
    Processing,
    Done(DoneMarker),
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct DoneMarker {
    pub task_id: CommitTaskId,
    pub finished_at: chrono::DateTime<chrono::Utc>,
    pub abstract_uri: ContextUri,
    pub overview_uri: ContextUri,
    pub memory_diff_uri: Option<ContextUri>,
}

/// 记忆变更审计。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryDiff {
    pub adds: Vec<MemoryChange>,
    pub updates: Vec<MemoryChange>,
    pub deletes: Vec<MemoryChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChange {
    pub uri: ContextUri,
    pub class: MemoryClass,
    pub before: Option<serde_json::Value>,
    pub after: Option<serde_json::Value>,
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_diff_default_is_empty() {
        let d = MemoryDiff::default();
        assert!(d.adds.is_empty() && d.updates.is_empty() && d.deletes.is_empty());
    }

    #[test]
    fn task_id_is_unique() {
        assert_ne!(CommitTaskId::new(), CommitTaskId::new());
    }
}
