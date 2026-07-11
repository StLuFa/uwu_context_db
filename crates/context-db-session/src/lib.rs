//! 持久化、可恢复的会话压缩状态机。

pub mod compressor;

pub use compressor::SessionCompressorImpl;

use agent_context_db_core::{ContentType, ContextUri};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    pub timestamp: DateTime<Utc>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Pending,
    Processing {
        attempt: u32,
        started_at: DateTime<Utc>,
    },
    Done(DoneMarker),
    Failed(FailureMetadata),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FailureMetadata {
    pub message: String,
    pub attempt: u32,
    pub failed_at: DateTime<Utc>,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DoneMarker {
    pub task_id: CommitTaskId,
    pub finished_at: DateTime<Utc>,
    pub abstract_uri: ContextUri,
    pub overview_uri: ContextUri,
    pub memory_diff_uri: ContextUri,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryDiff {
    pub adds: Vec<MemoryChange>,
    pub updates: Vec<MemoryChange>,
    pub deletes: Vec<MemoryChange>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryChange {
    pub uri: ContextUri,
    pub content_type: ContentType,
    pub before: Option<serde_json::Value>,
    pub after: Option<serde_json::Value>,
    pub reason: String,
}
