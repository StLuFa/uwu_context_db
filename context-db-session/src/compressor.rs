//! `SessionCompressorImpl`：两阶段 commit 会话压缩器实现。
//!
//! - Phase1（同步）：归档消息 → 写入 FS → 返回 task_id
//! - Phase2（异步）：语义处理（去重 → L0/L1 生成 → 写 .done 标记）

use agent_context_db_core::{
    ContentRepo, ContentType, ContextEntry, ContextMeta, ContextUri, MemoryClass, MvccVersion,
    Result, TenantId,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    CommitTaskId, DoneMarker, MemoryChange, MemoryDiff, SessionCompressor,
    SessionHandle, TaskStatus,
};

/// 会话压缩器实现。
///
/// Phase1 将消息写入 FS 归档目录；Phase2 执行完整的语义管线：
/// 记忆提取 → 去重 → L0/L1 生成 → 写 memory_diff → 标记完成。
pub struct SessionCompressorImpl {
    store: Arc<dyn ContentRepo>,
    /// Phase1 → Phase2 间暂存的会话句柄。
    pending: Mutex<HashMap<CommitTaskId, PendingSession>>,
}

pub struct PendingSession {
    pub handle: SessionHandle,
    pub archive_uri: ContextUri,
}

impl SessionCompressorImpl {
    pub fn new(store: Arc<dyn ContentRepo>) -> Self {
        Self {
            store,
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// 获取暂存的会话句柄（供上层编排 Phase2 语义处理）。
    pub fn take_pending(&self, task_id: &CommitTaskId) -> Option<PendingSession> {
        self.pending.lock().remove(task_id)
    }

    /// 归档 URI：`{archive_dir}/{compression_index}/messages.jsonl`
    pub fn archive_file_uri(session: &SessionHandle) -> ContextUri {
        session
            .archive_dir
            .join(&session.compression_index.to_string())
            .join("messages.jsonl")
    }
}

#[async_trait]
impl SessionCompressor for SessionCompressorImpl {
    async fn commit_phase1(&self, session: &SessionHandle) -> Result<CommitTaskId> {
        let task_id = CommitTaskId::new();
        let archive_uri = Self::archive_file_uri(session);

        // 将消息序列化为 JSONL
        let mut jsonl = String::new();
        for msg in &session.messages {
            let line = serde_json::to_string(msg).unwrap_or_default();
            jsonl.push_str(&line);
            jsonl.push('\n');
        }

        // 写入归档条目
        let entry = ContextEntry {
            uri: archive_uri.clone(),
            tenant: TenantId(uuid::Uuid::nil()),
            l0_abstract: format!(
                "session {} compression #{} with {} messages",
                session.session_id,
                session.compression_index,
                session.messages.len()
            ),
            l1_overview: Some(jsonl.clone()),
            l2_detail_uri: None,
            content_type: ContentType::Text,
            metadata: ContextMeta::default(),
            mvcc_version: MvccVersion(0),
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        self.store.write(entry).await?;

        // 暂存会话句柄供 Phase2 使用
        self.pending.lock().insert(
            task_id,
            PendingSession {
                handle: session.clone(),
                archive_uri,
            },
        );

        Ok(task_id)
    }

    async fn commit_phase2(&self, task_id: CommitTaskId) -> Result<DoneMarker> {
        // Phase2：上层负责在 Phase1 和 Phase2 之间调用语义管线
        // (MemoryExtractor + SemanticProcessor)，完成后再调用此方法标记完成。
        let pending = self
            .take_pending(&task_id)
            .ok_or_else(|| {
                agent_context_db_core::ContextError::NotFound(format!(
                    "no pending session for task {task_id:?}"
                ))
            })?;

        // 写 memory_diff（由上层预先生成）
        let memory_diff_uri = pending
            .handle
            .archive_dir
            .join(&pending.handle.compression_index.to_string())
            .join("memory_diff.json");

        // 写 .done 标记
        let done_uri = pending
            .handle
            .archive_dir
            .join(&pending.handle.compression_index.to_string())
            .join(".done");

        let done_marker = DoneMarker {
            task_id,
            finished_at: chrono::Utc::now(),
            abstract_uri: pending.archive_uri.clone(),
            overview_uri: pending.archive_uri.clone(),
            memory_diff_uri: Some(memory_diff_uri),
        };

        let done_entry = ContextEntry {
            uri: done_uri,
            tenant: TenantId(uuid::Uuid::nil()),
            l0_abstract: format!("done: {done_marker:?}"),
            l1_overview: None,
            l2_detail_uri: None,
            content_type: ContentType::Text,
            metadata: ContextMeta::default(),
            mvcc_version: MvccVersion(0),
            created_at: done_marker.finished_at,
            updated_at: done_marker.finished_at,
        };

        self.store.write(done_entry).await?;

        Ok(done_marker)
    }

    async fn poll_task(&self, task_id: CommitTaskId) -> Result<TaskStatus> {
        match self.pending.lock().get(&task_id) {
            Some(_) => Ok(TaskStatus::Processing),
            None => Ok(TaskStatus::Failed("task not found or already completed".into())),
        }
    }
}

// ===========================================================================
// 高层编排函数（组合 SessionCompressor + 语义管线）
// ===========================================================================

/// 完整的两阶段 commit 编排：Phase1 归档 → Phase2 语义处理 → 标记完成。
///
/// `extractor` 和 `semantic` 由上层注入。
/// 管线流程：
///   1. Phase1: 归档消息到 FS
///   2. 提取记忆候选项
///   3. 去重（合并/跳过/新建）
///   4. 生成 L0 摘要（每条新/更新的记忆）
///   5. 生成 L1 概览（聚合到父目录）
///   6. 写 memory_diff.json
///   7. Phase2: 标记完成
pub async fn run_full_compression(
    compressor: &SessionCompressorImpl,
    extractor: &dyn MemoryExtractorShim,
    semantic: &dyn SemanticProcessorShim,
    session: &SessionHandle,
) -> Result<DoneMarker> {
    // Phase1: 归档
    let task_id = compressor.commit_phase1(session).await?;

    let pending = compressor
        .take_pending(&task_id)
        .ok_or_else(|| {
            agent_context_db_core::ContextError::NotFound("task disappeared".into())
        })?;

    // Phase2: 语义处理管线

    // 1. 提取记忆候选项
    let candidates = extractor.extract(&pending.archive_uri).await?;

    // 2. 去重
    let decisions = extractor.deduplicate(candidates).await?;

    // 3. 分流：新建 / 合并 / 跳过
    let mut memory_diff = MemoryDiff::default();
    for dec in &decisions {
        match dec.action {
            ShimAction::Create => {
                // 为新记忆生成 L0 摘要
                let _abstract_ = semantic
                    .generate_abstract(&dec.target_uri)
                    .await?;
                memory_diff.adds.push(MemoryChange {
                    uri: dec.target_uri.clone(),
                    class: dec.class,
                    before: None,
                    after: Some(serde_json::json!({"abstract": _abstract_})),
                    reason: dec.reason.clone(),
                });
            }
            ShimAction::Merge => {
                memory_diff.updates.push(MemoryChange {
                    uri: dec.target_uri.clone(),
                    class: dec.class,
                    before: None,
                    after: Some(serde_json::json!({"merged": true})),
                    reason: dec.reason.clone(),
                });
            }
            ShimAction::Skip => {
                // 不需要操作
            }
        }
    }

    // 4. 聚合 L1 概览（对归档目录的父目录）
    if let Some(parent) = pending.archive_uri.parent() {
        let _ = semantic.aggregate_upward(&parent).await;
    }

    // 5. 写 memory_diff 到归档
    let diff_uri = pending
        .handle
        .archive_dir
        .join(&pending.handle.compression_index.to_string())
        .join("memory_diff.json");
    let diff_json = serde_json::to_string(&memory_diff)
        .map_err(|e| agent_context_db_core::ContextError::Serialization(
            format!("memory_diff serialize: {e}")
        ))?;

    let diff_entry = ContextEntry {
        uri: diff_uri,
        tenant: TenantId(uuid::Uuid::nil()),
        l0_abstract: format!(
            "memory diff: {} adds, {} updates, {} deletes",
            memory_diff.adds.len(),
            memory_diff.updates.len(),
            memory_diff.deletes.len()
        ),
        l1_overview: Some(diff_json),
        l2_detail_uri: None,
        content_type: ContentType::Text,
        metadata: ContextMeta::default(),
        mvcc_version: MvccVersion(0),
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };
    compressor.store.write(diff_entry).await?;

    // 6. Phase2: 标记完成
    let done = compressor.commit_phase2(task_id).await?;
    Ok(done)
}

// ===========================================================================
// Phase2 语义处理的 trait shim（避免 session 直接依赖 parse crate）
// ===========================================================================

/// 去重决策（shim 版本，不含 parse crate 的 MemoryCandidate）。
#[derive(Debug, Clone)]
pub struct ShimCandidateAction {
    pub target_uri: ContextUri,
    pub class: MemoryClass,
    pub action: ShimAction,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShimAction {
    Skip,
    Create,
    Merge,
}

/// Phase2 记忆提取 trait shim。
///
/// 避免 session crate 直接依赖 parse crate。
/// 实际实现在 context-db-parse crate 中，由 composition root 注入。
#[async_trait]
pub trait MemoryExtractorShim: Send + Sync {
    /// 从归档中提取记忆候选项（返回候选内容列表）。
    async fn extract(&self, archive: &ContextUri) -> Result<Vec<String>>;

    /// 对候选项集合去重，返回每个候选项的决策。
    async fn deduplicate(
        &self,
        candidates: Vec<String>,
    ) -> Result<Vec<ShimCandidateAction>>;
}

/// Phase2 语义处理 trait shim。
///
/// 避免 session crate 直接依赖 parse crate。
#[async_trait]
pub trait SemanticProcessorShim: Send + Sync {
    /// 为 URI 生成 L0 摘要。
    async fn generate_abstract(&self, uri: &ContextUri) -> Result<String>;

    /// 自底向上聚合：为目录生成 L1 概览，返回生成的文本。
    async fn aggregate_upward(&self, root: &ContextUri) -> Result<String>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Role, SessionMessage};

    fn make_session() -> SessionHandle {
        SessionHandle {
            session_id: uuid::Uuid::new_v4(),
            user_id: "u1".into(),
            agent_id: "a1".into(),
            messages: vec![
                SessionMessage {
                    role: Role::User,
                    content: "hello".into(),
                    timestamp: chrono::Utc::now(),
                    metadata: serde_json::Value::Null,
                },
                SessionMessage {
                    role: Role::Assistant,
                    content: "hi there".into(),
                    timestamp: chrono::Utc::now(),
                    metadata: serde_json::Value::Null,
                },
            ],
            compression_index: 0,
            archive_dir: ContextUri::parse("uwu://t1/sessions/s1/archive").unwrap(),
        }
    }

    #[test]
    fn archive_uri_contains_index_and_filename() {
        let s = make_session();
        let uri = SessionCompressorImpl::archive_file_uri(&s);
        assert!(uri.to_string().contains("messages.jsonl"));
        assert!(uri.to_string().contains("/0/"));
    }

    #[test]
    fn shim_action_display() {
        assert_eq!(ShimAction::Skip as u8, 0);
        assert_eq!(ShimAction::Create as u8, 1);
        assert_eq!(ShimAction::Merge as u8, 2);
    }
}
