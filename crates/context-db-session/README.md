# agent-context-db-session

两阶段 commit 会话压缩。

## 流程

```
Phase1 (同步): 归档消息 → 写 messages.jsonl → 返回 task_id
Phase2 (异步): MemoryExtractor::extract → deduplicate → SemanticProcessor::generate
               → 写入 ContextStore → memory_diff.json → .done 标记
```

## 核心类型

```rust
pub trait SessionCompressor: Send + Sync {
    async fn commit_phase1(&self, session: &SessionHandle) -> Result<CommitTaskId>;
    async fn commit_phase2(&self, task_id: CommitTaskId) -> Result<DoneMarker>;
    async fn poll_task(&self, task_id: CommitTaskId) -> Result<TaskStatus>;
}

pub struct SessionHandle {
    pub session_id: Uuid, pub user_id: String, pub agent_id: String,
    pub messages: Vec<SessionMessage>, pub compression_index: u64,
    pub archive_dir: ContextUri,
}
```

## 实现

`SessionCompressorImpl` — Phase1 归档到 ContentRepo，Phase2 标记完成。语义处理由上层编排注入。
