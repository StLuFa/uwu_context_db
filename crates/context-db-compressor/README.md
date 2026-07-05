# agent-context-db-compressor

tokio mpsc 异步语义处理队列。

## 核心类型

```rust
pub trait SemanticQueue: Send + Sync {
    async fn enqueue(&self, task: SemanticTask) -> Result<TaskId>;
    async fn dequeue(&self) -> Result<Option<(TaskId, SemanticTask)>>;
    async fn complete(&self, id: TaskId, outcome: TaskOutcome) -> Result<()>;
}

pub enum SemanticTask {
    GenerateAbstract(ContextUri), GenerateOverview(ContextUri),
    AggregateUpward(ContextUri), ExtractMemories { ... },
    DeduplicateMemories(Vec<MemoryCandidate>), ExtractTrajectory(ContextUri),
    InduceExperience(Vec<ContextUri>), MultimodalToText(ContextUri),
}
```

## 使用

```rust
let queue = Arc::new(TokioSemanticQueue::new());
let task_id = queue.enqueue(SemanticTask::GenerateAbstract(uri)).await?;
let worker = queue.spawn_worker(|id, task| async move { /* 处理 */ TaskOutcome::Success });
```

## 依赖

`context-db-core` / `context-db-session` / `context-db-parse` / `tokio` / `parking_lot`。
