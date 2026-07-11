# agent-context-db-session

持久化、幂等、可重试并支持进程重启恢复的会话压缩状态机。

## 状态与流程

```text
enqueue: messages.jsonl + Pending task record
run/retry/recover: Processing(attempt)
  -> MemoryExtractor::extract
  -> MemoryExtractor::deduplicate
  -> SemanticProcessor::generate_abstract
  -> SemanticProcessor::aggregate_upward
  -> memory_diff.json
  -> .done
  -> Done
failure: Failed(message, attempt, failed_at, retryable)
```

任务 ID 由 `session_id + compression_index + archive_dir` 确定性生成。任务记录和输出均写入注入的 `SessionTaskStore`；重建 `SessionCompressorImpl` 后可通过 `poll` 查询，通过 `recover` 恢复 Pending、Processing 或 Failed 任务。已完成任务再次执行 `run` 会直接返回持久化的 `DoneMarker`，不会重复执行语义管线。

## API

```rust
let compressor = SessionCompressorImpl::new(store, extractor, semantic);
let task_id = compressor.enqueue(&session).await?;
let status = compressor.poll(&session, task_id).await?;
let done = compressor.run(&session).await?;
let done = compressor.retry(&session, task_id).await?;
let done = compressor.recover(&session).await?;
```

`MemoryExtractor` 和 `SemanticProcessor` 是正式编排依赖，构造 compressor 时必须提供，不存在只写完成标记而跳过语义处理的 API。
