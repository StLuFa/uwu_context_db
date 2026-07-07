//! `TokioSemanticQueue`：基于 tokio mpsc 的异步语义处理队列实现。
//!
//! 替代原 `agent-sidecar-consolidator` 的独立进程模式，内嵌为 in-process worker。

use agent_context_db_core::Result;
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;
use tokio::sync::mpsc::{self, UnboundedReceiver, UnboundedSender};

use crate::{SemanticQueue, SemanticTask, TaskId, TaskOutcome};

/// 基于 tokio `mpsc` 的语义处理队列。
///
/// - `enqueue`: 推入任务到无界通道
/// - `dequeue`: worker 取出任务（AsyncMutex 保护 receiver）
/// - `complete`: 记录完成结果
pub struct TokioSemanticQueue {
    tx: UnboundedSender<(TaskId, SemanticTask)>,
    rx: AsyncMutex<UnboundedReceiver<(TaskId, SemanticTask)>>,
    outcomes: Mutex<HashMap<TaskId, TaskOutcome>>,
}

impl TokioSemanticQueue {
    pub fn new() -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        Self {
            tx,
            rx: AsyncMutex::new(rx),
            outcomes: Mutex::new(HashMap::new()),
        }
    }

    /// 返回 sender 的克隆，供其他组件持有以写入任务。
    pub fn sender(&self) -> UnboundedSender<(TaskId, SemanticTask)> {
        self.tx.clone()
    }

    /// 启动一个 worker，持续 dequeue 并回调 handler。
    ///
    /// `handler` 接收 (TaskId, SemanticTask) 并返回 TaskOutcome。
    /// 返回的 `JoinHandle` 可被 abort 以停止 worker。
    pub fn spawn_worker<F, Fut>(self: &Arc<Self>, handler: F) -> tokio::task::JoinHandle<()>
    where
        F: Fn(TaskId, SemanticTask) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = TaskOutcome> + Send + 'static,
    {
        let this = self.clone();
        tokio::spawn(async move {
            loop {
                match this.dequeue().await {
                    Ok(Some((id, task))) => {
                        let outcome = handler(id, task).await;
                        let _ = this.complete(id, outcome).await;
                    }
                    Ok(None) => {
                        // 通道关闭
                        break;
                    }
                    Err(_) => break,
                }
            }
        })
    }
}

impl Default for TokioSemanticQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SemanticQueue for TokioSemanticQueue {
    async fn enqueue(&self, task: SemanticTask) -> Result<TaskId> {
        let id = TaskId::new();
        self.tx.send((id, task)).map_err(|e| {
            agent_context_db_core::ContextError::Storage(format!("semantic queue closed: {e}"))
        })?;
        Ok(id)
    }

    async fn dequeue(&self) -> Result<Option<(TaskId, SemanticTask)>> {
        let mut rx = self.rx.lock().await;
        match rx.recv().await {
            Some(item) => Ok(Some(item)),
            None => Ok(None),
        }
    }

    async fn complete(&self, id: TaskId, outcome: TaskOutcome) -> Result<()> {
        self.outcomes.lock().insert(id, outcome);
        Ok(())
    }
}

// ===========================================================================
// 便利函数：启动 worker 并处理 SemanticTask 的每个变体
// ===========================================================================

/// 为 `TokioSemanticQueue` 启动一个批量 worker，逐个调用 handler。
///
/// 典型用法：
/// ```ignore
/// let queue = Arc::new(TokioSemanticQueue::new());
/// let worker = TokioSemanticQueue::spawn_worker(&queue, |id, task| async move {
///     handle_semantic_task(task).await
/// });
/// ```
pub fn spawn_semantic_worker<F, Fut>(
    queue: &Arc<TokioSemanticQueue>,
    handler: F,
) -> tokio::task::JoinHandle<()>
where
    F: Fn(TaskId, SemanticTask) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = TaskOutcome> + Send + 'static,
{
    queue.spawn_worker(handler)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_dequeue_roundtrip() {
        let q = TokioSemanticQueue::new();
        let uri =
            agent_context_db_core::ContextUri::parse("uwu://t/agent/a/memories/cases/c1").unwrap();

        let id = q
            .enqueue(SemanticTask::GenerateAbstract(uri.clone()))
            .await
            .unwrap();
        let (got_id, task) = q.dequeue().await.unwrap().unwrap();

        assert_eq!(got_id, id);
        assert!(matches!(task, SemanticTask::GenerateAbstract(_)));
    }

    #[tokio::test]
    async fn complete_stores_outcome() {
        let q = TokioSemanticQueue::new();
        let uri = agent_context_db_core::ContextUri::parse("uwu://t/x").unwrap();

        let id = q
            .enqueue(SemanticTask::GenerateAbstract(uri))
            .await
            .unwrap();
        q.complete(id, TaskOutcome::Success).await.unwrap();

        let outcomes = q.outcomes.lock();
        assert!(matches!(outcomes.get(&id), Some(TaskOutcome::Success)));
    }

    #[tokio::test]
    async fn worker_processes_tasks() {
        let q = Arc::new(TokioSemanticQueue::new());
        let uri = agent_context_db_core::ContextUri::parse("uwu://t/x").unwrap();

        // 入队两个任务
        q.enqueue(SemanticTask::GenerateAbstract(uri.clone()))
            .await
            .unwrap();
        q.enqueue(SemanticTask::GenerateOverview(uri.clone()))
            .await
            .unwrap();

        // 启动 worker
        let worker = q.spawn_worker(|_id, task| async move {
            match task {
                SemanticTask::GenerateAbstract(_) => TaskOutcome::Success,
                SemanticTask::GenerateOverview(_) => TaskOutcome::Success,
                _ => TaskOutcome::Failure("unexpected".into()),
            }
        });

        // 等 worker 处理完
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        worker.abort();

        let outcomes = q.outcomes.lock();
        assert_eq!(outcomes.len(), 2, "worker should have processed both tasks");
        assert!(outcomes.values().all(|o| matches!(o, TaskOutcome::Success)));
    }
}
