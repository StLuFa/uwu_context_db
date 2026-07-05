//! Redis Stream 语义队列 — 持久化任务队列 + 消费组 + 死信处理。
//!
//! 编译条件：`cargo build --features redis-backend`

use crate::{SemanticQueue, SemanticTask, TaskId, TaskOutcome};
use agent_context_db_core::Result;
use async_trait::async_trait;
use uuid::Uuid;

/// Redis Stream 任务队列。
pub struct RedisSemanticQueue {
    client: redis::Client,
    stream_key: String,
    consumer_group: String,
    consumer_id: String,
}

/// Lua 脚本：原子 XACK（确认完成）。
const XACK_LUA: &str = r#"
local stream = KEYS[1]
local group = ARGV[1]
local id = ARGV[2]
return redis.call('XACK', stream, group, id)
"#;

/// Lua 脚本：将失败任务移到死信 stream。
const DEAD_LETTER_LUA: &str = r#"
local src = KEYS[1]
local dst = KEYS[2]
local id = ARGV[1]
local entry = redis.call('XRANGE', src, id, id)
if #entry > 0 then
    redis.call('XADD', dst, '*', unpack(entry[1][2]))
    redis.call('XDEL', src, id)
    return 1
end
return 0
"#;

impl RedisSemanticQueue {
    pub fn connect(url: &str, key_prefix: &str) -> Result<Self> {
        let client = redis::Client::open(url)
            .map_err(|e| agent_context_db_core::ContextError::Storage(format!("redis connect: {e}")))?;
        Ok(Self {
            client,
            stream_key: format!("{}:tasks", key_prefix),
            consumer_group: format!("{}:workers", key_prefix),
            consumer_id: Uuid::new_v4().to_string(),
        })
    }

    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| agent_context_db_core::ContextError::Storage(format!("redis conn: {e}")))
    }

    /// 确保消费组存在（幂等创建）。
    async fn ensure_group(&self) -> Result<()> {
        let mut conn = self.conn().await?;
        let result: redis::RedisResult<()> = redis::cmd("XGROUP")
            .arg("CREATE")
            .arg(&self.stream_key)
            .arg(&self.consumer_group)
            .arg("0")
            .arg("MKSTREAM") // 自动创建 stream
            .exec_async(&mut conn)
            .await;
        // BUSYGROUP = group already exists = success
        match result {
            Ok(_) => Ok(()),
            Err(e) if e.to_string().contains("BUSYGROUP") => Ok(()),
            Err(e) => Err(agent_context_db_core::ContextError::Storage(format!("xgroup create: {e}"))),
        }
    }
}

#[async_trait]
impl SemanticQueue for RedisSemanticQueue {
    async fn enqueue(&self, task: SemanticTask) -> Result<TaskId> {
        self.ensure_group().await?;
        let id = TaskId::new();
        let payload = serde_json::to_vec(&task)
            .map_err(|e| agent_context_db_core::ContextError::Serialization(e.to_string()))?;

        let mut conn = self.conn().await?;
        redis::cmd("XADD")
            .arg(&self.stream_key)
            .arg("*") // auto-generate entry ID
            .arg("task_id")
            .arg(id.0.to_string())
            .arg("payload")
            .arg(&payload)
            .exec_async(&mut conn)
            .await
            .map_err(|e| agent_context_db_core::ContextError::Storage(format!("xadd: {e}")))?;

        Ok(id)
    }

    async fn dequeue(&self) -> Result<Option<(TaskId, SemanticTask)>> {
        self.ensure_group().await?;

        let mut conn = self.conn().await?;
        // XREADGROUP: 阻塞读取，消费组保证每条任务只被一个 worker 消费
        let entries: Vec<(String, Vec<(String, Vec<(String, Vec<u8>)>)>)> = redis::cmd("XREADGROUP")
            .arg("GROUP")
            .arg(&self.consumer_group)
            .arg(&self.consumer_id)
            .arg("COUNT")
            .arg(1)
            .arg("BLOCK")
            .arg(5000) // 5s 超时
            .arg("STREAMS")
            .arg(&self.stream_key)
            .arg(">") // 只读未处理的新消息
            .query_async(&mut conn)
            .await
            .map_err(|e| agent_context_db_core::ContextError::Storage(format!("xreadgroup: {e}")))?;

        if entries.is_empty() || entries[0].1.is_empty() {
            return Ok(None);
        }

        // 解析第一条消息
        let fields = &entries[0].1[0].1;
        let mut task_id: Option<TaskId> = None;
        let mut payload: Option<SemanticTask> = None;

        for (key, value) in fields {
            let val = String::from_utf8_lossy(value).to_string();
            match key.as_str() {
                "task_id" => {
                    if let Ok(u) = Uuid::parse_str(&val) {
                        task_id = Some(TaskId(u));
                    }
                }
                "payload" => {
                    payload = serde_json::from_slice(value).ok();
                }
                _ => {}
            }
        }

        match (task_id, payload) {
            (Some(tid), Some(task)) => Ok(Some((tid, task))),
            _ => Ok(None),
        }
    }

    async fn complete(&self, id: TaskId, outcome: TaskOutcome) -> Result<()> {
        let mut conn = self.conn().await?;

        match outcome {
            TaskOutcome::Success => {
                // XACK 确认完成
                redis::cmd("XACK")
                    .arg(&self.stream_key)
                    .arg(&self.consumer_group)
                    .arg(id.0.to_string())
                    .exec_async(&mut conn)
                    .await
                    .map_err(|e| agent_context_db_core::ContextError::Storage(format!("xack: {e}")))?;
            }
            TaskOutcome::Failure(_) | TaskOutcome::PartialFailure(_) => {
                // 移到死信 stream 进行人工排查
                let dead_stream = format!("{}:dead", self.stream_key);
                redis::cmd("EVAL")
                    .arg(DEAD_LETTER_LUA)
                    .arg(2)
                    .arg(&self.stream_key)
                    .arg(&dead_stream)
                    .arg(id.0.to_string())
                    .exec_async(&mut conn)
                    .await
                    .map_err(|e| agent_context_db_core::ContextError::Storage(format!("dead letter: {e}")))?;
            }
        }
        Ok(())
    }
}
