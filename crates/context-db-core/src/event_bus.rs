//! 事件总线 trait — 跨进程事件广播（可选 Redis 后端）。

use crate::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// 事件总线端口 — 替代 ContextPubSub。
#[async_trait]
pub trait EventBus: Send + Sync {
    async fn publish(&self, topic: &str, payload: &[u8]) -> Result<()>;
    async fn subscribe(&self, topic: &str) -> Result<Box<dyn EventStream>>;
}

/// 事件流。
#[async_trait]
pub trait EventStream: Send + Sync {
    async fn next(&mut self) -> Option<Vec<u8>>;
}

/// 内存实现（默认，单机）。
pub struct MemoryEventBus {
    subscribers: parking_lot::RwLock<
        std::collections::HashMap<String, Vec<tokio::sync::mpsc::UnboundedSender<Vec<u8>>>>,
    >,
}

impl MemoryEventBus {
    pub fn new() -> Self {
        Self { subscribers: parking_lot::RwLock::new(std::collections::HashMap::new()) }
    }
}

#[async_trait]
impl EventBus for MemoryEventBus {
    async fn publish(&self, topic: &str, payload: &[u8]) -> Result<()> {
        if let Some(senders) = self.subscribers.read().get(topic) {
            for sender in senders {
                let _ = sender.send(payload.to_vec());
            }
        }
        Ok(())
    }

    async fn subscribe(&self, topic: &str) -> Result<Box<dyn EventStream>> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        self.subscribers.write().entry(topic.to_string()).or_default().push(tx);
        Ok(Box::new(MemoryEventStream { rx: Some(rx) }))
    }
}

struct MemoryEventStream {
    rx: Option<tokio::sync::mpsc::UnboundedReceiver<Vec<u8>>>,
}

#[async_trait]
impl EventStream for MemoryEventStream {
    async fn next(&mut self) -> Option<Vec<u8>> {
        self.rx.as_mut()?.recv().await
    }
}
