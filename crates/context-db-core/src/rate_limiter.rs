//! 限流器 trait — LLM API 配额控制（Semaphore 或 Redis 令牌桶）。

use crate::Result;
use async_trait::async_trait;
use std::sync::Arc;

/// 限流器端口。
#[async_trait]
pub trait RateLimiter: Send + Sync {
    async fn acquire(&self) -> Result<()>;
    async fn try_acquire(&self) -> bool;
}

/// 内存实现（tokio Semaphore）。
pub struct MemoryRateLimiter {
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl MemoryRateLimiter {
    pub fn new(max_concurrency: usize) -> Self {
        Self { semaphore: Arc::new(tokio::sync::Semaphore::new(max_concurrency.max(1))) }
    }
}

#[async_trait]
impl RateLimiter for MemoryRateLimiter {
    async fn acquire(&self) -> Result<()> {
        let _permit = self.semaphore.acquire().await.map_err(|e| {
            crate::ContextError::Storage(format!("rate limiter closed: {e}"))
        })?;
        Ok(())
    }

    async fn try_acquire(&self) -> bool {
        self.semaphore.try_acquire().is_ok()
    }
}
