//! 检索缓存层 — 接入 ReadCache trait，含穿透/雪崩防护。

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri, ReadCache};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// 带防护的检索缓存包装器。
pub struct RetrievalCache {
    inner: Arc<dyn ReadCache>,
    /// 默认 TTL。
    default_ttl: Duration,
    /// 空结果 TTL（防穿透）。
    empty_ttl: Duration,
}

impl RetrievalCache {
    pub fn new(inner: Arc<dyn ReadCache>) -> Self {
        Self {
            inner,
            default_ttl: Duration::from_secs(300),
            empty_ttl: Duration::from_secs(30),
        }
    }

    /// 读取缓存（带 TTL 检查）。
    pub async fn get(
        &self,
        uri: &ContextUri,
        level: ContentLevel,
    ) -> Option<ContentPayload> {
        self.inner.get(uri, level).await
    }

    /// 写入缓存（带随机抖动防止雪崩）。
    pub async fn put(
        &self,
        uri: &ContextUri,
        level: ContentLevel,
        payload: ContentPayload,
    ) {
        let ttl = self.jittered_ttl(self.default_ttl);
        self.inner.put(uri, level, payload, ttl).await;
    }

    /// 写入空结果（短 TTL 防穿透）。
    pub async fn put_empty(
        &self,
        uri: &ContextUri,
        level: ContentLevel,
    ) {
        let ttl = self.jittered_ttl(self.empty_ttl);
        let empty = ContentPayload::Text {
            sparse: String::new(),
            dense: String::new(),
            full: String::new(),
        };
        self.inner.put(uri, level, empty, ttl).await;
    }

    /// 缓存失效。
    pub async fn invalidate(&self, uri: &ContextUri) {
        self.inner.invalidate(uri).await;
    }

    /// TTL 加随机抖动（±10%），防止缓存雪崩。
    fn jittered_ttl(&self, base: Duration) -> Duration {
        // 基于当前时间的简单抖动（不需要 rand crate）
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;
        let jitter = (base.as_millis() as u64 * (nanos % 20)) / 100;
        base + Duration::from_millis(jitter)
    }
}
