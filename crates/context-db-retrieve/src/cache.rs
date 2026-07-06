//! 检索缓存层 — 接入 ReadCache trait，含穿透/雪崩防护。
//!
//! 穿透/雪崩防护主体已下沉到 `ReadCache` 实现（`MemoryReadCache` / `UwuCacheAdapter` /
//! `RedisReadCache`）。本 wrapper 只提供便捷 API 与默认 TTL 语义。

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri, ReadCache};
use std::sync::Arc;
use std::time::Duration;

/// 带防护的检索缓存包装器。
pub struct RetrievalCache {
    inner: Arc<dyn ReadCache>,
    default_ttl: Duration,
}

/// 缓存查询结果，区分未命中、命中有值、命中空值。
#[derive(Debug, Clone)]
pub enum CacheLookup {
    /// 未命中，需要回源。
    Miss,
    /// 命中并返回内容。
    Hit(ContentPayload),
    /// 命中负缓存 —— 已知不存在，直接放弃回源。
    KnownMissing,
}

impl RetrievalCache {
    pub fn new(inner: Arc<dyn ReadCache>) -> Self {
        Self {
            inner,
            default_ttl: Duration::from_secs(300),
        }
    }

    /// 读取缓存 —— 语义化返回。
    pub async fn get(&self, uri: &ContextUri, level: ContentLevel) -> CacheLookup {
        match self.inner.get(uri, level).await {
            None => CacheLookup::Miss,
            Some(None) => CacheLookup::KnownMissing,
            Some(Some(payload)) => CacheLookup::Hit(payload),
        }
    }

    /// 写入正缓存 —— TTL 抖动由底层 ReadCache 处理。
    pub async fn put(&self, uri: &ContextUri, level: ContentLevel, payload: ContentPayload) {
        self.inner.put(uri, level, payload, self.default_ttl).await;
    }

    /// 写入负缓存 —— 回源发现 URI 不存在时调用（穿透防护）。
    pub async fn put_missing(&self, uri: &ContextUri, level: ContentLevel) {
        self.inner.put_negative(uri, level).await;
    }

    /// 缓存失效。
    pub async fn invalidate(&self, uri: &ContextUri) {
        self.inner.invalidate(uri).await;
    }
}
