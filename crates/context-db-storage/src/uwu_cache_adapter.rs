//! uwu_database 缓存适配器 — 桥接 `uwu_database::Cache` → `ReadCache`。
//!
//! 消除 ReadCache trait 与 uwu_database 内置 Cache 之间的重复，
//! 使 PgContextStore 和 ContextRetriever 可直接使用 uwu_database 的 Memory/Redis 缓存。
//!
//! ## 穿透/雪崩防护
//!
//! - **负缓存**：`put_negative` 写入固定 marker `b"\0NEG\0"`，`get` 命中该 marker 时
//!   返回 `Some(None)`（区别于未命中 `None`）。
//! - **TTL 抖动**：调用方（如 `MemoryReadCache`）已在传入 TTL 前抖动；本层直接透传。

use agent_context_db_core::read_cache::ReadCache;
use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

/// 负缓存标记 —— 存入 `uwu_database::Cache` 的哨兵值，表示"已知缺失"。
const NEG_MARKER: &[u8] = b"\0NEG\0";

/// 适配 `uwu_database::Cache` 实现 context-db 的 `ReadCache` trait。
pub struct UwuCacheAdapter {
    inner: Arc<dyn uwu_database::Cache>,
    prefix: String,
    default_ttl: Duration,
    /// 负缓存 TTL（穿透防护窗口）。
    negative_ttl: Duration,
}

impl UwuCacheAdapter {
    pub fn new(cache: Arc<dyn uwu_database::Cache>, prefix: &str, ttl: Duration) -> Self {
        Self {
            inner: cache,
            prefix: prefix.to_string(),
            default_ttl: ttl,
            negative_ttl: Duration::from_secs(30),
        }
    }

    /// 覆盖负缓存 TTL。
    pub fn with_negative_ttl(mut self, ttl: Duration) -> Self {
        self.negative_ttl = ttl;
        self
    }

    fn key(&self, uri: &ContextUri, level: ContentLevel) -> String {
        format!("{}:ctx:{}:{}", self.prefix, uri.as_str(), level.as_str())
    }
}

#[async_trait]
impl ReadCache for UwuCacheAdapter {
    async fn get(&self, uri: &ContextUri, level: ContentLevel) -> Option<Option<ContentPayload>> {
        let key = self.key(uri, level);
        let data = self.inner.get(&key).await.ok()??;
        if data.as_slice() == NEG_MARKER {
            return Some(None); // 负缓存命中
        }
        Some(serde_json::from_slice(&data).ok())
    }

    async fn put(
        &self,
        uri: &ContextUri,
        level: ContentLevel,
        payload: ContentPayload,
        ttl: Duration,
    ) {
        let effective = if ttl.is_zero() { self.default_ttl } else { ttl };
        let key = self.key(uri, level);
        if let Ok(data) = serde_json::to_vec(&payload) {
            self.inner.set(&key, &data, Some(effective)).await.ok();
        }
    }

    async fn put_negative(&self, uri: &ContextUri, level: ContentLevel) {
        let key = self.key(uri, level);
        self.inner
            .set(&key, NEG_MARKER, Some(self.negative_ttl))
            .await
            .ok();
    }

    async fn invalidate(&self, uri: &ContextUri) {
        for level in [ContentLevel::L0, ContentLevel::L1, ContentLevel::L2] {
            let key = self.key(uri, level);
            self.inner.del(&key).await.ok();
        }
    }
}
