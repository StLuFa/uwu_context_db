//! uwu_database 缓存适配器 — 桥接 `uwu_database::Cache` → `ReadCache`。
//!
//! 消除 ReadCache trait 与 uwu_database 内置 Cache 之间的重复，
//! 使 PgContextStore 和 ContextRetriever 可直接使用 uwu_database 的 Memory/Redis 缓存。

use agent_context_db_core::read_cache::ReadCache;
use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri, Result};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

/// 适配 `uwu_database::Cache` 实现 context-db 的 `ReadCache` trait。
pub struct UwuCacheAdapter {
    inner: Arc<dyn uwu_database::Cache>,
    prefix: String,
    default_ttl: Duration,
}

impl UwuCacheAdapter {
    pub fn new(cache: Arc<dyn uwu_database::Cache>, prefix: &str, ttl: Duration) -> Self {
        Self { inner: cache, prefix: prefix.to_string(), default_ttl: ttl }
    }

    fn key(&self, uri: &ContextUri, level: ContentLevel) -> String {
        format!("{}:ctx:{}:{}", self.prefix, uri.as_str(), level.as_str())
    }
}

#[async_trait]
impl ReadCache for UwuCacheAdapter {
    async fn get(&self, uri: &ContextUri, level: ContentLevel) -> Option<ContentPayload> {
        let key = self.key(uri, level);
        let data = self.inner.get(&key).await.ok()??;
        serde_json::from_slice(&data).ok()
    }

    async fn put(
        &self,
        uri: &ContextUri,
        level: ContentLevel,
        payload: ContentPayload,
        ttl: Duration,
    ) {
        let key = self.key(uri, level);
        if let Ok(data) = serde_json::to_vec(&payload) {
            self.inner.set(&key, &data, Some(ttl.as_secs())).await.ok();
        }
    }

    async fn invalidate(&self, uri: &ContextUri) {
        for level in [ContentLevel::L0, ContentLevel::L1, ContentLevel::L2] {
            let key = self.key(uri, level);
            self.inner.del(&key).await.ok();
        }
    }
}
