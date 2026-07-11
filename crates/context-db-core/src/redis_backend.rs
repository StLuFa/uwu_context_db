//! Redis 后端实现 — 可选（feature gate `redis-backend`）。
//!
//! 提供 ReadCache、EmbeddingCache 的 Redis 实现（跨进程共享缓存）。
//! 事件总线已迁移到 `uwu_event_mesh` + `uwu_nats_bridge`。
//!
//! 编译条件：
//! ```bash
//! cargo build --features redis-backend
//! ```

use crate::embedding_cache::EmbeddingCache;
use crate::read_cache::ReadCache;
use crate::{ContentLevel, ContentPayload, ContextUri, Result};
use async_trait::async_trait;
use std::time::Duration;

// ===========================================================================
// Redis 客户端配置
// ===========================================================================

/// Redis 连接配置。
#[derive(Debug, Clone)]
pub struct RedisConfig {
    pub url: String,
    pub key_prefix: String,
    pub default_ttl_secs: u64,
    pub embedding_ttl_secs: u64,
    pub pool_size: usize,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".into(),
            key_prefix: "uwu".into(),
            default_ttl_secs: 300,
            embedding_ttl_secs: 86_400,
            pool_size: 10,
        }
    }
}

/// 辅助：构建带前缀的 Redis key。
fn prefixed(prefix: &str, key: &str) -> String {
    format!("{}:{}", prefix, key)
}

// ===========================================================================
// RedisReadCache — Redis 读取缓存
// ===========================================================================

/// Redis 读取缓存 — 跨进程共享 L0/L1 结果。
///
/// 穿透防护：`put_negative` 存入哨兵值 `\0NEG\0`；`get` 命中该值时返回 `Some(None)`。
pub struct RedisReadCache {
    client: redis::Client,
    prefix: String,
    default_ttl: Duration,
    negative_ttl: Duration,
}

/// 负缓存哨兵值。
const NEG_MARKER: &[u8] = b"\0NEG\0";

impl RedisReadCache {
    pub fn connect(config: &RedisConfig) -> Result<Self> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| crate::ContextError::Storage(format!("redis connect: {e}")))?;
        Ok(Self {
            client,
            prefix: config.key_prefix.clone(),
            default_ttl: Duration::from_secs(config.default_ttl_secs),
            negative_ttl: Duration::from_secs(30),
        })
    }

    pub fn with_negative_ttl(mut self, ttl: Duration) -> Self {
        self.negative_ttl = ttl;
        self
    }

    fn cache_key(&self, uri: &ContextUri, level: ContentLevel) -> String {
        prefixed(
            &self.prefix,
            &format!("cache:{}:{}", uri.as_str(), level.as_str()),
        )
    }

    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::ContextError::Storage(format!("redis conn: {e}")))
    }
}

#[async_trait]
impl ReadCache for RedisReadCache {
    async fn get(&self, uri: &ContextUri, level: ContentLevel) -> Option<Option<ContentPayload>> {
        let key = self.cache_key(uri, level);
        let mut conn = self.conn().await.ok()?;
        let data: Option<Vec<u8>> = redis::cmd("GET")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .ok()?;
        let bytes = data?;
        if bytes.as_slice() == NEG_MARKER {
            return Some(None);
        }
        Some(serde_json::from_slice(&bytes).ok())
    }

    async fn put(
        &self,
        uri: &ContextUri,
        level: ContentLevel,
        payload: ContentPayload,
        ttl: Duration,
    ) {
        let effective = if ttl.is_zero() { self.default_ttl } else { ttl };
        let key = self.cache_key(uri, level);
        if let Ok(data) = serde_json::to_vec(&payload) {
            if let Ok(mut conn) = self.conn().await {
                let ttl_secs = effective.as_secs().max(1);
                let _: () = redis::cmd("SETEX")
                    .arg(&key)
                    .arg(ttl_secs)
                    .arg(&data)
                    .query_async(&mut conn)
                    .await
                    .unwrap_or(());
            }
        }
    }

    async fn put_negative(&self, uri: &ContextUri, level: ContentLevel) {
        let key = self.cache_key(uri, level);
        if let Ok(mut conn) = self.conn().await {
            let ttl_secs = self.negative_ttl.as_secs().max(1);
            let _: () = redis::cmd("SETEX")
                .arg(&key)
                .arg(ttl_secs)
                .arg(NEG_MARKER)
                .query_async(&mut conn)
                .await
                .unwrap_or(());
        }
    }

    async fn invalidate(&self, uri: &ContextUri) {
        if let Ok(mut conn) = self.conn().await {
            // 删除所有 level 的缓存
            for level in [ContentLevel::L0, ContentLevel::L1, ContentLevel::L2] {
                let key = self.cache_key(uri, level);
                let _: () = redis::cmd("DEL")
                    .arg(&key)
                    .query_async(&mut conn)
                    .await
                    .unwrap_or(());
            }
        }
    }
}

// ===========================================================================
// RedisEmbeddingCache — Redis embedding 缓存
// ===========================================================================

/// Redis embedding 缓存 — 按 blake3(content) 跨进程共享向量结果。
pub struct RedisEmbeddingCache {
    client: redis::Client,
    prefix: String,
    default_ttl: Duration,
}

impl RedisEmbeddingCache {
    pub fn connect(config: &RedisConfig) -> Result<Self> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| crate::ContextError::Storage(format!("redis connect: {e}")))?;
        Ok(Self {
            client,
            prefix: config.key_prefix.clone(),
            default_ttl: Duration::from_secs(config.embedding_ttl_secs),
        })
    }

    fn cache_key(&self, content_hash: &str) -> String {
        prefixed(&self.prefix, &format!("embedding:{content_hash}"))
    }

    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::ContextError::Storage(format!("redis conn: {e}")))
    }
}

#[async_trait]
impl EmbeddingCache for RedisEmbeddingCache {
    async fn get(&self, content_hash: &str) -> Option<Vec<f32>> {
        let key = self.cache_key(content_hash);
        let mut conn = self.conn().await.ok()?;
        let data: Option<Vec<u8>> = redis::cmd("GET")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .ok()?;
        serde_json::from_slice(&data?).ok()
    }

    async fn put(&self, content_hash: &str, embedding: Vec<f32>, ttl: Duration) {
        let effective = if ttl.is_zero() { self.default_ttl } else { ttl };
        if let Ok(data) = serde_json::to_vec(&embedding) {
            if let Ok(mut conn) = self.conn().await {
                let key = self.cache_key(content_hash);
                let ttl_secs = effective.as_secs().max(1);
                let _: () = redis::cmd("SETEX")
                    .arg(&key)
                    .arg(ttl_secs)
                    .arg(&data)
                    .query_async(&mut conn)
                    .await
                    .unwrap_or(());
            }
        }
    }

    async fn invalidate(&self, content_hash: &str) {
        if let Ok(mut conn) = self.conn().await {
            let key = self.cache_key(content_hash);
            let _: () = redis::cmd("DEL")
                .arg(&key)
                .query_async(&mut conn)
                .await
                .unwrap_or(());
        }
    }
}
