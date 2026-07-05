//! Redis 后端实现 — 可选（feature gate `redis-backend`）。
//!
//! 提供 EventBus、ReadCache、RateLimiter 的 Redis 实现，
//! 启用多进程共享事件、缓存、限流能力。
//!
//! 编译条件：
//! ```bash
//! cargo build --features redis-backend
//! ```

use crate::event_bus::{EventBus, EventStream};
use crate::read_cache::ReadCache;
use crate::rate_limiter::RateLimiter;
use crate::{ContentLevel, ContentPayload, ContextUri, Result};
use async_trait::async_trait;
use std::sync::Arc;
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
    pub pool_size: usize,
}

impl Default for RedisConfig {
    fn default() -> Self {
        Self {
            url: "redis://127.0.0.1:6379".into(),
            key_prefix: "uwu".into(),
            default_ttl_secs: 300,
            pool_size: 10,
        }
    }
}

/// 辅助：构建带前缀的 Redis key。
fn prefixed(prefix: &str, key: &str) -> String {
    format!("{}:{}", prefix, key)
}

// ===========================================================================
// RedisEventBus — Redis Pub/Sub 事件广播
// ===========================================================================

/// Redis 事件总线（Pub/Sub + Stream 双写）。
///
/// - Pub/Sub 做实时推送（fire-and-forget）
/// - Stream 做持久化（新订阅者/崩溃恢复可回放）
pub struct RedisEventBus {
    client: redis::Client,
    prefix: String,
}

impl RedisEventBus {
    pub fn connect(config: &RedisConfig) -> Result<Self> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| crate::ContextError::Storage(format!("redis connect: {e}")))?;
        Ok(Self { client, prefix: config.key_prefix.clone() })
    }

    fn channel(&self, topic: &str) -> String {
        prefixed(&self.prefix, &format!("events:{}", topic))
    }

    fn stream_key(&self, topic: &str) -> String {
        format!("{}:stream", self.channel(topic))
    }

    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::ContextError::Storage(format!("redis conn: {e}")))
    }
}

#[async_trait]
impl EventBus for RedisEventBus {
    async fn publish(&self, topic: &str, payload: &[u8]) -> Result<()> {
        let mut conn = self.conn().await?;
        let chan = self.channel(topic);

        // Pub/Sub 实时广播
        redis::cmd("PUBLISH")
            .arg(&chan)
            .arg(payload)
            .exec_async(&mut conn)
            .await
            .map_err(|e| crate::ContextError::Storage(format!("redis publish: {e}")))?;

        // Stream 持久化（新订阅者可回放）
        redis::cmd("XADD")
            .arg(self.stream_key(topic))
            .arg("*")  // auto-generate ID
            .arg("data")
            .arg(payload)
            .exec_async(&mut conn)
            .await
            .map_err(|e| crate::ContextError::Storage(format!("redis xadd: {e}")))?;

        Ok(())
    }

    async fn subscribe(&self, topic: &str) -> Result<Box<dyn EventStream>> {
        let client = self.client.clone();
        let channel = self.channel(topic);
        let mut pubsub = client
            .get_async_connection()
            .await
            .map_err(|e| crate::ContextError::Storage(format!("redis pubsub conn: {e}")))?
            .into_pubsub();

        pubsub
            .subscribe(&channel)
            .await
            .map_err(|e| crate::ContextError::Storage(format!("redis subscribe: {e}")))?;

        Ok(Box::new(RedisEventStream { pubsub, channel }))
    }
}

/// Redis 事件流（通过 Pub/Sub 接收）。
struct RedisEventStream {
    pubsub: redis::aio::PubSub,
    channel: String,
}

#[async_trait]
impl EventStream for RedisEventStream {
    async fn next(&mut self) -> Option<Vec<u8>> {
        let msg = self.pubsub.on_message().next().await?;
        msg.get_payload_bytes().map(|b| b.to_vec())
    }
}

// ===========================================================================
// RedisReadCache — Redis 读取缓存
// ===========================================================================

/// Redis 读取缓存 — 跨进程共享 L0/L1 结果。
pub struct RedisReadCache {
    client: redis::Client,
    prefix: String,
    default_ttl: Duration,
}

impl RedisReadCache {
    pub fn connect(config: &RedisConfig) -> Result<Self> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| crate::ContextError::Storage(format!("redis connect: {e}")))?;
        Ok(Self {
            client,
            prefix: config.key_prefix.clone(),
            default_ttl: Duration::from_secs(config.default_ttl_secs),
        })
    }

    fn cache_key(&self, uri: &ContextUri, level: ContentLevel) -> String {
        prefixed(&self.prefix, &format!("cache:{}:{}", uri.as_str(), level.as_str()))
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
    async fn get(&self, uri: &ContextUri, level: ContentLevel) -> Option<ContentPayload> {
        let key = self.cache_key(uri, level);
        let mut conn = self.conn().await.ok()?;
        let data: Option<Vec<u8>> = redis::cmd("GET")
            .arg(&key)
            .query_async(&mut conn)
            .await
            .ok()?;
        data.and_then(|bytes| serde_json::from_slice(&bytes).ok())
    }

    async fn put(
        &self,
        uri: &ContextUri,
        level: ContentLevel,
        payload: ContentPayload,
        ttl: Duration,
    ) {
        let key = self.cache_key(uri, level);
        if let Ok(data) = serde_json::to_vec(&payload) {
            if let Ok(mut conn) = self.conn().await {
                let ttl_secs = ttl.as_secs().max(1);
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
// RedisRateLimiter — Redis 令牌桶限流
// ===========================================================================

/// Redis 令牌桶限流器 — Lua 原子操作，跨进程共享 API 配额。
pub struct RedisRateLimiter {
    client: redis::Client,
    key: String,
    capacity: u32,
    refill_interval_secs: u64,
}

/// 令牌桶 Lua 脚本 — 原子操作避免竞态。
const TOKEN_BUCKET_LUA: &str = r#"
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local interval = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1]) or capacity
local last_refill = tonumber(bucket[2]) or now

-- 补充令牌
local elapsed = now - last_refill
local refill = math.floor(elapsed / interval * capacity)
tokens = math.min(capacity, tokens + refill)

if tokens >= 1 then
    tokens = tokens - 1
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    redis.call('EXPIRE', key, interval * 2)
    return 1  -- 成功
else
    redis.call('HMSET', key, 'tokens', tokens, 'last_refill', now)
    return 0  -- 限流
end
"#;

impl RedisRateLimiter {
    pub fn new(config: &RedisConfig, capacity: u32, refill_interval_secs: u64) -> Result<Self> {
        let client = redis::Client::open(config.url.as_str())
            .map_err(|e| crate::ContextError::Storage(format!("redis connect: {e}")))?;
        Ok(Self {
            client,
            key: prefixed(&config.key_prefix, "ratelimit:llm"),
            capacity,
            refill_interval_secs,
        })
    }

    async fn conn(&self) -> Result<redis::aio::MultiplexedConnection> {
        self.client
            .get_multiplexed_async_connection()
            .await
            .map_err(|e| crate::ContextError::Storage(format!("redis conn: {e}")))
    }
}

#[async_trait]
impl RateLimiter for RedisRateLimiter {
    async fn acquire(&self) -> Result<()> {
        loop {
            if self.try_acquire().await {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    async fn try_acquire(&self) -> bool {
        let mut conn = match self.conn().await {
            Ok(c) => c,
            Err(_) => return true, // Fail open: 无法连 Redis 时放行
        };
        let now = chrono::Utc::now().timestamp();
        let result: i32 = redis::cmd("EVAL")
            .arg(TOKEN_BUCKET_LUA)
            .arg(1) // number of keys
            .arg(&self.key)
            .arg(self.capacity)
            .arg(self.refill_interval_secs)
            .arg(now)
            .query_async(&mut conn)
            .await
            .unwrap_or(1); // Fail open
        result == 1
    }
}

// ===========================================================================
// 组合注入辅助
// ===========================================================================

/// Redis 后端聚合 — 一次性创建所有 Redis 实现。
pub struct RedisBackend {
    pub event_bus: Arc<dyn EventBus>,
    pub read_cache: Arc<dyn ReadCache>,
    pub rate_limiter: Arc<dyn RateLimiter>,
}

impl RedisBackend {
    /// 从配置创建全套 Redis 后端。
    pub fn new(config: &RedisConfig) -> Result<Self> {
        let event_bus = Arc::new(RedisEventBus::connect(config)?) as Arc<dyn EventBus>;
        let read_cache = Arc::new(RedisReadCache::connect(config)?) as Arc<dyn ReadCache>;
        let rate_limiter = Arc::new(RedisRateLimiter::new(config, 60, 60)?) as Arc<dyn RateLimiter>;
        Ok(Self { event_bus, read_cache, rate_limiter })
    }
}

// ===========================================================================
// Memory fallback 组合（无 Redis 时）
// ===========================================================================

/// 内存后端聚合 — 所有默认实现。
pub struct MemoryBackend {
    pub event_bus: Arc<dyn EventBus>,
    pub read_cache: Arc<dyn ReadCache>,
    pub rate_limiter: Arc<dyn RateLimiter>,
}

impl MemoryBackend {
    pub fn new(cache_capacity: usize, max_concurrency: usize) -> Self {
        use crate::event_bus::MemoryEventBus;
        use crate::read_cache::MemoryReadCache;
        use crate::rate_limiter::MemoryRateLimiter;

        Self {
            event_bus: Arc::new(MemoryEventBus::new()) as Arc<dyn EventBus>,
            read_cache: Arc::new(MemoryReadCache::new(cache_capacity, Duration::from_secs(300)))
                as Arc<dyn ReadCache>,
            rate_limiter: Arc::new(MemoryRateLimiter::new(max_concurrency)) as Arc<dyn RateLimiter>,
        }
    }
}

// ===========================================================================
// 配置驱动的后端创建
// ===========================================================================

/// 后端类型。
pub enum CacheBackend {
    Memory(MemoryBackend),
    #[cfg(feature = "redis-backend")]
    Redis(RedisBackend),
}

impl CacheBackend {
    pub fn event_bus(&self) -> Arc<dyn EventBus> {
        match self {
            CacheBackend::Memory(m) => m.event_bus.clone(),
            #[cfg(feature = "redis-backend")]
            CacheBackend::Redis(r) => r.event_bus.clone(),
        }
    }

    pub fn read_cache(&self) -> Arc<dyn ReadCache> {
        match self {
            CacheBackend::Memory(m) => m.read_cache.clone(),
            #[cfg(feature = "redis-backend")]
            CacheBackend::Redis(r) => r.read_cache.clone(),
        }
    }

    pub fn rate_limiter(&self) -> Arc<dyn RateLimiter> {
        match self {
            CacheBackend::Memory(m) => m.rate_limiter.clone(),
            #[cfg(feature = "redis-backend")]
            CacheBackend::Redis(r) => r.rate_limiter.clone(),
        }
    }
}
