//! 读取缓存 trait — L0/L1 内容缓存（moka async cache + 穿透/雪崩防护）。
//!
//! - **穿透防护**：`put_negative()` 记录"已知缺失"标记，短 TTL（默认 30s），
//!   避免重复回源查询同一个不存在的 URI。
//! - **雪崩防护**：`put()` 对 TTL 加 ±10% 均匀分布抖动，避免大批条目同时过期
//!   触发缓存击穿。

use crate::{ContentLevel, ContentPayload, ContextUri, Result};
use async_trait::async_trait;
use moka::policy::Expiry;
use std::time::{Duration, Instant};

/// 读取缓存端口。
#[async_trait]
pub trait ReadCache: Send + Sync {
    /// 查询缓存。
    ///
    /// 返回：
    /// - `Some(Some(payload))` — 命中且有值
    /// - `Some(None)` — 命中负缓存（已知缺失，无需回源）
    /// - `None` — 未命中，需回源
    async fn get(&self, uri: &ContextUri, level: ContentLevel) -> Option<Option<ContentPayload>>;
    /// 写入正缓存 —— TTL 会附加 ±10% 抖动。
    async fn put(
        &self,
        uri: &ContextUri,
        level: ContentLevel,
        payload: ContentPayload,
        ttl: Duration,
    );
    /// 写入负缓存 —— 标记该 URI 当前不存在，短 TTL（默认 30s）。
    async fn put_negative(&self, uri: &ContextUri, level: ContentLevel);
    async fn invalidate(&self, uri: &ContextUri);
}

#[derive(Debug, Clone)]
struct CacheEntry {
    payload: Option<ContentPayload>,
    ttl: Duration,
}

#[derive(Debug, Clone)]
struct ReadCacheExpiry;

impl Expiry<String, CacheEntry> for ReadCacheExpiry {
    fn expire_after_create(
        &self,
        _key: &String,
        value: &CacheEntry,
        _created_at: Instant,
    ) -> Option<Duration> {
        Some(value.ttl)
    }

    fn expire_after_update(
        &self,
        _key: &String,
        value: &CacheEntry,
        _updated_at: Instant,
        _duration_until_expiry: Option<Duration>,
    ) -> Option<Duration> {
        Some(value.ttl)
    }
}

/// 内存实现（moka async cache + jitter TTL + 负缓存）。
pub struct MemoryReadCache {
    /// URI → payload option。`None` 表示负缓存；moka 负责容量淘汰与 TTL 过期。
    l0: moka::future::Cache<String, CacheEntry>,
    /// 默认正缓存 TTL（`put` 传入的 ttl 优先）。
    default_ttl: Duration,
    /// 负缓存 TTL（穿透防护）。
    negative_ttl: Duration,
}

impl MemoryReadCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            l0: moka::future::Cache::builder()
                .max_capacity(capacity.max(1) as u64)
                .expire_after(ReadCacheExpiry)
                .build(),
            default_ttl: ttl,
            negative_ttl: Duration::from_secs(30),
        }
    }

    /// 自定义负缓存 TTL（穿透防护窗口）。
    pub fn with_negative_ttl(mut self, ttl: Duration) -> Self {
        self.negative_ttl = ttl;
        self
    }

    /// 给 TTL 加 ±10% 均匀抖动（雪崩防护）。
    fn jittered(ttl: Duration) -> Duration {
        // 简单确定性伪随机：基于纳秒的低位。避免引入 rand 依赖。
        let nanos = ttl.as_nanos() as u64;
        // 抖动幅度：TTL 的 ±10%
        let jitter_bound = ttl.as_millis().max(1) as u64 / 10;
        if jitter_bound == 0 {
            return ttl;
        }
        // 使用 SystemTime 纳秒做随机源；range [-jitter_bound, jitter_bound]
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u64)
            .unwrap_or(nanos);
        let signed_offset = (now_ns % (2 * jitter_bound + 1)) as i64 - jitter_bound as i64;
        let base_ms = ttl.as_millis() as i64;
        let final_ms = (base_ms + signed_offset).max(1) as u64;
        Duration::from_millis(final_ms)
    }
}

#[async_trait]
impl ReadCache for MemoryReadCache {
    async fn get(&self, uri: &ContextUri, _level: ContentLevel) -> Option<Option<ContentPayload>> {
        self.l0
            .get(&uri.to_string())
            .await
            .map(|entry| entry.payload)
    }

    async fn put(
        &self,
        uri: &ContextUri,
        _level: ContentLevel,
        payload: ContentPayload,
        ttl: Duration,
    ) {
        let effective_ttl = if ttl.is_zero() { self.default_ttl } else { ttl };
        self.l0
            .insert(
                uri.to_string(),
                CacheEntry {
                    payload: Some(payload),
                    ttl: Self::jittered(effective_ttl),
                },
            )
            .await;
    }

    async fn put_negative(&self, uri: &ContextUri, _level: ContentLevel) {
        self.l0
            .insert(
                uri.to_string(),
                CacheEntry {
                    payload: None,
                    ttl: Self::jittered(self.negative_ttl),
                },
            )
            .await;
    }

    async fn invalidate(&self, uri: &ContextUri) {
        self.l0.invalidate(&uri.to_string()).await;
    }
}
