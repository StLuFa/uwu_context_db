//! 读取缓存 trait — L0/L1 内容缓存（LRU + 穿透/雪崩防护）。
//!
//! - **穿透防护**：`put_negative()` 记录"已知缺失"标记，短 TTL（默认 30s），
//!   避免重复回源查询同一个不存在的 URI。
//! - **雪崩防护**：`put()` 对 TTL 加 ±10% 均匀分布抖动，避免大批条目同时过期
//!   触发缓存击穿。

use crate::{ContentLevel, ContentPayload, ContextUri, Result};
use async_trait::async_trait;
use std::time::Duration;

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
    async fn put(&self, uri: &ContextUri, level: ContentLevel, payload: ContentPayload, ttl: Duration);
    /// 写入负缓存 —— 标记该 URI 当前不存在，短 TTL（默认 30s）。
    async fn put_negative(&self, uri: &ContextUri, level: ContentLevel);
    async fn invalidate(&self, uri: &ContextUri);
}

/// 内存实现（LRU + jitter TTL + 负缓存）。
pub struct MemoryReadCache {
    /// LRU：URI → (payload_option, deadline)。None 表示负缓存。
    l0: parking_lot::Mutex<lru::LruCache<String, (Option<ContentPayload>, std::time::Instant)>>,
    /// 默认正缓存 TTL（`put` 传入的 ttl 优先）。
    default_ttl: Duration,
    /// 负缓存 TTL（穿透防护）。
    negative_ttl: Duration,
}

impl MemoryReadCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            l0: parking_lot::Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(capacity.max(1)).unwrap())),
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
        let mut cache = self.l0.lock();
        let key = uri.to_string();
        if let Some((payload_opt, deadline)) = cache.get(&key) {
            if *deadline > std::time::Instant::now() {
                return Some(payload_opt.clone());
            }
            cache.pop(&key);
        }
        None
    }

    async fn put(&self, uri: &ContextUri, _level: ContentLevel, payload: ContentPayload, ttl: Duration) {
        let effective_ttl = if ttl.is_zero() { self.default_ttl } else { ttl };
        let deadline = std::time::Instant::now() + Self::jittered(effective_ttl);
        self.l0.lock().put(uri.to_string(), (Some(payload), deadline));
    }

    async fn put_negative(&self, uri: &ContextUri, _level: ContentLevel) {
        let deadline = std::time::Instant::now() + Self::jittered(self.negative_ttl);
        self.l0.lock().put(uri.to_string(), (None, deadline));
    }

    async fn invalidate(&self, uri: &ContextUri) {
        self.l0.lock().pop(&uri.to_string());
    }
}
