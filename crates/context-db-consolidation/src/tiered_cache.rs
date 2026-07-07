//! TieredCache — 三层冷热缓存分离（内存 L1 → 磁盘 L2 → PG 兜底）。

use agent_context_db_core::{ContentPayload, ContextUri};
use moka::policy::Expiry;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 缓存层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier {
    Hot,
    Warm,
    Cold,
}

/// 缓存条目。
#[derive(Clone)]
struct CacheEntry {
    payload: ContentPayload,
    inserted: Instant,
    ttl: Duration,
    access_count: u64,
}

#[derive(Clone)]
struct CacheEntryExpiry;

impl Expiry<String, CacheEntry> for CacheEntryExpiry {
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

/// 三层冷热缓存。
pub struct TieredCache {
    hot: moka::future::Cache<String, CacheEntry>,
    warm: parking_lot::Mutex<HashMap<String, CacheEntry>>,
    warm_capacity: usize,
    hot_ttl: Duration,
    warm_ttl: Duration,
}

impl TieredCache {
    pub fn new(hot_cap: usize, warm_cap: usize) -> Self {
        Self {
            hot: moka::future::Cache::builder()
                .max_capacity(hot_cap.max(1) as u64)
                .expire_after(CacheEntryExpiry)
                .build(),
            warm: parking_lot::Mutex::new(HashMap::new()),
            warm_capacity: warm_cap,
            hot_ttl: Duration::from_secs(300),
            warm_ttl: Duration::from_secs(3600),
        }
    }

    /// 读取 — 先查热缓存，再查温缓存，未命中走 PG（由调用方处理）。
    pub async fn get(&self, uri: &ContextUri) -> Option<ContentPayload> {
        let key = uri.to_string();
        if let Some(entry) = self.hot.get(&key).await {
            return Some(entry.payload);
        }

        let warm_entry = self.warm.lock().get(&key).cloned();
        if let Some(entry) = warm_entry {
            if entry.inserted.elapsed() < entry.ttl {
                let payload = entry.payload.clone();
                self.promote_to_hot(&key, payload.clone(), self.hot_ttl)
                    .await;
                return Some(payload);
            }
            self.warm.lock().remove(&key);
        }
        None
    }

    /// 写入 — 进热缓存，TTL 后由 moka 过期淘汰。
    pub async fn put(&self, uri: &ContextUri, payload: ContentPayload) {
        self.promote_to_hot(&uri.to_string(), payload, self.hot_ttl)
            .await;
    }

    async fn promote_to_hot(&self, key: &str, payload: ContentPayload, ttl: Duration) {
        self.hot
            .insert(
                key.to_string(),
                CacheEntry {
                    payload,
                    inserted: Instant::now(),
                    ttl,
                    access_count: 1,
                },
            )
            .await;
    }

    /// Sleeptime 再平衡 — 清理热/温缓存中过期条目。
    pub async fn rebalance(&self) {
        self.hot.run_pending_tasks().await;
        let now = Instant::now();
        self.warm.lock().retain(|_, e| now - e.inserted < e.ttl);
    }
}
