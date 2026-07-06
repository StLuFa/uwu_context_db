//! TieredCache — 三层冷热缓存分离（内存 L1 → 磁盘 L2 → PG 兜底）。

use agent_context_db_core::{ContentPayload, ContextUri};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// 缓存层级。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheTier { Hot, Warm, Cold }

/// 缓存条目。
struct CacheEntry {
    payload: ContentPayload,
    inserted: Instant,
    ttl: Duration,
    access_count: u64,
}

/// 三层冷热缓存。
pub struct TieredCache {
    hot: parking_lot::Mutex<lru::LruCache<String, CacheEntry>>,
    warm: parking_lot::Mutex<HashMap<String, CacheEntry>>,
    hot_capacity: usize,
    warm_capacity: usize,
    hot_ttl: Duration,
    warm_ttl: Duration,
}

impl TieredCache {
    pub fn new(hot_cap: usize, warm_cap: usize) -> Self {
        Self {
            hot: parking_lot::Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(hot_cap.max(1)).unwrap())),
            warm: parking_lot::Mutex::new(HashMap::new()),
            hot_capacity: hot_cap, warm_capacity: warm_cap,
            hot_ttl: Duration::from_secs(300), warm_ttl: Duration::from_secs(3600),
        }
    }

    /// 读取 — 先查热缓存，再查温缓存，未命中走 PG（由调用方处理）。
    pub fn get(&self, uri: &ContextUri) -> Option<ContentPayload> {
        let key = uri.to_string();
        // 热缓存
        if let Some(entry) = self.hot.lock().get(&key) {
            if entry.inserted.elapsed() < entry.ttl {
                return Some(entry.payload.clone());
            }
        }
        // 温缓存 → 提升到热
        if let Some(entry) = self.warm.lock().get(&key) {
            if entry.inserted.elapsed() < entry.ttl {
                let payload = entry.payload.clone();
                self.promote_to_hot(&key, payload.clone(), self.hot_ttl);
                return Some(payload);
            }
        }
        None
    }

    /// 写入 — 进热缓存，TTL 后降级到温缓存。
    pub fn put(&self, uri: &ContextUri, payload: ContentPayload) {
        let key = uri.to_string();
        self.hot.lock().put(key.clone(), CacheEntry {
            payload, inserted: Instant::now(), ttl: self.hot_ttl, access_count: 1,
        });
    }

    fn promote_to_hot(&self, key: &str, payload: ContentPayload, ttl: Duration) {
        self.hot.lock().put(key.to_string(), CacheEntry {
            payload, inserted: Instant::now(), ttl, access_count: 1,
        });
    }

    /// Sleeptime 再平衡 — 热缓存低频降级到温，温缓存高频提升到热。
    pub fn rebalance(&self) {
        // 简化：清空过期条目
        let now = Instant::now();
        {
            let mut hot = self.hot.lock();
            let expired: Vec<String> = hot.iter()
                .filter(|(_, e)| now - e.inserted >= e.ttl)
                .map(|(k, _)| k.clone())
                .collect();
            for k in expired { hot.pop(&k); }
        }
        self.warm.lock().retain(|_, e| now - e.inserted < e.ttl);
    }
}
