//! 读取缓存 trait — L0/L1 内容缓存（moka LRU 或 Redis）。

use crate::{ContentLevel, ContentPayload, ContextUri, Result};
use async_trait::async_trait;
use std::time::Duration;

/// 读取缓存端口。
#[async_trait]
pub trait ReadCache: Send + Sync {
    async fn get(&self, uri: &ContextUri, level: ContentLevel) -> Option<ContentPayload>;
    async fn put(&self, uri: &ContextUri, level: ContentLevel, payload: ContentPayload, ttl: Duration);
    async fn invalidate(&self, uri: &ContextUri);
}

/// 内存实现（moka LRU）。
pub struct MemoryReadCache {
    l0: parking_lot::Mutex<lru::LruCache<String, (ContentPayload, std::time::Instant)>>,
    ttl: Duration,
}

impl MemoryReadCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            l0: parking_lot::Mutex::new(lru::LruCache::new(std::num::NonZeroUsize::new(capacity.max(1)).unwrap())),
            ttl,
        }
    }
}

#[async_trait]
impl ReadCache for MemoryReadCache {
    async fn get(&self, uri: &ContextUri, _level: ContentLevel) -> Option<ContentPayload> {
        let mut cache = self.l0.lock();
        if let Some((payload, ts)) = cache.get(&uri.to_string()) {
            if ts.elapsed() < self.ttl {
                return Some(payload.clone());
            }
            cache.pop(&uri.to_string());
        }
        None
    }

    async fn put(&self, uri: &ContextUri, _level: ContentLevel, payload: ContentPayload, _ttl: Duration) {
        self.l0.lock().put(uri.to_string(), (payload, std::time::Instant::now()));
    }

    async fn invalidate(&self, uri: &ContextUri) {
        self.l0.lock().pop(&uri.to_string());
    }
}
