//! 检索缓存层 — 接入 ReadCache trait，含穿透/雪崩防护。
//!
//! 穿透/雪崩防护主体已下沉到 `ReadCache` 实现（`MemoryReadCache` / `UwuCacheAdapter` /
//! `RedisReadCache`）。本 wrapper 只提供便捷 API 与默认 TTL 语义。

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri, ReadCache};
use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::Arc;
use std::time::Duration;

/// Thread-safe callers wrap this bounded LRU in their local mutex. Cache operations are
/// synchronous and never span I/O or an await point.
pub(crate) struct BoundedLruCache<K, V> {
    capacity: usize,
    entries: HashMap<K, V>,
    recency: VecDeque<K>,
}

impl<K: Eq + Hash + Clone, V> BoundedLruCache<K, V> {
    pub(crate) fn new(capacity: usize) -> Self {
        debug_assert!(capacity > 0);
        Self {
            capacity,
            entries: HashMap::with_capacity(capacity),
            recency: VecDeque::with_capacity(capacity),
        }
    }

    pub(crate) fn get(&mut self, key: &K) -> Option<&V> {
        if self.entries.contains_key(key) {
            self.touch(key);
        }
        self.entries.get(key)
    }

    pub(crate) fn insert(&mut self, key: K, value: V) {
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), value);
            self.touch(&key);
            return;
        }
        if self.entries.len() == self.capacity
            && let Some(evicted) = self.recency.pop_front()
        {
            self.entries.remove(&evicted);
        }
        self.recency.push_back(key.clone());
        self.entries.insert(key, value);
    }

    fn touch(&mut self, key: &K) {
        if let Some(position) = self.recency.iter().position(|candidate| candidate == key) {
            self.recency.remove(position);
        }
        self.recency.push_back(key.clone());
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod bounded_tests {
    use super::BoundedLruCache;

    #[test]
    fn bounded_lru_reuses_recent_entries_and_evicts_oldest() {
        let mut cache = BoundedLruCache::new(2);
        cache.insert("a", 1);
        cache.insert("b", 2);
        assert_eq!(cache.get(&"a"), Some(&1));
        cache.insert("c", 3);

        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), None);
        assert_eq!(cache.get(&"c"), Some(&3));
    }
}

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
