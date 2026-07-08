//! Embedding 缓存端口 — 按 blake3(content) 去重 LLM/embedding API 调用。

use async_trait::async_trait;
use moka::policy::Expiry;
use std::time::{Duration, Instant};

/// 计算 embedding 缓存内容哈希。
pub fn embedding_content_hash(content: &str) -> String {
    blake3::hash(content.as_bytes()).to_hex().to_string()
}

/// Embedding 缓存端口。
#[async_trait]
pub trait EmbeddingCache: Send + Sync {
    /// 按 `blake3(content)` 哈希读取 embedding。
    async fn get(&self, content_hash: &str) -> Option<Vec<f32>>;

    /// 按 `blake3(content)` 哈希写入 embedding。
    async fn put(&self, content_hash: &str, embedding: Vec<f32>, ttl: Duration);

    /// 删除指定内容哈希的 embedding。
    async fn invalidate(&self, content_hash: &str);
}

#[derive(Debug, Clone)]
struct EmbeddingCacheEntry {
    embedding: Vec<f32>,
    ttl: Duration,
}

#[derive(Debug, Clone)]
struct EmbeddingCacheExpiry;

impl Expiry<String, EmbeddingCacheEntry> for EmbeddingCacheExpiry {
    fn expire_after_create(
        &self,
        _key: &String,
        value: &EmbeddingCacheEntry,
        _created_at: Instant,
    ) -> Option<Duration> {
        Some(value.ttl)
    }

    fn expire_after_update(
        &self,
        _key: &String,
        value: &EmbeddingCacheEntry,
        _updated_at: Instant,
        _duration_until_expiry: Option<Duration>,
    ) -> Option<Duration> {
        Some(value.ttl)
    }
}

/// 进程内 embedding 缓存。
pub struct MemoryEmbeddingCache {
    entries: moka::future::Cache<String, EmbeddingCacheEntry>,
    default_ttl: Duration,
}

impl MemoryEmbeddingCache {
    pub fn new(capacity: usize, ttl: Duration) -> Self {
        Self {
            entries: moka::future::Cache::builder()
                .max_capacity(capacity.max(1) as u64)
                .expire_after(EmbeddingCacheExpiry)
                .build(),
            default_ttl: ttl,
        }
    }
}

#[async_trait]
impl EmbeddingCache for MemoryEmbeddingCache {
    async fn get(&self, content_hash: &str) -> Option<Vec<f32>> {
        self.entries
            .get(content_hash)
            .await
            .map(|entry| entry.embedding)
    }

    async fn put(&self, content_hash: &str, embedding: Vec<f32>, ttl: Duration) {
        let effective = if ttl.is_zero() { self.default_ttl } else { ttl };
        self.entries
            .insert(
                content_hash.to_string(),
                EmbeddingCacheEntry {
                    embedding,
                    ttl: effective,
                },
            )
            .await;
    }

    async fn invalidate(&self, content_hash: &str) {
        self.entries.invalidate(content_hash).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_hash_is_stable_and_content_addressed() {
        assert_eq!(
            embedding_content_hash("same"),
            embedding_content_hash("same")
        );
        assert_ne!(
            embedding_content_hash("same"),
            embedding_content_hash("other")
        );
    }

    #[tokio::test]
    async fn memory_embedding_cache_round_trips_by_hash() {
        let cache = MemoryEmbeddingCache::new(16, Duration::from_secs(60));
        let hash = embedding_content_hash("hello");
        assert!(cache.get(&hash).await.is_none());
        cache.put(&hash, vec![1.0, 2.0], Duration::ZERO).await;
        assert_eq!(cache.get(&hash).await, Some(vec![1.0, 2.0]));
        cache.invalidate(&hash).await;
        assert!(cache.get(&hash).await.is_none());
    }
}
