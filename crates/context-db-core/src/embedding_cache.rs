//! Embedding cache keyed by the complete embedding-space identity and input content.

use crate::{EmbeddingSpaceId, EncodedEmbedding};
use async_trait::async_trait;
use moka::policy::Expiry;
use std::time::{Duration, Instant};

/// Stable, unambiguous cache key over every field that can change vector semantics.
pub fn embedding_content_hash(space: &EmbeddingSpaceId, content: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    let normalization = match space.normalization {
        crate::EmbeddingNormalization::None => b"none".as_slice(),
        crate::EmbeddingNormalization::L2 => b"l2".as_slice(),
    };
    let dim = (space.dim as u64).to_le_bytes();
    for field in [
        space.model.as_bytes(),
        space.checkpoint.as_bytes(),
        space.preprocess.as_bytes(),
        dim.as_slice(),
        normalization,
        content,
    ] {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hasher.finalize().to_hex().to_string()
}

#[async_trait]
pub trait EmbeddingCache: Send + Sync {
    async fn get(&self, space: &EmbeddingSpaceId, content: &[u8]) -> Option<EncodedEmbedding>;
    async fn put(&self, content: &[u8], embedding: EncodedEmbedding, ttl: Duration);
    async fn invalidate(&self, space: &EmbeddingSpaceId, content: &[u8]);
}

#[derive(Debug, Clone)]
struct EmbeddingCacheEntry {
    embedding: EncodedEmbedding,
    ttl: Duration,
}
#[derive(Debug, Clone)]
struct EmbeddingCacheExpiry;
impl Expiry<String, EmbeddingCacheEntry> for EmbeddingCacheExpiry {
    fn expire_after_create(
        &self,
        _: &String,
        value: &EmbeddingCacheEntry,
        _: Instant,
    ) -> Option<Duration> {
        Some(value.ttl)
    }
    fn expire_after_update(
        &self,
        _: &String,
        value: &EmbeddingCacheEntry,
        _: Instant,
        _: Option<Duration>,
    ) -> Option<Duration> {
        Some(value.ttl)
    }
}

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
    async fn get(&self, space: &EmbeddingSpaceId, content: &[u8]) -> Option<EncodedEmbedding> {
        self.entries
            .get(&embedding_content_hash(space, content))
            .await
            .map(|entry| entry.embedding)
            .filter(|embedding| embedding.space == *space)
    }
    async fn put(&self, content: &[u8], embedding: EncodedEmbedding, ttl: Duration) {
        let key = embedding_content_hash(&embedding.space, content);
        self.entries
            .insert(
                key,
                EmbeddingCacheEntry {
                    embedding,
                    ttl: if ttl.is_zero() { self.default_ttl } else { ttl },
                },
            )
            .await;
    }
    async fn invalidate(&self, space: &EmbeddingSpaceId, content: &[u8]) {
        self.entries
            .invalidate(&embedding_content_hash(space, content))
            .await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EmbeddingNormalization;
    fn space(checkpoint: &str) -> EmbeddingSpaceId {
        EmbeddingSpaceId {
            model: "clip".into(),
            checkpoint: checkpoint.into(),
            preprocess: "resize224-v1".into(),
            dim: 2,
            normalization: EmbeddingNormalization::None,
        }
    }
    #[tokio::test]
    async fn complete_space_identity_is_part_of_cache_key() {
        let cache = MemoryEmbeddingCache::new(16, Duration::from_secs(60));
        let a = EncodedEmbedding::new(space("v1"), vec![1.0, 0.0]).unwrap();
        cache.put(b"same", a.clone(), Duration::ZERO).await;
        assert_eq!(cache.get(&space("v1"), b"same").await, Some(a));
        assert!(cache.get(&space("v2"), b"same").await.is_none());
    }
}
