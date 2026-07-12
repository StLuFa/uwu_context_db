//! Vector index ports, including a space-safe typed API.
use crate::llm::EmbeddingVector;
use crate::{ContextError, ContextUri, EmbeddingSpaceId, EncodedEmbedding, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexPoint {
    pub uri: ContextUri,
    pub vector: Vec<f32>,
    pub embedding_model_id: Option<String>,
    pub embedding_dim: Option<usize>,
    pub embedding_version: Option<u64>,
    #[serde(default)]
    pub payload: serde_json::Value,
}
impl IndexPoint {
    pub fn from_embedding(
        uri: ContextUri,
        embedding: EmbeddingVector,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            uri,
            vector: embedding.vector,
            embedding_model_id: Some(embedding.model_id),
            embedding_dim: Some(embedding.dim),
            embedding_version: Some(embedding.version),
            payload,
        }
    }
    pub fn from_encoded(
        uri: ContextUri,
        embedding: EncodedEmbedding,
        payload: serde_json::Value,
    ) -> Self {
        Self {
            uri,
            embedding_model_id: Some(embedding.space.model.clone()),
            embedding_dim: Some(embedding.space.dim),
            embedding_version: None,
            vector: embedding.values,
            payload,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexVector {
    pub uri: ContextUri,
    pub vector: Vec<f32>,
    pub embedding_model_id: Option<String>,
    pub embedding_dim: Option<usize>,
    pub embedding_version: Option<u64>,
    pub payload: serde_json::Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHit {
    pub uri: ContextUri,
    pub score: f32,
    pub payload: serde_json::Value,
}

#[async_trait]
pub trait VectorIndex: Send + Sync {
    async fn upsert(&self, collection: &str, point: IndexPoint) -> Result<()>;
    async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<IndexHit>>;
    async fn get_many(&self, _collection: &str, _uris: &[ContextUri]) -> Result<Vec<IndexVector>> {
        Err(ContextError::Storage(
            "vector backend does not support batch retrieval".into(),
        ))
    }
    async fn delete(&self, collection: &str, uri: &ContextUri) -> Result<()>;
}

#[async_trait]
impl<T: VectorIndex + ?Sized> VectorIndex for &T {
    async fn upsert(&self, collection: &str, point: IndexPoint) -> Result<()> {
        (**self).upsert(collection, point).await
    }

    async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<IndexHit>> {
        (**self).search(collection, query, top_k, filter).await
    }

    async fn get_many(&self, collection: &str, uris: &[ContextUri]) -> Result<Vec<IndexVector>> {
        (**self).get_many(collection, uris).await
    }

    async fn delete(&self, collection: &str, uri: &ContextUri) -> Result<()> {
        (**self).delete(collection, uri).await
    }
}

/// Mandatory space-aware facade for all new query/upsert paths. A collection is bound to one exact space.
pub struct SpaceCheckedVectorIndex<I> {
    inner: I,
    collection: String,
    space: EmbeddingSpaceId,
}
impl<I> SpaceCheckedVectorIndex<I> {
    pub fn new(inner: I, collection: impl Into<String>, space: EmbeddingSpaceId) -> Result<Self> {
        space.validate()?;
        Ok(Self {
            inner,
            collection: collection.into(),
            space,
        })
    }
    pub fn space(&self) -> &EmbeddingSpaceId {
        &self.space
    }
}
impl<I: VectorIndex> SpaceCheckedVectorIndex<I> {
    pub async fn upsert(
        &self,
        uri: ContextUri,
        embedding: EncodedEmbedding,
        payload: serde_json::Value,
    ) -> Result<()> {
        embedding.ensure_space(&self.space)?;
        self.inner
            .upsert(
                &self.collection,
                IndexPoint::from_encoded(uri, embedding, payload),
            )
            .await
    }
    pub async fn search(
        &self,
        query: EncodedEmbedding,
        top_k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<IndexHit>> {
        query.ensure_space(&self.space)?;
        self.inner
            .search(&self.collection, query.values, top_k, filter)
            .await
    }
    pub async fn delete(&self, uri: &ContextUri) -> Result<()> {
        self.inner.delete(&self.collection, uri).await
    }
}
