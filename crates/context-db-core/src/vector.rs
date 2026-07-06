//! 向量索引端口（M0 轻量 trait，零外部依赖）。
//!
//! 从 `context-db-storage` 提升至 core，使检索层可依赖此端口而不反向依赖存储层。
//! 后端适配器（Qdrant/Pgvector/Memory）由 storage 层实现。

use crate::error::Result;
use crate::uri::ContextUri;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// 索引写入点：URI + 向量 + 可选 payload。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexPoint {
    /// 指向内容层的 uwu:// URI。
    pub uri: ContextUri,
    pub vector: Vec<f32>,
    #[serde(default)]
    pub payload: serde_json::Value,
}

/// 索引命中结果。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexHit {
    pub uri: ContextUri,
    pub score: f32,
    pub payload: serde_json::Value,
}

/// 向量索引端口 —— 检索层通过它做向量召回，不感知具体后端。
#[async_trait]
pub trait VectorIndex: Send + Sync {
    /// 写入/更新一个索引点。
    async fn upsert(&self, collection: &str, point: IndexPoint) -> Result<()>;

    /// 相似度检索，返回 top_k 结果。
    async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<IndexHit>>;

    /// 按 URI 删除索引点。
    async fn delete(&self, collection: &str, uri: &ContextUri) -> Result<()>;
}
