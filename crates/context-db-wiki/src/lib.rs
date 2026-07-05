//! # agent-context-db-wiki (L7 wiki 层)
//!
//! `ContextDbWikiStorage` —— 把 `wiki-core` 的 7 个存储端口桥接到 context-db 的
//! 双层存储（内容层 + 索引层）。**无 uwu 依赖**，属通用核心 L7。
//!
//! ## 本 crate 的职责边界
//!
//! - 提供**跨层桥接的关键适配器** [`WikiVectorStoreAdapter`]：把
//!   `wiki_core::VectorStore` 映射到 context-db 的 `VectorIndex`（Qdrant）。
//! - [`ContextDbWikiStorage`] 是装配壳：vector_store 由本层桥接构造，
//!   其余 6 个端口（doc/op/text/link/blob/version）由宿主注入（对接 PG）。
//!
//! 这样 wiki 子域完全不自持存储，真值源唯一（context-db 的 PG+Qdrant）。

use agent_context_db_storage::{IndexPoint, VectorIndex};
use async_trait::async_trait;
use std::sync::Arc;
use wiki_core::storage::{
    BlobStore, DocStore, DocVersionStore, LinkStore, OpLog, TextIndex, VectorSearchResult,
    VectorStore, WikiStorage,
};
use wiki_core::Result;

// ===========================================================================
// 关键桥接：wiki VectorStore → context-db VectorIndex
// ===========================================================================

/// 把 wiki 的 `VectorStore` 端口翻译成 context-db 的 `VectorIndex`。
///
/// wiki 侧的 block id 映射为索引层的 `uri`（`wiki://<collection>/<id>` 语义键）。
pub struct WikiVectorStoreAdapter {
    index: Arc<dyn VectorIndex>,
}

impl WikiVectorStoreAdapter {
    pub fn new(index: Arc<dyn VectorIndex>) -> Self {
        Self { index }
    }
}

#[async_trait]
impl VectorStore for WikiVectorStoreAdapter {
    async fn upsert(
        &self,
        collection: &str,
        id: &str,
        vector: Vec<f32>,
        metadata: serde_json::Value,
    ) -> Result<()> {
        self.index
            .upsert(
                collection,
                IndexPoint {
                    uri: id.to_string(),
                    vector,
                    payload: metadata,
                },
            )
            .await
            .map_err(|e| wiki_core::WikiError::Storage(e.to_string()))
    }

    async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<VectorSearchResult>> {
        let hits = self
            .index
            .search(collection, query, top_k, filter)
            .await
            .map_err(|e| wiki_core::WikiError::Storage(e.to_string()))?;
        Ok(hits
            .into_iter()
            .map(|h| VectorSearchResult {
                id: h.uri,
                score: h.score,
                metadata: h.payload,
            })
            .collect())
    }

    async fn delete(&self, collection: &str, id: &str) -> Result<()> {
        self.index
            .delete(collection, id)
            .await
            .map_err(|e| wiki_core::WikiError::Storage(e.to_string()))
    }
}

// ===========================================================================
// 装配壳：vector_store 走桥接，其余 6 端口由宿主注入（对接 PG）
// ===========================================================================

/// context-db 实现的 `WikiStorage`。
///
/// vector_store 复用索引层（Qdrant）；doc/op/text/link/blob/version 由宿主注入的
/// PG 适配器提供。此壳不自持存储，只做装配。
pub struct ContextDbWikiStorage {
    vector: Arc<WikiVectorStoreAdapter>,
    docs: Arc<dyn DocStore>,
    oplog: Arc<dyn OpLog>,
    text: Arc<dyn TextIndex>,
    links: Arc<dyn LinkStore>,
    blobs: Arc<dyn BlobStore>,
    versions: Arc<dyn DocVersionStore>,
}

impl ContextDbWikiStorage {
    /// `index` 为 context-db 索引层；其余为宿主注入的 PG 适配器。
    pub fn new(
        index: Arc<dyn VectorIndex>,
        docs: Arc<dyn DocStore>,
        oplog: Arc<dyn OpLog>,
        text: Arc<dyn TextIndex>,
        links: Arc<dyn LinkStore>,
        blobs: Arc<dyn BlobStore>,
        versions: Arc<dyn DocVersionStore>,
    ) -> Self {
        Self {
            vector: Arc::new(WikiVectorStoreAdapter::new(index)),
            docs,
            oplog,
            text,
            links,
            blobs,
            versions,
        }
    }
}

impl WikiStorage for ContextDbWikiStorage {
    fn vector_store(&self) -> Arc<dyn VectorStore> {
        self.vector.clone()
    }
    fn doc_store(&self) -> Arc<dyn DocStore> {
        self.docs.clone()
    }
    fn op_log(&self) -> Arc<dyn OpLog> {
        self.oplog.clone()
    }
    fn text_index(&self) -> Arc<dyn TextIndex> {
        self.text.clone()
    }
    fn link_store(&self) -> Arc<dyn LinkStore> {
        self.links.clone()
    }
    fn blob_store(&self) -> Arc<dyn BlobStore> {
        self.blobs.clone()
    }
    fn version_store(&self) -> Arc<dyn DocVersionStore> {
        self.versions.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_context_db_core::Result as CdbResult;
    use agent_context_db_storage::IndexHit;
    use parking_lot::Mutex;
    use std::collections::HashMap;

    // 内存 VectorIndex，验证桥接方向正确。
    #[derive(Default)]
    struct MemIndex {
        data: Mutex<HashMap<String, (Vec<f32>, serde_json::Value)>>,
    }
    #[async_trait]
    impl VectorIndex for MemIndex {
        async fn upsert(&self, _c: &str, p: IndexPoint) -> CdbResult<()> {
            self.data.lock().insert(p.uri, (p.vector, p.payload));
            Ok(())
        }
        async fn search(
            &self,
            _c: &str,
            _q: Vec<f32>,
            _k: usize,
            _f: Option<serde_json::Value>,
        ) -> CdbResult<Vec<IndexHit>> {
            Ok(self
                .data
                .lock()
                .iter()
                .map(|(uri, (_, payload))| IndexHit {
                    uri: uri.clone(),
                    score: 1.0,
                    payload: payload.clone(),
                })
                .collect())
        }
        async fn delete(&self, _c: &str, uri: &str) -> CdbResult<()> {
            self.data.lock().remove(uri);
            Ok(())
        }
    }

    #[tokio::test]
    async fn vector_bridge_roundtrips_through_index() {
        let index: Arc<dyn VectorIndex> = Arc::new(MemIndex::default());
        let adapter = WikiVectorStoreAdapter::new(index);

        adapter
            .upsert("wiki_blocks", "b1", vec![1.0, 0.0], serde_json::json!({"doc": "d1"}))
            .await
            .unwrap();
        let hits = adapter
            .search("wiki_blocks", vec![1.0, 0.0], 5, None)
            .await
            .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "b1");

        adapter.delete("wiki_blocks", "b1").await.unwrap();
        assert!(adapter
            .search("wiki_blocks", vec![1.0, 0.0], 5, None)
            .await
            .unwrap()
            .is_empty());
    }
}
