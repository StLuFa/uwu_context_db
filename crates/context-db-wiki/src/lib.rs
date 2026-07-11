//! # agent-context-db-wiki (L7 wiki 层)
//!
//! Wiki 向量桥接在构造时绑定 tenant 与 wiki space，并从二者内部派生物理
//! collection。调用方传入的 collection 仅作为 wiki-core 协议字段校验，不能选择
//! 其他租户或 space 的索引。所有写入、删除和检索结果都经过 URI 命名空间校验。

use agent_context_db_core::{ContextUri, UriCategory};
use agent_context_db_storage::{IndexPoint, VectorIndex};
use async_trait::async_trait;
use std::sync::Arc;
use wiki_core::Result;
use wiki_core::storage::{
    BlobStore, DocStore, DocVersionStore, LinkStore, OpLog, TextIndex, VectorSearchResult,
    VectorStore, WikiStorage,
};

/// 把 wiki 的 `VectorStore` 端口翻译成绑定命名空间的 context-db `VectorIndex`。
pub struct WikiVectorStoreAdapter {
    index: Arc<dyn VectorIndex>,
    tenant: String,
    space: String,
    collection: String,
}

impl WikiVectorStoreAdapter {
    /// 创建 tenant/space 专属适配器。collection 由适配器内部派生，不接受外部配置。
    pub fn new(
        index: Arc<dyn VectorIndex>,
        tenant: impl Into<String>,
        space: impl Into<String>,
    ) -> Result<Self> {
        let tenant = tenant.into();
        let space = space.into();
        Self::validate_namespace_segment("tenant", &tenant)?;
        Self::validate_namespace_segment("wiki space", &space)?;
        let collection = format!("wiki__{}__{}", hex_segment(&tenant), hex_segment(&space));
        Ok(Self {
            index,
            tenant,
            space,
            collection,
        })
    }

    fn validate_namespace_segment(name: &str, value: &str) -> Result<()> {
        if value.is_empty()
            || value == "."
            || value == ".."
            || value.contains('/')
            || value.contains('?')
            || value.contains('#')
        {
            return Err(storage_error(format!("invalid {name}: {value:?}")));
        }
        Ok(())
    }

    fn validate_collection(&self, supplied: &str) -> Result<()> {
        if supplied != self.collection {
            return Err(storage_error(format!(
                "wiki collection is bound to tenant {:?} space {:?}; expected {:?}, got {:?}",
                self.tenant, self.space, self.collection, supplied
            )));
        }
        Ok(())
    }

    /// 强制 URI 形状为 `uwu://{tenant}/wiki/{space}/...`。至少要求一个文档/块段，
    /// 防止把 space 根本身伪装成可索引对象。
    fn parse_and_validate_id(&self, id: &str) -> Result<ContextUri> {
        let uri = ContextUri::parse(id)
            .map_err(|e| storage_error(format!("wiki id is not a valid uwu URI: {id} — {e}")))?;
        self.validate_uri(&uri)?;
        Ok(uri)
    }

    fn validate_uri(&self, uri: &ContextUri) -> Result<()> {
        let segments = uri.segments();
        if uri.tenant() != self.tenant
            || uri.category() != UriCategory::Wiki
            || segments.get(2).map(String::as_str) != Some(self.space.as_str())
            || segments.len() < 4
        {
            return Err(storage_error(format!(
                "wiki URI is outside bound namespace uwu://{}/wiki/{}/: {}",
                self.tenant, self.space, uri
            )));
        }
        Ok(())
    }

    /// 返回 wiki-core 必须使用的、不可由调用方选择的物理 collection 名。
    pub fn collection(&self) -> &str {
        &self.collection
    }
}

fn hex_segment(value: &str) -> String {
    value
        .as_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn storage_error(message: String) -> wiki_core::WikiError {
    wiki_core::WikiError::Storage(message)
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
        self.validate_collection(collection)?;
        let uri = self.parse_and_validate_id(id)?;
        self.index
            .upsert(
                &self.collection,
                IndexPoint {
                    uri,
                    vector,
                    embedding_model_id: None,
                    embedding_dim: None,
                    embedding_version: None,
                    payload: metadata,
                },
            )
            .await
            .map_err(|e| storage_error(e.to_string()))
    }

    async fn search(
        &self,
        collection: &str,
        query: Vec<f32>,
        top_k: usize,
        filter: Option<serde_json::Value>,
    ) -> Result<Vec<VectorSearchResult>> {
        self.validate_collection(collection)?;
        let hits = self
            .index
            .search(&self.collection, query, top_k, filter)
            .await
            .map_err(|e| storage_error(e.to_string()))?;
        hits.into_iter()
            .map(|hit| {
                // 后端 collection 配置错误或被污染时 fail closed，不向 Wiki 泄露跨域命中。
                self.validate_uri(&hit.uri)?;
                Ok(VectorSearchResult {
                    id: hit.uri.to_string(),
                    score: hit.score,
                    metadata: hit.payload,
                })
            })
            .collect()
    }

    async fn delete(&self, collection: &str, id: &str) -> Result<()> {
        self.validate_collection(collection)?;
        let uri = self.parse_and_validate_id(id)?;
        self.index
            .delete(&self.collection, &uri)
            .await
            .map_err(|e| storage_error(e.to_string()))
    }
}

/// context-db 实现的 `WikiStorage`。真实运行装配必须显式提供 tenant/space，且构造
/// 可能因非法命名空间失败；不存在不绑定命名空间的旧构造路径。
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: Arc<dyn VectorIndex>,
        tenant: impl Into<String>,
        space: impl Into<String>,
        docs: Arc<dyn DocStore>,
        oplog: Arc<dyn OpLog>,
        text: Arc<dyn TextIndex>,
        links: Arc<dyn LinkStore>,
        blobs: Arc<dyn BlobStore>,
        versions: Arc<dyn DocVersionStore>,
    ) -> Result<Self> {
        Ok(Self {
            vector: Arc::new(WikiVectorStoreAdapter::new(index, tenant, space)?),
            docs,
            oplog,
            text,
            links,
            blobs,
            versions,
        })
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

    #[derive(Default)]
    struct MemIndex {
        data: Mutex<HashMap<ContextUri, (Vec<f32>, serde_json::Value)>>,
        collections: Mutex<Vec<String>>,
        injected_hits: Mutex<Vec<IndexHit>>,
    }

    #[async_trait]
    impl VectorIndex for MemIndex {
        async fn upsert(&self, collection: &str, point: IndexPoint) -> CdbResult<()> {
            self.collections.lock().push(collection.to_owned());
            self.data
                .lock()
                .insert(point.uri, (point.vector, point.payload));
            Ok(())
        }
        async fn search(
            &self,
            collection: &str,
            _query: Vec<f32>,
            _top_k: usize,
            _filter: Option<serde_json::Value>,
        ) -> CdbResult<Vec<IndexHit>> {
            self.collections.lock().push(collection.to_owned());
            let injected = self.injected_hits.lock().clone();
            if !injected.is_empty() {
                return Ok(injected);
            }
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
        async fn delete(&self, collection: &str, uri: &ContextUri) -> CdbResult<()> {
            self.collections.lock().push(collection.to_owned());
            self.data.lock().remove(uri);
            Ok(())
        }
    }

    fn adapter(index: Arc<MemIndex>) -> WikiVectorStoreAdapter {
        WikiVectorStoreAdapter::new(index, "tenant-a", "engineering").unwrap()
    }

    #[tokio::test]
    async fn vector_bridge_uses_only_derived_collection() {
        let index = Arc::new(MemIndex::default());
        let adapter = adapter(index.clone());
        let collection = adapter.collection().to_owned();
        adapter
            .upsert(
                &collection,
                "uwu://tenant-a/wiki/engineering/doc-1/block-1",
                vec![1.0],
                serde_json::json!({}),
            )
            .await
            .unwrap();
        assert_eq!(index.collections.lock().as_slice(), &[collection]);
    }

    #[tokio::test]
    async fn rejects_caller_selected_collection() {
        let index = Arc::new(MemIndex::default());
        let adapter = adapter(index.clone());
        assert!(
            adapter
                .upsert(
                    "wiki_blocks",
                    "uwu://tenant-a/wiki/engineering/doc-1",
                    vec![1.0],
                    serde_json::json!({})
                )
                .await
                .is_err()
        );
        assert!(
            adapter
                .search("other_collection", vec![1.0], 5, None)
                .await
                .is_err()
        );
        assert!(index.collections.lock().is_empty());
    }

    #[tokio::test]
    async fn rejects_cross_tenant_space_category_and_forged_uri() {
        let index = Arc::new(MemIndex::default());
        let adapter = adapter(index);
        let collection = adapter.collection().to_owned();
        for id in [
            "uwu://tenant-b/wiki/engineering/doc-1",
            "uwu://tenant-a/wiki/hr/doc-1",
            "uwu://tenant-a/agent/engineering/doc-1",
            "uwu://tenant-a/wiki/engineering",
            "https://tenant-a/wiki/engineering/doc-1",
        ] {
            assert!(
                adapter
                    .upsert(&collection, id, vec![1.0], serde_json::json!({}))
                    .await
                    .is_err(),
                "accepted {id}"
            );
            assert!(
                adapter.delete(&collection, id).await.is_err(),
                "accepted {id}"
            );
        }
    }

    #[tokio::test]
    async fn search_rejects_cross_namespace_backend_hit() {
        let index = Arc::new(MemIndex::default());
        index.injected_hits.lock().push(IndexHit {
            uri: ContextUri::parse("uwu://tenant-b/wiki/engineering/stolen").unwrap(),
            score: 1.0,
            payload: serde_json::json!({}),
        });
        let adapter = adapter(index);
        let collection = adapter.collection().to_owned();
        assert!(
            adapter
                .search(&collection, vec![1.0], 5, None)
                .await
                .is_err()
        );
    }

    #[test]
    fn collection_derivation_is_unambiguous_and_namespace_bound() {
        let index = Arc::new(MemIndex::default());
        let a = WikiVectorStoreAdapter::new(index.clone(), "ab", "c").unwrap();
        let b = WikiVectorStoreAdapter::new(index, "a", "bc").unwrap();
        assert_ne!(a.collection(), b.collection());
        assert!(
            WikiVectorStoreAdapter::new(Arc::new(MemIndex::default()), "tenant/a", "space")
                .is_err()
        );
    }
}
