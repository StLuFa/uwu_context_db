# agent-context-db-wiki

wiki-core → context-db 存储桥接，使 uwu_wiki 不持有存储。

## 桥接模式

```
wiki-core (纯逻辑, 7 个端口: DocStore/OpLog/TextIndex/LinkStore/BlobStore/DocVersionStore/VectorStore)
  └── WikiVectorStoreAdapter — VectorStore 端口 → context-db VectorIndex(Qdrant)
  └── 其余 6 端口 — 由宿主注入 PG 适配器
```

## 关键类型

```rust
pub struct ContextDbWikiStorage {
    vector: Arc<WikiVectorStoreAdapter>,  // context-db Qdrant
    docs: Arc<dyn DocStore>,             // PG (宿主注入)
    oplog: Arc<dyn OpLog>,               // PG
    text: Arc<dyn TextIndex>,            // PG
    links: Arc<dyn LinkStore>,           // PG
    blobs: Arc<dyn BlobStore>,           // PG
    versions: Arc<dyn DocVersionStore>,  // PG
}
```

## URI 映射

```
uwu://.../wiki/{space}/{doc_id}           → DocStore
uwu://.../wiki/{space}/{doc_id}/{block}   → DocStore
uwu://.../wiki/{space}/index              → IngestPipeline 维护
```

零 uwu 依赖，复用 L7 storage 的 PG/Qdrant 连接。
