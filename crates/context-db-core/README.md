# agent-context-db-core

Agent 上下文数据库最小内核，零 uwu 依赖，可独立发布。

## 模块

| 模块 | 内容                                                                                  |
|------|-------------------------------------------------------------------------------------|
| `uri` | `uwu://` URI 强类型寻址 + `UriCategory` 分类                                               |
| `model` | 三层信息模型(L0/L1/L2) + 多种记忆分类 + ContextEntry/DirEntry/TreeNode                          |
| `store` | 四个窄端口：`FsOps`(只读寻址) + `ContentRepo`(写入) + `VersionOps`(版本) + `TenantOps`(租户)        |
| `llm` | `LlmClient` 端口(complete/embed/complete_json/stream/batch/speculative) + `LlmStream` |
| `vector` | `VectorIndex` 端口(upsert/search/delete) + IndexPoint/IndexHit                        |
| `lifecycle` | F22 遗忘曲线 + F26 Token 预算经济模型                                                         |
| `pack` | F7 ContextPack 导出导入 + F8 路径级 ACL                                                    |
| `observability` | F9 订阅推送 + F13 质量评分 + F15 血缘图                                                        |
| `event` | F5 事件流因果链 + F11 上下文继承 + F12 上下文模板                                                   |
| `similarity` | F14 跨 Agent 去重与相似度聚类                                                                |
| `error` | `ContextError` 统一错误类型                                                               |

## 窄端口

```rust
pub trait FsOps: Send + Sync {      // 检索层唯一依赖
    async fn ls(&self, dir) -> Result<Vec<DirEntry>>;
    async fn find(&self, pattern) -> Result<Vec<ContextUri>>;
    async fn grep(&self, regex, scope) -> Result<Vec<GrepHit>>;
    async fn tree(&self, root, depth) -> Result<TreeNode>;
    async fn read(&self, uri, level) -> Result<ContentPayload>;
}
pub trait ContentRepo: Send + Sync { // M0 必需
    async fn write(&self, entry) -> Result<MvccVersion>;
    async fn delete(&self, uri) -> Result<()>;
    async fn rename(&self, from, to) -> Result<()>;
}
pub trait VersionOps: Send + Sync { // M2 独立 crate
    async fn version_history(&self, uri) -> Result<Vec<VersionEntry>>;
    async fn rollback(&self, uri, to) -> Result<()>;
    async fn diff(&self, uri, a, b) -> Result<ContextDiff>;
}
pub trait TenantOps: Send + Sync {
    async fn list_tenants(&self) -> Result<Vec<TenantId>>;
}
```

## LlmClient 端口

```rust
pub trait LlmClient: Send + Sync {
    async fn complete(&self, prompt, opts) -> Result<String>;
    async fn embed(&self, text) -> Result<Vec<f32>>;
    async fn complete_json(&self, prompt, schema, opts) -> Result<String>;
    async fn stream_complete(&self, prompt, opts) -> Result<Box<dyn LlmStream>>;
    async fn batch_complete(&self, prompts, opts) -> Result<Vec<String>>;
    async fn speculative_complete(&self, prompt, opts) -> Result<String>;
}
```

## 实现

- `context-db-testkit`：`MemoryContextStore`（四个窄端口的内存实现）
- `context-db-storage`：`PgContextStore`（基于 `uwu_database::DbPool`）
- `context-db-storage`：`UwuVectorIndex`（适配 `uwu_database::VectorStore`）
