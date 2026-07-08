# agent-context-db-storage

双层存储适配器 + composition root，基于 `uwu_database`。

## 模块

| 模块 | 内容 |
|------|------|
| `pg` | `PgContextStore` — 实现全部四个窄端口(FsOps/ContentRepo/VersionOps/TenantOps) |
| `vector_index` | `UwuVectorIndex` — 适配 `uwu_database::VectorStore` → core `VectorIndex` |
| `migrations` | PG schema migration（context_entries + context_versions 表） |
| `lib` | `ContextDbService` composition root + `service_from_uwu_db` 一键构造 |

## 使用

```rust
let db = uwu_database::Database::connect_with_vector(&cfg).await?;
let service = service_from_uwu_db(db).await?;
let fs: Arc<PgContextStore> = service.fs_ops();
let idx: Arc<dyn VectorIndex> = service.vector_index();
```

## Schema

```sql
context_entries (uri PK, tenant_id, l0_abstract, l1_overview, l2_detail_ref,
                 content_type, state_scope, tags JSONB, custom JSONB,
                 mvcc_version, created_at, updated_at)
context_versions (uri + mvcc_version PK, entry_json JSONB, ...)
```

## 依赖

`context-db-core` / `uwu_database` / `sqlx`。
