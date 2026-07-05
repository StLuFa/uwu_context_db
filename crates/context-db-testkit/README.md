# agent-context-db-testkit

存储端口内存实现 — 仅测试/开发。

## 组件

### MemoryContextStore

同时实现 `FsOps` + `ContentRepo` + `VersionOps` + `TenantOps` 四个窄端口。

```rust
let store = MemoryContextStore::new();
store.write(entry).await?;
let entries = store.ls(&dir).await?;
let payload = store.read(&uri, ContentLevel::L0).await?;
```

### MemoryVersionStore

完整 DAG 版本存储内存实现，支持全部 17 个 `VersionStore` trait 方法：

```rust
let store = MemoryVersionStore::new();
let c1 = store.commit(&scope, changes, meta).await?;
let branch = store.create_branch(&scope, name, c1, BranchType::StateFork).await?;
store.merge(&scope, &fork, &main, MergeStrategy::FastForward).await?;
store.cherry_pick(&scope, &commit, &branch).await?;
store.squash(&scope, commits, "merged").await?;
let payload = store.asof_read(&uri, AsOfTime::Commit(c1), ContentLevel::L0).await?;
```

- DAG 模型：每个 commit 存储完整内容快照
- `commit_on_parent` 支持并行推演场景
- `is_ancestor` 沿 parent 链回溯验证合并条件

## 依赖

`context-db-core` / `context-db-version` / `parking_lot` / `serde_json`。
