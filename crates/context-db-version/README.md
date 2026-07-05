# agent-context-db-version

类 Git DAG 版本管理系统 + CRDT 合并 + 版本推理。

## 模块

| 模块 | 内容 |
|------|------|
| `model` | Commit DAG + Branch(Main/StateFork/Experiment/Collaboration/Staging) + Tag(Immutable/Mutable/Semantic) + ChangeSet |
| `crdt_merge` | `CrdtMerger`(LwwMap + SetUnion) — 多 Agent 并发写入的零冲突合并 |
| `reasoning` | `DiffReasoner`(F24 语义差异推理) + `TemporalReasoner`(F30 时态推理) |
| `innovation` | `CrystalDistiller`(F19 知识晶体) + `SelfHealer`(F21 自修复) + `DreamConsolidator`(F23 梦境巩固) + `CausalInference`(F27 因果推断) |

## VersionStore trait

```rust
pub trait VersionStore: Send + Sync {
    // 提交
    async fn commit(&self, scope, changes, meta) -> Result<CommitId>;
    // 分支
    async fn create_branch(&self, scope, name, from, bt) -> Result<Branch>;
    async fn list_branches(&self, scope) -> Result<Vec<Branch>>;
    async fn delete_branch(&self, scope, name) -> Result<()>;
    async fn switch_head(&self, scope, branch) -> Result<()>;
    // 标签
    async fn create_tag(&self, scope, tag) -> Result<()>;
    async fn list_tags(&self, scope) -> Result<Vec<Tag>>;
    async fn evaluate_semantic_tags(&self, scope) -> Result<Vec<(TagName, CommitId)>>;
    // 读取/时间旅行
    async fn log(&self, scope, opts) -> Result<Vec<Commit>>;
    async fn read_at(&self, uri, ref_, level) -> Result<ContentPayload>;
    async fn asof_read(&self, uri, when, level) -> Result<ContentPayload>;
    // 合并/Diff
    async fn merge(&self, scope, from, into, strategy) -> Result<MergeResult>;
    async fn diff_commits(&self, scope, a, b) -> Result<TreeDiff>;
    // 高级操作
    async fn cherry_pick(&self, scope, commit, onto) -> Result<CommitId>;
    async fn rebase(&self, scope, branch, onto) -> Result<Vec<CommitId>>;
    async fn squash(&self, scope, commits, message) -> Result<SquashResult>;
    async fn gc(&self, scope, policy) -> Result<GcReport>;
    // 因果分析
    async fn provenance(&self, uri) -> Result<ProvenanceGraph>;
    async fn impact_analysis(&self, commit) -> Result<ImpactAnalysis>;
}
```

## 实现

`MemoryVersionStore`(testkit) — 完整的 DAG 内存实现，支持全部 17 个 trait 方法。
