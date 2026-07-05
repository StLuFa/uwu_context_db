//! 存储端口 —— 按职责拆分的窄 trait（接口隔离原则，见 ARCHITECTURE.md §2.2）。
//!
//! 各上层只依赖它真正用到的窄端口；实现由 composition root 注入。
//! 库内部严禁依赖聚合 [`ContextStore`]，只依赖窄端口。

use crate::error::Result;
use crate::model::{
    ContentLevel, ContentPayload, ContextDiff, ContextEntry, DirEntry, FindPattern, GrepHit,
    MvccVersion, TenantId, TreeNode, VersionEntry,
};
use crate::uri::ContextUri;
use async_trait::async_trait;

/// 端口 1：只读 FS 寻址（检索层唯一依赖此端口）。
#[async_trait]
pub trait FsOps: Send + Sync {
    async fn ls(&self, dir: &ContextUri) -> Result<Vec<DirEntry>>;
    async fn find(&self, pattern: &FindPattern) -> Result<Vec<ContextUri>>;
    async fn grep(&self, regex: &str, scope: &ContextUri) -> Result<Vec<GrepHit>>;
    async fn tree(&self, root: &ContextUri, depth: usize) -> Result<TreeNode>;
    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload>;
}

/// 端口 2：内容写（M0 必需）。
#[async_trait]
pub trait ContentRepo: Send + Sync {
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion>;
    async fn delete(&self, uri: &ContextUri) -> Result<()>;
    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()>;
    /// D.1: 批量写入 — 默认逐条 write，后端可 override 用 UNNEST 优化。
    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>> {
        let mut versions = Vec::with_capacity(entries.len());
        for entry in entries { versions.push(self.write(entry.clone()).await?); }
        Ok(versions)
    }
}

/// 端口 3：版本操作（M2 独立 crate，feature 开关；M0/M1 可用空实现）。
#[async_trait]
pub trait VersionOps: Send + Sync {
    async fn version_history(&self, uri: &ContextUri) -> Result<Vec<VersionEntry>>;
    async fn rollback(&self, uri: &ContextUri, to: MvccVersion) -> Result<()>;
    async fn diff(&self, uri: &ContextUri, a: MvccVersion, b: MvccVersion) -> Result<ContextDiff>;
}

/// 端口 4：租户隔离。
#[async_trait]
pub trait TenantOps: Send + Sync {
    async fn list_tenants(&self) -> Result<Vec<TenantId>>;
}

/// 便利 supertrait 别名：需要"完整存储"能力的调用方用它，
/// 但库内部各层只依赖上面的窄端口。
pub trait ContextStore: FsOps + ContentRepo + VersionOps + TenantOps {}
impl<T: FsOps + ContentRepo + VersionOps + TenantOps> ContextStore for T {}

// ===========================================================================
// 6 域存储端口 — 替代旧 4 trait
// ===========================================================================

/// 内容存储（替代 ContentRepo + FsOps::read）。
#[async_trait]
pub trait ContentStore: Send + Sync {
    async fn read(&self, uri: &ContextUri, level: ContentLevel) -> Result<ContentPayload>;
    async fn write(&self, entry: ContextEntry) -> Result<MvccVersion>;
    async fn delete(&self, uri: &ContextUri) -> Result<()>;
    async fn rename(&self, from: &ContextUri, to: &ContextUri) -> Result<()>;
    async fn batch_write(&self, entries: &[ContextEntry]) -> Result<Vec<MvccVersion>>;
    async fn scan_by_prefix(&self, prefix: &str, limit: usize) -> Result<Vec<ContextEntry>>;
}

/// 目录浏览（从 FsOps 拆分）。
#[async_trait]
pub trait BrowsingOps: Send + Sync {
    async fn ls(&self, dir: &ContextUri) -> Result<Vec<DirEntry>>;
    async fn tree(&self, dir: &ContextUri, depth: usize) -> Result<TreeNode>;
    async fn find(&self, scope: &ContextUri, pattern: &str) -> Result<Vec<ContextUri>>;
    async fn grep(&self, scope: &ContextUri, pattern: &str) -> Result<Vec<GrepHit>>;
}

/// 图存储（新增）。
#[async_trait]
pub trait GraphStore: Send + Sync {
    async fn add_edge(&self, from: &ContextUri, to: &ContextUri, kind: GraphRelation) -> Result<()>;
    async fn remove_edge(&self, from: &ContextUri, to: &ContextUri) -> Result<()>;
    async fn neighbors(&self, uri: &ContextUri, kind: Option<GraphRelation>) -> Result<Vec<ContextUri>>;
    async fn batch_traverse(
        &self,
        seeds: &[ContextUri],
        kinds: &[GraphRelation],
        max_hops: usize,
    ) -> Result<Vec<(ContextUri, ContextUri, GraphRelation)>>;
    async fn centrality(&self, uri: &ContextUri) -> Result<f32>;
}

/// 图关系类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphRelation {
    EvolvedFrom,
    EvolvedTo,
    EvidenceOf,
    EntangledWith,
    Contradicts,
    Corroborates,
    DerivedFrom,
    Supersedes,
    DrivesPolicy,
}

/// Blob 存储（多模态原始载荷）。
#[async_trait]
pub trait BlobStore: Send + Sync {
    async fn put(&self, data: &[u8], mime_type: &str) -> Result<crate::BlobRef>;
    async fn get(&self, blob_ref: &crate::BlobRef) -> Result<Vec<u8>>;
    async fn dedup_check(&self, hash: &crate::ContentHash) -> Result<bool>;
}

// ===========================================================================
// 存储引擎 — 组合全部域端口
// ===========================================================================

/// 存储引擎 — composition root，组合 6 域端口。
pub trait StorageEngine: Send + Sync {
    fn content(&self) -> &dyn ContentStore;
    fn browsing(&self) -> &dyn BrowsingOps;
    fn version(&self) -> &dyn VersionOps;
    fn tenant(&self) -> &dyn TenantOps;
    fn graph(&self) -> Option<&dyn GraphStore>;
    fn blob(&self) -> &dyn BlobStore;
}
