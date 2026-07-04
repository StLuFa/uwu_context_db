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
