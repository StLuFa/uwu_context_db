//! # agent-context-db-version (M2 版本层)
//!
//! 类 Git 的 DAG 版本模型：Commit / Branch / Tag + `VersionStore` 端口。
//!
//! ## 解耦约束
//!
//! - 独立 crate，通过 feature 开关；关闭时 M0/M1 仍可编译。
//! - 依赖 core 的类型（`ContextUri` / `ContentLevel` / `ContentPayload`），
//!   不反向依赖检索层或 uwu 扩展。
//! - `VersionStore` 是端口（零实现）；线性快照 / DAG 后端由宿主注入。

pub mod belief_revision;
pub mod crdt_merge;
pub mod innovation;
pub mod model;
pub mod reasoning;

pub use belief_revision::{
    AgmBeliefReviser, AgmRevisionConfig, BeliefConflict, BeliefLiteral, BeliefPolarity,
    BeliefPredicate, BeliefRevisionAction, BeliefRevisionDecision, BeliefSentence,
    revision_to_resolution,
};
pub use crdt_merge::{CrdtMergeResult, CrdtMerger, CrdtStrategy};
pub use innovation::{
    CausalDiscoveryConfig, CausalEdge, CausalGraph, CausalHypothesis, CausalInference,
    CausalIntervention, CounterfactualImpact, CrystalDistiller, DreamConsolidator,
    InterventionResult, KnowledgeCrystal, RepairAction, SelfHealer,
};
pub use model::{
    Author, Branch, BranchLifecycle, BranchName, BranchType, CausalDag, ChangeSet, ChangeType,
    Commit, CommitId, CommitMeta, CommitTrigger, ConflictResolver, CorrectionType, EntityChange,
    FactCorrection, KnowledgeMergeStrategy, ProvenanceLink, ProvenanceRelation, RelationChange,
    RelationChangeType, RelationKind, RenameOp, Resolution, SemanticCondition, SemanticConflict,
    StructuredDiff, Tag, TagName, TagType, TemporalIndex, TemporalVersion, UriChange, VersionRef,
};
pub use reasoning::{
    ChangeCategory, DiffChangeType, DiffImpact, DiffReasoner, SemanticChange, SemanticDiff,
    TemporalPattern, TemporalReasoner, TimelineEvent,
};

pub use agent_context_db_core::ContentHash;

use agent_context_db_core::{ContentLevel, ContentPayload, ContextUri};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum VersionError {
    #[error("not found: {0}")]
    NotFound(String),
    #[error("branch exists: {0}")]
    BranchExists(String),
    #[error("merge conflict: {0}")]
    MergeConflict(String),
    #[error("conflict session not available: {0}")]
    ConflictSessionUnavailable(String),
    #[error("conflict session incomplete: {0}")]
    ConflictSessionIncomplete(String),
    #[error("storage: {0}")]
    Storage(String),
}

pub type Result<T> = std::result::Result<T, VersionError>;

/// 时间旅行读取的时间点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AsOfTime {
    Timestamp(chrono::DateTime<chrono::Utc>),
    Commit(CommitId),
}

#[derive(Debug, Clone, Default)]
pub struct LogOpts {
    pub max_count: Option<usize>,
    pub branch: Option<BranchName>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TreeDiff {
    pub adds: Vec<ContextUri>,
    pub updates: Vec<ContextUri>,
    pub deletes: Vec<ContextUri>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// 快进（无分叉）。
    FastForward,
    /// 三方合并。
    ThreeWay,
    /// 冲突时优先目标分支。
    Ours,
    /// 冲突时优先来源分支。
    Theirs,
}

/// cherry_pick / rebase 遇冲突时的解决策略。
///
/// - `Fail`：报 `MergeConflict` 错误并停止（默认，与 Git 语义一致）。
/// - `Ours`：保留 target 分支的当前值（放弃 cherry commit 对冲突 URI 的修改）。
/// - `Theirs`：采用 cherry commit 的新值（覆盖 target 独立修改）。
///
/// 无冲突的 URI 在任何策略下都按 delta 正常应用。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictStrategy {
    #[default]
    Fail,
    Ours,
    Theirs,
}

/// 交互式冲突 session 的持久化策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConflictSessionPersistence {
    /// 只把 session 返回给调用方，不写入后端。
    #[default]
    Disabled,
    /// 后端同时保存 session，使调用方可用 session id 恢复/继续/中止。
    Enabled,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ConflictSessionId(pub uuid::Uuid);

impl ConflictSessionId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4())
    }
}

impl Default for ConflictSessionId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ConflictSessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// 需要人工/Agent 分步处理的版本改写操作。
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InteractiveOperation {
    CherryPick {
        commit: CommitId,
        onto: BranchName,
    },
    Rebase {
        branch: BranchName,
        onto: BranchName,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictValueOp {
    Set,
    Delete,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictItem {
    pub uri: ContextUri,
    pub base: Option<ContentPayload>,
    pub ours: Option<ContentPayload>,
    pub theirs: Option<ContentPayload>,
    pub op: ConflictValueOp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictSession {
    pub id: ConflictSessionId,
    pub scope: ContextUri,
    pub operation: InteractiveOperation,
    pub base: Option<CommitId>,
    pub target: CommitId,
    pub commits: Vec<CommitId>,
    pub conflicts: Vec<ConflictItem>,
    pub clean_snapshot: Vec<(String, ContentPayload)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConflictResolution {
    Ours,
    Theirs,
    Delete,
    Manual(ContentPayload),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConflictResolutionSet {
    pub resolutions: std::collections::BTreeMap<String, ConflictResolution>,
}

impl ConflictResolutionSet {
    pub fn insert(&mut self, uri: &ContextUri, resolution: ConflictResolution) {
        self.resolutions.insert(uri.to_string(), resolution);
    }

    pub fn get(&self, uri: &ContextUri) -> Option<&ConflictResolution> {
        self.resolutions.get(&uri.to_string())
    }
}

#[async_trait]
pub trait InteractiveVersionStore: Send + Sync {
    async fn begin_cherry_pick(
        &self,
        scope: &ContextUri,
        commit: &CommitId,
        onto: &BranchName,
        persistence: ConflictSessionPersistence,
    ) -> Result<ConflictSession>;

    async fn begin_rebase(
        &self,
        scope: &ContextUri,
        branch: &BranchName,
        onto: &BranchName,
        persistence: ConflictSessionPersistence,
    ) -> Result<ConflictSession>;

    async fn continue_conflict_session(
        &self,
        session: ConflictSession,
        resolutions: ConflictResolutionSet,
    ) -> Result<Vec<CommitId>>;

    async fn load_conflict_session(&self, id: &ConflictSessionId) -> Result<ConflictSession>;

    async fn continue_conflict_session_by_id(
        &self,
        id: &ConflictSessionId,
        resolutions: ConflictResolutionSet,
    ) -> Result<Vec<CommitId>>;

    async fn abort_conflict_session(&self, id: &ConflictSessionId) -> Result<()>;
}

#[derive(Debug, Clone)]
pub struct MergeResult {
    pub commit: CommitId,
    pub conflicts: Vec<ContextUri>,
}

/// Squash 结果。
#[derive(Debug, Clone)]
pub struct SquashResult {
    pub new_commit: CommitId,
    pub squashed_count: usize,
}

/// GC 策略。
#[derive(Debug, Clone)]
pub struct GcPolicy {
    /// 保留最近 N 个 commit
    pub keep_recent: usize,
    /// 超过此天数的清理
    pub max_age_days: i64,
}

/// GC 报告。
#[derive(Debug, Clone, Default)]
pub struct GcReport {
    pub removed_commits: usize,
    pub freed_snapshots: usize,
}

/// 因果链图。
#[derive(Debug, Clone)]
pub struct ProvenanceGraph {
    pub root_uri: ContextUri,
    pub nodes: Vec<crate::model::ProvenanceLink>,
    pub depth: usize,
}

/// 影响分析结果。
#[derive(Debug, Clone)]
pub struct ImpactAnalysis {
    pub commit: CommitId,
    pub downstream_uris: Vec<ContextUri>,
    pub affected_branches: Vec<BranchName>,
}

/// 版本存储端口（M2）。见 ARCHITECTURE.md §1.6（此为骨架子集，聚焦交付验收面）。
#[async_trait]
pub trait VersionStore: Send + Sync {
    // === 提交 ===
    async fn commit(
        &self,
        scope: &ContextUri,
        changes: ChangeSet,
        meta: CommitMeta,
    ) -> Result<CommitId>;

    // === 分支（含 State fork 沙盒）===
    async fn create_branch(
        &self,
        scope: &ContextUri,
        name: BranchName,
        from: CommitId,
        bt: BranchType,
    ) -> Result<Branch>;
    async fn list_branches(&self, scope: &ContextUri) -> Result<Vec<Branch>>;
    async fn delete_branch(&self, scope: &ContextUri, name: &BranchName) -> Result<()>;

    // === 标签 ===
    async fn create_tag(&self, scope: &ContextUri, tag: Tag) -> Result<()>;
    async fn list_tags(&self, scope: &ContextUri) -> Result<Vec<Tag>>;

    // === 读取 / 时间旅行 ===
    async fn log(&self, scope: &ContextUri, opts: &LogOpts) -> Result<Vec<Commit>>;
    async fn read_at(
        &self,
        uri: &ContextUri,
        ref_: VersionRef,
        level: ContentLevel,
    ) -> Result<ContentPayload>;
    async fn asof_read(
        &self,
        uri: &ContextUri,
        when: AsOfTime,
        level: ContentLevel,
    ) -> Result<ContentPayload>;

    // === 合并 / Diff ===
    async fn merge(
        &self,
        scope: &ContextUri,
        from: &BranchName,
        into: &BranchName,
        strategy: MergeStrategy,
    ) -> Result<MergeResult>;
    async fn diff_commits(
        &self,
        scope: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> Result<TreeDiff>;

    // === 高级分支操作 ===
    async fn switch_head(&self, scope: &ContextUri, branch: &BranchName) -> Result<()>;
    async fn cherry_pick(
        &self,
        scope: &ContextUri,
        commit: &CommitId,
        onto: &BranchName,
        strategy: ConflictStrategy,
    ) -> Result<CommitId>;

    // === 历史改写 ===
    async fn rebase(
        &self,
        scope: &ContextUri,
        branch: &BranchName,
        onto: &BranchName,
        strategy: ConflictStrategy,
    ) -> Result<Vec<CommitId>>;
    async fn squash(
        &self,
        scope: &ContextUri,
        commits: Vec<CommitId>,
        message: &str,
    ) -> Result<SquashResult>;

    // === 生命周期 ===
    async fn gc(&self, scope: &ContextUri, policy: &GcPolicy) -> Result<GcReport>;

    // === 语义标签 ===
    async fn evaluate_semantic_tags(
        &self,
        scope: &ContextUri,
    ) -> Result<Vec<(crate::model::TagName, CommitId)>>;

    // === 因果分析 ===
    /// 返回 URI 的**显式证据溯源**（CommitMeta.provenance 中写入方主动声明的链接）。
    ///
    /// 语义边界（重要）：
    /// - 本方法**只读**每个修改此 URI 的 commit 的 `CommitMeta.provenance` 字段；
    ///   不沿 commit parent DAG 回溯祖先。
    /// - 显式证据 = 写入方在 commit 时声明的"这条知识来自 session/URI/外部证据 X"。
    /// - 若需要"版本祖先链"（这个 commit 的父提交是谁）请用
    ///   [`CausalDag`](crate::model::CausalDag) / [`impact_analysis`] / [`evolution`]。
    ///
    /// 两者故意保持正交：显式证据是**认知层**溯源，parent DAG 是**版本层**溯源，
    /// 合并会丢失"哪些是显式声明的、哪些是推导出来的"这一区分。
    async fn provenance(&self, uri: &ContextUri) -> Result<ProvenanceGraph>;
    async fn impact_analysis(&self, commit: &CommitId) -> Result<ImpactAnalysis>;

    // === 语义 diff（新增 — 替代 diff_commits 的 TreeDiff） ===
    async fn semantic_diff(
        &self,
        scope: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> Result<crate::model::StructuredDiff>;

    // === 时态演化（新增） ===
    async fn evolution(&self, uri: &ContextUri) -> Result<Vec<crate::model::TemporalVersion>>;

    // === 知识图谱合并（新增 — 替代 merge with MergeStrategy） ===
    async fn knowledge_merge(
        &self,
        scope: &ContextUri,
        from: &BranchName,
        into: &BranchName,
        strategy: crate::model::KnowledgeMergeStrategy,
    ) -> Result<MergeResult>;
}

// ===========================================================================
// G.3: VersionStore 拆为 5 个窄 trait + blanket impl
// ===========================================================================

/// Commit 操作（提交 + 日志）。
#[async_trait]
pub trait CommitOps: Send + Sync {
    async fn commit(
        &self,
        scope: &ContextUri,
        changes: ChangeSet,
        meta: CommitMeta,
    ) -> Result<CommitId>;
    async fn log(&self, scope: &ContextUri, opts: &LogOpts) -> Result<Vec<Commit>>;
}

/// 分支操作（CRUD + switch）。
#[async_trait]
pub trait BranchOps: Send + Sync {
    async fn create_branch(
        &self,
        scope: &ContextUri,
        name: BranchName,
        from: CommitId,
        bt: BranchType,
    ) -> Result<Branch>;
    async fn list_branches(&self, scope: &ContextUri) -> Result<Vec<Branch>>;
    async fn delete_branch(&self, scope: &ContextUri, name: &BranchName) -> Result<()>;
    async fn switch_head(&self, scope: &ContextUri, branch: &BranchName) -> Result<()>;
}

/// 标签操作。
#[async_trait]
pub trait TagOps: Send + Sync {
    async fn create_tag(&self, scope: &ContextUri, tag: Tag) -> Result<()>;
    async fn list_tags(&self, scope: &ContextUri) -> Result<Vec<Tag>>;
}

/// 合并/改写操作。
#[async_trait]
pub trait MergeOps: Send + Sync {
    async fn merge(
        &self,
        scope: &ContextUri,
        from: &BranchName,
        into: &BranchName,
        strategy: MergeStrategy,
    ) -> Result<MergeResult>;
    async fn diff_commits(
        &self,
        scope: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> Result<TreeDiff>;
    async fn cherry_pick(
        &self,
        scope: &ContextUri,
        commit: &CommitId,
        onto: &BranchName,
        strategy: ConflictStrategy,
    ) -> Result<CommitId>;
    async fn rebase(
        &self,
        scope: &ContextUri,
        branch: &BranchName,
        onto: &BranchName,
        strategy: ConflictStrategy,
    ) -> Result<Vec<CommitId>>;
    async fn squash(
        &self,
        scope: &ContextUri,
        commits: Vec<CommitId>,
        message: &str,
    ) -> Result<SquashResult>;
    async fn knowledge_merge(
        &self,
        scope: &ContextUri,
        from: &BranchName,
        into: &BranchName,
        strategy: crate::model::KnowledgeMergeStrategy,
    ) -> Result<MergeResult>;
}

/// 历史/分析操作（读取 + 时间旅行 + 血缘）。
#[async_trait]
pub trait HistoryOps: Send + Sync {
    async fn read_at(
        &self,
        uri: &ContextUri,
        ref_: VersionRef,
        level: ContentLevel,
    ) -> Result<ContentPayload>;
    async fn asof_read(
        &self,
        uri: &ContextUri,
        when: AsOfTime,
        level: ContentLevel,
    ) -> Result<ContentPayload>;
    async fn provenance(&self, uri: &ContextUri) -> Result<ProvenanceGraph>;
    async fn impact_analysis(&self, commit: &CommitId) -> Result<ImpactAnalysis>;
    async fn evolution(&self, uri: &ContextUri) -> Result<Vec<crate::model::TemporalVersion>>;
    async fn semantic_diff(
        &self,
        scope: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> Result<crate::model::StructuredDiff>;
    async fn gc(&self, scope: &ContextUri, policy: &GcPolicy) -> Result<GcReport>;
}

// Blanket impl: 任何实现 VersionStore 的自动实现全部 5 个窄 trait
#[async_trait]
impl<T: VersionStore + Send + Sync> CommitOps for T {
    async fn commit(&self, s: &ContextUri, c: ChangeSet, m: CommitMeta) -> Result<CommitId> {
        VersionStore::commit(self, s, c, m).await
    }
    async fn log(&self, s: &ContextUri, o: &LogOpts) -> Result<Vec<Commit>> {
        VersionStore::log(self, s, o).await
    }
}
#[async_trait]
impl<T: VersionStore + Send + Sync> BranchOps for T {
    async fn create_branch(
        &self,
        s: &ContextUri,
        n: BranchName,
        f: CommitId,
        bt: BranchType,
    ) -> Result<Branch> {
        VersionStore::create_branch(self, s, n, f, bt).await
    }
    async fn list_branches(&self, s: &ContextUri) -> Result<Vec<Branch>> {
        VersionStore::list_branches(self, s).await
    }
    async fn delete_branch(&self, s: &ContextUri, n: &BranchName) -> Result<()> {
        VersionStore::delete_branch(self, s, n).await
    }
    async fn switch_head(&self, s: &ContextUri, b: &BranchName) -> Result<()> {
        VersionStore::switch_head(self, s, b).await
    }
}
#[async_trait]
impl<T: VersionStore + Send + Sync> TagOps for T {
    async fn create_tag(&self, s: &ContextUri, t: Tag) -> Result<()> {
        VersionStore::create_tag(self, s, t).await
    }
    async fn list_tags(&self, s: &ContextUri) -> Result<Vec<Tag>> {
        VersionStore::list_tags(self, s).await
    }
}
#[async_trait]
impl<T: VersionStore + Send + Sync> MergeOps for T {
    async fn merge(
        &self,
        s: &ContextUri,
        f: &BranchName,
        i: &BranchName,
        st: MergeStrategy,
    ) -> Result<MergeResult> {
        VersionStore::merge(self, s, f, i, st).await
    }
    async fn diff_commits(&self, s: &ContextUri, a: &CommitId, b: &CommitId) -> Result<TreeDiff> {
        VersionStore::diff_commits(self, s, a, b).await
    }
    async fn cherry_pick(
        &self,
        s: &ContextUri,
        c: &CommitId,
        o: &BranchName,
        st: ConflictStrategy,
    ) -> Result<CommitId> {
        VersionStore::cherry_pick(self, s, c, o, st).await
    }
    async fn rebase(
        &self,
        s: &ContextUri,
        b: &BranchName,
        o: &BranchName,
        st: ConflictStrategy,
    ) -> Result<Vec<CommitId>> {
        VersionStore::rebase(self, s, b, o, st).await
    }
    async fn squash(&self, s: &ContextUri, cs: Vec<CommitId>, m: &str) -> Result<SquashResult> {
        VersionStore::squash(self, s, cs, m).await
    }
    async fn knowledge_merge(
        &self,
        s: &ContextUri,
        f: &BranchName,
        i: &BranchName,
        st: crate::model::KnowledgeMergeStrategy,
    ) -> Result<MergeResult> {
        VersionStore::knowledge_merge(self, s, f, i, st).await
    }
}
#[async_trait]
impl<T: VersionStore + Send + Sync> HistoryOps for T {
    async fn read_at(
        &self,
        u: &ContextUri,
        r: VersionRef,
        l: ContentLevel,
    ) -> Result<ContentPayload> {
        VersionStore::read_at(self, u, r, l).await
    }
    async fn asof_read(
        &self,
        u: &ContextUri,
        w: AsOfTime,
        l: ContentLevel,
    ) -> Result<ContentPayload> {
        VersionStore::asof_read(self, u, w, l).await
    }
    async fn provenance(&self, u: &ContextUri) -> Result<ProvenanceGraph> {
        VersionStore::provenance(self, u).await
    }
    async fn impact_analysis(&self, c: &CommitId) -> Result<ImpactAnalysis> {
        VersionStore::impact_analysis(self, c).await
    }
    async fn evolution(&self, u: &ContextUri) -> Result<Vec<crate::model::TemporalVersion>> {
        VersionStore::evolution(self, u).await
    }
    async fn semantic_diff(
        &self,
        s: &ContextUri,
        a: &CommitId,
        b: &CommitId,
    ) -> Result<crate::model::StructuredDiff> {
        VersionStore::semantic_diff(self, s, a, b).await
    }
    async fn gc(&self, s: &ContextUri, p: &GcPolicy) -> Result<GcReport> {
        VersionStore::gc(self, s, p).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_dag_parents() {
        let root = CommitId::new();
        let merge = Commit {
            id: CommitId::new(),
            parents: vec![root.clone(), CommitId::new()],
            tree_hash: ContentHash("abc".into()),
            author: Author {
                agent_id: Some("a1".into()),
                user_id: None,
                system: false,
            },
            message: "merge".into(),
            timestamp: chrono::Utc::now(),
            metadata: CommitMeta {
                trigger: CommitTrigger::Merge {
                    branches: vec![BranchName::new("main")],
                },
                changes: ChangeSet::default(),
                provenance: vec![],
            },
        };
        assert_eq!(merge.parents.len(), 2);
        assert_eq!(merge.parents[0], root);
    }
}
