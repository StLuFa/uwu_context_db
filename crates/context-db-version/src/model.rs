//! 版本 DAG 模型（M2）：Commit / Branch / Tag。见 ARCHITECTURE.md §1.2-1.4。

use agent_context_db_core::ContextUri;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 内容寻址哈希（类 Git SHA，blake3）。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ContentHash(pub String);

/// 版本号（替代 M0 的 MvccVersion）。
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CommitId(pub Uuid);

impl CommitId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for CommitId {
    fn default() -> Self {
        Self::new()
    }
}

/// 提交：版本 DAG 中的一个节点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Commit {
    pub id: CommitId,
    /// DAG：可有多个 parent（merge commit）。
    pub parents: Vec<CommitId>,
    pub tree_hash: ContentHash,
    pub author: Author,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metadata: CommitMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Author {
    pub agent_id: Option<String>,
    pub user_id: Option<String>,
    pub system: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CommitMeta {
    #[serde(default)]
    pub trigger: CommitTrigger,
    #[serde(default)]
    pub changes: ChangeSet,
    #[serde(default)]
    pub provenance: Vec<ProvenanceLink>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub enum CommitTrigger {
    #[default]
    AutoConsolidation,
    SessionCommit { session_id: Uuid, compression_index: u64 },
    AgentWrite { agent_id: String, action: String },
    ForkPromotion { fork_name: String },
    Merge { branches: Vec<BranchName> },
    UserExplicit,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChangeSet {
    pub adds: Vec<ContextUri>,
    pub updates: Vec<UriChange>,
    pub deletes: Vec<ContextUri>,
    pub renames: Vec<RenameOp>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UriChange {
    pub uri: ContextUri,
    pub old_hash: Option<ContentHash>,
    pub new_hash: ContentHash,
    pub diff_summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenameOp {
    pub from: ContextUri,
    pub to: ContextUri,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceLink {
    pub source_uri: ContextUri,
    pub source_commit: CommitId,
    pub relation: ProvenanceRelation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProvenanceRelation {
    DerivedFrom,
    ExtractedFrom,
    MergedFrom,
    ForkedFrom,
    TriggeredBy,
}

// ===========================================================================
// 分支
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    pub name: BranchName,
    pub head: CommitId,
    pub created_from: CommitId,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub branch_type: BranchType,
    pub lifecycle: BranchLifecycle,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BranchName(pub(crate) String); // F.3

impl BranchName {
    pub fn new(name: impl Into<String>) -> Self { Self(name.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl std::fmt::Display for BranchName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BranchType {
    Main,
    StateFork,
    Experiment,
    Collaboration,
    Staging,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BranchLifecycle {
    Active,
    Merged { into: BranchName, at: CommitId },
    Abandoned,
    Archived,
}

// ===========================================================================
// 标签
// ===========================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub name: TagName,
    pub target: CommitId,
    pub tag_type: TagType,
    pub message: String,
    pub created_by: Author,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TagName(pub(crate) String); // F.3

impl TagName {
    pub fn new(name: impl Into<String>) -> Self { Self(name.into()) }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl std::fmt::Display for TagName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", self.0) }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TagType {
    Immutable,
    Mutable,
    Semantic { condition: SemanticCondition },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCondition {
    pub metric: String,
    pub threshold: f32,
    pub window_size: usize,
}

/// 版本引用：可指向 commit/branch/tag。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VersionRef {
    Commit(CommitId),
    Branch(BranchName),
    Tag(TagName),
    Head,
}

// ===========================================================================
// 语义 diff — 替代 TreeDiff（URI 列表）
// ===========================================================================

/// 结构化语义 diff — 机器可读的变更集（替代 TreeDiff）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredDiff {
    pub entity_changes: Vec<EntityChange>,
    pub relation_changes: Vec<RelationChange>,
    pub fact_corrections: Vec<FactCorrection>,
    pub confidence_delta: f32,
    pub summary: String,
}

/// 实体属性变更。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityChange {
    pub entity_uri: ContextUri,
    pub field: String,
    pub old_value: Option<serde_json::Value>,
    pub new_value: Option<serde_json::Value>,
    pub change_type: ChangeType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    Set,
    Remove,
    ArrayAppend,
    ArrayRemove,
}

/// 关系变更。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelationChange {
    pub relation: RelationKind,
    pub from: ContextUri,
    pub to: ContextUri,
    pub change: RelationChangeType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationChangeType {
    Added,
    Removed,
    Strengthened,
    Weakened,
}

/// 关系类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationKind {
    EvolvedFrom,
    EvolvedTo,
    EvidenceOf,
    EntangledWith,
    Contradicts,
    Corroborates,
    DerivedFrom,
    Supersedes,
}

/// 事实修正。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactCorrection {
    pub fact_uri: ContextUri,
    pub old_claim: String,
    pub new_claim: String,
    pub correction_type: CorrectionType,
    pub evidence: Vec<ContextUri>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrectionType {
    Refinement,
    Contradiction,
    Update,
}

// ===========================================================================
// 知识图谱合并 — 替代 MergeStrategy 文件级合并
// ===========================================================================

/// 知识图谱合并策略。
pub enum KnowledgeMergeStrategy {
    EntityAutoMerge,
    GraphMerge { edge_policy: EdgePolicy },
    ContradictionDetection { threshold: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgePolicy {
    AutoMerge,
    RequireConsensus,
    ManualOnly,
}

/// 冲突解决器 trait。
pub trait ConflictResolver: Send + Sync {
    fn resolve(&self, conflict: SemanticConflict) -> Resolution;
}

/// 语义冲突。
#[derive(Debug, Clone)]
pub enum SemanticConflict {
    ContradictoryFact { uri: ContextUri, a: String, b: String },
    ConflictingRelation { from: ContextUri, to: ContextUri, a: RelationKind, b: RelationKind },
    OverlappingEntity { a: ContextUri, b: ContextUri, similarity: f32 },
}

/// 冲突解决。
#[derive(Debug, Clone)]
pub enum Resolution {
    KeepBoth { reason: String },
    PreferA { reason: String },
    PreferB { reason: String },
    Fuse { merged: serde_json::Value, reason: String },
    DeferToHuman { reason: String },
}

// ===========================================================================
// 时态版本 — TemporalIndex
// ===========================================================================

/// 时态版本条目。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalVersion {
    pub commit_id: CommitId,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub content_hash: ContentHash,
    pub valid_from: chrono::DateTime<chrono::Utc>,
    pub valid_until: Option<chrono::DateTime<chrono::Utc>>,
}

/// 时态索引 — URI → 有序版本时间线，O(log n) 查询。
#[derive(Debug, Clone, Default)]
pub struct TemporalIndex {
    timelines: std::collections::HashMap<String, Vec<TemporalVersion>>,
}

impl TemporalIndex {
    pub fn new() -> Self {
        Self { timelines: std::collections::HashMap::new() }
    }

    /// 注册一个版本。
    pub fn record(&mut self, uri: &ContextUri, version: TemporalVersion) {
        let key = uri.to_string();
        let timeline = self.timelines.entry(key).or_default();
        // 有序插入（按 timestamp）
        let pos = timeline.binary_search_by(|v| v.timestamp.cmp(&version.timestamp))
            .unwrap_or_else(|e| e);
        timeline.insert(pos, version);
    }

    /// AS OF 查询：某时间点的版本。二分查找 O(log n)。
    pub fn as_of(&self, uri: &ContextUri, at: chrono::DateTime<chrono::Utc>) -> Option<&TemporalVersion> {
        let timeline = self.timelines.get(&uri.to_string())?;
        let pos = timeline
            .binary_search_by(|v| v.timestamp.cmp(&at))
            .unwrap_or_else(|e| e.saturating_sub(1));
        timeline.get(pos)
    }

    /// BETWEEN 查询：时间范围内的版本。
    pub fn between(
        &self,
        uri: &ContextUri,
        from: chrono::DateTime<chrono::Utc>,
        to: chrono::DateTime<chrono::Utc>,
    ) -> Vec<&TemporalVersion> {
        let timeline = match self.timelines.get(&uri.to_string()) {
            Some(t) => t,
            None => return vec![],
        };
        timeline
            .iter()
            .filter(|v| v.timestamp >= from && v.timestamp <= to)
            .collect()
    }

    /// EVOLUTION OF：完整演化历史。
    pub fn evolution(&self, uri: &ContextUri) -> Vec<&TemporalVersion> {
        self.timelines
            .get(&uri.to_string())
            .map(|t| t.iter().collect())
            .unwrap_or_default()
    }
}
