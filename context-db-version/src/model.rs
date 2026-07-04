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
pub struct BranchName(pub String);

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
pub struct TagName(pub String);

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
