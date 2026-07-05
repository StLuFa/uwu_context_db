//! 核心数据模型（M0）：三层信息模型 + 记忆分类 + 内容载荷。

use crate::uri::ContextUri;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// 单调递增版本号（M0 用；M2 起版本管理迁移到 `agent-context-db-version` 的 CommitId）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MvccVersion(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub Uuid);

/// AGFS 内容 blob 引用（L2 原始内容指针）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentRef(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    Text,
    Image,
    Audio,
    Video,
    Binary,
}

/// 三层内容级别：L0 摘要 / L1 概览 / L2 原始详情。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ContentLevel {
    #[default]
    L0,
    L1,
    L2,
}

/// 读取内容的载荷。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPayload {
    /// L0：~100 tokens 摘要。
    Abstract(String),
    /// L1：~2k tokens 概览。
    Overview(String),
    /// L2：原始字节（多模态）。
    Detail(Vec<u8>),
}

/// 8 种记忆分类。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryClass {
    // user 类
    Profile,
    Preferences,
    Entities,
    Events,
    // agent 类
    Cases,
    Patterns,
    Tools,
    Skills,
}

impl MemoryClass {
    /// 是否可与既有条目合并（vs 追加新条目）。
    pub fn mergeable(&self) -> bool {
        matches!(
            self,
            Self::Profile
                | Self::Preferences
                | Self::Entities
                | Self::Patterns
                | Self::Tools
                | Self::Skills
        )
    }
}

/// State 作用域（短/中/长程 WS）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateScope {
    Short,
    Mid,
    Long,
}

/// 条目元数据。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextMeta {
    pub memory_class: Option<MemoryClass>,
    pub state_scope: Option<StateScope>,
    pub tags: Vec<String>,
    #[serde(default)]
    pub custom: serde_json::Value,
}

/// 三层信息模型条目。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextEntry {
    pub uri: ContextUri,
    pub tenant: TenantId,
    /// L0：~100 tokens Markdown 摘要。
    pub l0_abstract: String,
    /// L1：~2k tokens 概览（含章节导航）。
    pub l1_overview: Option<String>,
    /// L2：原始内容指针（多模态）。
    pub l2_detail_uri: Option<ContentRef>,
    pub content_type: ContentType,
    pub metadata: ContextMeta,
    pub mvcc_version: MvccVersion,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

impl ContextEntry {
    /// 构造一个最小 L0 文本条目（便于测试 / 快速写入）。
    pub fn new_text(uri: ContextUri, tenant: TenantId, abstract_: impl Into<String>) -> Self {
        let now = chrono::Utc::now();
        Self {
            uri,
            tenant,
            l0_abstract: abstract_.into(),
            l1_overview: None,
            l2_detail_uri: None,
            content_type: ContentType::Text,
            metadata: ContextMeta::default(),
            mvcc_version: MvccVersion(0),
            created_at: now,
            updated_at: now,
        }
    }
}

// ===========================================================================
// FS 寻址返回类型
// ===========================================================================

/// 目录项（`ls` 返回）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirEntry {
    pub uri: ContextUri,
    pub is_dir: bool,
    /// L0 摘要（目录/文件都有）。
    pub abstract_: String,
    /// 记忆分类（从条目元数据中获取）。
    pub memory_class: Option<MemoryClass>,
}

/// `find` 匹配模式。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FindPattern {
    /// 起点目录。
    pub scope: Option<ContextUri>,
    /// 名称 glob（如 `*.md`）。
    pub name_glob: Option<String>,
    /// 按分类过滤。
    pub class: Option<MemoryClass>,
    /// 最大递归深度。
    pub max_depth: Option<usize>,
}

/// `grep` 命中。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepHit {
    pub uri: ContextUri,
    pub line: String,
    pub level: ContentLevel,
}

/// `tree` 返回的节点。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeNode {
    pub uri: ContextUri,
    pub is_dir: bool,
    pub children: Vec<TreeNode>,
}

/// 版本历史条目（M0 占位；M2 由 version crate 扩展）。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionEntry {
    pub version: MvccVersion,
    pub message: String,
    pub ts: chrono::DateTime<chrono::Utc>,
}

/// 两版本间差异（M0 占位）。
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ContextDiff {
    pub summary: String,
}
