//! 查询 DSL（WQL — uwu 查询语言）。
//!
//! 声明式上下文查询，编译为 LogicalPlan → CBO 优化 → PhysicalPlan 执行。

use agent_context_db_core::{ContentLevel, ContentType, ContextUri};

// ===========================================================================
// 查询 AST
// ===========================================================================

/// uwu 查询语言 AST。
#[derive(Debug, Clone)]
pub enum Query {
    /// 关键词/路径查找。
    /// ```text
    /// FIND memories WHERE type = fact WITH budget = 4000
    /// ```
    Find {
        scope: Option<ContextUri>,
        predicate: Predicate,
        budget: usize,
        order: SortKey,
        expand: Option<RelationExpand>,
    },
    /// 语义相似查找。
    Similar {
        query_embedding: Vec<f32>,
        predicate: Predicate,
        budget: usize,
        expand: Option<RelationExpand>,
    },
    /// 时态查询（AS OF / BETWEEN）。
    AsOf {
        uri: ContextUri,
        at: AsOfTime,
        level: ContentLevel,
    },
    /// 图遍历查询。
    Traverse {
        start: ContextUri,
        edges: Vec<RelationKind>,
        max_hops: usize,
        predicate: Predicate,
    },
    /// 组合查询（多查询并行 + 合并策略）。
    Composite {
        queries: Vec<Query>,
        merge: MergeStrategy,
    },
}

// ===========================================================================
// 谓词（WHERE 子句）
// ===========================================================================

/// 谓词 — WHERE 条件的集合。
#[derive(Debug, Clone, Default)]
pub struct Predicate {
    pub conditions: Vec<Condition>,
}

impl Predicate {
    pub fn new() -> Self {
        Self { conditions: vec![] }
    }

    pub fn with(mut self, condition: Condition) -> Self {
        self.conditions.push(condition);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.conditions.is_empty()
    }
}

/// 查询条件。
#[derive(Debug, Clone)]
pub enum Condition {
    TypeEquals(ContentType),
    ScopeEquals(Scope),
    TimeBetween(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>),
    TagsContains(Vec<String>),
    QualityAbove(f32),
    ValidOnly,
}

/// 查询作用域。
#[derive(Debug, Clone)]
pub enum Scope {
    Agent(String),
    Tenant(String),
    All,
}

// ===========================================================================
// 排序 + 分页
// ===========================================================================

/// 排序键。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortKey {
    Relevance,
    Recency,
    Quality,
    Natural,
}

// ===========================================================================
// 关系扩展（图遍历）
// ===========================================================================

/// 关系扩展 — 沿关系图 N 跳扩展。
#[derive(Debug, Clone)]
pub struct RelationExpand {
    pub kinds: Vec<RelationKind>,
    pub max_hops: usize,
}

/// 关系类型（用于图遍历查询）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationKind {
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

// ===========================================================================
// 时态查询
// ===========================================================================

/// 时态查询时间点。
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AsOfTime {
    Commit(String),
    Timestamp(chrono::DateTime<chrono::Utc>),
    Latest,
}

// ===========================================================================
// 合并策略
// ===========================================================================

/// 多查询结果合并策略。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MergeStrategy {
    /// 取并集，按 relevance 排序。
    Union,
    /// 取交集。
    Intersect,
    /// 取第一个非空结果。
    First,
    /// 去重合并，保留最高分。
    Dedup,
}
