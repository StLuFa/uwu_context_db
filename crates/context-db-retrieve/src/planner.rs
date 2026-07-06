//! 查询规划器 — LogicalPlan → CBO 优化 → PhysicalPlan。
//!
//! 核心流程：
//! 1. 查询 DSL/自然语言 → LogicalPlan（关系代数表示）
//! 2. CBO 优化器枚举候选 PhysicalPlan，估算成本，选最优
//! 3. PhysicalPlan 交由物理算子执行

use crate::query::{Condition, MergeStrategy, Predicate, RelationExpand, SortKey};
use agent_context_db_core::{ContentLevel, ContentType, ContextUri};
use std::collections::HashMap;
use std::sync::RwLock;

// ===========================================================================
// 逻辑计划
// ===========================================================================

/// 逻辑计划 — 与物理执行无关的关系代数表示。
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// 全表扫描。
    Scan {
        scope: Option<ContextUri>,
        level: ContentLevel,
    },
    /// 向量语义搜索。
    VectorSearch {
        collection: String,
        query: Vec<f32>,
        top_k: usize,
    },
    /// 谓词过滤。
    Filter {
        input: Box<LogicalPlan>,
        predicate: Predicate,
    },
    /// 连接。
    Join {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        on: JoinKey,
    },
    /// 排序。
    Sort {
        input: Box<LogicalPlan>,
        key: SortKey,
        desc: bool,
    },
    /// 限制数量。
    Limit {
        input: Box<LogicalPlan>,
        budget: usize,
    },
    /// 图遍历扩展。
    Traverse {
        input: Box<LogicalPlan>,
        edges: Vec<crate::query::RelationKind>,
        max_hops: usize,
    },
    /// 时态扫描。
    TemporalScan {
        uri: ContextUri,
        at: crate::query::AsOfTime,
    },
}

/// 连接键。
#[derive(Debug, Clone)]
pub enum JoinKey {
    Uri,
    Tenant,
    ContentHash,
}

// ===========================================================================
// 物理计划
// ===========================================================================

/// 物理计划 — CBO 优化后的可执行算子树。
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// 按类型前缀扫描（PG WHERE uri LIKE）。
    TypeScan {
        content_type: ContentType,
        scope: Option<ScopeFilter>,
        limit: usize,
    },
    /// PG 前缀扫描。
    PgPrefixScan {
        uri_prefix: String,
        limit: usize,
    },
    /// 向量搜索（带 payload 过滤）。
    VectorSearch {
        embedding: Vec<f32>,
        filter: VectorFilter,
        limit: usize,
    },
    /// 谓词过滤。
    Filter {
        input: Box<PhysicalPlan>,
        predicate: Predicate,
    },
    /// Hash 连接。
    HashJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
    },
    /// 嵌套循环连接。
    NestedLoopJoin {
        left: Box<PhysicalPlan>,
        right: Box<PhysicalPlan>,
    },
    /// 排序。
    Sort {
        input: Box<PhysicalPlan>,
        key: SortKey,
    },
    /// 限制。
    Limit {
        input: Box<PhysicalPlan>,
        budget: usize,
    },
    /// 图遍历。
    GraphTraverse {
        seeds: Vec<ContextUri>,
        edges: Vec<crate::query::RelationKind>,
        max_hops: usize,
    },
    /// 并行执行（多计划 + 合并）。
    Parallel {
        plans: Vec<PhysicalPlan>,
        merge: MergeStrategy,
    },
    /// 全表扫描（fallback）。
    FullScan {
        scope: Option<ScopeFilter>,
        limit: usize,
    },
}

/// 作用域过滤。
#[derive(Debug, Clone)]
pub enum ScopeFilter {
    Agent(String),
    Tenant(String),
    UriPrefix(String),
}

/// 向量搜索的 payload 过滤条件。
#[derive(Debug, Clone, Default)]
pub struct VectorFilter {
    pub uri_prefix: Option<String>,
    pub content_type: Option<ContentType>,
    pub only_valid: bool,
}

// ===========================================================================
// 统计信息收集器
// ===========================================================================

/// 统计信息收集器 — 为 CBO 提供代价估算数据。
pub struct StatisticsCollector {
    /// 每个 scope 的条目数。
    row_counts: RwLock<HashMap<String, usize>>,
    /// 每种 ContentType 的条目数。
    type_counts: RwLock<HashMap<ContentType, usize>>,
    /// 每个 scope 的平均深度。
    avg_depth: RwLock<HashMap<String, usize>>,
    /// 向量索引选择性。
    vector_selectivity: RwLock<f64>,
}

impl StatisticsCollector {
    pub fn new() -> Self {
        Self {
            row_counts: RwLock::new(HashMap::new()),
            type_counts: RwLock::new(HashMap::new()),
            avg_depth: RwLock::new(HashMap::new()),
            vector_selectivity: RwLock::new(0.1),
        }
    }

    /// 更新统计信息（Sleeptime 或写入后调用）。
    pub fn update_row_count(&self, scope: &str, count: usize) {
        if let Ok(mut m) = self.row_counts.write() {
            m.insert(scope.to_string(), count);
        }
    }

    pub fn update_type_count(&self, ct: ContentType, count: usize) {
        if let Ok(mut m) = self.type_counts.write() {
            m.insert(ct, count);
        }
    }

    /// 估算按类型过滤后的行数。
    pub fn estimate_rows_by_type(&self, ct: &ContentType) -> usize {
        self.type_counts
            .read()
            .ok()
            .and_then(|m| m.get(ct).copied())
            .unwrap_or(100)
    }

    /// 估算 scope 内的行数。
    pub fn estimate_rows_in_scope(&self, scope: &str) -> usize {
        self.row_counts
            .read()
            .ok()
            .and_then(|m| m.get(scope).copied())
            .unwrap_or(1000)
    }

    /// 向量索引选择性。
    pub fn vector_selectivity(&self) -> f64 {
        *self.vector_selectivity.read().unwrap()
    }
}

impl Default for StatisticsCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// CBO 优化器
// ===========================================================================

/// CBO 优化器 — 基于统计信息选择最优物理计划。
pub struct CboOptimizer {
    stats: std::sync::Arc<StatisticsCollector>,
}

impl CboOptimizer {
    pub fn new(stats: std::sync::Arc<StatisticsCollector>) -> Self {
        Self { stats }
    }

    /// 逻辑计划 → 物理计划（经 CBO 优化）。
    pub fn optimize(&self, logical: &LogicalPlan) -> PhysicalPlan {
        match logical {
            LogicalPlan::Scan { scope, level } => {
                let limit = 1000;
                // 快路径：如果有 scope，用前缀扫描
                if let Some(uri) = scope {
                    PhysicalPlan::PgPrefixScan {
                        uri_prefix: uri.to_string(),
                        limit,
                    }
                } else {
                    PhysicalPlan::FullScan {
                        scope: None,
                        limit,
                    }
                }
            }

            LogicalPlan::VectorSearch {
                collection: _,
                query,
                top_k,
            } => PhysicalPlan::VectorSearch {
                embedding: query.clone(),
                filter: VectorFilter::default(),
                limit: *top_k,
            },

            LogicalPlan::Filter { input, predicate } => {
                // 优化：如果谓词只含 TypeEquals，下推到 TypeScan
                if let Some(ct) = predicate_type_only(predicate) {
                    let scope = extract_scope(input);
                    return PhysicalPlan::TypeScan {
                        content_type: ct,
                        scope,
                        limit: 100,
                    };
                }
                // 否则：先执行 input，再 filter
                let inner = self.optimize(input);
                PhysicalPlan::Filter {
                    input: Box::new(inner),
                    predicate: predicate.clone(),
                }
            }

            LogicalPlan::Sort { input, key, desc } => {
                let inner = self.optimize(input);
                PhysicalPlan::Sort {
                    input: Box::new(inner),
                    key: *key,
                }
            }

            LogicalPlan::Limit { input, budget } => {
                let inner = self.optimize(input);
                PhysicalPlan::Limit {
                    input: Box::new(inner),
                    budget: *budget,
                }
            }

            LogicalPlan::Traverse {
                input,
                edges,
                max_hops,
            } => {
                let seeds = extract_uris(input);
                PhysicalPlan::GraphTraverse {
                    seeds,
                    edges: edges.clone(),
                    max_hops: *max_hops,
                }
            }

            LogicalPlan::TemporalScan { uri, at } => {
                // 时态查询 → PgPrefixScan（URI + 版本查询参数）
                PhysicalPlan::PgPrefixScan {
                    uri_prefix: uri.to_string(),
                    limit: 10,
                }
            }

            LogicalPlan::Join { left, right, on: _ } => {
                let l = self.optimize(left);
                let r = self.optimize(right);
                PhysicalPlan::HashJoin {
                    left: Box::new(l),
                    right: Box::new(r),
                }
            }
        }
    }

    /// 估算物理计划的执行成本。
    pub fn estimate_cost(&self, plan: &PhysicalPlan) -> f64 {
        match plan {
            PhysicalPlan::TypeScan {
                content_type, limit, ..
            } => {
                let rows = self.stats.estimate_rows_by_type(content_type);
                ((*limit).min(rows) as f64) * 0.001 // PG WHERE 前缀扫描，很便宜
            }
            PhysicalPlan::PgPrefixScan { limit, .. } => (*limit as f64) * 0.002,
            PhysicalPlan::VectorSearch { limit, .. } => {
                (*limit as f64) * 0.01 // 向量搜索，中等成本
            }
            PhysicalPlan::GraphTraverse {
                max_hops, edges, ..
            } => {
                // 图遍历成本随 hops 指数增长
                10.0 * (edges.len() as f64) * 2_f64.powi(*max_hops as i32)
            }
            PhysicalPlan::Filter { input, .. } => self.estimate_cost(input) * 1.1,
            PhysicalPlan::Sort { input, .. } => self.estimate_cost(input) * 1.5,
            PhysicalPlan::Limit { input, .. } => self.estimate_cost(input),
            PhysicalPlan::HashJoin { left, right } => {
                self.estimate_cost(left) + self.estimate_cost(right) + 5.0
            }
            PhysicalPlan::NestedLoopJoin { left, right } => {
                self.estimate_cost(left) * self.estimate_cost(right)
            }
            PhysicalPlan::Parallel { plans, .. } => {
                // 并行执行：取最大单计划成本
                plans
                    .iter()
                    .map(|p| self.estimate_cost(p))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0)
            }
            PhysicalPlan::FullScan { limit, .. } => {
                let rows = 1000_f64; // 默认估算
                ((*limit).min(rows as usize) as f64) * 0.05
            }
        }
    }
}

// ===========================================================================
// 辅助函数
// ===========================================================================

/// 如果谓词只包含 TypeEquals，返回该类型。
fn predicate_type_only(predicate: &Predicate) -> Option<ContentType> {
    if predicate.conditions.len() == 1 {
        if let Condition::TypeEquals(ct) = &predicate.conditions[0] {
            return Some(*ct);
        }
    }
    None
}

/// 从逻辑计划中提取 scope。
fn extract_scope(plan: &LogicalPlan) -> Option<ScopeFilter> {
    match plan {
        LogicalPlan::Scan { scope, .. } => scope.as_ref().map(|u| {
            ScopeFilter::UriPrefix(u.to_string())
        }),
        _ => None,
    }
}

/// 从逻辑计划中提取 URI 列表（用于图遍历的 seeds）。
fn extract_uris(plan: &LogicalPlan) -> Vec<ContextUri> {
    match plan {
        LogicalPlan::Scan { scope, .. } => scope.clone().into_iter().collect(),
        _ => vec![],
    }
}
