//! 查询规划器 — LogicalPlan → CBO 优化 → PhysicalPlan。
//!
//! 核心流程：
//! 1. 查询 DSL/自然语言 → LogicalPlan（关系代数表示）
//! 2. CBO 优化器枚举候选 PhysicalPlan，估算成本，选最优
//! 3. PhysicalPlan 交由物理算子执行

use crate::intent::{IntentDecision, IntentExecutionNodeKind, IntentRoute};
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
    PgPrefixScan { uri_prefix: String, limit: usize },
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

#[derive(Debug, Clone)]
pub struct IntentPlannerHint {
    pub route: IntentRoute,
    pub prefer_exact: bool,
    pub prefer_vector: bool,
    pub prefer_graph: bool,
    pub prefer_temporal: bool,
    pub max_graph_depth: usize,
    pub top_k_multiplier: f32,
}

impl From<&IntentDecision> for IntentPlannerHint {
    fn from(decision: &IntentDecision) -> Self {
        let mut hint = Self {
            route: decision.route,
            prefer_exact: false,
            prefer_vector: false,
            prefer_graph: false,
            prefer_temporal: false,
            max_graph_depth: decision.execution_graph.budget.max_graph_depth,
            top_k_multiplier: decision
                .candidates
                .first()
                .map(|candidate| candidate.breakdown.final_score.max(0.1))
                .unwrap_or(1.0),
        };
        for node in &decision.execution_graph.nodes {
            match node.kind {
                IntentExecutionNodeKind::ExactLookup => hint.prefer_exact = true,
                IntentExecutionNodeKind::VectorRetrieve => hint.prefer_vector = true,
                IntentExecutionNodeKind::GraphTraversal => hint.prefer_graph = true,
                IntentExecutionNodeKind::TemporalReplay => hint.prefer_temporal = true,
                IntentExecutionNodeKind::KnowledgeNetworkProbe
                | IntentExecutionNodeKind::KnowledgeNetworkFetch => {}
            }
        }
        hint
    }
}

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
        self.optimize_inner(logical, None)
    }

    /// 带 intent execution graph hint 的优化入口。
    pub fn optimize_with_intent(
        &self,
        logical: &LogicalPlan,
        hint: Option<&IntentPlannerHint>,
    ) -> PhysicalPlan {
        self.optimize_inner(logical, hint)
    }

    fn optimize_inner(&self, logical: &LogicalPlan, hint: Option<&IntentPlannerHint>) -> PhysicalPlan {
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
                    PhysicalPlan::FullScan { scope: None, limit }
                }
            }

            LogicalPlan::VectorSearch {
                collection: _,
                query,
                top_k,
            } => {
                let multiplier = hint
                    .filter(|hint| hint.prefer_vector || matches!(hint.route, IntentRoute::LocalVector | IntentRoute::LocalHybrid))
                    .map(|hint| hint.top_k_multiplier.clamp(1.0, 3.0))
                    .unwrap_or(1.0);
                PhysicalPlan::VectorSearch {
                    embedding: query.clone(),
                    filter: VectorFilter::default(),
                    limit: ((*top_k as f32) * multiplier).ceil() as usize,
                }
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
                let inner = self.optimize_inner(input, hint);
                PhysicalPlan::Filter {
                    input: Box::new(inner),
                    predicate: predicate.clone(),
                }
            }

            LogicalPlan::Sort { input, key, desc } => {
                let inner = self.optimize_inner(input, hint);
                PhysicalPlan::Sort {
                    input: Box::new(inner),
                    key: *key,
                }
            }

            LogicalPlan::Limit { input, budget } => {
                let inner = self.optimize_inner(input, hint);
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
                let max_hops = hint
                    .filter(|hint| hint.prefer_graph || matches!(hint.route, IntentRoute::GraphTraversal))
                    .map(|hint| (*max_hops).min(hint.max_graph_depth.max(1)))
                    .unwrap_or(*max_hops);
                PhysicalPlan::GraphTraverse {
                    seeds,
                    edges: edges.clone(),
                    max_hops,
                }
            }

            LogicalPlan::TemporalScan { uri, at } => {
                // 时态查询 → PgPrefixScan（URI + 版本查询参数）
                let limit = if hint
                    .map(|hint| hint.prefer_temporal || matches!(hint.route, IntentRoute::TemporalIndex))
                    .unwrap_or(false)
                {
                    25
                } else {
                    10
                };
                PhysicalPlan::PgPrefixScan {
                    uri_prefix: uri.to_string(),
                    limit,
                }
            }

            LogicalPlan::Join { left, right, on: _ } => {
                let l = self.optimize_inner(left, hint);
                let r = self.optimize_inner(right, hint);
                PhysicalPlan::HashJoin {
                    left: Box::new(l),
                    right: Box::new(r),
                }
            }
        }
    }

    /// 估算物理计划的执行成本。
    pub fn estimate_cost(&self, plan: &PhysicalPlan) -> f64 {
        self.estimate_cost_with_intent(plan, None)
    }

    /// 带 intent hint 的成本估算：execution graph 偏好的算子会降低成本，冲突算子会升高成本。
    pub fn estimate_cost_with_intent(
        &self,
        plan: &PhysicalPlan,
        hint: Option<&IntentPlannerHint>,
    ) -> f64 {
        let base = match plan {
            PhysicalPlan::TypeScan {
                content_type,
                limit,
                ..
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
            PhysicalPlan::Filter { input, .. } => self.estimate_cost_with_intent(input, hint) * 1.1,
            PhysicalPlan::Sort { input, .. } => self.estimate_cost_with_intent(input, hint) * 1.5,
            PhysicalPlan::Limit { input, .. } => self.estimate_cost_with_intent(input, hint),
            PhysicalPlan::HashJoin { left, right } => {
                self.estimate_cost_with_intent(left, hint)
                    + self.estimate_cost_with_intent(right, hint)
                    + 5.0
            }
            PhysicalPlan::NestedLoopJoin { left, right } => {
                self.estimate_cost_with_intent(left, hint)
                    * self.estimate_cost_with_intent(right, hint)
            }
            PhysicalPlan::Parallel { plans, .. } => {
                // 并行执行：取最大单计划成本
                plans
                    .iter()
                    .map(|p| self.estimate_cost_with_intent(p, hint))
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap_or(0.0)
            }
            PhysicalPlan::FullScan { limit, .. } => {
                let rows = 1000_f64; // 默认估算
                ((*limit).min(rows as usize) as f64) * 0.05
            }
        };
        base * intent_cost_multiplier(plan, hint)
    }
}

// ===========================================================================
// 辅助函数
// ===========================================================================

fn intent_cost_multiplier(plan: &PhysicalPlan, hint: Option<&IntentPlannerHint>) -> f64 {
    let Some(hint) = hint else {
        return 1.0;
    };
    match plan {
        PhysicalPlan::TypeScan { .. } | PhysicalPlan::PgPrefixScan { .. } => {
            if hint.prefer_exact
                || matches!(hint.route, IntentRoute::LocalExact | IntentRoute::TemporalIndex)
            {
                0.75
            } else {
                1.0
            }
        }
        PhysicalPlan::VectorSearch { .. } => {
            if hint.prefer_vector
                || matches!(hint.route, IntentRoute::LocalVector | IntentRoute::LocalHybrid)
            {
                0.8
            } else {
                1.1
            }
        }
        PhysicalPlan::GraphTraverse { .. } => {
            if hint.prefer_graph || matches!(hint.route, IntentRoute::GraphTraversal) {
                0.7
            } else {
                1.25
            }
        }
        _ => 1.0,
    }
}

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
        LogicalPlan::Scan { scope, .. } => scope
            .as_ref()
            .map(|u| ScopeFilter::UriPrefix(u.to_string())),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::{
        IntentCandidate, IntentDecision, IntentExecutionBudget, IntentExecutionGraph,
        IntentExecutionNode, IntentExecutionNodeKind, IntentExplanation, IntentKind,
        IntentPolicyRef, IntentScoreBreakdown,
    };
    use crate::query::RelationKind;

    #[test]
    fn intent_hint_limits_graph_depth() {
        let optimizer = CboOptimizer::new(std::sync::Arc::new(StatisticsCollector::new()));
        let logical = LogicalPlan::Traverse {
            input: Box::new(LogicalPlan::Scan {
                scope: Some(ContextUri::parse("uwu://u/agent/a/memories").unwrap()),
                level: ContentLevel::L0,
            }),
            edges: vec![RelationKind::DerivedFrom],
            max_hops: 5,
        };
        let decision = IntentDecision {
            primary: IntentKind::PatternMatch,
            secondary: Vec::new(),
            route: IntentRoute::GraphTraversal,
            confidence: 0.9,
            ambiguity: 0.0,
            candidates: vec![IntentCandidate {
                intent: IntentKind::PatternMatch,
                route: IntentRoute::GraphTraversal,
                score: 0.9,
                priority: 10,
                matched_terms: vec!["pattern".into()],
                matched_patterns: Vec::new(),
                breakdown: IntentScoreBreakdown {
                    final_score: 0.9,
                    ..Default::default()
                },
            }],
            execution_graph: IntentExecutionGraph {
                nodes: vec![IntentExecutionNode {
                    id: "graph".into(),
                    kind: IntentExecutionNodeKind::GraphTraversal,
                    route: IntentRoute::GraphTraversal,
                }],
                edges: Vec::new(),
                budget: IntentExecutionBudget {
                    max_graph_depth: 2,
                    ..Default::default()
                },
            },
            explanation: IntentExplanation {
                policy_pack: "test".into(),
                policy_version: "1".into(),
                matched_rule_ids: Vec::new(),
                notes: Vec::new(),
            },
            policy: IntentPolicyRef {
                id: "test".into(),
                version: "1".into(),
                engine_version: 1,
            },
        };
        let hint = IntentPlannerHint::from(&decision);
        let physical = optimizer.optimize_with_intent(&logical, Some(&hint));
        match physical {
            PhysicalPlan::GraphTraverse { max_hops, .. } => assert_eq!(max_hops, 2),
            other => panic!("unexpected plan: {other:?}"),
        }
    }
}
