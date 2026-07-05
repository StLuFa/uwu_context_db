//! `ContextRetriever`：计划驱动检索器 + `HierarchicalRetrieverImpl`（向后兼容）。
//!
//! 新版流程：Query DSL → LogicalPlan → CBO 优化 → PhysicalPlan → 执行。
//! 旧版流程（保留）：意图分析 → 目录定位 → 目录内搜索 → 递归深入 → Rerank → 按预算加载。

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContentType, ContextUri, FsOps, LlmClient, Result, VectorIndex,
};
use async_trait::async_trait;
use std::sync::Arc;

use crate::intent::{LlmIntentAnalyzer, RuleBasedIntentAnalyzer};
use crate::operators::ExecContext;
use crate::planner::{CboOptimizer, LogicalPlan, PhysicalPlan, StatisticsCollector};
use crate::query::{Condition, Predicate, Query};
use crate::{
    HierarchicalRetriever, IntentAnalyzer, PlanRetriever, QueryPlanner, Reranker, RetrievalHit,
    RetrievalResult, RetrievalTrace, RetrieveContext, TraceStep, TypedQuery,
};

// ===========================================================================
// ContextRetriever — 计划驱动检索器
// ===========================================================================

/// 计划驱动检索器：Query DSL → Plan → Execute。
pub struct ContextRetriever {
    fs: Arc<dyn FsOps>,
    index: Option<Arc<dyn VectorIndex>>,
    planner: Arc<dyn QueryPlanner>,
    reranker: Arc<dyn Reranker>,
    optimizer: CboOptimizer,
    /// 计划缓存（相同 query 不重复优化）。
    plan_cache: parking_lot::Mutex<std::collections::HashMap<Vec<u8>, PhysicalPlan>>,
}

impl ContextRetriever {
    pub fn new(
        fs: Arc<dyn FsOps>,
        index: Option<Arc<dyn VectorIndex>>,
        planner: Arc<dyn QueryPlanner>,
        reranker: Arc<dyn Reranker>,
    ) -> Self {
        let stats = Arc::new(StatisticsCollector::new());
        Self {
            fs,
            index,
            planner,
            reranker,
            optimizer: CboOptimizer::new(stats),
            plan_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// G.1: Builder 入口。
    pub fn builder(fs: Arc<dyn FsOps>) -> ContextRetrieverBuilder {
        ContextRetrieverBuilder::new(fs)
    }

    /// 自然语言查询 → 检索结果。
    pub async fn retrieve(
        &self,
        query: &str,
        ctx: &RetrieveContext,
    ) -> Result<RetrievalResult> {
        let budget = ctx.budget_tokens.unwrap_or(8000);
        let mut trace = RetrievalTrace::default();

        // 1. 自然语言 → LogicalPlan
        let logical = self.planner.parse(query, ctx).await?;

        // 2. CBO 优化 → PhysicalPlan
        let physical = self.optimizer.optimize(&logical);
        trace.steps.push(TraceStep::PlanOptimized {
            logical: format!("{logical:?}"),
            physical: format!("{physical:?}"),
        });

        // 3. 执行物理计划
        let exec_ctx = ExecContext {
            fs: self.fs.clone(),
            index: self.index.clone(),
        };
        let batch = physical.execute(&exec_ctx).await?;
        trace.steps.push(TraceStep::Execute {
            plan: "physical".into(),
            hits: batch.records.len(),
            duration_ms: batch.stats.duration.as_millis() as u64,
        });

        // 4. Rerank
        let reranked = self.reranker.rerank(query, batch.records).await?;
        trace.steps.push(TraceStep::Rerank {
            input: batch.stats.rows_scanned,
            kept: reranked.len(),
            model: "score".into(),
        });

        // 5. Budget loading
        let (hits, tokens_used) = load_within_budget(reranked, budget);

        Ok(RetrievalResult {
            hits,
            trace,
            tokens_used,
        })
    }

    /// 结构化 Query → 检索结果。
    pub async fn retrieve_query(
        &self,
        query: &Query,
        ctx: &RetrieveContext,
    ) -> Result<RetrievalResult> {
        let budget = match query {
            Query::Find { budget, .. } => *budget,
            Query::Similar { budget, .. } => *budget,
            _ => ctx.budget_tokens.unwrap_or(8000),
        };
        let mut trace = RetrievalTrace::default();

        let logical = query_to_logical(query);
        let physical = self.optimizer.optimize(&logical);
        trace.steps.push(TraceStep::PlanOptimized {
            logical: format!("{logical:?}"),
            physical: format!("{physical:?}"),
        });

        let exec_ctx = ExecContext {
            fs: self.fs.clone(),
            index: self.index.clone(),
        };
        let batch = physical.execute(&exec_ctx).await?;
        trace.steps.push(TraceStep::Execute {
            plan: "physical".into(),
            hits: batch.records.len(),
            duration_ms: batch.stats.duration.as_millis() as u64,
        });

        let reranked = self.reranker.rerank("", batch.records).await?;
        let (hits, tokens_used) = load_within_budget(reranked, budget);

        Ok(RetrievalResult {
            hits,
            trace,
            tokens_used,
        })
    }
}

#[async_trait]
impl PlanRetriever for ContextRetriever {
    async fn retrieve_plan(
        &self,
        query: &Query,
        ctx: &RetrieveContext,
    ) -> Result<RetrievalResult> {
        self.retrieve_query(query, ctx).await
    }
}

#[async_trait]
impl HierarchicalRetriever for ContextRetriever {
    async fn retrieve(&self, query: &str, ctx: &RetrieveContext) -> Result<RetrievalResult> {
        self.retrieve(query, ctx).await
    }
}

// ===========================================================================
// Query → LogicalPlan 转换
// ===========================================================================

fn query_to_logical(query: &Query) -> LogicalPlan {
    match query {
        Query::Find {
            scope,
            predicate,
            budget,
            expand,
            ..
        } => {
            let scan = LogicalPlan::Scan {
                scope: scope.clone(),
                level: ContentLevel::L0,
            };
            let plan = if predicate.is_empty() {
                scan
            } else {
                LogicalPlan::Filter {
                    input: Box::new(scan),
                    predicate: predicate.clone(),
                }
            };
            let plan = LogicalPlan::Limit {
                input: Box::new(plan),
                budget: *budget,
            };
            if let Some(exp) = expand {
                LogicalPlan::Traverse {
                    input: Box::new(plan),
                    edges: exp.kinds.clone(),
                    max_hops: exp.max_hops,
                }
            } else {
                plan
            }
        }
        Query::Similar {
            query_embedding,
            predicate,
            budget,
            expand,
        } => {
            let vs = LogicalPlan::VectorSearch {
                collection: "memories".into(),
                query: query_embedding.clone(),
                top_k: 50,
            };
            let plan = if predicate.is_empty() {
                vs
            } else {
                LogicalPlan::Filter {
                    input: Box::new(vs),
                    predicate: predicate.clone(),
                }
            };
            let plan = LogicalPlan::Limit {
                input: Box::new(plan),
                budget: *budget,
            };
            if let Some(exp) = expand {
                LogicalPlan::Traverse {
                    input: Box::new(plan),
                    edges: exp.kinds.clone(),
                    max_hops: exp.max_hops,
                }
            } else {
                plan
            }
        }
        Query::AsOf { uri, at, level } => LogicalPlan::TemporalScan {
            uri: uri.clone(),
            at: at.clone(),
        },
        Query::Traverse {
            start,
            edges,
            max_hops,
            predicate,
        } => {
            let scan = LogicalPlan::Scan {
                scope: Some(start.clone()),
                level: ContentLevel::L0,
            };
            LogicalPlan::Traverse {
                input: Box::new(scan),
                edges: edges.clone(),
                max_hops: *max_hops,
            }
        },
        Query::Composite { queries, merge: _ } => {
            // 组合查询 → 多个 LogicalPlan 并行
            // 简化：合并所有子查询的 scan scopes
            let first = queries.first().map(|q| query_to_logical(q));
            first.unwrap_or(LogicalPlan::Scan {
                scope: None,
                level: ContentLevel::L0,
            })
        }
    }
}

// ===========================================================================
// Budget loading
// ===========================================================================

fn load_within_budget(
    mut hits: Vec<RetrievalHit>,
    budget: usize,
) -> (Vec<RetrievalHit>, usize) {
    let mut tokens = 0;
    let mut result = Vec::new();
    hits.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
    for hit in hits {
        let cost = estimate_tokens(&hit.content);
        if tokens + cost <= budget {
            tokens += cost;
            result.push(hit);
        } else {
            // 降级到 L0
            let l0_text = hit.content.sparse_text().to_string();
            let l0_cost = l0_text.len() / 4;
            if tokens + l0_cost <= budget {
                tokens += l0_cost;
                result.push(RetrievalHit {
                    level: ContentLevel::L0,
                    content: ContentPayload::Text {
                        sparse: l0_text.clone(),
                        dense: l0_text.clone(),
                        full: l0_text,
                    },
                    ..hit
                });
            }
        }
    }
    (result, tokens)
}

fn estimate_tokens(content: &ContentPayload) -> usize {
    match content {
        ContentPayload::Text { sparse, dense, full: _ } => {
            std::cmp::max(sparse.len() / 4, 100)
        }
        _ => 100,
    }
}

// ===========================================================================
// 基于规则的 QueryPlanner 实现
// ===========================================================================

/// 基于规则的查询规划器 — 从自然语言生成 LogicalPlan。
pub struct RuleBasedPlanner {
    default_tenant: String,
    default_agent: String,
}

impl RuleBasedPlanner {
    pub fn new(
        default_tenant: impl Into<String>,
        default_agent: impl Into<String>,
    ) -> Self {
        Self {
            default_tenant: default_tenant.into(),
            default_agent: default_agent.into(),
        }
    }
}

#[async_trait]
impl QueryPlanner for RuleBasedPlanner {
    async fn parse(
        &self,
        query: &str,
        ctx: &RetrieveContext,
    ) -> Result<LogicalPlan> {
        let lower = query.to_lowercase();
        let scope = ContextUri::parse(&format!(
            "uwu://{}/agent/{}",
            ctx.user_id.as_deref().unwrap_or(&self.default_tenant),
            ctx.agent_id.as_deref().unwrap_or(&self.default_agent),
        ))?;

        // 简单关键词 → Predicate 映射
        let mut predicate = Predicate::new();
        if lower.contains("fact") || lower.contains("事实") {
            predicate = predicate.with(Condition::TypeEquals(ContentType::Fact));
        }
        if lower.contains("error") || lower.contains("错误") || lower.contains("失败") {
            predicate = predicate.with(Condition::TypeEquals(ContentType::Error));
        }
        predicate = predicate.with(Condition::ValidOnly);

        Ok(LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Scan {
                scope: Some(scope),
                level: ContentLevel::L0,
            }),
            predicate,
        })
    }

    async fn plan(&self, logical: &LogicalPlan) -> Result<PhysicalPlan> {
        let stats = Arc::new(StatisticsCollector::new());
        let optimizer = CboOptimizer::new(stats);
        Ok(optimizer.optimize(logical))
    }
}

// ===========================================================================
// HierarchicalRetrieverImpl（向后兼容的旧版实现，委托给 ContextRetriever）
// ===========================================================================

#[deprecated(note = "使用 ContextRetriever 替代")]
pub struct HierarchicalRetrieverImpl {
    pub fs: Arc<dyn FsOps>,
    pub index: Option<Arc<dyn VectorIndex>>,
    pub llm: Option<Arc<dyn LlmClient>>,
    pub intent: Arc<dyn IntentAnalyzer>,
    pub reranker: Arc<dyn Reranker>,
}

#[allow(deprecated)]
impl HierarchicalRetrieverImpl {
    pub fn new(
        fs: Arc<dyn FsOps>,
        intent: Arc<dyn IntentAnalyzer>,
        reranker: Arc<dyn Reranker>,
    ) -> Self {
        Self {
            fs,
            index: None,
            llm: None,
            intent,
            reranker,
        }
    }

    pub fn with_index(
        fs: Arc<dyn FsOps>,
        index: Arc<dyn VectorIndex>,
        intent: Arc<dyn IntentAnalyzer>,
        reranker: Arc<dyn Reranker>,
    ) -> Self {
        Self {
            fs,
            index: Some(index),
            llm: None,
            intent,
            reranker,
        }
    }
}

#[allow(deprecated)]
#[async_trait]
impl HierarchicalRetriever for HierarchicalRetrieverImpl {
    async fn retrieve(&self, query: &str, _ctx: &RetrieveContext) -> Result<RetrievalResult> {
        // 委托到新的计划驱动方式
        let scope = ContextUri::parse("uwu://t/")?;
        let logical = LogicalPlan::Scan {
            scope: Some(scope),
            level: ContentLevel::L0,
        };
        let stats = Arc::new(StatisticsCollector::new());
        let optimizer = CboOptimizer::new(stats);
        let physical = optimizer.optimize(&logical);

        let exec_ctx = ExecContext {
            fs: self.fs.clone(),
            index: self.index.clone(),
        };
        let batch = physical.execute(&exec_ctx).await?;
        let (hits, tokens) = load_within_budget(batch.records, 8000);
        Ok(RetrievalResult {
            hits,
            trace: RetrievalTrace::default(),
            tokens_used: tokens,
        })
    }
}

// ===========================================================================
// G.1: ContextRetrieverBuilder
// ===========================================================================

/// Builder for ContextRetriever — 替代多构造器模式。
pub struct ContextRetrieverBuilder {
    fs: Arc<dyn FsOps>,
    index: Option<Arc<dyn VectorIndex>>,
    planner: Option<Arc<dyn QueryPlanner>>,
    reranker: Option<Arc<dyn Reranker>>,
}

impl ContextRetrieverBuilder {
    pub fn new(fs: Arc<dyn FsOps>) -> Self {
        Self { fs, index: None, planner: None, reranker: None }
    }

    pub fn with_vector_index(mut self, index: Arc<dyn VectorIndex>) -> Self {
        self.index = Some(index);
        self
    }

    pub fn with_planner(mut self, planner: Arc<dyn QueryPlanner>) -> Self {
        self.planner = Some(planner);
        self
    }

    pub fn with_reranker(mut self, reranker: Arc<dyn Reranker>) -> Self {
        self.reranker = Some(reranker);
        self
    }

    pub fn build(self) -> ContextRetriever {
        let planner = self.planner.unwrap_or_else(|| {
            Arc::new(RuleBasedPlanner::new("default", "default"))
        });
        let reranker = self.reranker.unwrap_or_else(|| {
            Arc::new(crate::ScoreReranker { keep: 20 })
        });
        ContextRetriever::new(self.fs, self.index, planner, reranker)
    }
}
