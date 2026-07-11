//! RetrieverCompiler — 将查询编译为可执行物理计划，compile 与 execute 两阶段分离。
//!
//! 设计来自 agent_context_db_complete.md §9（性能重构）：
//! > ContextRetriever 拆分为编译器（RetrieverCompiler）+ 执行器（PhysicalPlan::execute）。
//!
//! ## 两阶段职责
//! - **compile**：Query DSL / 自然语言 → LogicalPlan → CBO 优化 → PhysicalPlan（纯计算，无 I/O）
//! - **execute**：PhysicalPlan + ExecContext → RecordBatch（实际 I/O）
//!
//! 这样 PhysicalPlan 可以被缓存（同 query 不重复 compile），也可以被解释打印或做批量优化。
//!
//! ## 用法
//! ```ignore
//! let compiler = RetrieverCompiler::new(planner)?;
//! let plan = compiler.compile_text("给我所有 fact", &ctx)?;
//! let plan2 = compiler.compile_query(&Query::Find { .. }, &ctx)?;
//! let batch = plan.execute(&exec_ctx).await?;
//! ```

use crate::budget::load_hits_within_budget;
use crate::operators::ExecContext;
use crate::planner::{CboOptimizer, LogicalPlan, PhysicalPlan, StatisticsCollector};
use crate::query::{Condition, Predicate, Query};
use crate::{
    QueryPlanner, RetrievalHit, RetrievalResult, RetrievalTrace, RetrieveContext, TraceStep,
    operators::RecordBatch,
};
use agent_context_db_core::{ContentLevel, Result};
use std::sync::Arc;

// ===========================================================================
// CompiledPlan — compile 阶段产物
// ===========================================================================

/// Compile 阶段产物 — 持有物理计划及调试信息，可缓存复用。
#[derive(Debug, Clone)]
pub struct CompiledPlan {
    pub logical: LogicalPlan,
    pub physical: PhysicalPlan,
    /// 估算执行成本（用于上层决策）。
    pub estimated_cost: f64,
}

impl CompiledPlan {
    /// 执行已编译的物理计划。
    pub async fn execute(&self, ctx: &ExecContext) -> Result<RecordBatch> {
        self.physical.execute(ctx).await
    }

    /// 人类可读的计划描述（调试 / explain 用）。
    pub fn explain(&self) -> String {
        format!(
            "LogicalPlan: {:?}\nPhysicalPlan: {:?}\nEstimatedCost: {:.4}",
            self.logical, self.physical, self.estimated_cost
        )
    }
}

// ===========================================================================
// RetrieverCompiler — 纯计算编译器
// ===========================================================================

/// 检索编译器 — 将查询编译为 CompiledPlan，不执行任何 I/O。
pub struct RetrieverCompiler {
    planner: Arc<dyn QueryPlanner>,
    optimizer: CboOptimizer,
}

impl RetrieverCompiler {
<<<<<<< Updated upstream
    pub fn new(planner: Arc<dyn QueryPlanner>) -> Self {
        let stats = Arc::new(StatisticsCollector::new(crate::QueryPlanConfig::default()).unwrap());
        Self {
            planner,
            optimizer: CboOptimizer::new(stats, crate::QueryPlanConfig::default()).unwrap(),
        }
=======
    pub fn new(
        planner: Arc<dyn QueryPlanner>,
    ) -> std::result::Result<Self, crate::RetrieveConfigError> {
        let config = crate::QueryPlanConfig::default();
        let stats = Arc::new(StatisticsCollector::new(config)?);
        Self::with_stats(planner, stats)
>>>>>>> Stashed changes
    }

    /// 使用外部统计信息构建（用于 CBO 代价估算更准确）。
    pub fn with_stats(
        planner: Arc<dyn QueryPlanner>,
        stats: Arc<StatisticsCollector>,
    ) -> std::result::Result<Self, crate::RetrieveConfigError> {
        Ok(Self {
            planner,
<<<<<<< Updated upstream
            optimizer: CboOptimizer::new(stats, crate::QueryPlanConfig::default()).unwrap(),
        }
=======
            optimizer: CboOptimizer::new(stats, crate::QueryPlanConfig::default())?,
        })
>>>>>>> Stashed changes
    }

    /// 编译自然语言查询 → CompiledPlan（异步，需 LLM/规则解析）。
    pub async fn compile_text(&self, query: &str, ctx: &RetrieveContext) -> Result<CompiledPlan> {
        let logical = self.planner.parse(query, ctx).await?;
        Ok(self.compile_logical(logical))
    }

    /// 编译结构化 Query DSL → CompiledPlan（同步）。
    pub fn compile_query(&self, query: &Query, _ctx: &RetrieveContext) -> CompiledPlan {
        let logical = query_to_logical(query);
        self.compile_logical(logical)
    }

    /// 直接从 LogicalPlan 编译（内部共用）。
    pub fn compile_logical(&self, logical: LogicalPlan) -> CompiledPlan {
        let physical = self.optimizer.optimize(&logical);
        let estimated_cost = self.optimizer.estimate_cost(&physical);
        CompiledPlan {
            logical,
            physical,
            estimated_cost,
        }
    }
}

// ===========================================================================
// PlanExecutor — 执行阶段（薄封装，注入依赖）
// ===========================================================================

/// 计划执行器 — 持有 I/O 依赖，执行 CompiledPlan。
///
/// 与 RetrieverCompiler 分离，职责单一：
/// - Compiler：Query → Plan（纯计算）
/// - PlanExecutor：Plan + Deps → Results（实际 I/O）
pub struct PlanExecutor {
    exec_ctx: ExecContext,
    budget_tokens: usize,
}

impl PlanExecutor {
    pub fn new(exec_ctx: ExecContext) -> Self {
        Self {
            exec_ctx,
            budget_tokens: 8000,
        }
    }

    pub fn with_budget(mut self, budget: usize) -> Self {
        self.budget_tokens = budget;
        self
    }

    /// 执行已编译计划，返回检索结果（含 trace）。
    pub async fn execute(
        &self,
        compiled: &CompiledPlan,
        query_hint: &str,
        reranker: &dyn crate::Reranker,
    ) -> Result<RetrievalResult> {
        let mut trace = RetrievalTrace::default();
        trace.steps.push(TraceStep::PlanOptimized {
            logical: format!("{:?}", compiled.logical),
            physical: format!("{:?}", compiled.physical),
        });

        let batch = compiled.execute(&self.exec_ctx).await?;
        trace.steps.push(TraceStep::Execute {
            plan: "compiled".into(),
            hits: batch.records.len(),
            duration_ms: batch.stats.duration.as_millis() as u64,
        });

        let reranked = reranker.rerank(query_hint, batch.records).await?;
        trace.steps.push(TraceStep::Rerank {
            input: batch.stats.rows_scanned,
            kept: reranked.len(),
            model: "score".into(),
        });

        let (hits, tokens_used) = load_within_budget(reranked, self.budget_tokens)?;

        Ok(RetrievalResult {
            hits,
            trace,
            tokens_used,
        })
    }
}

// ===========================================================================
// CompiledRetriever — 组合编译器 + 执行器（高层便利 API）
// ===========================================================================

/// 组合检索器 — Compiler + Executor，供调用方一步调用。
pub struct CompiledRetriever {
    compiler: RetrieverCompiler,
    executor: PlanExecutor,
    reranker: Arc<dyn crate::Reranker>,
    /// 已编译计划缓存（query hash → CompiledPlan）。
    plan_cache: parking_lot::Mutex<std::collections::HashMap<u64, CompiledPlan>>,
}

impl CompiledRetriever {
    pub fn new(
        planner: Arc<dyn QueryPlanner>,
        exec_ctx: ExecContext,
        reranker: Arc<dyn crate::Reranker>,
    ) -> std::result::Result<Self, crate::RetrieveConfigError> {
        Ok(Self {
            compiler: RetrieverCompiler::new(planner)?,
            executor: PlanExecutor::new(exec_ctx),
            reranker,
            plan_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        })
    }

    /// 自然语言查询（带计划缓存）。
    pub async fn retrieve(&self, query: &str, ctx: &RetrieveContext) -> Result<RetrievalResult> {
        let cache_key = simple_hash(query);

        // 尝试从缓存读取已编译计划
        let compiled = {
            let cache = self.plan_cache.lock();
            cache.get(&cache_key).cloned()
        };

        let compiled = match compiled {
            Some(c) => c,
            None => {
                let c = self.compiler.compile_text(query, ctx).await?;
                self.plan_cache.lock().insert(cache_key, c.clone());
                c
            }
        };

        self.executor
            .execute(&compiled, query, self.reranker.as_ref())
            .await
    }

    /// 结构化 Query 查询（同步 compile，无缓存）。
    pub async fn retrieve_query(
        &self,
        query: &Query,
        ctx: &RetrieveContext,
    ) -> Result<RetrievalResult> {
        let compiled = self.compiler.compile_query(query, ctx);
        let hint = query_text_hint(query);
        self.executor
            .execute(&compiled, &hint, self.reranker.as_ref())
            .await
    }
}

// ===========================================================================
// Query → LogicalPlan（与 retriever.rs 共用逻辑，独立维护）
// ===========================================================================

pub fn query_to_logical(query: &Query) -> LogicalPlan {
    match query {
        Query::Find {
            scope,
            predicate,
            budget,
            order,
            expand,
        } => {
            let scan = LogicalPlan::Scan {
                scope: scope.clone(),
                level: ContentLevel::L0,
                limit: Some(scan_budget(predicate, *budget)),
            };
            let plan = if predicate.is_empty() {
                scan
            } else {
                LogicalPlan::Filter {
                    input: Box::new(scan),
                    predicate: predicate.clone(),
                }
            };
            let plan = if *order == crate::query::SortKey::Natural {
                plan
            } else {
                LogicalPlan::Sort {
                    input: Box::new(plan),
                    key: *order,
                    desc: true,
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
                top_k: vector_budget(predicate, *budget),
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
        Query::AsOf { uri, at, .. } => LogicalPlan::TemporalScan {
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
                limit: Some(1),
            };
            let input = if predicate.is_empty() {
                scan
            } else {
                LogicalPlan::Filter {
                    input: Box::new(scan),
                    predicate: predicate.clone(),
                }
            };
            LogicalPlan::Traverse {
                input: Box::new(input),
                edges: edges.clone(),
                max_hops: *max_hops,
            }
        }
        Query::Composite { queries, merge } => LogicalPlan::Parallel {
            plans: queries.iter().map(query_to_logical).collect(),
            merge: *merge,
        },
    }
}

// ===========================================================================
// 工具函数
// ===========================================================================

fn load_within_budget(
    hits: Vec<RetrievalHit>,
    budget: usize,
) -> Result<(Vec<RetrievalHit>, usize)> {
    let plan = load_hits_within_budget(hits, budget, crate::TokenBudgetConfig::default())?;
    Ok((plan.hits, plan.tokens_used))
}

fn scan_budget(predicate: &Predicate, budget: usize) -> usize {
    let selectivity_multiplier = if predicate.conditions.iter().any(|condition| {
        matches!(
            condition,
            Condition::QualityAbove(_)
                | Condition::TagsContains(_)
                | Condition::ValidOnly
                | Condition::TransactionTimeBetween(_, _)
                | Condition::ValidTimeContains(_)
                | Condition::ValidTimeOverlaps(_, _)
                | Condition::Bitemporal { .. }
        )
    }) {
        4
    } else {
        2
    };
    budget.saturating_mul(selectivity_multiplier).clamp(1, 4096)
}

fn vector_budget(predicate: &Predicate, budget: usize) -> usize {
    let multiplier = if predicate.is_empty() { 2 } else { 5 };
    budget.saturating_mul(multiplier).clamp(10, 512)
}

fn simple_hash(s: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

fn query_text_hint(query: &Query) -> String {
    match query {
        Query::Find { scope, .. } => format!(
            "find:{}",
            scope.as_ref().map(|u| u.to_string()).unwrap_or_default()
        ),
        Query::Similar { .. } => "similar".into(),
        Query::AsOf { uri, .. } => format!("asof:{uri}"),
        Query::Traverse { start, .. } => format!("traverse:{start}"),
        Query::Composite { .. } => "composite".into(),
    }
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{Condition, Predicate};
    use crate::retriever::RuleBasedPlanner;
    use agent_context_db_core::ContentType;

    #[test]
    fn compile_query_produces_physical_plan() -> std::result::Result<(), crate::RetrieveConfigError>
    {
        let planner: Arc<dyn QueryPlanner> = Arc::new(RuleBasedPlanner::new("t", "a"));
        let compiler = RetrieverCompiler::new(planner)?;
        let ctx = RetrieveContext::default();
        let query = Query::Find {
            scope: None,
            predicate: Predicate::new().with(Condition::TypeEquals(ContentType::Fact)),
            budget: 100,
            order: crate::query::SortKey::Relevance,
            expand: None,
        };
        let plan = compiler.compile_query(&query, &ctx);
        // 谓词下推：执行树内部应该包含 TypeScan。
        assert!(contains_type_scan(&plan.physical));
        assert!(plan.estimated_cost > 0.0);
        Ok(())
    }

    fn contains_type_scan(plan: &PhysicalPlan) -> bool {
        match plan {
            PhysicalPlan::TypeScan { .. } => true,
            PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::GraphTraverse { input, .. } => contains_type_scan(input),
            PhysicalPlan::HashJoin { left, right }
            | PhysicalPlan::NestedLoopJoin { left, right } => {
                contains_type_scan(left) || contains_type_scan(right)
            }
            PhysicalPlan::Parallel { plans, .. } => plans.iter().any(contains_type_scan),
            _ => false,
        }
    }

    #[test]
    fn explain_is_non_empty() -> std::result::Result<(), crate::RetrieveConfigError> {
        let planner: Arc<dyn QueryPlanner> = Arc::new(RuleBasedPlanner::new("t", "a"));
        let compiler = RetrieverCompiler::new(planner)?;
        let ctx = RetrieveContext::default();
        let query = Query::Find {
            scope: None,
            predicate: Predicate::new(),
            budget: 50,
            order: crate::query::SortKey::Natural,
            expand: None,
        };
        let plan = compiler.compile_query(&query, &ctx);
        let explain = plan.explain();
        assert!(explain.contains("LogicalPlan"));
        assert!(explain.contains("PhysicalPlan"));
        Ok(())
    }
}
