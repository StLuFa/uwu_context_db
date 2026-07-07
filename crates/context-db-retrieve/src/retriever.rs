//! `ContextRetriever`：计划驱动检索器。
//!
//! 流程：Query DSL → LogicalPlan → CBO 优化 → PhysicalPlan → 执行。

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContentType, ContextUri, FsOps, GraphStore, Result, VectorIndex,
};
use async_trait::async_trait;
use std::sync::Arc;
use tracing::Instrument;

use crate::intent::RuleBasedIntentAnalyzer;
use crate::operators::ExecContext;
use crate::planner::{CboOptimizer, IntentPlannerHint, LogicalPlan, PhysicalPlan, StatisticsCollector};
use crate::query::{Condition, Predicate, Query};
use crate::{
    PlanRetriever, QueryPlanner, Reranker, RetrievalHit, RetrievalResult, RetrievalTrace,
    RetrieveContext, TraceStep,
};

// ===========================================================================
// ContextRetriever — 计划驱动检索器
// ===========================================================================

/// 计划驱动检索器：Query DSL → Plan → Execute。
pub struct ContextRetriever {
    fs: Arc<dyn FsOps>,
    index: Option<Arc<dyn VectorIndex>>,
    /// 关系图存储 — 若注入，`LogicalPlan::Traverse` 会走 `GraphTraverse` 算子；否则退化。
    graph: Option<Arc<dyn GraphStore>>,
    /// 联想扩展开关 —— 若为 true 且 `graph` 存在，主计划输出后自动沿联想图扩展。
    associative_enabled: bool,
    planner: Arc<dyn QueryPlanner>,
    reranker: Arc<dyn Reranker>,
    intent_analyzer: Option<Arc<RuleBasedIntentAnalyzer>>,
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
            graph: None,
            associative_enabled: false,
            planner,
            reranker,
            intent_analyzer: None,
            optimizer: CboOptimizer::new(stats),
            plan_cache: parking_lot::Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// 注入新版 intent policy engine，让检索 trace 携带可解释执行图。
    pub fn with_intent_analyzer(mut self, intent_analyzer: Arc<RuleBasedIntentAnalyzer>) -> Self {
        self.intent_analyzer = Some(intent_analyzer);
        self
    }

    /// 若配置了图与联想扩展开关，则对已有命中做联想扩展并合并到结果尾部。
    async fn maybe_expand_associative(&self, hits: Vec<RetrievalHit>) -> Vec<RetrievalHit> {
        if !self.associative_enabled {
            return hits;
        }
        let graph = match &self.graph {
            Some(g) => g.clone(),
            None => return hits,
        };
        let expander = crate::AssociativeExpander::new(self.fs.clone(), graph);
        match expander.expand(&hits).await {
            Ok(extra) => {
                let mut merged = hits;
                merged.extend(extra);
                merged
            }
            Err(_) => hits,
        }
    }

    /// G.1: Builder 入口。
    pub fn builder(fs: Arc<dyn FsOps>) -> ContextRetrieverBuilder {
        ContextRetrieverBuilder::new(fs)
    }

    /// 自然语言查询 → 检索结果。
    ///
    /// 检索管线 6 阶段（每阶段都有独立 tracing span）：
    /// 1. `plan.parse` — NL → LogicalPlan
    /// 2. `plan.optimize` — CBO → PhysicalPlan
    /// 3. `plan.execute` — 物理算子执行
    /// 4. `rerank` — 结果重排
    /// 5. `expand.associative` — 图边联想扩展（可选）
    /// 6. `budget.load` — Token 预算加载
    #[tracing::instrument(
        skip(self, ctx),
        fields(query_len = query.len(), budget = ctx.budget_tokens.unwrap_or(8000)),
    )]
    pub async fn retrieve(&self, query: &str, ctx: &RetrieveContext) -> Result<RetrievalResult> {
        let budget = ctx.budget_tokens.unwrap_or(8000);
        let mut trace = RetrievalTrace::default();
        let intent_decision = self
            .intent_analyzer
            .as_ref()
            .map(|analyzer| analyzer.decide(query, ctx));
        if let Some(decision) = &intent_decision {
            trace.steps.push(TraceStep::IntentAnalysis {
                raw: query.to_string(),
                num_queries: decision.candidates.len(),
                decision: Some(decision.clone()),
            });
        }

        // 1. 自然语言 → LogicalPlan
        let logical = self
            .planner
            .parse(query, ctx)
            .instrument(tracing::info_span!("plan.parse"))
            .await?;

        // 2. CBO 优化 → PhysicalPlan
        let intent_hint = intent_decision.as_ref().map(IntentPlannerHint::from);
        let physical = tracing::info_span!("plan.optimize").in_scope(|| {
            self.optimizer
                .optimize_with_intent(&logical, intent_hint.as_ref())
        });
        let adjusted_cost = self
            .optimizer
            .estimate_cost_with_intent(&physical, intent_hint.as_ref());
        trace.steps.push(TraceStep::PlanOptimized {
            logical: format!("{:?}", logical),
            physical: format!("{:?}; intent_cost={adjusted_cost:.3}", physical),
        });

        // 3. 执行物理计划
        let exec_ctx = ExecContext {
            fs: self.fs.clone(),
            index: self.index.clone(),
            graph: self.graph.clone(),
        };
        let batch = physical
            .execute(&exec_ctx)
            .instrument(tracing::info_span!("plan.execute"))
            .await?;
        trace.steps.push(TraceStep::Execute {
            plan: "physical".into(),
            hits: batch.records.len(),
            duration_ms: batch.stats.duration.as_millis() as u64,
        });

        // 4. Rerank
        let rerank_input = batch.records.len();
        let reranked = self
            .reranker
            .rerank(query, batch.records)
            .instrument(tracing::info_span!("rerank", input = rerank_input))
            .await?;
        trace.steps.push(TraceStep::Rerank {
            input: batch.stats.rows_scanned,
            kept: reranked.len(),
            model: "score".into(),
        });

        // 4b. 联想扩展（可选）
        let expand_input = reranked.len();
        let expanded = self
            .maybe_expand_associative(reranked)
            .instrument(tracing::info_span!(
                "expand.associative",
                input = expand_input
            ))
            .await;

        // 5. Budget loading
        let (hits, tokens_used) = tracing::info_span!("budget.load", budget)
            .in_scope(|| load_within_budget(expanded, budget));

        tracing::info!(hits = hits.len(), tokens_used, "retrieve complete");
        Ok(RetrievalResult {
            hits,
            trace,
            tokens_used,
        })
    }

    /// 结构化 Query → 检索结果。
    #[tracing::instrument(skip(self, ctx))]
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

        let logical = tracing::info_span!("plan.parse").in_scope(|| query_to_logical(query));
        let physical =
            tracing::info_span!("plan.optimize").in_scope(|| self.optimizer.optimize(&logical));
        trace.steps.push(TraceStep::PlanOptimized {
            logical: format!("{logical:?}"),
            physical: format!("{physical:?}"),
        });

        let exec_ctx = ExecContext {
            fs: self.fs.clone(),
            index: self.index.clone(),
            graph: self.graph.clone(),
        };
        let batch = physical
            .execute(&exec_ctx)
            .instrument(tracing::info_span!("plan.execute"))
            .await?;
        trace.steps.push(TraceStep::Execute {
            plan: "physical".into(),
            hits: batch.records.len(),
            duration_ms: batch.stats.duration.as_millis() as u64,
        });

        let rerank_input = batch.records.len();
        let reranked = self
            .reranker
            .rerank("", batch.records)
            .instrument(tracing::info_span!("rerank", input = rerank_input))
            .await?;
        let expand_input = reranked.len();
        let expanded = self
            .maybe_expand_associative(reranked)
            .instrument(tracing::info_span!(
                "expand.associative",
                input = expand_input
            ))
            .await;
        let (hits, tokens_used) = tracing::info_span!("budget.load", budget)
            .in_scope(|| load_within_budget(expanded, budget));

        tracing::info!(hits = hits.len(), tokens_used, "retrieve_query complete");
        Ok(RetrievalResult {
            hits,
            trace,
            tokens_used,
        })
    }
}

#[async_trait]
impl PlanRetriever for ContextRetriever {
    async fn retrieve_plan(&self, query: &Query, ctx: &RetrieveContext) -> Result<RetrievalResult> {
        self.retrieve_query(query, ctx).await
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
        }
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

fn load_within_budget(mut hits: Vec<RetrievalHit>, budget: usize) -> (Vec<RetrievalHit>, usize) {
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
        ContentPayload::Text {
            sparse,
            dense,
            full: _,
        } => std::cmp::max(sparse.len() / 4, 100),
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
    pub fn new(default_tenant: impl Into<String>, default_agent: impl Into<String>) -> Self {
        Self {
            default_tenant: default_tenant.into(),
            default_agent: default_agent.into(),
        }
    }
}

#[async_trait]
impl QueryPlanner for RuleBasedPlanner {
    async fn parse(&self, query: &str, ctx: &RetrieveContext) -> Result<LogicalPlan> {
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
// G.1: ContextRetrieverBuilder
// ===========================================================================

/// Builder for ContextRetriever — 替代多构造器模式。
pub struct ContextRetrieverBuilder {
    fs: Arc<dyn FsOps>,
    index: Option<Arc<dyn VectorIndex>>,
    graph: Option<Arc<dyn GraphStore>>,
    associative_enabled: bool,
    planner: Option<Arc<dyn QueryPlanner>>,
    reranker: Option<Arc<dyn Reranker>>,
}

impl ContextRetrieverBuilder {
    pub fn new(fs: Arc<dyn FsOps>) -> Self {
        Self {
            fs,
            index: None,
            graph: None,
            associative_enabled: false,
            planner: None,
            reranker: None,
        }
    }

    pub fn with_vector_index(mut self, index: Arc<dyn VectorIndex>) -> Self {
        self.index = Some(index);
        self
    }

    /// 注入关系图存储 — 启用 `LogicalPlan::Traverse` 的图遍历执行。
    pub fn with_graph(mut self, graph: Arc<dyn GraphStore>) -> Self {
        self.graph = Some(graph);
        self
    }

    /// 启用主计划输出后的联想扩展（需先注入 graph，否则无效）。
    pub fn enable_associative(mut self) -> Self {
        self.associative_enabled = true;
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
        let planner = self
            .planner
            .unwrap_or_else(|| Arc::new(RuleBasedPlanner::new("default", "default")));
        let reranker = self
            .reranker
            .unwrap_or_else(|| Arc::new(crate::ScoreReranker { keep: 20 }));
        let mut r = ContextRetriever::new(self.fs, self.index, planner, reranker);
        r.graph = self.graph;
        r.associative_enabled = self.associative_enabled;
        r
    }
}
