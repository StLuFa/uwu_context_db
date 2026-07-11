//! # agent-context-db-retrieve (M1 检索层)
//!
//! 查询引擎：Query DSL → LogicalPlan → CBO 优化 → PhysicalPlan → 物理算子执行。
//!
//! ## 模块
//! - [`query`]：查询 DSL AST
//! - [`planner`]：LogicalPlan + CBO 优化器
//! - [`operators`]：物理算子
//! - [`intent`]：意图分析器（LLM / 规则）
//! - [`retriever`]：ContextRetriever（计划驱动的检索器）
//!
//! ## 解耦约束
//! - 仅依赖 core 的 `FsOps` 窄端口和可选的 `VectorIndex`，不依赖具体后端。

pub mod associative;
pub mod budget;
pub mod cache;
pub mod compiler;
pub mod config;
pub mod graph_rag;
pub mod innovation;
pub mod intent;
pub mod operators;
pub mod perf;
pub mod planner;
pub mod quality;
pub mod query;
pub mod rag;
pub mod retriever;
pub mod theory_of_mind;

pub use associative::AssociativeExpander;
pub use budget::{BudgetLoadPlan, LevelAllocation, allocate_hit_levels, load_hits_within_budget};
pub use config::{
    GraphRagConfig, InnovationConfig, QueryPlanConfig, RagSynthesisConfig, RetrieveConfigError,
    TokenBudgetConfig,
};
pub use graph_rag::{
    GraphRagCommunity, GraphRagEngine, GraphRagIndex, GraphRagIndexConfig, GraphRagIndexStats,
    GraphRagIndexer, GraphRagRequest,
};
pub use innovation::{
    AccessPattern, IncrementalRetrievalLearner, PredictivePrefetcher, PrefetchPrediction,
    RelevanceFeedback,
};
pub use intent::{
    BuiltinIntentPolicyProvider, CompiledIntentPolicy, EventMeshIntentTraceSink,
    FileIntentPolicyProvider, IntentCaller, IntentCandidate, IntentDecision, IntentExecutionGraph,
    IntentExecutionPlan, IntentFeedbackEvent, IntentFeedbackLearning, IntentInput, IntentKind,
    IntentPolicyLayer, IntentPolicyLayerKind, IntentPolicyPack, IntentPolicyProvider,
    IntentPolicyRef, IntentPolicyReloadReport, IntentPolicyReloadStatus, IntentPolicySignature,
    IntentPolicySignatureVerifier, IntentPolicySnapshot, IntentRoute, IntentTraceEvent,
    IntentTraceSink, LayeredIntentPolicyProvider, RuleBasedIntentAnalyzer, SignedIntentPolicyPack,
    TracingIntentTraceSink,
};
pub use operators::{ExecContext, ExecStats, RecordBatch};
pub use perf::{
    BatchFsRequest, CacheTier, MaterializedView, ParallelGenerator, PartitionedRetriever,
    QueryCompiler, TieredVectorIndex, VectorQuantizer,
};
pub use planner::{
    CboOptimizer, IntentPlannerHint, LogicalPlan, PhysicalPlan, StatisticsCollector, VectorFilter,
};
pub use quality::{CompressionAwareLoader, HallucinationDetector, PressureLevel, QualityReport};
pub use query::{Condition, Predicate, Query, QueryMergeStrategy, SortKey};
pub use rag::{
    AnswerCitation, AnswerConfidence, AnswerDecision, AnswerSynthesisConfig, CalibratedAnswer,
    CalibratedAnswerSynthesizer,
};
pub use retriever::{ContextRetriever, ContextRetrieverBuilder, RuleBasedPlanner};
pub use theory_of_mind::{
    BeliefFacet, TheoryOfMindModel, TomObservation, TomObservationKind, TomRetrievalHint,
    model_from_entry,
};

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContentType, ContextMeta, ContextUri, LlmClient, LlmOpts, Result,
};
use async_trait::async_trait;
use std::sync::Arc;

// ===========================================================================
// 检索器端口（分层 + 计划驱动）
// ===========================================================================

/// 计划驱动检索器 — 查询 DSL → CBO 优化 → 物理计划执行（推荐）。
#[async_trait]
pub trait PlanRetriever: Send + Sync {
    async fn retrieve_plan(&self, query: &Query, ctx: &RetrieveContext) -> Result<RetrievalResult>;
}

#[derive(Debug, Clone, Default)]
pub struct RetrieveContext {
    pub user_id: Option<String>,
    pub agent_id: Option<String>,
    pub budget_tokens: Option<usize>,
    pub prefer_level: ContentLevel,
    pub trace_enabled: bool,
}

#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub hits: Vec<RetrievalHit>,
    pub trace: RetrievalTrace,
    pub tokens_used: usize,
}

#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub uri: ContextUri,
    pub level: ContentLevel,
    pub content: ContentPayload,
    pub relevance: f32,
    /// 父目录链（递归深入路径）。
    pub parent_chain: Vec<ContextUri>,
    /// 内容类型（从 URI 或元数据中获取）。
    pub content_type: Option<ContentType>,
    /// 条目元数据，用于 WQL 条件完整执行。
    pub metadata: ContextMeta,
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    pub updated_at: Option<chrono::DateTime<chrono::Utc>>,
}

// ===========================================================================
// 检索轨迹
// ===========================================================================

#[derive(Debug, Clone, Default)]
pub struct RetrievalTrace {
    pub steps: Vec<TraceStep>,
}

#[derive(Debug, Clone)]
pub enum TraceStep {
    IntentAnalysis {
        raw: String,
        num_queries: usize,
        decision: Option<Box<IntentDecision>>,
    },
    PlanOptimized {
        logical: String,
        physical: String,
    },
    Execute {
        plan: String,
        hits: usize,
        duration_ms: u64,
    },
    Rerank {
        input: usize,
        kept: usize,
        model: String,
    },
    Load {
        uri: ContextUri,
        level: ContentLevel,
        tokens: usize,
    },
}

// ===========================================================================
// Rerank
// ===========================================================================

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, hits: Vec<RetrievalHit>) -> Result<Vec<RetrievalHit>>;
}

pub struct ScoreReranker {
    pub keep: usize,
}

#[async_trait]
impl Reranker for ScoreReranker {
    async fn rerank(&self, _query: &str, mut hits: Vec<RetrievalHit>) -> Result<Vec<RetrievalHit>> {
        hits.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(self.keep);
        Ok(hits)
    }
}

pub struct LlmReranker {
    llm: Arc<dyn LlmClient>,
    pub keep: usize,
}

impl LlmReranker {
    pub fn new(llm: Arc<dyn LlmClient>, keep: usize) -> Self {
        Self { llm, keep }
    }
}

#[async_trait]
impl Reranker for LlmReranker {
    async fn rerank(&self, query: &str, hits: Vec<RetrievalHit>) -> Result<Vec<RetrievalHit>> {
        if hits.is_empty() {
            return Ok(hits);
        }
        let mut rescored: Vec<(RetrievalHit, f32)> = Vec::new();
        for hit in hits {
            let content_text = content_preview(&hit.content, 500);
            let prompt = format!(
                "Query: {query}\n\nDocument: {content_text}\n\n\
                 Rate the semantic relevance of this document to the query on a scale of 0.0 to 1.0.\n\
                 - 1.0 = directly answers the query\n\
                 - 0.5 = somewhat related\n\
                 - 0.0 = completely unrelated\n\
                 Respond with ONLY a number between 0.0 and 1.0."
            );
            let score = match self
                .llm
                .complete(
                    &prompt,
                    &LlmOpts {
                        max_tokens: Some(10),
                        temperature: Some(0.0),
                        ..Default::default()
                    },
                )
                .await
            {
                Ok(text) => text.trim().parse::<f32>().unwrap_or(hit.relevance),
                Err(_) => hit.relevance,
            };
            rescored.push((hit, score));
        }
        rescored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        rescored.truncate(self.keep);
        Ok(rescored
            .into_iter()
            .map(|(h, s)| RetrievalHit { relevance: s, ..h })
            .collect())
    }
}

fn content_preview(content: &ContentPayload, max_chars: usize) -> String {
    let text = content.sparse_text();
    let mut chars = text.chars();
    let preview = chars.by_ref().take(max_chars).collect::<String>();
    if chars.next().is_some() {
        format!("{preview}...")
    } else {
        preview
    }
}

// ===========================================================================
// 查询规划器 trait（QueryPlanner）
// ===========================================================================

/// 查询规划器 — 将自然语言或 Query DSL 编译为物理计划。
#[async_trait]
pub trait QueryPlanner: Send + Sync {
    /// 自然语言 → LogicalPlan。
    async fn parse(&self, query: &str, ctx: &RetrieveContext) -> Result<LogicalPlan>;
    /// LogicalPlan → PhysicalPlan（经 CBO）。
    async fn plan(&self, logical: &LogicalPlan) -> Result<PhysicalPlan>;
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn score_reranker_sorts_and_truncates() {
        let mk = |score: f32| RetrievalHit {
            uri: ContextUri::parse("uwu://t/agent/a/memories/cases/c1").unwrap(),
            level: ContentLevel::L0,
            content: ContentPayload::Text {
                sparse: "x".into(),
                dense: String::new(),
                full: String::new(),
            },
            relevance: score,
            parent_chain: vec![],
            content_type: None,
            metadata: Default::default(),
            created_at: None,
            updated_at: None,
        };
        let rr = ScoreReranker { keep: 2 };
        let out = rr
            .rerank("q", vec![mk(0.1), mk(0.9), mk(0.5)])
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(out[0].relevance >= out[1].relevance);
    }
}
