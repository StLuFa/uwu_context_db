//! # agent-context-db-retrieve (M1 检索层)
//!
//! 分层检索：意图分析 → 目录递归 → Rerank，全程可产生检索轨迹。
//!
//! ## 模块
//!
//! - [`intent`]：`RuleBasedIntentAnalyzer`（关键词匹配） + `IntentAnalyzer` trait
//! - [`retriever`]：`HierarchicalRetrieverImpl`（完整检索管线实现）
//! - 本模块：trait 定义 + `ScoreReranker`
//!
//! ## 解耦约束
//!
//! - 仅依赖 core 的 **`FsOps` 窄端口**（只读寻址）和可选的 `VectorIndex`，不依赖具体后端。
//! - 可用内存版 `FsOps` mock 单测，不启 PG（见 dev-tests）。

pub mod innovation;
pub mod intent;
pub mod perf;
pub mod quality;
pub mod retriever;

pub use innovation::{IncrementalRetrievalLearner, PredictivePrefetcher, RelevanceFeedback};
pub use intent::{LlmIntentAnalyzer, RuleBasedIntentAnalyzer};
pub use perf::{
    BatchFsRequest, CacheTier, MaterializedView, ParallelGenerator, PartitionedRetriever,
    QueryCompiler, TieredVectorIndex, VectorQuantizer,
};
pub use quality::{CompressionAwareLoader, HallucinationDetector, PressureLevel, QualityReport};
pub use retriever::HierarchicalRetrieverImpl;

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContextUri, LlmClient, LlmOpts, MemoryClass, Result,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ===========================================================================
// 检索器端口
// ===========================================================================

/// 分层检索器：意图分析 → 目录递归 → Rerank。
#[async_trait]
pub trait HierarchicalRetriever: Send + Sync {
    async fn retrieve(&self, query: &str, ctx: &RetrieveContext) -> Result<RetrievalResult>;
    async fn retrieve_typed(
        &self,
        query: &str,
        class: MemoryClass,
        ctx: &RetrieveContext,
    ) -> Result<RetrievalResult>;
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
    pub intent: Vec<TypedQuery>,
}

#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub uri: ContextUri,
    pub level: ContentLevel,
    pub content: ContentPayload,
    pub relevance: f32,
    /// 父目录链（递归深入路径）。
    pub parent_chain: Vec<ContextUri>,
    /// 记忆分类（从条目元数据中获取，用于 class-aware 过滤）。
    pub memory_class: Option<MemoryClass>,
}

// ===========================================================================
// 检索轨迹可视化
// ===========================================================================

#[derive(Debug, Clone, Default)]
pub struct RetrievalTrace {
    pub steps: Vec<TraceStep>,
}

#[derive(Debug, Clone)]
pub enum TraceStep {
    IntentAnalysis { raw: String, typed: Vec<TypedQuery> },
    InitialLocate { query: TypedQuery, top_dirs: Vec<(ContextUri, f32)> },
    IntraDirSearch { dir: ContextUri, candidates: Vec<ContextUri> },
    RecursiveDescent { from: ContextUri, into: ContextUri, reason: String },
    Rerank { input: usize, kept: usize, model: String },
    Load { uri: ContextUri, level: ContentLevel, tokens: usize },
}

// ===========================================================================
// 意图分析
// ===========================================================================

/// 意图分析器：将自然语言查询拆为 0-N 个类型化查询。
#[async_trait]
pub trait IntentAnalyzer: Send + Sync {
    async fn analyze(&self, query: &str, ctx: &RetrieveContext) -> Result<Vec<TypedQuery>>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypedQuery {
    pub kind: QueryKind,
    pub text: String,
    pub target_dirs: Vec<ContextUri>,
    pub expected_class: Option<MemoryClass>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryKind {
    SemanticSearch,
    EntityLookup,
    EventRecall,
    SkillReuse,
    PatternMatch,
    StateSnapshot,
    PersonaRelation,
}

// ===========================================================================
// Rerank
// ===========================================================================

#[async_trait]
pub trait Reranker: Send + Sync {
    async fn rerank(&self, query: &str, hits: Vec<RetrievalHit>) -> Result<Vec<RetrievalHit>>;
}

/// 分数降序、按 budget 截断的朴素 reranker（轻量默认实现）。
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

/// LLM 驱动的语义重排器。
///
/// 对每个命中的内容调用 LLM，评估其与查询的语义相关性（0.0–1.0），
/// 然后按语义分数降序排列。相比 `ScoreReranker` 的纯分数排序，
/// 能捕捉到词语不匹配但语义相关的命中。
pub struct LlmReranker {
    llm: Arc<dyn LlmClient>,
    /// 保留的最大命中数
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

        // 为每个 hit 调用 LLM 评估语义相关性
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

            let score = match self.llm.complete(&prompt, &LlmOpts {
                max_tokens: Some(10),
                temperature: Some(0.0),
                ..Default::default()
            }).await {
                Ok(text) => text.trim().parse::<f32>().unwrap_or(hit.relevance),
                Err(_) => hit.relevance, // LLM 失败时保留原始分数
            };

            rescored.push((hit, score));
        }

        // 按语义分数降序
        rescored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        rescored.truncate(self.keep);

        Ok(rescored.into_iter().map(|(h, s)| {
            RetrievalHit { relevance: s, ..h }
        }).collect())
    }
}

/// 提取 ContentPayload 的文本预览（最多 max_chars 字符）。
fn content_preview(content: &ContentPayload, max_chars: usize) -> String {
    let text = match content {
        ContentPayload::Abstract(s) => s.clone(),
        ContentPayload::Overview(s) => s.clone(),
        ContentPayload::Detail(bytes) => {
            String::from_utf8_lossy(bytes).into_owned()
        }
    };
    if text.len() > max_chars {
        format!("{}...", &text[..max_chars])
    } else {
        text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn score_reranker_sorts_and_truncates() {
        let mk = |score: f32| RetrievalHit {
            uri: ContextUri::parse("uwu://t/agent/a/memories/cases/c1").unwrap(),
            level: ContentLevel::L0,
            content: ContentPayload::Abstract("x".into()),
            relevance: score,
            parent_chain: vec![],
            memory_class: None,
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
