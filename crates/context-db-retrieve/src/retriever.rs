//! `HierarchicalRetrieverImpl`：M1 检索管线完整实现。
//!
//! 流程：意图分析 → 目录定位 → 目录内搜索 → 递归深入 → Rerank → 按预算加载。
//!
//! ## 端口依赖
//!
//! - `FsOps`（core 窄端口）：只读寻址，所有 ls/find/grep/read 走此端口。
//! - `VectorIndex`（core 窄端口，可选）：向量召回定位高分目录；`None` 时纯 FS 检索。
//! - `IntentAnalyzer` + `Reranker`：本层端口，由调用方注入。
//!
//! 检索层不依赖任何具体后端（PG/Qdrant），可用 MemoryContextStore 完整单测。

use agent_context_db_core::{
    ContentLevel, ContentPayload, ContextUri, FsOps, LlmClient, MemoryClass, Result, VectorIndex,
};
use async_trait::async_trait;
use std::sync::Arc;

use crate::{
    HierarchicalRetriever, IntentAnalyzer, QueryKind, Reranker, RetrievalHit, RetrievalResult,
    RetrievalTrace, RetrieveContext, TraceStep, TypedQuery,
};

// ===========================================================================
// 默认 token 预算估算常量
// ===========================================================================

/// L0 摘要估算：~100 tokens
const L0_TOKENS: usize = 100;
/// L1 概览估算：~2k tokens
const L1_TOKENS: usize = 2000;
/// 默认检索 budget（无 ctx 指定时）
const DEFAULT_BUDGET: usize = 8000;

// ===========================================================================
// HierarchicalRetrieverImpl
// ===========================================================================

pub struct HierarchicalRetrieverImpl {
    fs: Arc<dyn FsOps>,
    index: Option<Arc<dyn VectorIndex>>,
    llm: Option<Arc<dyn LlmClient>>,
    intent: Arc<dyn IntentAnalyzer>,
    reranker: Arc<dyn Reranker>,
}

impl HierarchicalRetrieverImpl {
    /// 无向量索引的构造器（纯 FS 检索）。
    pub fn new(
        fs: Arc<dyn FsOps>,
        intent: Arc<dyn IntentAnalyzer>,
        reranker: Arc<dyn Reranker>,
    ) -> Self {
        Self { fs, index: None, llm: None, intent, reranker }
    }

    /// 带向量索引的构造器。
    pub fn with_index(
        fs: Arc<dyn FsOps>,
        index: Arc<dyn VectorIndex>,
        intent: Arc<dyn IntentAnalyzer>,
        reranker: Arc<dyn Reranker>,
    ) -> Self {
        Self { fs, index: Some(index), llm: None, intent, reranker }
    }

    /// 带向量索引 + LLM embedding 的构造器（完整向量召回链路）。
    pub fn with_full_vector(
        fs: Arc<dyn FsOps>,
        index: Arc<dyn VectorIndex>,
        llm: Arc<dyn LlmClient>,
        intent: Arc<dyn IntentAnalyzer>,
        reranker: Arc<dyn Reranker>,
    ) -> Self {
        Self { fs, index: Some(index), llm: Some(llm), intent, reranker }
    }
}

#[async_trait]
impl HierarchicalRetriever for HierarchicalRetrieverImpl {
    async fn retrieve(&self, query: &str, ctx: &RetrieveContext) -> Result<RetrievalResult> {
        let mut trace = RetrievalTrace::default();
        let budget = ctx.budget_tokens.unwrap_or(DEFAULT_BUDGET);

        // ── 阶段 1：意图分析 ──
        let typed_queries = self.intent.analyze(query, ctx).await?;
        trace.steps.push(TraceStep::IntentAnalysis {
            raw: query.to_string(),
            typed: typed_queries.clone(),
        });

        let mut all_hits: Vec<RetrievalHit> = Vec::new();

        for tq in &typed_queries {
            // ── 阶段 2：定位目录 ──
            let top_dirs = self.locate_dirs(tq, ctx).await?;
            trace.steps.push(TraceStep::InitialLocate {
                query: tq.clone(),
                top_dirs: top_dirs.clone(),
            });

            // ── 阶段 3 + 4：目录内搜索 + 递归深入 ──
            for (dir, _score) in &top_dirs {
                let candidates = self.intra_dir_search(dir, tq).await?;
                let uris: Vec<ContextUri> = candidates.iter().map(|(u, _)| u.clone()).collect();
                trace.steps.push(TraceStep::IntraDirSearch {
                    dir: dir.clone(),
                    candidates: uris,
                });

                for (cand_uri, cand_mc) in &candidates {
                    let deeper = self
                        .recursive_descent(cand_uri, *cand_mc, tq, ctx, &mut trace)
                        .await?;
                    all_hits.extend(deeper);
                }
            }
        }

        // ── 阶段 5：Rerank ──
        let before = all_hits.len();
        let reranked = self.reranker.rerank(query, all_hits).await?;
        trace.steps.push(TraceStep::Rerank {
            input: before,
            kept: reranked.len(),
            model: "score-reranker".into(),
        });

        // ── 阶段 6：按预算加载 ──
        let (hits, tokens_used) = self.load_within_budget(reranked, budget).await?;
        for h in &hits {
            trace.steps.push(TraceStep::Load {
                uri: h.uri.clone(),
                level: h.level,
                tokens: estimate_tokens(&h.content),
            });
        }

        Ok(RetrievalResult {
            hits,
            trace,
            tokens_used,
            intent: typed_queries,
        })
    }

    async fn retrieve_typed(
        &self,
        query: &str,
        class: MemoryClass,
        ctx: &RetrieveContext,
    ) -> Result<RetrievalResult> {
        // 调用通用 retrieve，然后按 class 过滤
        let mut result = self.retrieve(query, ctx).await?;

        // Post-filter: 只保留匹配 class 或无 class 标注的 hits
        result.hits.retain(|h| {
            h.memory_class.map_or(true, |mc| mc == class)
        });

        // 如果过滤后为空，保留原始结果（宽松降级）
        if result.hits.is_empty() {
            return self.retrieve(query, ctx).await;
        }

        Ok(result)
    }
}

// ===========================================================================
// 私有管线阶段
// ===========================================================================

impl HierarchicalRetrieverImpl {
    /// 阶段 2：定位候选目录。
    ///
    /// 有向量索引时通过 embedding 搜索定位；
    /// 无索引时直接用 typed_query 的 target_dirs。
    async fn locate_dirs(
        &self,
        tq: &TypedQuery,
        _ctx: &RetrieveContext,
    ) -> Result<Vec<(ContextUri, f32)>> {
        // 向量检索：有 index + llm 时用 embedding 搜索
        if let (Some(index), Some(llm)) = (&self.index, &self.llm) {
            match llm.embed(&tq.text).await {
                Ok(vec) => {
                    let filter = tq.expected_class.map(|c| {
                        serde_json::json!({"memory_class": memory_class_str(c)})
                    });
                    match index.search("memories", vec, 5, filter).await {
                        Ok(hits) if !hits.is_empty() => {
                            return Ok(hits
                                .into_iter()
                                .map(|h| {
                                    (ContextUri::parse(h.uri).unwrap_or_else(|_| ContextUri("".into())), h.score)
                                })
                                .filter(|(u, _)| !u.0.is_empty())
                                .collect());
                        }
                        _ => {} // fall through to target_dirs
                    }
                }
                Err(_) => {} // embed failed, fall through
            }
        }

        // Fallback：target_dirs（无向量或向量搜索失败）
        let score = if self.index.is_some() { 0.5 } else { 0.8 };
        Ok(tq.target_dirs.iter().cloned().map(|d| (d, score)).collect())
    }

    /// 阶段 3：目录内搜索。
    ///
    /// 对指定目录 ls，按 tq.kind 和 tq.expected_class 筛选候选。
    /// 返回 (uri, memory_class) 对。
    async fn intra_dir_search(
        &self,
        dir: &ContextUri,
        tq: &TypedQuery,
    ) -> Result<Vec<(ContextUri, Option<MemoryClass>)>> {
        let entries = self.fs.ls(dir).await?;
        let candidates: Vec<(ContextUri, Option<MemoryClass>)> = entries
            .into_iter()
            .filter(|e| !e.is_dir) // 只取文件
            .filter(|e| {
                // 按 expected_class 过滤（有指定 class 时只保留匹配项）
                match tq.expected_class {
                    Some(expected) => e.memory_class.map_or(true, |mc| mc == expected),
                    None => true,
                }
            })
            .map(|e| (e.uri, e.memory_class))
            .collect();
        Ok(candidates)
    }

    /// 阶段 4：递归深入子目录。
    ///
    /// 如果候选目录下有子目录且预算允许，深入读取。
    async fn recursive_descent(
        &self,
        uri: &ContextUri,
        memory_class: Option<MemoryClass>,
        tq: &TypedQuery,
        ctx: &RetrieveContext,
        trace: &mut RetrievalTrace,
    ) -> Result<Vec<RetrievalHit>> {
        let mut hits = Vec::new();
        let mut parent_chain = Vec::new();

        // 按 ctx.prefer_level 读取内容
        let level = ctx.prefer_level;
        match self.fs.read(uri, level).await {
            Ok(content) => {
                hits.push(RetrievalHit {
                    uri: uri.clone(),
                    level,
                    content,
                    relevance: initial_relevance(tq),
                    parent_chain: parent_chain.clone(),
                    memory_class,
                });
            }
            Err(_) => {
                // 文件不可读时尝试 L0 降级
                if level != ContentLevel::L0 {
                    if let Ok(l0) = self.fs.read(uri, ContentLevel::L0).await {
                        hits.push(RetrievalHit {
                            uri: uri.clone(),
                            level: ContentLevel::L0,
                            content: l0,
                            relevance: initial_relevance(tq) * 0.5,
                            parent_chain: parent_chain.clone(),
                            memory_class,
                        });
                    }
                }
            }
        }

        // 检查是否有子目录：ls uri 的父目录看有无同名目录下的内容
        if let Some(parent) = uri.parent() {
            if let Ok(children) = self.fs.ls(&parent).await {
                for child in children {
                    if child.is_dir && child.uri.0.starts_with(&uri.0) {
                        if ctx.trace_enabled {
                            trace.steps.push(TraceStep::RecursiveDescent {
                                from: uri.clone(),
                                into: child.uri.clone(),
                                reason: "subdir hit".into(),
                            });
                        }
                        parent_chain.push(uri.clone());
                        // 递归进去
                        let sub_hits = Box::pin(
                            self.recursive_descent(
                                &child.uri,
                                child.memory_class,
                                tq, ctx, trace,
                            ),
                        )
                        .await?;
                        hits.extend(sub_hits);
                    }
                }
            }
        }

        Ok(hits)
    }

    /// 阶段 6：按 token 预算加载内容。
    ///
    /// 按 relevance 从高到低消耗 budget。
    async fn load_within_budget(
        &self,
        hits: Vec<RetrievalHit>,
        budget: usize,
    ) -> Result<(Vec<RetrievalHit>, usize)> {
        let mut loaded = Vec::new();
        let mut used = 0usize;

        for h in hits {
            let cost = estimate_tokens(&h.content);
            if used + cost > budget && !loaded.is_empty() {
                // 超预算：后续命中降级为 L0
                if h.level != ContentLevel::L0 {
                    if let Ok(l0) = self.fs.read(&h.uri, ContentLevel::L0).await {
                        let l0_cost = L0_TOKENS;
                        if used + l0_cost <= budget {
                            loaded.push(RetrievalHit {
                                uri: h.uri,
                                level: ContentLevel::L0,
                                content: l0,
                                relevance: h.relevance,
                                parent_chain: h.parent_chain,
                                memory_class: h.memory_class,
                            });
                            used += l0_cost;
                        }
                    }
                }
                continue;
            }
            used += cost;
            loaded.push(h);
        }

        Ok((loaded, used))
    }
}

// ===========================================================================
// 辅助函数
// ===========================================================================

/// 根据 query type 给出初始相关性。
fn initial_relevance(tq: &TypedQuery) -> f32 {
    match tq.kind {
        QueryKind::EntityLookup => 0.9,
        QueryKind::EventRecall => 0.85,
        QueryKind::SkillReuse => 0.8,
        QueryKind::SemanticSearch => 0.6,
        QueryKind::PatternMatch => 0.75,
        QueryKind::StateSnapshot => 0.7,
        QueryKind::PersonaRelation => 0.8,
    }
}

fn memory_class_str(c: MemoryClass) -> &'static str {
    match c {
        MemoryClass::Profile => "profile",
        MemoryClass::Preferences => "preferences",
        MemoryClass::Entities => "entities",
        MemoryClass::Events => "events",
        MemoryClass::Cases => "cases",
        MemoryClass::Patterns => "patterns",
        MemoryClass::Tools => "tools",
        MemoryClass::Skills => "skills",
    }
}

/// 估算 ContentPayload 的 token 数。
fn estimate_tokens(content: &ContentPayload) -> usize {
    match content {
        ContentPayload::Abstract(s) => {
            // ~1 token / 4 chars for English
            s.len().max(L0_TOKENS.min(s.len() / 4))
        }
        ContentPayload::Overview(s) => {
            s.len().max(L1_TOKENS.min(s.len() / 4))
        }
        ContentPayload::Detail(b) => b.len() / 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_estimate_l0_is_reasonable() {
        let c = ContentPayload::Abstract("short abstract".into());
        let t = estimate_tokens(&c);
        assert!(t > 0 && t < 500);
    }

    #[test]
    fn token_estimate_l2_is_bytes_based() {
        let c = ContentPayload::Detail(vec![0u8; 4000]);
        let t = estimate_tokens(&c);
        assert_eq!(t, 1000); // 4000 / 4
    }
}
