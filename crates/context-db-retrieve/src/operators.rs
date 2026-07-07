//! 物理算子 — PhysicalPlan 的执行器。
//!
//! 每个物理计划节点对应一个算子实现，通过 `PhysicalPlan::execute()` 分发。

use crate::planner::{PhysicalPlan, ScopeFilter, VectorFilter};
use crate::query::{MergeStrategy, Predicate, RelationExpand};
use crate::{RetrievalHit, RetrievalResult, RetrieveContext};
use agent_context_db_core::{
    ContentLevel, ContentPayload, ContentType, ContextUri, FsOps, IndexHit, Result, VectorIndex,
};
use async_trait::async_trait;
use std::sync::Arc;
use std::time::Duration;

// ===========================================================================
// 执行上下文 + 结果批次
// ===========================================================================

/// 执行上下文 — 注入依赖。
#[derive(Clone)]
pub struct ExecContext {
    pub fs: Arc<dyn FsOps>,
    pub index: Option<Arc<dyn VectorIndex>>,
    /// 可选的关系图存储（用于图遍历查询）。
    pub graph: Option<Arc<dyn agent_context_db_core::GraphStore>>,
}

/// 记录批次 — 算子执行结果。
#[derive(Debug, Clone)]
pub struct RecordBatch {
    pub records: Vec<RetrievalHit>,
    pub stats: ExecStats,
}

/// 执行统计。
#[derive(Debug, Clone, Default)]
pub struct ExecStats {
    pub rows_scanned: usize,
    pub tokens_consumed: usize,
    pub duration: Duration,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

// ===========================================================================
// PhysicalPlan 分发执行
// ===========================================================================

impl PhysicalPlan {
    /// 执行物理计划 — 分发到对应算子。
    /// 执行物理计划 — 分发到对应算子。
    pub fn execute<'a>(
        &'a self,
        ctx: &'a ExecContext,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<RecordBatch>> + Send + 'a>> {
        Box::pin(async move { self.execute_inner(ctx).await })
    }

    async fn execute_inner(&self, ctx: &ExecContext) -> Result<RecordBatch> {
        match self {
            PhysicalPlan::TypeScan {
                content_type,
                scope,
                limit,
            } => TypeScanOp::execute_scan(content_type, scope, *limit, ctx).await,
            PhysicalPlan::PgPrefixScan { uri_prefix, limit } => {
                PgPrefixScanOp::execute_scan(uri_prefix, *limit, ctx).await
            }
            PhysicalPlan::VectorSearch {
                embedding,
                filter,
                limit,
            } => VectorSearchOp::execute_search(embedding, filter, *limit, ctx).await,
            PhysicalPlan::Filter { input, predicate } => {
                let inner = input.execute(ctx).await?;
                FilterOp::apply(inner, predicate).await
            }
            PhysicalPlan::Sort { input, key } => {
                let inner = input.execute(ctx).await?;
                SortOp::apply(inner, *key).await
            }
            PhysicalPlan::Limit { input, budget } => {
                let inner = input.execute(ctx).await?;
                LimitOp::apply(inner, *budget).await
            }
            PhysicalPlan::GraphTraverse {
                seeds,
                edges,
                max_hops,
            } => GraphTraverseOp::execute_traverse(seeds, edges, *max_hops, ctx).await,
            PhysicalPlan::Parallel { plans, merge } => {
                ParallelOp::execute_parallel(plans, *merge, ctx).await
            }
            PhysicalPlan::FullScan { scope, limit } => {
                FullScanOp::execute_scan(scope, *limit, ctx).await
            }
            PhysicalPlan::HashJoin { left, right } => {
                let l = left.execute(ctx).await?;
                let r = right.execute(ctx).await?;
                JoinOp::hash_join(l, r).await
            }
            PhysicalPlan::NestedLoopJoin { left, right } => {
                let l = left.execute(ctx).await?;
                let r = right.execute(ctx).await?;
                JoinOp::nested_loop(l, r).await
            }
        }
    }
}

// ===========================================================================
// 物理算子实现
// ===========================================================================

/// 按类型前缀扫描（快路径 1）。
pub struct TypeScanOp;

impl TypeScanOp {
    async fn execute_scan(
        content_type: &ContentType,
        scope: &Option<ScopeFilter>,
        limit: usize,
        ctx: &ExecContext,
    ) -> Result<RecordBatch> {
        let prefix = match scope {
            Some(ScopeFilter::UriPrefix(p)) => p.clone(),
            _ => format!("uwu://t/"),
        };
        let _dir_uri = ContextUri::parse(&prefix)?;
        let entries = ctx.fs.ls(&_dir_uri).await?;

        let hits: Vec<RetrievalHit> = entries
            .into_iter()
            .take(limit)
            .filter(|e| e.content_type == Some(*content_type))
            .filter_map(|e| {
                Some(RetrievalHit {
                    uri: e.uri,
                    level: ContentLevel::L0,
                    content: ContentPayload::Text {
                        sparse: e.abstract_,
                        dense: String::new(),
                        full: String::new(),
                    },
                    relevance: 0.9,
                    parent_chain: vec![],
                    content_type: e.content_type,
                })
            })
            .collect();

        Ok(RecordBatch {
            records: hits,
            stats: ExecStats::default(),
        })
    }
}

/// PG 前缀扫描。
pub struct PgPrefixScanOp;

impl PgPrefixScanOp {
    async fn execute_scan(
        _uri_prefix: &str,
        limit: usize,
        ctx: &ExecContext,
    ) -> Result<RecordBatch> {
        let dir_uri = ContextUri::parse(_uri_prefix)?;
        let entries = ctx.fs.ls(&dir_uri).await?;
        let hits: Vec<RetrievalHit> = entries
            .into_iter()
            .take(limit)
            .map(|e| RetrievalHit {
                uri: e.uri,
                level: ContentLevel::L0,
                content: ContentPayload::Text {
                    sparse: e.abstract_,
                    dense: String::new(),
                    full: String::new(),
                },
                relevance: 0.8,
                parent_chain: vec![],
                content_type: e.content_type,
            })
            .collect();
        Ok(RecordBatch {
            records: hits,
            stats: ExecStats::default(),
        })
    }
}

/// 向量搜索算子。
pub struct VectorSearchOp;

impl VectorSearchOp {
    async fn execute_search(
        embedding: &[f32],
        filter: &VectorFilter,
        limit: usize,
        ctx: &ExecContext,
    ) -> Result<RecordBatch> {
        let index = ctx.index.as_ref().ok_or_else(|| {
            agent_context_db_core::ContextError::Unsupported("no vector index".into())
        })?;

        // 构建 JSON filter（向量索引原生支持 payload 过滤）
        let filter_json = if filter.uri_prefix.is_some() || filter.content_type.is_some() {
            let mut f = serde_json::Map::new();
            if let Some(prefix) = &filter.uri_prefix {
                f.insert(
                    "uri_prefix".into(),
                    serde_json::Value::String(prefix.clone()),
                );
            }
            if let Some(ct) = &filter.content_type {
                f.insert(
                    "content_type".into(),
                    serde_json::Value::String(ct.as_path_segment().into()),
                );
            }
            Some(serde_json::Value::Object(f))
        } else {
            None
        };

        let collection = filter.uri_prefix.as_deref().unwrap_or("default");
        let index_hits = index
            .search(collection, embedding.to_vec(), limit, filter_json)
            .await?;

        let hits: Vec<RetrievalHit> = index_hits
            .into_iter()
            .map(|h| RetrievalHit {
                uri: h.uri,
                level: ContentLevel::L0,
                content: ContentPayload::Text {
                    sparse: String::new(),
                    dense: String::new(),
                    full: String::new(),
                },
                relevance: h.score,
                parent_chain: vec![],
                content_type: None,
            })
            .collect();

        Ok(RecordBatch {
            records: hits,
            stats: ExecStats::default(),
        })
    }
}

/// 过滤算子。
pub struct FilterOp;

impl FilterOp {
    async fn apply(batch: RecordBatch, predicate: &Predicate) -> Result<RecordBatch> {
        let filtered: Vec<RetrievalHit> = batch
            .records
            .into_iter()
            .filter(|hit| predicate_matches(hit, predicate))
            .collect();
        Ok(RecordBatch {
            records: filtered,
            stats: batch.stats,
        })
    }
}

fn predicate_matches(_hit: &RetrievalHit, predicate: &Predicate) -> bool {
    predicate.conditions.iter().all(|c| match c {
        crate::query::Condition::TypeEquals(ct) => _hit.content_type == Some(*ct),
        _ => true, // 其他条件在详细实现中处理
    })
}

/// 排序算子。
pub struct SortOp;

impl SortOp {
    async fn apply(mut batch: RecordBatch, key: crate::query::SortKey) -> Result<RecordBatch> {
        match key {
            crate::query::SortKey::Relevance => {
                batch
                    .records
                    .sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
            }
            crate::query::SortKey::Natural => {} // 保持原序
            _ => {
                batch
                    .records
                    .sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
            }
        }
        Ok(batch)
    }
}

/// 限制算子。
pub struct LimitOp;

impl LimitOp {
    async fn apply(mut batch: RecordBatch, budget: usize) -> Result<RecordBatch> {
        batch.records.truncate(budget);
        Ok(batch)
    }
}

/// 图遍历算子 — 使用 GraphStore 的关系图扩展检索结果。
pub struct GraphTraverseOp;

impl GraphTraverseOp {
    async fn execute_traverse(
        seeds: &[ContextUri],
        edges: &[crate::query::RelationKind],
        max_hops: usize,
        ctx: &ExecContext,
    ) -> Result<RecordBatch> {
        let graph = match &ctx.graph {
            Some(g) => g.clone(),
            None => {
                // 无图存储 → 返回 seeds 本身
                return Ok(RecordBatch {
                    records: seeds
                        .iter()
                        .map(|s| RetrievalHit {
                            uri: s.clone(),
                            level: ContentLevel::L0,
                            content: ContentPayload::Text {
                                sparse: String::new(),
                                dense: String::new(),
                                full: String::new(),
                            },
                            relevance: 0.5,
                            parent_chain: vec![],
                            content_type: None,
                        })
                        .collect(),
                    stats: ExecStats::default(),
                });
            }
        };

        // 将 RelationKind 转换为 GraphRelation
        let kinds: Vec<agent_context_db_core::GraphRelation> = edges
            .iter()
            .map(|e| match e {
                crate::query::RelationKind::EvolvedFrom => {
                    agent_context_db_core::GraphRelation::EvolvedFrom
                }
                crate::query::RelationKind::EvolvedTo => {
                    agent_context_db_core::GraphRelation::EvolvedTo
                }
                crate::query::RelationKind::EvidenceOf => {
                    agent_context_db_core::GraphRelation::EvidenceOf
                }
                crate::query::RelationKind::EntangledWith => {
                    agent_context_db_core::GraphRelation::EntangledWith
                }
                crate::query::RelationKind::Contradicts => {
                    agent_context_db_core::GraphRelation::Contradicts
                }
                crate::query::RelationKind::Corroborates => {
                    agent_context_db_core::GraphRelation::Corroborates
                }
                crate::query::RelationKind::DerivedFrom => {
                    agent_context_db_core::GraphRelation::DerivedFrom
                }
                crate::query::RelationKind::Supersedes => {
                    agent_context_db_core::GraphRelation::Supersedes
                }
                crate::query::RelationKind::DrivesPolicy => {
                    agent_context_db_core::GraphRelation::DrivesPolicy
                }
            })
            .collect();

        match graph.batch_traverse(seeds, &kinds, max_hops).await {
            Ok(edges_result) => {
                let hits: Vec<RetrievalHit> = edges_result
                    .into_iter()
                    .map(|(from, to, kind)| RetrievalHit {
                        uri: to.clone(),
                        level: ContentLevel::L0,
                        content: ContentPayload::Text {
                            sparse: format!("{:?} {}", kind, from),
                            dense: String::new(),
                            full: String::new(),
                        },
                        relevance: 0.6,
                        parent_chain: vec![from],
                        content_type: None,
                    })
                    .collect();

                let count = hits.len();
                Ok(RecordBatch {
                    records: hits,
                    stats: ExecStats {
                        rows_scanned: count,
                        ..Default::default()
                    },
                })
            }
            Err(e) => {
                tracing::warn!(error=%e, "graph traverse failed, returning seeds");
                Ok(RecordBatch {
                    records: vec![],
                    stats: ExecStats::default(),
                })
            }
        }
    }
}

/// 并行执行算子。
pub struct ParallelOp;

impl ParallelOp {
    async fn execute_parallel(
        plans: &[PhysicalPlan],
        merge: MergeStrategy,
        ctx: &ExecContext,
    ) -> Result<RecordBatch> {
        let mut handles = Vec::new();
        for plan in plans {
            let plan = plan.clone();
            let ctx = ctx.clone();
            handles.push(tokio::spawn(async move { plan.execute(&ctx).await }));
        }

        let mut all_hits = Vec::new();
        for handle in handles {
            if let Ok(Ok(batch)) = handle.await {
                all_hits.extend(batch.records);
            }
        }

        match merge {
            MergeStrategy::Dedup => {
                all_hits.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
                all_hits.dedup_by(|a, b| a.uri == b.uri);
            }
            MergeStrategy::Union => {
                all_hits.sort_by(|a, b| b.relevance.partial_cmp(&a.relevance).unwrap());
            }
            _ => {}
        }

        Ok(RecordBatch {
            records: all_hits,
            stats: ExecStats::default(),
        })
    }
}

/// 全表扫描算子（fallback）。
pub struct FullScanOp;

impl FullScanOp {
    async fn execute_scan(
        scope: &Option<ScopeFilter>,
        limit: usize,
        ctx: &ExecContext,
    ) -> Result<RecordBatch> {
        // 确定扫描范围
        let prefix = match scope {
            Some(ScopeFilter::UriPrefix(p)) => p.clone(),
            Some(ScopeFilter::Agent(a)) => format!("uwu://{a}"),
            Some(ScopeFilter::Tenant(t)) => format!("uwu://{t}"),
            None => "uwu://".to_string(),
        };

        // 用 FsOps::find 执行扫描
        let pattern = agent_context_db_core::FindPattern {
            scope: Some(
                ContextUri::parse(&prefix).unwrap_or_else(|_| ContextUri::parse("uwu://").unwrap()),
            ),
            ..Default::default()
        };

        match ctx.fs.find(&pattern).await {
            Ok(uris) => {
                let count = uris.len().min(limit);
                let hits: Vec<RetrievalHit> = uris
                    .into_iter()
                    .take(limit)
                    .map(|uri| RetrievalHit {
                        uri,
                        level: ContentLevel::L0,
                        content: ContentPayload::Text {
                            sparse: String::new(),
                            dense: String::new(),
                            full: String::new(),
                        },
                        relevance: 0.5,
                        parent_chain: vec![],
                        content_type: None,
                    })
                    .collect();

                Ok(RecordBatch {
                    records: hits,
                    stats: ExecStats {
                        rows_scanned: count,
                        ..Default::default()
                    },
                })
            }
            Err(e) => {
                tracing::warn!(error=%e, "full scan failed");
                Ok(RecordBatch {
                    records: vec![],
                    stats: ExecStats::default(),
                })
            }
        }
    }
}

/// 连接算子。
pub struct JoinOp;

impl JoinOp {
    async fn hash_join(l: RecordBatch, r: RecordBatch) -> Result<RecordBatch> {
        let mut merged = l.records;
        merged.extend(r.records);
        merged.dedup_by(|a, b| a.uri == b.uri);
        Ok(RecordBatch {
            records: merged,
            stats: ExecStats::default(),
        })
    }

    async fn nested_loop(l: RecordBatch, r: RecordBatch) -> Result<RecordBatch> {
        // 简单合并 + 去重
        JoinOp::hash_join(l, r).await
    }
}
