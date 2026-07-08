//! 物理算子 — PhysicalPlan 的执行器。
//!
//! 每个物理计划节点对应一个算子实现，通过 `PhysicalPlan::execute()` 分发。

use crate::RetrievalHit;
use crate::planner::{PhysicalPlan, ScopeFilter, VectorFilter};
use crate::query::{Condition, MergeStrategy, Predicate, Scope, SortKey};
use agent_context_db_core::{
    ContentLevel, ContentPayload, ContentStore, ContentType, ContextEntry, ContextError,
    ContextUri, FsOps, Result, VectorIndex,
};
use std::sync::Arc;
use std::time::Duration;

// ===========================================================================
// 执行上下文 + 结果批次
// ===========================================================================

/// 执行上下文 — 注入依赖。
#[derive(Clone)]
pub struct ExecContext {
    pub fs: Arc<dyn FsOps>,
    /// 写入/读取主内容端口。WQL 的 metadata 条件需要从这里读取完整条目。
    pub content: Option<Arc<dyn ContentStore>>,
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
                input,
                edges,
                max_hops,
            } => {
                let seeds = input.execute(ctx).await?;
                GraphTraverseOp::execute_traverse(seeds, edges, *max_hops, ctx).await
            }
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
        let prefix = prefix_for_scope(scope);
        let mut batch = scan_prefix(&prefix, limit, 0.9, ctx).await?;
        batch
            .records
            .retain(|hit| hit.content_type == Some(*content_type));
        batch.stats.rows_scanned = batch.records.len();
        Ok(batch)
    }
}

/// PG 前缀扫描。
pub struct PgPrefixScanOp;

impl PgPrefixScanOp {
    async fn execute_scan(
        uri_prefix: &str,
        limit: usize,
        ctx: &ExecContext,
    ) -> Result<RecordBatch> {
        scan_prefix(uri_prefix, limit, 0.8, ctx).await
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
                metadata: Default::default(),
                created_at: None,
                updated_at: None,
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

fn predicate_matches(hit: &RetrievalHit, predicate: &Predicate) -> bool {
    predicate
        .conditions
        .iter()
        .all(|condition| match condition {
            Condition::TypeEquals(ct) => hit.content_type == Some(*ct),
            Condition::ScopeEquals(scope) => scope_matches(&hit.uri, scope),
            Condition::TimeBetween(start, end) => hit
                .updated_at
                .or(hit.created_at)
                .map(|ts| ts >= *start && ts <= *end)
                .unwrap_or(false),
            Condition::TagsContains(tags) => tags
                .iter()
                .all(|tag| hit.metadata.tags.iter().any(|existing| existing == tag)),
            Condition::QualityAbove(min) => hit
                .metadata
                .quality_score
                .map(|score| score >= *min)
                .unwrap_or(false),
            Condition::ValidOnly => hit
                .metadata
                .validity
                .as_ref()
                .map(|validity| validity.invalidated_by.is_none() && validity.valid_until.is_none())
                .unwrap_or(true),
        })
}

/// 排序算子。
pub struct SortOp;

impl SortOp {
    async fn apply(mut batch: RecordBatch, key: SortKey) -> Result<RecordBatch> {
        match key {
            SortKey::Relevance => batch.records.sort_by(compare_relevance),
            SortKey::Recency => batch.records.sort_by(compare_recency),
            SortKey::Quality => batch.records.sort_by(compare_quality),
            SortKey::Natural => {}
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
        seed_batch: RecordBatch,
        edges: &[crate::query::RelationKind],
        max_hops: usize,
        ctx: &ExecContext,
    ) -> Result<RecordBatch> {
        let graph = match &ctx.graph {
            Some(g) => g.clone(),
            None => return Ok(seed_batch),
        };
        let seeds: Vec<ContextUri> = seed_batch
            .records
            .iter()
            .map(|hit| hit.uri.clone())
            .collect();

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

        match graph.batch_traverse(&seeds, &kinds, max_hops).await {
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
                        metadata: Default::default(),
                        created_at: None,
                        updated_at: None,
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

        let merged = match merge {
            MergeStrategy::Union => {
                all_hits.sort_by(compare_relevance);
                all_hits
            }
            MergeStrategy::Dedup => dedup_best(all_hits),
            MergeStrategy::Intersect => intersect_hits(all_hits, plans.len()),
            MergeStrategy::First => all_hits,
        };

        Ok(RecordBatch {
            records: merged,
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
        let prefix = prefix_for_scope(scope);
        scan_prefix(&prefix, limit, 0.5, ctx).await
    }
}

fn prefix_for_scope(scope: &Option<ScopeFilter>) -> String {
    match scope {
        Some(ScopeFilter::UriPrefix(prefix)) => prefix.clone(),
        Some(ScopeFilter::Agent(_)) => "uwu://".to_string(),
        Some(ScopeFilter::Tenant(tenant)) => format!("uwu://{tenant}"),
        None => "uwu://".to_string(),
    }
}

async fn scan_prefix(
    prefix: &str,
    limit: usize,
    relevance: f32,
    ctx: &ExecContext,
) -> Result<RecordBatch> {
    if let Some(content) = &ctx.content {
        let entries = content.scan_by_prefix(prefix, limit).await?;
        let rows_scanned = entries.len();
        return Ok(RecordBatch {
            records: entries
                .into_iter()
                .map(|entry| hit_from_entry(entry, ContentLevel::L0, relevance))
                .collect(),
            stats: ExecStats {
                rows_scanned,
                ..Default::default()
            },
        });
    }

    let scope = ContextUri::parse(prefix).map_err(|err| {
        ContextError::Unsupported(format!(
            "WQL prefix scan requires ContentStore for non-concrete prefix `{prefix}`: {err}"
        ))
    })?;
    let uris = ctx
        .fs
        .find(&agent_context_db_core::FindPattern {
            scope: Some(scope),
            ..Default::default()
        })
        .await?;
    let rows_scanned = uris.len().min(limit);
    let mut records = Vec::with_capacity(rows_scanned);
    for uri in uris.into_iter().take(limit) {
        let content = ctx.fs.read(&uri, ContentLevel::L0).await?;
        records.push(RetrievalHit {
            uri,
            level: ContentLevel::L0,
            content,
            relevance,
            parent_chain: Vec::new(),
            content_type: None,
            metadata: Default::default(),
            created_at: None,
            updated_at: None,
        });
    }
    Ok(RecordBatch {
        records,
        stats: ExecStats {
            rows_scanned,
            ..Default::default()
        },
    })
}

fn hit_from_entry(entry: ContextEntry, level: ContentLevel, relevance: f32) -> RetrievalHit {
    RetrievalHit {
        uri: entry.uri,
        level,
        content: entry.payload,
        relevance,
        parent_chain: Vec::new(),
        content_type: entry.metadata.content_type,
        metadata: entry.metadata,
        created_at: Some(entry.created_at),
        updated_at: Some(entry.updated_at),
    }
}

fn scope_matches(uri: &ContextUri, scope: &Scope) -> bool {
    match scope {
        Scope::All => true,
        Scope::Tenant(tenant) => uri.tenant() == tenant,
        Scope::Agent(agent) => {
            let segments = uri.segments();
            segments
                .windows(2)
                .any(|pair| pair[0] == "agent" && pair[1] == *agent)
        }
    }
}

fn compare_relevance(a: &RetrievalHit, b: &RetrievalHit) -> std::cmp::Ordering {
    b.relevance
        .partial_cmp(&a.relevance)
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn compare_recency(a: &RetrievalHit, b: &RetrievalHit) -> std::cmp::Ordering {
    b.updated_at
        .or(b.created_at)
        .cmp(&a.updated_at.or(a.created_at))
        .then_with(|| compare_relevance(a, b))
}

fn compare_quality(a: &RetrievalHit, b: &RetrievalHit) -> std::cmp::Ordering {
    b.metadata
        .quality_score
        .partial_cmp(&a.metadata.quality_score)
        .unwrap_or_else(|| compare_relevance(a, b))
}

fn dedup_best(mut hits: Vec<RetrievalHit>) -> Vec<RetrievalHit> {
    hits.sort_by(compare_relevance);
    hits.dedup_by(|a, b| a.uri == b.uri);
    hits
}

fn intersect_hits(hits: Vec<RetrievalHit>, required_count: usize) -> Vec<RetrievalHit> {
    use std::collections::HashMap;
    let mut grouped: HashMap<ContextUri, (usize, RetrievalHit)> = HashMap::new();
    for hit in hits {
        grouped
            .entry(hit.uri.clone())
            .and_modify(|(count, best)| {
                *count += 1;
                if hit.relevance > best.relevance {
                    *best = hit.clone();
                }
            })
            .or_insert((1, hit));
    }
    let mut out: Vec<_> = grouped
        .into_iter()
        .filter_map(|(_, (count, hit))| (count == required_count).then_some(hit))
        .collect();
    out.sort_by(compare_relevance);
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compiler::query_to_logical;
    use crate::planner::CboOptimizer;
    use crate::query::{Condition, MergeStrategy, Predicate, Query, SortKey};
    use agent_context_db_core::{ContentRepo, TenantId};
    use agent_context_db_testkit::MemoryContextStore;
    use std::sync::Arc;
    use uuid::Uuid;

    fn entry(uri: &str, text: &str, content_type: ContentType, quality: f32) -> ContextEntry {
        let mut entry =
            ContextEntry::new_text(ContextUri::parse(uri).unwrap(), TenantId(Uuid::nil()), text);
        entry.metadata.content_type = Some(content_type);
        entry.metadata.quality_score = Some(quality);
        entry.metadata.tags = vec!["p2".into()];
        entry
    }

    #[tokio::test]
    async fn wql_executes_filters_sort_and_composite_merge() {
        let store = Arc::new(MemoryContextStore::new());
        ContentRepo::write(
            store.as_ref(),
            entry(
                "uwu://t/agent/a/memories/fact/high",
                "high quality fact",
                ContentType::Fact,
                0.95,
            ),
        )
        .await
        .unwrap();
        ContentRepo::write(
            store.as_ref(),
            entry(
                "uwu://t/agent/a/memories/fact/low",
                "low quality fact",
                ContentType::Fact,
                0.40,
            ),
        )
        .await
        .unwrap();
        ContentRepo::write(
            store.as_ref(),
            entry(
                "uwu://t/agent/a/memories/error/e1",
                "error evidence",
                ContentType::Error,
                0.80,
            ),
        )
        .await
        .unwrap();

        let fact_query = Query::Find {
            scope: Some(ContextUri::parse("uwu://t/agent/a/memories").unwrap()),
            predicate: Predicate::new()
                .with(Condition::TypeEquals(ContentType::Fact))
                .with(Condition::QualityAbove(0.9)),
            budget: 10,
            order: SortKey::Quality,
            expand: None,
        };
        let error_query = Query::Find {
            scope: Some(ContextUri::parse("uwu://t/agent/a/memories").unwrap()),
            predicate: Predicate::new().with(Condition::TypeEquals(ContentType::Error)),
            budget: 10,
            order: SortKey::Natural,
            expand: None,
        };
        let composite = Query::Composite {
            queries: vec![fact_query, error_query],
            merge: MergeStrategy::Union,
        };

        let optimizer = CboOptimizer::new(Arc::new(crate::planner::StatisticsCollector::new()));
        let physical = optimizer.optimize(&query_to_logical(&composite));
        let ctx = ExecContext {
            fs: store.clone(),
            content: Some(store),
            index: None,
            graph: None,
        };
        let batch = physical.execute(&ctx).await.unwrap();
        let uris: Vec<_> = batch
            .records
            .iter()
            .map(|hit| hit.uri.to_string())
            .collect();

        assert_eq!(uris.len(), 2);
        assert!(uris.contains(&"uwu://t/agent/a/memories/fact/high".to_string()));
        assert!(uris.contains(&"uwu://t/agent/a/memories/error/e1".to_string()));
        assert!(!uris.contains(&"uwu://t/agent/a/memories/fact/low".to_string()));
        assert!(
            batch
                .records
                .iter()
                .all(|hit| hit.metadata.quality_score.is_some())
        );
    }
}
