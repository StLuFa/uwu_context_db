//! 三级查询计划 — URI 原生三轴分类法的查询入口。
//!
//! 核心思想（来自 context-db-consolidation-design.md §12.6）：
//! - **快路径 1（TypeScan）**：80% 的查询只需类型过滤，PG WHERE uri LIKE 前缀扫描。
//! - **快路径 2（SemanticWithType）**：语义近邻 + 类型内联过滤，向量 payload 过滤。
//! - **慢路径（FullTriaxis）**：三轴全用——向量 + 类型 + 关系图扩展，约 5% 的复杂查询。
//!
//! ## 用法
//! ```ignore
//! let req = RetrievalRequest { content_type: Some(ContentType::Fact), ..Default::default() };
//! let plan = QueryPlan::from_request(&req);
//! let hits = plan.execute(&executor).await?;
//! ```

use crate::relational_axis::RelKind;
use agent_context_db_core::{ContentType, ContextUri, FsOps, Result, VectorIndex};
use std::sync::Arc;

// ===========================================================================
// Scope / Lifecycle（与 core 保持一致的本地类型）
// ===========================================================================

/// 查询作用域。
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryScope {
    Agent(String),
    Tenant(String),
    All,
}

/// 内容生命周期过滤。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lifecycle {
    ValidOnly,
    All,
    InvalidatedOnly,
}

// ===========================================================================
// RetrievalRequest — 用户请求输入
// ===========================================================================

/// 三轴检索请求。
#[derive(Debug, Clone, Default)]
pub struct RetrievalRequest {
    /// 查询文本向量（None → 纯类型扫描）。
    pub query_embedding: Option<Vec<f32>>,
    /// 类型轴：精确类型过滤。
    pub content_type: Option<ContentType>,
    /// 作用域过滤。
    pub scope: Option<QueryScope>,
    /// 生命周期过滤（默认只取有效条目）。
    pub lifecycle: Option<Lifecycle>,
    /// 关系轴扩展（空 → 不走图）。
    pub expand_relations: Vec<RelKind>,
    /// 最大跳数（关系轴扩展用）。
    pub max_hops: Option<usize>,
    /// 返回条目数上限。
    pub limit: usize,
}

// ===========================================================================
// RetrievalHit — 单条命中
// ===========================================================================

/// 三轴检索命中。
#[derive(Debug, Clone)]
pub struct RetrievalHit {
    pub uri: ContextUri,
    /// 相关性分数（0.0–1.0）。
    pub relevance: f32,
    /// 内容摘要（L0 sparse text）。
    pub abstract_: String,
    /// 来源路径："type_scan" | "semantic" | "relational"。
    pub source: &'static str,
}

// ===========================================================================
// QueryPlan — 三级计划
// ===========================================================================

/// 三级查询计划。
#[derive(Debug, Clone)]
pub enum QueryPlan {
    /// 快路径 1：纯类型扫描（~80% 的查询）。
    /// 引擎：PG WHERE uri LIKE 'uwu://t/a/memory/{type}/%'
    TypeScan {
        content_type: ContentType,
        scope: Option<QueryScope>,
        lifecycle: Lifecycle,
        limit: usize,
    },

    /// 快路径 2：类型 + 语义向量（~15% 的查询）。
    /// 引擎：向量搜索 + payload 内联类型过滤
    SemanticWithType {
        query_embedding: Vec<f32>,
        content_type: ContentType,
        scope: Option<QueryScope>,
        lifecycle: Lifecycle,
        limit: usize,
    },

    /// 慢路径：三轴全用（~5% 的复杂查询）。
    /// 引擎：向量搜索 + payload 过滤 + 批量图查询
    FullTriaxis {
        query_embedding: Vec<f32>,
        content_type: Option<ContentType>,
        scope: Option<QueryScope>,
        lifecycle: Lifecycle,
        expand_relations: Vec<RelKind>,
        max_hops: usize,
        limit: usize,
    },
}

impl QueryPlan {
    /// 根据请求自动选择最优计划。
    ///
    /// 选择逻辑：
    /// - 无向量 + 无关系扩展 → TypeScan（要求 content_type 非空）
    /// - 有向量 + 有 content_type + 无关系扩展 → SemanticWithType
    /// - 任何带关系扩展的 → FullTriaxis
    /// - 有向量 + 无 content_type → FullTriaxis（无类型约束的语义搜索）
    pub fn from_request(req: &RetrievalRequest) -> Self {
        let lifecycle = req.lifecycle.unwrap_or(Lifecycle::ValidOnly);
        let limit = if req.limit == 0 { 20 } else { req.limit };
        match (
            req.query_embedding.as_ref(),
            req.expand_relations.is_empty(),
            req.content_type,
        ) {
            // 快路径 1：纯类型过滤
            (None, true, Some(content_type)) => QueryPlan::TypeScan {
                content_type,
                scope: req.scope.clone(),
                lifecycle,
                limit,
            },

            // 快路径 2：类型 + 语义，无关系扩展
            (Some(query_embedding), true, Some(content_type)) => QueryPlan::SemanticWithType {
                query_embedding: query_embedding.clone(),
                content_type,
                scope: req.scope.clone(),
                lifecycle,
                limit,
            },

            // 慢路径：任何带关系扩展的（或无类型约束的语义搜索）
            (query_embedding, _, content_type) => QueryPlan::FullTriaxis {
                query_embedding: query_embedding.cloned().unwrap_or_default(),
                content_type,
                scope: req.scope.clone(),
                lifecycle,
                expand_relations: req.expand_relations.clone(),
                max_hops: req.max_hops.unwrap_or(2),
                limit,
            },
        }
    }

    /// 返回该计划的估算相对成本（供上层调度参考）。
    pub fn estimated_cost(&self) -> f32 {
        match self {
            QueryPlan::TypeScan { limit, .. } => *limit as f32 * 0.001,
            QueryPlan::SemanticWithType { limit, .. } => *limit as f32 * 0.01,
            QueryPlan::FullTriaxis {
                limit,
                max_hops,
                expand_relations,
                ..
            } => {
                *limit as f32 * 0.01
                    + (expand_relations.len() as f32) * 2_f32.powi(*max_hops as i32) * 10.0
            }
        }
    }
}

// ===========================================================================
// QueryExecutor — 执行三级计划
// ===========================================================================

/// 三轴查询执行器。
pub struct QueryExecutor {
    fs: Arc<dyn FsOps>,
    vector_index: Option<Arc<dyn VectorIndex>>,
    /// 关系图存储（可选，FullTriaxis 路径需要）。
    graph: Option<Arc<dyn agent_context_db_core::GraphStore>>,
}

impl QueryExecutor {
    pub fn new(fs: Arc<dyn FsOps>) -> Self {
        Self {
            fs,
            vector_index: None,
            graph: None,
        }
    }

    pub fn with_vector_index(mut self, idx: Arc<dyn VectorIndex>) -> Self {
        self.vector_index = Some(idx);
        self
    }

    pub fn with_graph(mut self, graph: Arc<dyn agent_context_db_core::GraphStore>) -> Self {
        self.graph = Some(graph);
        self
    }

    /// 执行查询计划，返回命中列表。
    pub async fn execute(&self, plan: QueryPlan) -> Result<Vec<RetrievalHit>> {
        match plan {
            QueryPlan::TypeScan {
                content_type,
                scope,
                lifecycle,
                limit,
            } => {
                self.execute_type_scan(content_type, scope, lifecycle, limit)
                    .await
            }
            QueryPlan::SemanticWithType {
                query_embedding,
                content_type,
                scope,
                lifecycle,
                limit,
            } => {
                self.execute_semantic_with_type(
                    &query_embedding,
                    content_type,
                    scope,
                    lifecycle,
                    limit,
                )
                .await
            }
            QueryPlan::FullTriaxis {
                query_embedding,
                content_type,
                scope,
                lifecycle,
                expand_relations,
                max_hops,
                limit,
            } => {
                self.execute_full_triaxis(
                    &query_embedding,
                    content_type,
                    scope,
                    lifecycle,
                    &expand_relations,
                    max_hops,
                    limit,
                )
                .await
            }
        }
    }

    // -----------------------------------------------------------------------
    // 快路径 1 — 纯类型扫描（PG WHERE uri LIKE 前缀）
    // -----------------------------------------------------------------------

    async fn execute_type_scan(
        &self,
        content_type: ContentType,
        scope: Option<QueryScope>,
        lifecycle: Lifecycle,
        limit: usize,
    ) -> Result<Vec<RetrievalHit>> {
        let prefix = self.build_uri_prefix(Some(content_type), &scope);
        let dir_uri = ContextUri::parse(&prefix)?;
        let entries = self.fs.ls(&dir_uri).await?;

        let hits = entries
            .into_iter()
            .filter(|e| lifecycle_matches(e, lifecycle))
            .take(limit)
            .map(|e| RetrievalHit {
                uri: e.uri,
                relevance: 0.9,
                abstract_: e.abstract_,
                source: "type_scan",
            })
            .collect();

        Ok(hits)
    }

    // -----------------------------------------------------------------------
    // 快路径 2 — 语义 + 类型内联过滤
    // -----------------------------------------------------------------------

    async fn execute_semantic_with_type(
        &self,
        embedding: &[f32],
        content_type: ContentType,
        scope: Option<QueryScope>,
        lifecycle: Lifecycle,
        limit: usize,
    ) -> Result<Vec<RetrievalHit>> {
        let index = match &self.vector_index {
            Some(idx) => idx,
            None => {
                // 降级到类型扫描
                return self
                    .execute_type_scan(content_type, scope, lifecycle, limit)
                    .await;
            }
        };

        let raw_hits = index
            .search("default", embedding.to_vec(), limit * 2, None)
            .await?;

        let type_seg = content_type.as_path_segment();
        let hits = raw_hits
            .into_iter()
            .filter(|h| {
                // 类型轴内联过滤：URI 路径含 /memory/{type}/
                h.uri.as_str().contains(&format!("/memory/{type_seg}/"))
            })
            .filter_map(|h| {
                if !lifecycle_filter_by_uri(&h.uri, lifecycle) {
                    return None;
                }
                Some(RetrievalHit {
                    uri: h.uri,
                    relevance: h.score,
                    abstract_: String::new(),
                    source: "semantic",
                })
            })
            .take(limit)
            .collect();

        Ok(hits)
    }

    // -----------------------------------------------------------------------
    // 慢路径 — 三轴全用
    // -----------------------------------------------------------------------

    async fn execute_full_triaxis(
        &self,
        embedding: &[f32],
        content_type: Option<ContentType>,
        scope: Option<QueryScope>,
        lifecycle: Lifecycle,
        expand_relations: &[RelKind],
        max_hops: usize,
        limit: usize,
    ) -> Result<Vec<RetrievalHit>> {
        // 阶段 1：向量召回（或类型扫描）
        let mut hits: Vec<RetrievalHit> = if !embedding.is_empty() {
            if let Some(index) = &self.vector_index {
                let raw = index
                    .search("default", embedding.to_vec(), limit * 2, None)
                    .await?;
                // 可选类型过滤
                raw.into_iter()
                    .filter(|h| {
                        if let Some(ct) = content_type {
                            h.uri
                                .as_str()
                                .contains(&format!("/memory/{}/", ct.as_path_segment()))
                        } else {
                            true
                        }
                    })
                    .filter_map(|h| {
                        if !lifecycle_filter_by_uri(&h.uri, lifecycle) {
                            return None;
                        }
                        Some(RetrievalHit {
                            uri: h.uri,
                            relevance: h.score,
                            abstract_: String::new(),
                            source: "semantic",
                        })
                    })
                    .take(limit)
                    .collect()
            } else if let Some(ct) = content_type {
                self.execute_type_scan(ct, scope.clone(), lifecycle, limit)
                    .await?
            } else {
                vec![]
            }
        } else if let Some(ct) = content_type {
            self.execute_type_scan(ct, scope.clone(), lifecycle, limit)
                .await?
        } else {
            vec![]
        };

        // 阶段 2：关系轴批量扩展（避免 N+1）
        if !expand_relations.is_empty() {
            if let Some(graph) = &self.graph {
                let seed_uris: Vec<ContextUri> = hits.iter().map(|h| h.uri.clone()).collect();
                let graph_relations: Vec<agent_context_db_core::GraphRelation> =
                    expand_relations.iter().map(relkind_to_graph).collect();

                let edges = graph
                    .batch_traverse(&seed_uris, &graph_relations, max_hops)
                    .await?;
                let mut extra: Vec<RetrievalHit> = edges
                    .into_iter()
                    .map(|(from, to, kind)| RetrievalHit {
                        // 关系命中的相关性来自种子节点
                        relevance: hits
                            .iter()
                            .find(|h| h.uri == from)
                            .map(|h| h.relevance * 0.7)
                            .unwrap_or(0.4),
                        uri: to,
                        abstract_: format!("{:?}", kind),
                        source: "relational",
                    })
                    .collect();
                hits.append(&mut extra);
            }
        }

        // 阶段 3：去重 + 按相关性排序 + 截断
        hits.sort_by(|a, b| {
            b.relevance
                .partial_cmp(&a.relevance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.dedup_by(|a, b| a.uri == b.uri);
        hits.truncate(limit);

        Ok(hits)
    }

    // -----------------------------------------------------------------------
    // 工具函数
    // -----------------------------------------------------------------------

    /// 构建类型前缀 URI。
    fn build_uri_prefix(
        &self,
        content_type: Option<ContentType>,
        scope: &Option<QueryScope>,
    ) -> String {
        let agent_part = match scope {
            Some(QueryScope::Agent(a)) => a.clone(),
            Some(QueryScope::Tenant(t)) => format!("{t}/default"),
            _ => "default/default".to_string(),
        };
        match content_type {
            Some(ct) => format!("uwu://t/{agent_part}/memory/{}/", ct.as_path_segment()),
            None => format!("uwu://t/{agent_part}/"),
        }
    }
}

// ===========================================================================
// 工具函数
// ===========================================================================

fn lifecycle_matches(entry: &agent_context_db_core::DirEntry, lifecycle: Lifecycle) -> bool {
    match lifecycle {
        Lifecycle::ValidOnly => true, // DirEntry 不含有效期信息，交给存储层
        Lifecycle::All => true,
        Lifecycle::InvalidatedOnly => false,
    }
}

fn lifecycle_filter_by_uri(_uri: &ContextUri, _lifecycle: Lifecycle) -> bool {
    // 有效期过滤由存储层 / 向量索引 payload 处理；这里始终放行
    true
}

fn relkind_to_graph(k: &RelKind) -> agent_context_db_core::GraphRelation {
    use agent_context_db_core::GraphRelation::*;
    match k {
        RelKind::EvolvedFrom => EvolvedFrom,
        RelKind::EvolvedTo => EvolvedTo,
        RelKind::EvidenceOf => EvidenceOf,
        RelKind::EntangledWith => EntangledWith,
        RelKind::Contradicts => Contradicts,
        RelKind::Corroborates => Corroborates,
        RelKind::DerivedFrom => DerivedFrom,
        RelKind::Supersedes => Supersedes,
    }
}

// ===========================================================================
// 测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_req_type_only() -> RetrievalRequest {
        RetrievalRequest {
            content_type: Some(ContentType::Fact),
            limit: 10,
            ..Default::default()
        }
    }

    fn make_req_semantic() -> RetrievalRequest {
        RetrievalRequest {
            content_type: Some(ContentType::Error),
            query_embedding: Some(vec![0.1_f32; 8]),
            limit: 5,
            ..Default::default()
        }
    }

    fn make_req_relational() -> RetrievalRequest {
        RetrievalRequest {
            content_type: Some(ContentType::Fact),
            query_embedding: Some(vec![0.1_f32; 8]),
            expand_relations: vec![RelKind::EvidenceOf],
            limit: 10,
            ..Default::default()
        }
    }

    #[test]
    fn plan_routing_type_scan() {
        let plan = QueryPlan::from_request(&make_req_type_only());
        assert!(matches!(plan, QueryPlan::TypeScan { .. }));
    }

    #[test]
    fn plan_routing_semantic_with_type() {
        let plan = QueryPlan::from_request(&make_req_semantic());
        assert!(matches!(plan, QueryPlan::SemanticWithType { .. }));
    }

    #[test]
    fn plan_routing_full_triaxis() {
        let plan = QueryPlan::from_request(&make_req_relational());
        assert!(matches!(plan, QueryPlan::FullTriaxis { .. }));
    }

    #[test]
    fn type_scan_cheapest() {
        let ts = QueryPlan::from_request(&make_req_type_only());
        let st = QueryPlan::from_request(&make_req_semantic());
        let ft = QueryPlan::from_request(&make_req_relational());
        assert!(ts.estimated_cost() < st.estimated_cost());
        assert!(st.estimated_cost() < ft.estimated_cost());
    }
}
