//! 联想检索 —— 沿 GraphStore 联想图 1-2 跳扩展检索命中，按跳数衰减权重。
//!
//! 与 `PhysicalPlan::GraphTraverse` 的区别：
//! - GraphTraverse 是**计划节点**，在物理计划里显式声明
//! - AssociativeExpander 是**后处理钩子**，对主计划的输出统一扩边，与 CBO 解耦

use crate::RetrievalHit;
use agent_context_db_core::{ContentLevel, ContentPayload, FsOps, GraphStore, Result};
use std::sync::Arc;

/// 联想扩展器 —— 在向量/类型召回基础上沿联想图扩展。
pub struct AssociativeExpander {
    fs: Arc<dyn FsOps>,
    graph: Arc<dyn GraphStore>,
    max_hops: usize,
    decay_factor: f32,
}

impl AssociativeExpander {
    pub fn new(fs: Arc<dyn FsOps>, graph: Arc<dyn GraphStore>) -> Self {
        Self {
            fs,
            graph,
            max_hops: 2,
            decay_factor: 0.5,
        }
    }

    pub fn with_max_hops(mut self, hops: usize) -> Self {
        self.max_hops = hops;
        self
    }

    pub fn with_decay(mut self, decay: f32) -> Self {
        self.decay_factor = decay.clamp(0.0, 1.0);
        self
    }

    /// 对现有命中做联想扩展，返回**新增**的命中（不含种子本身）。
    ///
    /// 每次 hop 相关性乘以 `decay_factor^hop`；低于 0.1 的邻居丢弃。
    /// 若 fs.read 失败，仍返回带空 payload 的命中（不阻塞检索）。
    pub async fn expand(&self, base_hits: &[RetrievalHit]) -> Result<Vec<RetrievalHit>> {
        let mut expanded = Vec::new();
        let mut seen: std::collections::HashSet<String> =
            base_hits.iter().map(|h| h.uri.to_string()).collect();

        for hit in base_hits {
            let neighbors = match self.graph.neighbors(&hit.uri, None).await {
                Ok(n) => n,
                Err(_) => continue,
            };
            for (hop, neighbor_uri) in neighbors.iter().enumerate() {
                if hop >= self.max_hops {
                    break;
                }
                if !seen.insert(neighbor_uri.to_string()) {
                    continue; // 避免重复
                }
                let weight = self.decay_factor.powi(hop as i32);
                if weight < 0.1 {
                    continue;
                }
                let payload = self
                    .fs
                    .read(neighbor_uri, ContentLevel::L0)
                    .await
                    .unwrap_or(ContentPayload::Text {
                        sparse: String::new(),
                        dense: String::new(),
                        full: String::new(),
                    });
                expanded.push(RetrievalHit {
                    uri: neighbor_uri.clone(),
                    level: ContentLevel::L0,
                    content: payload,
                    relevance: hit.relevance * weight,
                    parent_chain: vec![hit.uri.clone()],
                    content_type: None,
                    metadata: Default::default(),
                    created_at: None,
                    updated_at: None,
                });
            }
        }
        Ok(expanded)
    }
}
