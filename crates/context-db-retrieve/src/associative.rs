//! 联想检索 —— 沿 GraphStore 联想图 1-2 跳扩展检索命中，按跳数衰减权重。
//!
//! 与 `PhysicalPlan::GraphTraverse` 的区别：
//! - GraphTraverse 是**计划节点**，在物理计划里显式声明
//! - AssociativeExpander 是**后处理钩子**，对主计划的输出统一扩边，与 CBO 解耦

use crate::RetrievalHit;
use agent_context_db_core::{ContentLevel, ContentPayload, FsOps, GraphStore, Result};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct AssociativeConfig {
    pub max_hops: usize,
    pub decay_factor: f32,
    pub min_weight: f32,
}

impl Default for AssociativeConfig {
    fn default() -> Self {
        Self {
            max_hops: 2,
            decay_factor: 0.5,
            min_weight: 0.1,
        }
    }
}

impl AssociativeConfig {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.max_hops == 0 {
            return Err("max_hops must be non-zero".into());
        }
        if !self.decay_factor.is_finite() || !(0.0..=1.0).contains(&self.decay_factor) {
            return Err("decay_factor must be finite and in [0, 1]".into());
        }
        if !self.min_weight.is_finite() || !(0.0..=1.0).contains(&self.min_weight) {
            return Err("min_weight must be finite and in [0, 1]".into());
        }
        Ok(())
    }
}

/// 联想扩展器 —— 在向量/类型召回基础上沿联想图扩展。
pub struct AssociativeExpander {
    fs: Arc<dyn FsOps>,
    graph: Arc<dyn GraphStore>,
    config: AssociativeConfig,
}

impl AssociativeExpander {
    pub fn new(
        fs: Arc<dyn FsOps>,
        graph: Arc<dyn GraphStore>,
        config: AssociativeConfig,
    ) -> std::result::Result<Self, String> {
        config.validate()?;
        Ok(Self { fs, graph, config })
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
            let neighbors = match self.graph.outgoing_neighbors(&hit.uri, None).await {
                Ok(n) => n,
                Err(_) => continue,
            };
            for (hop, neighbor_uri) in neighbors.iter().enumerate() {
                if hop >= self.config.max_hops {
                    break;
                }
                if !seen.insert(neighbor_uri.to_string()) {
                    continue; // 避免重复
                }
                let weight = self.config.decay_factor.powi(hop as i32);
                if weight < self.config.min_weight {
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
